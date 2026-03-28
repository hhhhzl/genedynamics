"""
Stepping-stones scene generation.

The current layout uses two mostly regular support lanes, one per lateral side,
with random gap spacing along x. Stones are non-overlapping and lie flat on the
ground; river geometry is optional and disabled for the default layouts here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    radius_range: Tuple[float, float]
    lane_gap_range: Tuple[float, float]
    horizon_k_range: Tuple[int, int]
    map_x: Tuple[float, float]
    map_y: Tuple[float, float]
    river_x: Tuple[float, float]
    lane_y_jitter: float
    has_river: bool = False


@dataclass(frozen=True)
class SteppingStonesScene:
    difficulty: str
    level: int
    map_x: Tuple[float, float]
    map_y: Tuple[float, float]
    river_x: Tuple[float, float]
    has_river: bool
    stones_centers: np.ndarray  # (N, 2)
    stones_radii: np.ndarray  # (N,)
    start_left: np.ndarray  # (2,)
    start_right: np.ndarray  # (2,)
    goal_left: np.ndarray  # (2,)
    goal_right: np.ndarray  # (2,)
    k_horizon: int
    l_max: float

    @property
    def start_mid(self) -> np.ndarray:
        return 0.5 * (self.start_left + self.start_right)

    @property
    def goal_mid(self) -> np.ndarray:
        return 0.5 * (self.goal_left + self.goal_right)


def default_difficulty_profiles() -> Dict[str, DifficultyProfile]:
    return {
        "easy": DifficultyProfile(
            name="easy",
            radius_range=(0.076, 0.088),
            lane_gap_range=(0.22, 0.28),
            horizon_k_range=(12, 14),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(0.0, 0.0),
            lane_y_jitter=0.010,
            has_river=False,
        ),
        "medium": DifficultyProfile(
            name="medium",
            radius_range=(0.072, 0.084),
            lane_gap_range=(0.24, 0.30),
            horizon_k_range=(14, 16),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(0.0, 0.0),
            lane_y_jitter=0.012,
            has_river=False,
        ),
        "hard": DifficultyProfile(
            name="hard",
            radius_range=(0.068, 0.080),
            lane_gap_range=(0.26, 0.32),
            horizon_k_range=(16, 18),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(0.0, 0.0),
            lane_y_jitter=0.015,
            has_river=False,
        ),
    }


def level_to_bucket(level: int) -> str:
    if level <= 2:
        return "easy"
    if level <= 4:
        return "medium"
    return "hard"


def _sample_int(rng: np.random.Generator, lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi + 1))


def _sample_float(rng: np.random.Generator, lo_hi: Tuple[float, float]) -> float:
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi <= lo:
        return lo
    return float(rng.uniform(lo, hi))


def _build_lane_xs(
    rng: np.random.Generator,
    start_hind_x: float,
    start_front_x: float,
    goal_hind_x: float,
    goal_front_x: float,
    gap_range: Tuple[float, float],
    map_x: Tuple[float, float],
) -> np.ndarray:
    xs = [float(start_hind_x), float(start_front_x)]
    gap_lo, gap_hi = float(gap_range[0]), float(gap_range[1])
    curr = float(start_front_x)
    x_min = float(map_x[0] + 0.05)
    x_max = float(map_x[1] - 0.05)
    goal_hind_x = float(np.clip(goal_hind_x, x_min, x_max))
    goal_front_x = float(np.clip(goal_front_x, x_min, x_max))
    while curr < goal_hind_x - gap_hi:
        step = float(rng.uniform(gap_lo, gap_hi))
        curr = min(curr + step, goal_hind_x)
        if curr - xs[-1] >= 0.20:
            xs.append(float(curr))
    if goal_hind_x - xs[-1] >= 0.20:
        xs.append(goal_hind_x)
    else:
        xs[-1] = goal_hind_x
    if goal_front_x - xs[-1] >= 0.20:
        xs.append(goal_front_x)
    else:
        xs[-1] = goal_front_x
    xs = np.asarray(sorted(set(round(v, 4) for v in xs)), dtype=np.float32)
    return xs


def sample_stepping_stones_scene(
    *,
    level: int,
    seed: int,
    l_max: float = 0.35,
    stance_width: float = 0.30,
    start_mid: Tuple[float, float] = (-1.25, 0.0),
    goal_mid: Tuple[float, float] = (1.25, 0.0),
    fore_hind_offset: float = 0.18,
) -> SteppingStonesScene:
    profiles = default_difficulty_profiles()
    bucket = level_to_bucket(level)
    prof = profiles[bucket]
    rng = np.random.default_rng(int(seed))

    k_horizon = _sample_int(rng, prof.horizon_k_range)
    radius_lo, radius_hi = prof.radius_range
    gap_range = prof.lane_gap_range

    start_mid_np = np.asarray(start_mid, dtype=np.float32)
    goal_mid_np = np.asarray(goal_mid, dtype=np.float32)
    half_stance = 0.5 * float(stance_width)
    start_left = start_mid_np + np.array([0.0, half_stance], dtype=np.float32)
    start_right = start_mid_np + np.array([0.0, -half_stance], dtype=np.float32)
    goal_left = goal_mid_np + np.array([0.0, half_stance], dtype=np.float32)
    goal_right = goal_mid_np + np.array([0.0, -half_stance], dtype=np.float32)

    start_hind_x = float(start_mid_np[0] - fore_hind_offset)
    start_front_x = float(start_mid_np[0] + fore_hind_offset)
    goal_hind_x = float(goal_mid_np[0] - fore_hind_offset)
    goal_front_x = float(goal_mid_np[0] + fore_hind_offset)

    left_xs = _build_lane_xs(
        rng=rng,
        start_hind_x=start_hind_x,
        start_front_x=start_front_x,
        goal_hind_x=goal_hind_x,
        goal_front_x=goal_front_x,
        gap_range=gap_range,
        map_x=prof.map_x,
    )
    right_xs = _build_lane_xs(
        rng=rng,
        start_hind_x=start_hind_x,
        start_front_x=start_front_x,
        goal_hind_x=goal_hind_x,
        goal_front_x=goal_front_x,
        gap_range=gap_range,
        map_x=prof.map_x,
    )

    left_y = start_mid_np[1] + half_stance + float(rng.normal(0.0, prof.lane_y_jitter))
    right_y = start_mid_np[1] - half_stance + float(rng.normal(0.0, prof.lane_y_jitter))
    left_centers = np.stack([left_xs, np.full_like(left_xs, left_y)], axis=1)
    right_centers = np.stack([right_xs, np.full_like(right_xs, right_y)], axis=1)
    centers = np.concatenate([left_centers, right_centers], axis=0).astype(np.float32)
    radii = rng.uniform(radius_lo, radius_hi, size=(centers.shape[0],)).astype(np.float32)

    lane_terminal_radius = radius_lo
    radii[:2] = np.maximum(radii[:2], lane_terminal_radius)
    radii[len(left_xs) - 2 : len(left_xs)] = np.maximum(radii[len(left_xs) - 2 : len(left_xs)], lane_terminal_radius)
    right_start = len(left_xs)
    radii[right_start : right_start + 2] = np.maximum(radii[right_start : right_start + 2], lane_terminal_radius)
    radii[-2:] = np.maximum(radii[-2:], lane_terminal_radius)

    k_horizon = max(int(k_horizon), int(max(len(left_xs), len(right_xs)) + 4))

    return SteppingStonesScene(
        difficulty=bucket,
        level=int(level),
        map_x=(float(prof.map_x[0]), float(prof.map_x[1])),
        map_y=(float(prof.map_y[0]), float(prof.map_y[1])),
        river_x=(float(prof.river_x[0]), float(prof.river_x[1])),
        has_river=bool(prof.has_river),
        stones_centers=centers.astype(np.float32),
        stones_radii=radii.astype(np.float32),
        start_left=start_left.astype(np.float32),
        start_right=start_right.astype(np.float32),
        goal_left=goal_left.astype(np.float32),
        goal_right=goal_right.astype(np.float32),
        k_horizon=int(k_horizon),
        l_max=float(l_max),
    )

