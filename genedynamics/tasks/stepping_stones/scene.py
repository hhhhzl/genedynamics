"""
Stepping-stones scene generation.

The scene is sampled from difficulty buckets and is deterministic per seed.
It provides:
- river geometry (for rendering/metadata)
- circular stepping stones (safe footholds)
- start/goal biped stance
- task constants (K and step bound Lmax)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    num_stones_range: Tuple[int, int]
    radius_range: Tuple[float, float]
    center_gap_range: Tuple[float, float]
    horizon_k_range: Tuple[int, int]
    map_x: Tuple[float, float]
    map_y: Tuple[float, float]
    river_x: Tuple[float, float]
    lateral_noise_std: float


@dataclass(frozen=True)
class SteppingStonesScene:
    difficulty: str
    level: int
    map_x: Tuple[float, float]
    map_y: Tuple[float, float]
    river_x: Tuple[float, float]
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
            num_stones_range=(20, 28),
            radius_range=(0.16, 0.18),
            center_gap_range=(0.25, 0.32),
            horizon_k_range=(10, 12),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(-0.20, 0.20),
            lateral_noise_std=0.06,
        ),
        "medium": DifficultyProfile(
            name="medium",
            num_stones_range=(24, 32),
            radius_range=(0.14, 0.16),
            center_gap_range=(0.32, 0.39),
            horizon_k_range=(12, 14),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(-0.20, 0.20),
            lateral_noise_std=0.08,
        ),
        "hard": DifficultyProfile(
            name="hard",
            num_stones_range=(28, 40),
            radius_range=(0.12, 0.14),
            center_gap_range=(0.39, 0.45),
            horizon_k_range=(14, 16),
            map_x=(-1.6, 1.6),
            map_y=(-0.9, 0.9),
            river_x=(-0.20, 0.20),
            lateral_noise_std=0.10,
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


def _build_backbone_centers(
    rng: np.random.Generator,
    n_backbone: int,
    start_mid: np.ndarray,
    goal_mid: np.ndarray,
    gap_range: Tuple[float, float],
    map_x: Tuple[float, float],
    map_y: Tuple[float, float],
    lateral_noise_std: float,
) -> np.ndarray:
    centers = np.zeros((n_backbone, 2), dtype=np.float32)
    centers[0] = start_mid.astype(np.float32)
    centers[-1] = goal_mid.astype(np.float32)
    curr = start_mid.astype(np.float32).copy()
    remain_steps = max(1, n_backbone - 1)
    gap_lo, gap_hi = float(gap_range[0]), float(gap_range[1])
    for i in range(1, n_backbone - 1):
        remain_steps = n_backbone - i
        to_goal = goal_mid - curr
        dist_to_goal = float(np.linalg.norm(to_goal))
        if dist_to_goal < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = (to_goal / dist_to_goal).astype(np.float32)
        lateral = np.array([-direction[1], direction[0]], dtype=np.float32)
        # Keep progress toward the goal while preserving multimodality by lateral perturbation.
        step = min(
            gap_hi,
            max(
                gap_lo * 0.8,
                dist_to_goal / max(1, remain_steps),
            ),
        )
        noise = float(rng.normal(0.0, lateral_noise_std))
        proposal = curr + step * direction + noise * lateral
        proposal[0] = np.clip(proposal[0], map_x[0] + 0.08, map_x[1] - 0.08)
        proposal[1] = np.clip(proposal[1], map_y[0] + 0.08, map_y[1] - 0.08)
        centers[i] = proposal.astype(np.float32)
        curr = proposal.astype(np.float32)
    return centers


def sample_stepping_stones_scene(
    *,
    level: int,
    seed: int,
    l_max: float = 0.35,
    stance_width: float = 0.30,
    start_mid: Tuple[float, float] = (-1.25, 0.0),
    goal_mid: Tuple[float, float] = (1.25, 0.0),
) -> SteppingStonesScene:
    profiles = default_difficulty_profiles()
    bucket = level_to_bucket(level)
    prof = profiles[bucket]
    rng = np.random.default_rng(int(seed))

    n_stones = _sample_int(rng, prof.num_stones_range)
    k_horizon = _sample_int(rng, prof.horizon_k_range)
    radius_lo, radius_hi = prof.radius_range
    gap_range = prof.center_gap_range

    start_mid_np = np.asarray(start_mid, dtype=np.float32)
    goal_mid_np = np.asarray(goal_mid, dtype=np.float32)
    half_stance = 0.5 * float(stance_width)
    start_left = start_mid_np + np.array([0.0, half_stance], dtype=np.float32)
    start_right = start_mid_np + np.array([0.0, -half_stance], dtype=np.float32)
    goal_left = goal_mid_np + np.array([0.0, half_stance], dtype=np.float32)
    goal_right = goal_mid_np + np.array([0.0, -half_stance], dtype=np.float32)

    # Backbone stones ensure at least one plausible path.
    n_backbone = max(8, min(n_stones, k_horizon + 2))
    backbone = _build_backbone_centers(
        rng=rng,
        n_backbone=n_backbone,
        start_mid=start_mid_np,
        goal_mid=goal_mid_np,
        gap_range=gap_range,
        map_x=prof.map_x,
        map_y=prof.map_y,
        lateral_noise_std=prof.lateral_noise_std,
    )

    n_extra = max(0, n_stones - n_backbone)
    extras = np.zeros((n_extra, 2), dtype=np.float32)
    if n_extra > 0:
        extras[:, 0] = rng.uniform(prof.map_x[0] + 0.08, prof.map_x[1] - 0.08, size=(n_extra,)).astype(np.float32)
        extras[:, 1] = rng.uniform(prof.map_y[0] + 0.08, prof.map_y[1] - 0.08, size=(n_extra,)).astype(np.float32)

    centers = np.concatenate([backbone, extras], axis=0)
    radii = rng.uniform(radius_lo, radius_hi, size=(centers.shape[0],)).astype(np.float32)
    # Keep terminal footholds easier to hit.
    radii[0] = max(radii[0], min(radius_hi, radius_lo + 0.015))
    radii[n_backbone - 1] = max(radii[n_backbone - 1], min(radius_hi, radius_lo + 0.015))

    return SteppingStonesScene(
        difficulty=bucket,
        level=int(level),
        map_x=(float(prof.map_x[0]), float(prof.map_x[1])),
        map_y=(float(prof.map_y[0]), float(prof.map_y[1])),
        river_x=(float(prof.river_x[0]), float(prof.river_x[1])),
        stones_centers=centers.astype(np.float32),
        stones_radii=radii.astype(np.float32),
        start_left=start_left.astype(np.float32),
        start_right=start_right.astype(np.float32),
        goal_left=goal_left.astype(np.float32),
        goal_right=goal_right.astype(np.float32),
        k_horizon=int(k_horizon),
        l_max=float(l_max),
    )

