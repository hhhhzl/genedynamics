"""
Stepping-stones scene generation.

The current layout uses two mostly regular support lanes, one per lateral side,
with random gap spacing along x. Stones are non-overlapping and lie flat on the
ground; river geometry is optional and disabled for the default layouts here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    # Axis-aligned safe regions [xmin, xmax, ymin, ymax]: merged start deck, merged goal deck (left+right lanes).
    support_platforms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))

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


MIN_LANE_X_SEP = 0.18  # smallest along-x spacing between two stone centers on the same lane


def _bbox_two_circles(
    c0: np.ndarray,
    r0: float,
    c1: np.ndarray,
    r1: float,
) -> Tuple[float, float, float, float]:
    x0, y0 = float(c0[0]), float(c0[1])
    x1, y1 = float(c1[0]), float(c1[1])
    return (
        min(x0 - r0, x1 - r1),
        max(x0 + r0, x1 + r1),
        min(y0 - r0, y1 - r1),
        max(y0 + r0, y1 + r1),
    )


def _bbox_union(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def _bbox_almost_equal(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    atol: float = 0.03,
) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= float(atol) for i in range(4))


def _lane_circles_to_platforms_and_mid_stones(
    lane_xy: np.ndarray,
    lane_radii: np.ndarray,
) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray, np.ndarray]:
    """
    Merge the first two and last two stepping disks on a lane into axis-aligned rectangles.
    Remaining indices become ordinary circular stones.
    """
    ce = np.asarray(lane_xy, dtype=np.float32).reshape(-1, 2)
    rad = np.asarray(lane_radii, dtype=np.float32).reshape(-1)
    n = int(ce.shape[0])
    platforms: List[Tuple[float, float, float, float]] = []
    if n <= 1:
        return [], ce, rad
    if n == 2:
        platforms.append(_bbox_two_circles(ce[0], float(rad[0]), ce[1], float(rad[1])))
        return platforms, np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    platforms.append(_bbox_two_circles(ce[0], float(rad[0]), ce[1], float(rad[1])))
    platforms.append(_bbox_two_circles(ce[-2], float(rad[-2]), ce[-1], float(rad[-1])))
    if n == 3:
        keep_ce = ce[1:2].copy()
        keep_r = rad[1:2].copy()
    else:
        keep_ce = ce[2 : n - 2].copy()
        keep_r = rad[2 : n - 2].copy()
    return platforms, keep_ce, keep_r


def _extend_lane_segment(
    xs: list,
    end_x: float,
    *,
    rng: np.random.Generator,
    gap_lo: float,
    gap_hi: float,
    max_center_dx: float,
) -> None:
    """Walk from xs[-1] to end_x; each new stone is at most ``max_center_dx`` ahead of the previous."""
    end_x = float(end_x)
    g_lo = float(max(0.06, min(gap_lo, gap_hi)))
    g_hi = float(max(g_lo + 0.01, min(gap_hi, max_center_dx)))

    while xs[-1] < end_x - 1e-6:
        rem = end_x - xs[-1]
        if rem <= max_center_dx + 1e-6:
            xs.append(round(end_x, 4))
            return
        hi = min(g_hi, max_center_dx, rem - MIN_LANE_X_SEP)
        lo = min(g_lo, hi - 1e-6)
        if hi < lo + 1e-6:
            step = float(max_center_dx)
        else:
            step = float(rng.uniform(lo, hi))
        step = float(np.clip(step, MIN_LANE_X_SEP, min(max_center_dx, rem)))
        if rem - step < MIN_LANE_X_SEP and rem > max_center_dx + 1e-6:
            step = float(max(MIN_LANE_X_SEP, rem - MIN_LANE_X_SEP))
        step = float(min(step, max_center_dx, rem))
        nxt = xs[-1] + step
        xs.append(round(float(nxt), 4))


def _build_lane_xs(
    rng: np.random.Generator,
    start_hind_x: float,
    start_front_x: float,
    goal_hind_x: float,
    goal_front_x: float,
    gap_range: Tuple[float, float],
    map_x: Tuple[float, float],
    *,
    max_center_dx: float,
) -> np.ndarray:
    x_min = float(map_x[0] + 0.05)
    x_max = float(map_x[1] - 0.05)
    goal_hind_x = float(np.clip(goal_hind_x, x_min, x_max))
    goal_front_x = float(np.clip(goal_front_x, x_min, x_max))
    gap_lo, gap_hi = float(gap_range[0]), float(gap_range[1])

    xs: list[float] = [float(start_hind_x), float(start_front_x)]
    _extend_lane_segment(xs, goal_hind_x, rng=rng, gap_lo=gap_lo, gap_hi=gap_hi, max_center_dx=max_center_dx)
    _extend_lane_segment(xs, goal_front_x, rng=rng, gap_lo=gap_lo, gap_hi=gap_hi, max_center_dx=max_center_dx)

    out = np.asarray(sorted(set(round(v, 4) for v in xs)), dtype=np.float32)
    return out


def sample_stepping_stones_scene(
    *,
    level: int,
    seed: int,
    l_max: float = 0.35,
    stance_width: float = 0.30,
    start_mid: Tuple[float, float] = (-1.25, 0.0),
    goal_mid: Tuple[float, float] = (1.25, 0.0),
    fore_hind_offset: float = 0.18,
    max_lane_center_dx: Optional[float] = None,
    lane_gap_l_ref: float = 0.35,
    lane_gap_scale: float = 1.0,
    lane_tail_margin: int = 4,
    k_horizon_cap: Optional[int] = None,
) -> SteppingStonesScene:
    profiles = default_difficulty_profiles()
    bucket = level_to_bucket(level)
    prof = profiles["easy"]
    rng = np.random.default_rng(int(seed))

    k_horizon = _sample_int(rng, prof.horizon_k_range)
    radius_lo, radius_hi = prof.radius_range
    l_ref = float(lane_gap_l_ref) if float(lane_gap_l_ref) > 1e-6 else 0.35
    lm = float(l_max)
    ls = float(max(0.5, min(float(lane_gap_scale), 2.0)))
    gap_lo = max(0.06, float(prof.lane_gap_range[0]) / l_ref * lm * ls)
    gap_hi = max(gap_lo + 0.02, float(prof.lane_gap_range[1]) / l_ref * lm * ls)
    # Cap along-lane spacing so consecutive stones stay within one-step reach (see l_max).
    # Override with obstacle_config.max_lane_center_dx to add body travel (e.g. l_max + vs*dt).
    if max_lane_center_dx is None:
        max_dx = lm
    else:
        max_dx = float(max_lane_center_dx)
    gap_hi = min(gap_hi, max_dx)
    gap_lo = min(gap_lo, gap_hi - 0.02)
    gap_range = (gap_lo, gap_hi)

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
        max_center_dx=max_dx,
    )
    right_xs = _build_lane_xs(
        rng=rng,
        start_hind_x=start_hind_x,
        start_front_x=start_front_x,
        goal_hind_x=goal_hind_x,
        goal_front_x=goal_front_x,
        gap_range=gap_range,
        map_x=prof.map_x,
        max_center_dx=max_dx,
    )

    left_y = start_mid_np[1] + half_stance + float(rng.normal(0.0, prof.lane_y_jitter))
    right_y = start_mid_np[1] - half_stance + float(rng.normal(0.0, prof.lane_y_jitter))
    left_centers = np.stack([left_xs, np.full_like(left_xs, left_y)], axis=1)
    right_centers = np.stack([right_xs, np.full_like(right_xs, right_y)], axis=1)
    nL = int(left_centers.shape[0])
    nR = int(right_centers.shape[0])
    centers_full = np.concatenate([left_centers, right_centers], axis=0).astype(np.float32)
    radii_full = rng.uniform(radius_lo, radius_hi, size=(centers_full.shape[0],)).astype(np.float32)

    lane_terminal_radius = radius_lo
    radii_full[:2] = np.maximum(radii_full[:2], lane_terminal_radius)
    radii_full[nL - 2 : nL] = np.maximum(radii_full[nL - 2 : nL], lane_terminal_radius)
    right_start = nL
    radii_full[right_start : right_start + 2] = np.maximum(radii_full[right_start : right_start + 2], lane_terminal_radius)
    radii_full[-2:] = np.maximum(radii_full[-2:], lane_terminal_radius)

    _, c_l, r_l = _lane_circles_to_platforms_and_mid_stones(left_centers, radii_full[:nL])
    _, c_r, r_r = _lane_circles_to_platforms_and_mid_stones(right_centers, radii_full[nL:])

    rL = radii_full[:nL]
    rR = radii_full[nL:]
    # One connected start deck and one connected goal deck (union left + right lane foot pairs).
    start_full = _bbox_union(
        _bbox_two_circles(left_centers[0], float(rL[0]), left_centers[1], float(rL[1])),
        _bbox_two_circles(right_centers[0], float(rR[0]), right_centers[1], float(rR[1])),
    )
    goal_full = _bbox_union(
        _bbox_two_circles(left_centers[nL - 2], float(rL[nL - 2]), left_centers[nL - 1], float(rL[nL - 1])),
        _bbox_two_circles(right_centers[nR - 2], float(rR[nR - 2]), right_centers[nR - 1], float(rR[nR - 1])),
    )
    if _bbox_almost_equal(start_full, goal_full):
        support_platforms = np.asarray([start_full], dtype=np.float32).reshape(1, 4)
    else:
        support_platforms = np.asarray([start_full, goal_full], dtype=np.float32).reshape(2, 4)
    _parts_c = [x for x in (c_l, c_r) if x.size > 0]
    _parts_r = [x for x in (r_l, r_r) if x.size > 0]
    centers = np.concatenate(_parts_c, axis=0).astype(np.float32) if _parts_c else np.zeros((0, 2), dtype=np.float32)
    radii = np.concatenate(_parts_r, axis=0).astype(np.float32) if _parts_r else np.zeros((0,), dtype=np.float32)
    radius_shrink = {"easy": 0.0, "medium": 0.004, "hard": 0.008}[bucket]
    if radius_shrink > 0.0 and radii.size > 0:
        radii = np.maximum(0.01, radii - float(radius_shrink)).astype(np.float32)

    lane_m = int(max(0, min(int(lane_tail_margin), 32)))
    k_horizon = max(int(k_horizon), int(max(len(left_xs), len(right_xs)) + lane_m))
    if k_horizon_cap is not None:
        k_horizon = min(int(k_horizon), int(k_horizon_cap))

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
        support_platforms=support_platforms,
    )
