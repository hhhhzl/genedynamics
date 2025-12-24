"""
EDOC experiment on Single Integrator 2D with obstacles.

This script mirrors `experiments/double2d/run_experiment.py` but for a
single-integrator (position-only) 2D system.

It:
1) Generates 11 levels of obstacle configurations (0..10)
2) Runs EDOC with soft→hard constraint scheduling + (optional) CFS projection
3) Visualizes diffusion steps, final trajectory, energy/reward, and states
4) Computes SSR (Safety Success Rate)
5) Saves all results to results/single2d/
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# Add repo root to path (so the script can be run directly)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from enerdynamics.envs.single_integrator_box_2d import SingleIntegratorBox2DEnv
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from enerdynamics.envs.obstacles.nonconvex import UnionObstacle

from enerdynamics.core.constraints import (
    HardConstraint,
    ConstraintManager,
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
    CFSProjection,
    ConstraintScheduleManager,
)
from enerdynamics.core.types import Trajectory
from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_energy
from enerdynamics.core.backends.runtime import RuntimeBackendManager

# Alias for 2D: SphereObstacle is CircleObstacle in 2D
CircleObstacle = SphereObstacle


# ============================================================================
# Constants
# ============================================================================

EDOC_COLOR = "#1f77b4"  # Blue
OBSTACLE_COLOR = "gray"
OBSTACLE_ALPHA = 0.5
DIFFUSION_FRACTIONS = (0.1, 0.5, 0.9)  # 90%, 50%, 10%
MAX_SAMPLE_TRAJ_PLOT = 80

ROBOT_RADIUS = 0.05  # Robot radius for collision checking and visualization
MIN_OBSTACLE_MARGIN = 2.4 * ROBOT_RADIUS
OBSTACLE_RADIUS_SCALE = 1.3

# Experiment "map" bounds (requested)
MAP_X_MIN, MAP_X_MAX = -1.5, 1.0
MAP_Y_MIN, MAP_Y_MAX = -2.0, 0.5

# ============================================================================
# Action (speed) Constraint
# ============================================================================

class SpeedConstraint(HardConstraint):
    """
    Hard constraint on the single-integrator action (interpreted as velocity).

    IMPORTANT: The environment clips actions component-wise with `control_limit`,
    i.e. an L-infinity (box) bound. Therefore this constraint enforces:

        max_i |u_i| <= u_max

    rather than an L2 ball. This keeps SSR and hard projection semantics
    consistent with `SingleIntegratorBox2DEnv`.
    """

    def __init__(self, u_max: float = 1.0):
        self.u_max = float(u_max)

    def is_feasible(self, trajectory: Trajectory) -> bool:
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            if float(np.max(np.abs(a))) > self.u_max + 1e-8:
                return False
        return True

    def violations(self, trajectory: Trajectory) -> np.ndarray:
        violations = []
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            violations.append(max(0.0, float(np.max(np.abs(a)) - self.u_max)))
        return np.asarray(violations, dtype=np.float32)

    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Trajectory:
        projected_actions = []
        for action in trajectory.actions:
            a = np.asarray(action, dtype=np.float32)
            a = np.clip(a, -self.u_max, self.u_max)
            projected_actions.append(a)
        return Trajectory(states=trajectory.states, actions=projected_actions)


# ============================================================================
# Obstacle Configuration Generator (copied/adapted from double2d)
# ============================================================================

def _check_obstacle_spacing(
    new_center: np.ndarray,
    new_radius: float,
    existing_obstacles: List[Tuple[np.ndarray, float]],
    margin: float,
) -> bool:
    for existing_center, existing_radius in existing_obstacles:
        distance = np.linalg.norm(new_center - existing_center)
        min_distance = new_radius + existing_radius + margin
        if distance < min_distance:
            return False
    return True


def _get_obstacle_radius(obstacle) -> float:
    if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
        return float(obstacle.radius)
    if isinstance(obstacle, BoxObstacle):
        # Use circumscribed radius (center -> corner) so spacing is conservative.
        he = np.asarray(obstacle.half_extents, dtype=np.float32).reshape(-1)
        he = he[:2] if he.size >= 2 else np.array([float(he[0]), float(he[0])], dtype=np.float32)
        return float(np.linalg.norm(he))
    if isinstance(obstacle, UnionObstacle):
        max_r = 0.0
        for prim in obstacle.obstacles:
            max_r = max(max_r, _get_obstacle_radius(prim))
        return max_r
    return 0.0


def _check_start_target_clearance(
    obstacle,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    buffer: float = ROBOT_RADIUS,
) -> bool:
    buf = float(buffer)
    for p in (start_pos, target_pos):
        p = np.asarray(p, dtype=np.float32)
        sdf = obstacle.sdf(p)
        sdf = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
        if sdf < buf:
            return False
        try:
            if hasattr(obstacle, "contains") and obstacle.contains(p):
                return False
        except Exception:
            pass
    return True


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to segment a-b (2D)."""
    p = np.asarray(p, dtype=np.float32).reshape(2)
    a = np.asarray(a, dtype=np.float32).reshape(2)
    b = np.asarray(b, dtype=np.float32).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _place_union_obstacle(
    *,
    union_idx: int,
    primitives_per_union: int,
    base_radius_scale: float,
    center_region: np.ndarray,
    region_size: float,
    p_max: float,
    margin: float,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    existing: List[Tuple[np.ndarray, float]],
    max_attempts: int = 400,
    offset_range: float = 0.3,
    corridor_width: float = 0.35,
    region_low_override: Optional[np.ndarray] = None,
    region_high_override: Optional[np.ndarray] = None,
    bias_to_path: bool = False,
    path_bias_sigma: float = 0.18,
    path_perp_range: float = 0.35,
    sep_min_mult: float = 1.05,
    sep_max_mult: float = 1.45,
) -> Optional[Tuple[UnionObstacle, np.ndarray, float]]:
    """
    Robustly place a non-convex UnionObstacle.

    Key fix vs the previous implementation:
    - Sample primitive offsets + sizes first, compute feasible base_center bounds,
      then sample base_center *within* those bounds (instead of clipping later).
    - Keep primitive geometry consistent with the chosen base_center.
    """
    # Union primitive sizes should match levels <=6.
    # We do NOT inflate union primitive sizes beyond OBSTACLE_RADIUS_SCALE.
    base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE * float(base_radius_scale)
    center_region = np.asarray(center_region, dtype=np.float32)
    region_half = float(region_size) / 2.0
    region_low = center_region - region_half
    region_high = center_region + region_half

    for _ in range(int(max_attempts)):
        # 1) Sample primitive geometry in union-local coordinates.
        # For primitives_per_union==2 we sample a *controlled separation* so the union is non-convex
        # while keeping the overall footprint comparable to a single level<=6 obstacle.
        prim_specs: List[Tuple[str, float]] = []
        extents: List[float] = []

        for prim_idx in range(primitives_per_union):
            if np.random.rand() < 0.6:
                r = float(np.random.uniform(base_radius * 0.8, base_radius * 1.5))
                prim_specs.append(("circle", r))
                extents.append(r)
            else:
                size = float(np.random.uniform(base_radius * 0.8, base_radius * 1.5))
                prim_specs.append(("box", size))
                extents.append(size)

        extents = np.asarray(extents, dtype=np.float32)  # (K,)

        if primitives_per_union == 2:
            # Choose a direction and separation proportional to primitive size to induce concavity.
            theta = float(np.random.uniform(0.0, 2.0 * np.pi))
            dvec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            # For equal radii R, union becomes non-convex roughly when center distance > R.
            rmax = float(np.max(extents))
            lo = float(max(1.01, float(sep_min_mult))) * rmax
            hi = float(max(lo + 1e-6, float(sep_max_mult))) * rmax
            sep = float(np.random.uniform(lo, hi))
            offsets = np.stack([-0.5 * sep * dvec, 0.5 * sep * dvec], axis=0).astype(np.float32)
        else:
            offsets = np.random.uniform(-offset_range, offset_range, size=(primitives_per_union, 2)).astype(np.float32)

        # Union bounding radius for spacing checks
        union_radius = float(np.max(np.linalg.norm(offsets, axis=1) + extents))

        # 2) Compute feasible base_center bounds so that all primitives stay inside box
        # Require: |base + offset| <= p_max - (extent + margin)
        ext2 = (extents + float(margin))[:, None]  # (K,1) for broadcasting with (K,2)
        per_prim_low = -float(p_max) + ext2 - offsets  # (K,2)
        per_prim_high = float(p_max) - ext2 - offsets  # (K,2)
        low = np.max(per_prim_low, axis=0)
        high = np.min(per_prim_high, axis=0)

        # Also restrict to the intended region (optionally overridden to a rectangle).
        if region_low_override is not None and region_high_override is not None:
            rlo = np.asarray(region_low_override, dtype=np.float32).reshape(2)
            rhi = np.asarray(region_high_override, dtype=np.float32).reshape(2)
            low = np.maximum(low, rlo)
            high = np.minimum(high, rhi)
        else:
            low = np.maximum(low, region_low)
            high = np.minimum(high, region_high)

        if np.any(low > high):
            continue

        # 2.4) Sample base_center, optionally biased near the start->target path ("on the road").
        if bias_to_path:
            a = np.asarray(start_pos, dtype=np.float32).reshape(2)
            b = np.asarray(target_pos, dtype=np.float32).reshape(2)
            ab = b - a
            dn2 = float(np.dot(ab, ab))
            if dn2 > 1e-12:
                dn = float(np.sqrt(dn2))
                dir_u = ab / dn
                perp = np.array([-dir_u[1], dir_u[0]], dtype=np.float32)
                # Prefer centers along the segment with perpendicular jitter (slalom-style).
                t = float(np.random.uniform(0.05, 0.98))
                p_line = a + t * ab
                p_pref = p_line + perp * float(np.random.uniform(-path_perp_range, path_perp_range))

                # Truncated Gaussian around p_pref inside [low, high]
                base_center = None
                for _k in range(4):
                    cand = (p_pref + np.random.normal(scale=float(path_bias_sigma), size=(2,)).astype(np.float32))
                    if np.all(cand >= low) and np.all(cand <= high):
                        base_center = cand.astype(np.float32)
                        break
                if base_center is None:
                    base_center = np.clip(p_pref, low, high).astype(np.float32)
            else:
                base_center = np.random.uniform(low, high).astype(np.float32)
        else:
            base_center = np.random.uniform(low, high).astype(np.float32)

        # 2.5) Optional corridor keep-out around the start->target segment.
        if corridor_width is not None and corridor_width > 0.0:
            d_seg = _point_segment_distance(base_center, start_pos, target_pos)
            if d_seg < float(corridor_width) + float(union_radius) * 0.35:
                continue

        # 3) Spacing check using union bounding radius
        if not _check_obstacle_spacing(base_center, union_radius, existing, margin):
            continue

        # 4) Build primitives with the sampled base_center
        primitives = []
        for prim_idx, (kind, size) in enumerate(prim_specs):
            center = base_center + offsets[prim_idx]
            if kind == "circle":
                prim = SphereObstacle(center=center, radius=size, name=f"union_{union_idx}_circle_{prim_idx}")
            else:
                half_extents = np.array([size, size], dtype=np.float32)
                prim = BoxObstacle(center=center, half_extents=half_extents, name=f"union_{union_idx}_box_{prim_idx}")
            primitives.append(prim)

        union = UnionObstacle(obstacles=primitives, name=f"union_{union_idx}")

        # 5) Hard guarantee: start/target cannot be in (or too close to) obstacles
        if not _check_start_target_clearance(union, start_pos, target_pos, buffer=ROBOT_RADIUS):
            continue

        return union, base_center, union_radius

    return None


def generate_obstacle_config(
    level: int,
    seed: int,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
) -> ObstacleManager:
    np.random.seed(seed)
    manager = ObstacleManager()

    if level == 0:
        return manager

    p_max = 2.0
    margin = MIN_OBSTACLE_MARGIN
    # Prefer placing obstacles in this sub-region (soft preference via sampling/clipping).
    # Requested: x ∈ [-1.5, 1], y ∈ [-2, 0.5]
    pref_low = np.array([-1.5, -2.0], dtype=np.float32)
    pref_high = np.array([1.0, 0.5], dtype=np.float32)
    pref_low = np.maximum(pref_low, -p_max)
    pref_high = np.minimum(pref_high, p_max)

    direction = target_pos - start_pos
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        center_region = np.array([0.0, 0.0], dtype=np.float32)
        region_size = p_max * 0.8
    else:
        center_region = (start_pos + target_pos) / 2.0
        region_size = direction_norm * 0.8 + 0.5

    existing: List[Tuple[np.ndarray, float]] = []

    if level <= 3:
        num_obstacles = {1: 3, 2: 6, 3: 10}[level]
        max_attempts = 100
        for i in range(num_obstacles):
            base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE
            radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)

            placed = False
            for _ in range(max_attempts):
                offset = np.array(
                    [
                        np.random.uniform(-region_size / 2, region_size / 2),
                        np.random.uniform(-region_size / 2, region_size / 2),
                    ],
                    dtype=np.float32,
                )
                center = np.clip(
                    center_region + offset,
                    np.maximum(-p_max, pref_low) + radius + margin + ROBOT_RADIUS,
                    np.minimum(p_max, pref_high) - radius - margin - ROBOT_RADIUS,
                )
                if _check_obstacle_spacing(center, radius, existing, margin):
                    if np.random.rand() < 0.5:
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    else:
                        size = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=size, name=f"box_{i}")

                    if not _check_start_target_clearance(obstacle, start_pos, target_pos, buffer=ROBOT_RADIUS):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius + ROBOT_RADIUS))
                    placed = True
                    break
            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    elif level <= 6:
        num_obstacles = {4: 14, 5: 18, 6: 22}[level]
        max_attempts = 100
        for i in range(num_obstacles):
            obstacle_type = np.random.choice(["circle", "box", "small_circle"])
            base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE

            if obstacle_type == "circle":
                radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
            elif obstacle_type == "box":
                size = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
                radius = size
            else:
                radius = np.random.uniform(base_radius * 0.5, base_radius * 1.0)

            placed = False
            for _ in range(max_attempts):
                offset = np.array(
                    [
                        np.random.uniform(-region_size / 2, region_size / 2),
                        np.random.uniform(-region_size / 2, region_size / 2),
                    ],
                    dtype=np.float32,
                )
                center = np.clip(
                    center_region + offset,
                    np.maximum(-p_max, pref_low) + radius + margin + ROBOT_RADIUS,
                    np.minimum(p_max, pref_high) - radius - margin - ROBOT_RADIUS,
                )
                if _check_obstacle_spacing(center, radius, existing, margin):
                    if obstacle_type == "circle":
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    elif obstacle_type == "box":
                        half_extents = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=half_extents, name=f"box_{i}")
                    else:
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"small_circle_{i}")

                    if not _check_start_target_clearance(obstacle, start_pos, target_pos, buffer=ROBOT_RADIUS):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius + ROBOT_RADIUS))
                    placed = True
                    break
            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    else:
        # Level>=7: keep union primitive sizes comparable to level<=6 (OBSTACLE_RADIUS_SCALE=1.3),
        # and increase difficulty mainly by adding *more union obstacles* (not inflating each union).
        #
        # The user's intent: level 9 should have ~30 unions, each "about the same size" as level<=6 obstacles.
        num_unions = {7: 18, 8: 24, 9: 30, 10: 36}[level]
        primitives_per_union = 2  # small non-convex union footprint

        size_scale0 = 1.0
        # Small within-union spread so each union stays roughly "level6-sized"
        offset0 = 0.16
        region_size_union = float(min(2.0 * p_max, float(region_size)))
        # Stronger preference for unions: use the requested rectangular area.
        # Represent it as a square for the union sampler.
        # Also bias unions to be closer together by starting with a smaller region and expanding if needed.
        # Put the union region near the start->target corridor (centered on the path).
        dir_vec = np.asarray(target_pos - start_pos, dtype=np.float32)
        dn = float(np.linalg.norm(dir_vec))
        mid = ((np.asarray(start_pos, dtype=np.float32) + np.asarray(target_pos, dtype=np.float32)) / 2.0).astype(np.float32)
        center_region_union = mid if dn > 1e-6 else ((pref_low + pref_high) / 2.0).astype(np.float32)
        center_region_union = np.clip(center_region_union, pref_low + 0.25, pref_high - 0.25).astype(np.float32)

        # Base region size is the preferred box width; effective region will be smaller early to cluster unions.
        base_region_size_union = float(min(2.0 * p_max, float(pref_high[0] - pref_low[0])))
        region_size_union = float(min(region_size_union, base_region_size_union))
        # Allow unions to pack more tightly (we still account for ROBOT_RADIUS separately).
        margin_union = float(max(0.00, float(margin) * 0.30))
        spacing_shrink = 0.75  # allow unions to be closer than their full bounding radius

        existing_before = list(existing)
        added_unions: List[UnionObstacle] = []

        placed_all = False
        last_connect_ok: Optional[bool] = None
        last_nonconv_sdf_raw: Optional[float] = None

        # Enforce monotone + >0.1 nonconvexity_sdf for levels 7-9 via absolute targets.
        nonconv_targets = {7: 0.11, 8: 0.125, 9: 0.14}
        target_nonconv = nonconv_targets.get(int(level), None)

        # Outer loop: cluster unions first; if placement fails or nonconv target not met,
        # expand region / adjust separation to increase nonconvexity.
        cal_max = 16 if (target_nonconv is not None) else 10
        for cal in range(int(cal_max)):
            np.random.seed(int(seed) + 10000 * (cal + 1))
            # Start clustered; expand slowly across cal attempts to help placement if needed.
            region_mult = float(min(1.0, 0.55 + 0.06 * float(cal)))
            region_size_eff = float(max(0.9, float(region_size_union) * region_mult))
            for _round in range(12):
                # Reset unions placed in previous round
                for u in added_unions:
                    try:
                        manager.remove(u)
                    except Exception:
                        pass
                added_unions = []
                existing = list(existing_before)

                # Keep primitive sizes fixed; only adjust within-union spread to help packing.
                size_scale = float(size_scale0)
                offset_range = float(np.clip(offset0 * (0.95 ** _round), 0.08, 0.22))

                failed = False
                for union_idx in range(num_unions):
                    # Increase 2-primitive separation with level (and with retries) to increase SDF nonconvexity.
                    lvl = int(level)
                    base_sep_min = {7: 1.10, 8: 1.22, 9: 1.35, 10: 1.35}.get(lvl, 1.10)
                    base_sep_max = {7: 1.60, 8: 1.80, 9: 2.00, 10: 2.10}.get(lvl, 1.60)
                    sep_min = float(base_sep_min + 0.03 * float(cal))
                    sep_max = float(base_sep_max + 0.05 * float(cal))
                    placed = _place_union_obstacle(
                        union_idx=union_idx,
                        primitives_per_union=primitives_per_union,
                        base_radius_scale=size_scale,
                        center_region=center_region_union,
                        region_size=region_size_eff,
                        p_max=float(p_max),
                        margin=float(margin_union),
                        start_pos=start_pos,
                        target_pos=target_pos,
                        existing=existing,
                        max_attempts=900,
                        offset_range=offset_range,
                        # We WANT unions near the path; BFS connectivity check will prevent full blockage.
                        corridor_width=0.0,
                        # Hard preference: keep union centers inside requested rectangle.
                        region_low_override=pref_low,
                        region_high_override=pref_high,
                        # Mix: many unions near the path, but some spread across the whole rectangle
                        # to "fill" x∈[-1.5,1] as requested.
                        bias_to_path=(union_idx % 5 != 0),
                        path_bias_sigma=0.24,
                        path_perp_range=0.95,
                        sep_min_mult=sep_min,
                        sep_max_mult=sep_max,
                    )
                    if placed is None:
                        failed = True
                        break
                    union, base_center, union_radius = placed
                    manager.add(union)
                    added_unions.append(union)
                    existing.append((base_center, float(union_radius) * float(spacing_shrink) + ROBOT_RADIUS))

                if not failed:
                    # Connectivity gate
                    last_connect_ok = has_free_space_path(
                        manager,
                        start=start_pos,
                        target=target_pos,
                        p_max=float(p_max),
                        robot_radius=float(ROBOT_RADIUS),
                        grid_res=0.06 if level >= 9 else 0.05,
                    )
                    if not last_connect_ok:
                        failed = True

                if not failed and target_nonconv is not None:
                    # Nonconvexity gate (fast estimate)
                    last_nonconv_sdf_raw = float(
                        compute_nonconvexity_score_sdf(
                            manager,
                            p_max=float(p_max),
                            robot_radius=float(ROBOT_RADIUS),
                            seed=int(seed),
                            n_pairs=6000,
                            x_min=float(MAP_X_MIN),
                            x_max=float(MAP_X_MAX),
                            y_min=float(MAP_Y_MIN),
                            y_max=float(MAP_Y_MAX),
                        ).get("score_raw", 0.0)
                    )
                    if last_nonconv_sdf_raw < float(target_nonconv):
                        failed = True

                if not failed:
                    placed_all = True
                    break

            if placed_all:
                break

        if not placed_all:
            # Keep whatever was placed in the last round, but warn accurately.
            missing = num_unions - len(added_unions)
            if missing > 0:
                print(f"Warning: Could not place {missing}/{num_unions} union(s) with proper spacing after retries")
            else:
                if target_nonconv is not None:
                    n_msg = "n/a" if last_nonconv_sdf_raw is None else f"{last_nonconv_sdf_raw:.4f}"
                    print(
                        "Warning: placed all unions but gates failed in retries "
                        f"(connect_ok={last_connect_ok}, nonconv_target={target_nonconv:.3f}, got={n_msg})."
                    )
                else:
                    print(f"Warning: placed all unions but connectivity gate failed in retries (last_connect_ok={last_connect_ok}).")

    # Final sanity: start must be safe
    if len(manager) > 0:
        sdf_start = manager.sdf(np.asarray(start_pos, dtype=np.float32))
        sdf_start = float(np.asarray(sdf_start).item() if hasattr(sdf_start, "item") else sdf_start)
        if sdf_start < ROBOT_RADIUS or manager.contains(np.asarray(start_pos, dtype=np.float32)):
            raise RuntimeError(
                f"Start position is in/too close to an obstacle (sdf={sdf_start:.6f}). "
                "Obstacle generation should prevent this."
            )

    return manager


def generate_optimized_start_position(level: int, seed: int, target: np.ndarray, p_max: float) -> np.ndarray:
    np.random.seed(seed)
    target = np.asarray(target, dtype=np.float32)

    max_distance = 2.0 * np.sqrt(2.0)
    min_dist = 0.5 + (level / 10.0) * 1.0
    max_dist = 1.0 + (level / 10.0) * (max_distance - 1.0)
    distance = np.random.uniform(min_dist, max_dist)

    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    start = target + distance * direction

    margin = 0.2
    start = np.clip(start, -p_max + margin, p_max - margin)
    return start


# ============================================================================
# Visualization
# ============================================================================

def draw_obstacles(ax: plt.Axes, obstacles: ObstacleManager):
    for obstacle in obstacles:
        if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
            center = np.asarray(obstacle.center, dtype=np.float32)[:2]
            radius = float(obstacle.radius)
            circle = plt.Circle(
                tuple(center),
                radius,
                facecolor=OBSTACLE_COLOR,
                alpha=OBSTACLE_ALPHA,
                edgecolor="darkgray",
                linewidth=1.5,
            )
            ax.add_patch(circle)
        elif isinstance(obstacle, BoxObstacle):
            center = np.asarray(obstacle.center, dtype=np.float32)[:2]
            half_extents = np.asarray(obstacle.half_extents, dtype=np.float32)[:2]
            rect = mpatches.Rectangle(
                tuple(center - half_extents),
                2 * float(half_extents[0]),
                2 * float(half_extents[1]),
                facecolor=OBSTACLE_COLOR,
                alpha=OBSTACLE_ALPHA,
                edgecolor="darkgray",
                linewidth=1.5,
            )
            ax.add_patch(rect)
        elif isinstance(obstacle, UnionObstacle):
            for prim in obstacle.obstacles:
                if isinstance(prim, (SphereObstacle, CircleObstacle)):
                    center = np.asarray(prim.center, dtype=np.float32)[:2]
                    radius = float(prim.radius)
                    circle = plt.Circle(
                        tuple(center),
                        radius,
                        facecolor=OBSTACLE_COLOR,
                        alpha=OBSTACLE_ALPHA * 0.8,
                        edgecolor="darkgray",
                        linewidth=1.2,
                    )
                    ax.add_patch(circle)
                elif isinstance(prim, BoxObstacle):
                    center = np.asarray(prim.center, dtype=np.float32)[:2]
                    half_extents = np.asarray(prim.half_extents, dtype=np.float32)[:2]
                    rect = mpatches.Rectangle(
                        tuple(center - half_extents),
                        2 * float(half_extents[0]),
                        2 * float(half_extents[1]),
                        facecolor=OBSTACLE_COLOR,
                        alpha=OBSTACLE_ALPHA * 0.8,
                        edgecolor="darkgray",
                        linewidth=1.2,
                    )
                    ax.add_patch(rect)


def visualize_diffusion_step(
    ax: plt.Axes,
    env: SingleIntegratorBox2DEnv,
    obstacles: ObstacleManager,
    initial_state: np.ndarray,
    action_sequence: np.ndarray,
    sample_actions: Optional[np.ndarray] = None,
    robot_radius: float = ROBOT_RADIUS,
    title: Optional[str] = None,
):
    ax.set_aspect("equal")
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)

    draw_obstacles(ax, obstacles)

    if sample_actions is not None and len(sample_actions) > 0:
        num_samples = min(len(sample_actions), MAX_SAMPLE_TRAJ_PLOT)
        if num_samples > MAX_SAMPLE_TRAJ_PLOT:
            idx = np.linspace(0, len(sample_actions) - 1, MAX_SAMPLE_TRAJ_PLOT, dtype=int)
            sample_actions = sample_actions[idx]

        import matplotlib.colors as mcolors
        light_rgba = mcolors.to_rgba(EDOC_COLOR, alpha=0.15)
        for acts in sample_actions:
            states = env.rollout_actions(initial_state, acts)
            if len(states) > 1:
                ax.plot(states[:, 0], states[:, 1], color=light_rgba, linewidth=0.8)

    if action_sequence is not None and len(action_sequence) > 0:
        states = env.rollout_actions(initial_state, action_sequence)
        if len(states) > 1:
            ax.plot(states[:, 0], states[:, 1], color=EDOC_COLOR, linewidth=2.5)
            ax.scatter(
                states[0, 0],
                states[0, 1],
                marker="o",
                s=30,
                facecolors="white",
                edgecolors=EDOC_COLOR,
                linewidths=1.0,
            )
            ax.scatter(
                states[-1, 0],
                states[-1, 1],
                marker="*",
                s=55,
                facecolors=EDOC_COLOR,
                edgecolors="black",
                linewidths=0.5,
            )

    target = np.asarray(env.target, dtype=np.float32)
    ax.plot(target[0], target[1], "r*", markersize=15, label="Target", zorder=10)

    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)


def visualize_trajectory(
    ax: plt.Axes,
    env: SingleIntegratorBox2DEnv,
    obstacles: ObstacleManager,
    trajectory: Trajectory,
    robot_radius: float = ROBOT_RADIUS,
    title: str = "Final Trajectory",
):
    ax.set_aspect("equal")
    ax.set_xlim(MAP_X_MIN, MAP_X_MAX)
    ax.set_ylim(MAP_Y_MIN, MAP_Y_MAX)

    draw_obstacles(ax, obstacles)

    if len(trajectory.states) > 1:
        positions = np.array([np.asarray(s)[:2] for s in trajectory.states], dtype=np.float32)
        ax.plot(positions[:, 0], positions[:, 1], color=EDOC_COLOR, linewidth=2.0, label="Trajectory")

        start_circle = plt.Circle(
            tuple(positions[0]),
            robot_radius,
            facecolor="white",
            edgecolor=EDOC_COLOR,
            linewidth=1.5,
            label="Start",
        )
        ax.add_patch(start_circle)

        end_circle = plt.Circle(
            tuple(positions[-1]),
            robot_radius,
            facecolor=EDOC_COLOR,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(end_circle)

    target = np.asarray(env.target, dtype=np.float32)
    target_circle = plt.Circle(
        tuple(target),
        robot_radius * 2,
        facecolor="none",
        edgecolor="red",
        linewidth=1.5,
        linestyle="--",
        label="Target (margin)",
    )
    ax.add_patch(target_circle)
    ax.plot(target[0], target[1], "r*", markersize=15, label="Target")

    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_energy_reward(
    ax_energy: plt.Axes,
    ax_reward: plt.Axes,
    energies: np.ndarray,
    rewards: np.ndarray,
    dt: float = 0.1,
):
    time_axis = np.arange(len(energies)) * dt
    ax_energy.plot(time_axis, energies, color=EDOC_COLOR, linewidth=2.0, label="Energy")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy Over Time")
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    time_r = np.arange(len(rewards)) * dt
    ax_reward.plot(time_r, rewards, color=EDOC_COLOR, linewidth=2.0, label="Reward")
    ax_reward.set_xlabel("Time (s)")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("Reward Over Time")
    ax_reward.legend()
    ax_reward.grid(True, alpha=0.3)


def plot_states(axes: List[plt.Axes], trajectory: Trajectory, dt: float = 0.1):
    if len(trajectory.states) == 0:
        return
    states_array = np.array([np.asarray(s) for s in trajectory.states], dtype=np.float32)
    time_axis = np.arange(len(trajectory.states)) * dt

    labels = [("x position", 0), ("y position", 1)]
    for ax, (title, idx) in zip(axes, labels):
        ax.plot(time_axis, states_array[:, idx], color=EDOC_COLOR, linewidth=2.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


# ============================================================================
# JSON Serialization Helper
# ============================================================================

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    return obj


# (Removed) prior-level metric loading: we now use an absolute per-level target schedule.
# ============================================================================
# Environment metrics (density + nonconvexity)
# ============================================================================

def compute_obstacle_density(
    obstacles: ObstacleManager,
    *,
    p_max: float,
    robot_radius: float,
    seed: int,
    n_samples: int = 50000,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> float:
    """
    Approximate obstacle density as occupied area fraction in the box domain.

    "Occupied" means the robot center would be in collision:
        obstacles.sdf(p) < robot_radius
    """
    if len(obstacles) == 0:
        return 0.0
    rng = np.random.RandomState(int(seed) + 12345)
    xlo = float(-p_max) if x_min is None else float(x_min)
    xhi = float(p_max) if x_max is None else float(x_max)
    ylo = float(-p_max) if y_min is None else float(y_min)
    yhi = float(p_max) if y_max is None else float(y_max)
    pts = np.stack(
        [
            rng.uniform(xlo, xhi, size=(int(n_samples),)).astype(np.float32),
            rng.uniform(ylo, yhi, size=(int(n_samples),)).astype(np.float32),
        ],
        axis=-1,
    )
    sdf = obstacles.sdf(pts)
    sdf = np.asarray(sdf, dtype=np.float32).reshape(-1)
    return float(np.mean(sdf < float(robot_radius)))


def compute_nonconvexity_metrics(obstacles: ObstacleManager) -> Dict[str, Any]:
    """
    Simple, monotone-ish nonconvexity metrics:
    - union_count / obstacle_count
    - primitive_count (sum over unions) / union_count
    - total_primitives (treat convex obstacles as 1 primitive)
    """
    obstacles_list = list(obstacles)
    obstacle_count = int(len(obstacles_list))
    if obstacle_count == 0:
        return {
            "obstacle_count": 0,
            "union_count": 0,
            "total_primitives": 0,
            "union_fraction": 0.0,
            "avg_primitives_per_union": 0.0,
            "nonconvexity_score": 0.0,
        }
    union_count = 0
    union_primitives = 0
    total_primitives = 0
    for obs in obstacles_list:
        if isinstance(obs, UnionObstacle):
            union_count += 1
            k = int(len(obs.obstacles))
            union_primitives += k
            total_primitives += k
        else:
            total_primitives += 1
    union_fraction = float(union_count / obstacle_count)
    avg_prims = float(union_primitives / union_count) if union_count > 0 else 0.0
    # One simple scalar that increases with more unions and more primitives per union
    nonconvexity_score = float(union_fraction * max(0.0, avg_prims - 1.0))
    return {
        "obstacle_count": obstacle_count,
        "union_count": union_count,
        "total_primitives": int(total_primitives),
        "union_fraction": union_fraction,
        "avg_primitives_per_union": avg_prims,
        "nonconvexity_score": nonconvexity_score,
    }


def compute_nonconvexity_score_sdf(
    obstacles: ObstacleManager,
    *,
    p_max: float,
    robot_radius: float,
    seed: int,
    n_pairs: int = 20000,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Geometric non-convexity metric based on SDF midpoint inequality violations.

    Let d(p) be the (inflated) obstacle SDF: d(p) = obstacles.sdf(p) - robot_radius.
    For a convex set, a proxy inequality holds:
        d((x+y)/2) >= min(d(x), d(y))
    For non-convex obstacle sets, this is frequently violated.

    We sample random pairs (x,y) and measure:
        violation = max(0, min(d(x), d(y)) - d(mid))
    Returns mean violation and violation rate.
    """
    if len(obstacles) == 0:
        return {
            "score_raw": 0.0,
            "violation_rate": 0.0,
            "mean_violation": 0.0,
            "n_pairs": int(n_pairs),
            "robot_radius": float(robot_radius),
        }
    rng = np.random.RandomState(int(seed) + 54321)
    n = int(max(1, n_pairs))
    xlo = float(-p_max) if x_min is None else float(x_min)
    xhi = float(p_max) if x_max is None else float(x_max)
    ylo = float(-p_max) if y_min is None else float(y_min)
    yhi = float(p_max) if y_max is None else float(y_max)
    x = np.stack(
        [
            rng.uniform(xlo, xhi, size=(n,)).astype(np.float32),
            rng.uniform(ylo, yhi, size=(n,)).astype(np.float32),
        ],
        axis=-1,
    )
    y = np.stack(
        [
            rng.uniform(xlo, xhi, size=(n,)).astype(np.float32),
            rng.uniform(ylo, yhi, size=(n,)).astype(np.float32),
        ],
        axis=-1,
    )
    mid = 0.5 * (x + y)

    dx = np.asarray(obstacles.sdf(x), dtype=np.float32).reshape(-1) - float(robot_radius)
    dy = np.asarray(obstacles.sdf(y), dtype=np.float32).reshape(-1) - float(robot_radius)
    dm = np.asarray(obstacles.sdf(mid), dtype=np.float32).reshape(-1) - float(robot_radius)

    viol = np.maximum(0.0, np.minimum(dx, dy) - dm)
    mean_viol = float(np.mean(viol))
    rate = float(np.mean(viol > 1e-6))
    return {
        "score_raw": mean_viol,
        "violation_rate": rate,
        "mean_violation": mean_viol,
        "n_pairs": int(n),
        "robot_radius": float(robot_radius),
    }


def has_free_space_path(
    obstacles: ObstacleManager,
    *,
    start: np.ndarray,
    target: np.ndarray,
    p_max: float,
    robot_radius: float,
    grid_res: float = 0.06,
) -> bool:
    """
    Conservative connectivity check in robot-inflated free space.

    A grid cell is free iff obstacles.sdf(cell_center) >= robot_radius.
    Run BFS from start cell to target cell.
    """
    if len(obstacles) == 0:
        return True
    start = np.asarray(start, dtype=np.float32)[:2]
    target = np.asarray(target, dtype=np.float32)[:2]
    p_max = float(p_max)
    r = float(robot_radius)
    res = float(grid_res)
    if res <= 0:
        return True

    xs = np.arange(-p_max, p_max + 1e-6, res, dtype=np.float32)
    ys = np.arange(-p_max, p_max + 1e-6, res, dtype=np.float32)
    W = int(xs.size)
    H = int(ys.size)
    if W < 2 or H < 2:
        return True

    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
    sdf = np.asarray(obstacles.sdf(pts), dtype=np.float32).reshape(H, W)
    free = sdf >= r

    def to_idx(p):
        ix = int(np.clip(np.round((p[0] + p_max) / res), 0, W - 1))
        iy = int(np.clip(np.round((p[1] + p_max) / res), 0, H - 1))
        return iy, ix

    sy, sx = to_idx(start)
    ty, tx = to_idx(target)
    if not free[sy, sx] or not free[ty, tx]:
        return False

    q = deque([(sy, sx)])
    visited = np.zeros((H, W), dtype=bool)
    visited[sy, sx] = True
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while q:
        y, x = q.popleft()
        if y == ty and x == tx:
            return True
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and free[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return False


# ============================================================================
# SSR Computation
# ============================================================================

def compute_ssr(
    trajectory: Trajectory,
    env: SingleIntegratorBox2DEnv,
    obstacles: ObstacleManager,
    speed_constraint: SpeedConstraint,
    robot_radius: float = ROBOT_RADIUS,
    success_margin: Optional[float] = None,
) -> Dict[str, Any]:
    if success_margin is None:
        success_margin = 2 * robot_radius

    safe = True
    if len(obstacles) > 0:
        for state in trajectory.states:
            pos = np.asarray(state, dtype=np.float32)[:2]
            sdf = obstacles.sdf(pos)
            if float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf) < robot_radius:
                safe = False
                break
            if obstacles.contains(pos):
                safe = False
                break

    speed_feasible = bool(speed_constraint.is_feasible(trajectory))

    if len(trajectory.states) > 0:
        final_pos = np.asarray(trajectory.states[-1], dtype=np.float32)[:2]
        target = np.asarray(env.target, dtype=np.float32)
        distance_to_target = float(np.linalg.norm(final_pos - target))
        task_success = bool(distance_to_target < success_margin)
    else:
        distance_to_target = float("inf")
        task_success = False

    ssr = 1.0 if (safe and speed_feasible and task_success) else 0.0
    return {
        "ssr": float(ssr),
        "safe": bool(safe),
        "speed_feasible": bool(speed_feasible),
        "task_success": bool(task_success),
        "distance_to_target": float(distance_to_target),
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_experiment(
    level: int,
    seed: int,
    output_dir: Path,
    edoc_config: Dict[str, Any],
    backend_name: str = "jax",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run single experiment with given level and seed.
    
    Args:
        level: Obstacle level (0-10)
        seed: Random seed
        output_dir: Output directory for results
        edoc_config: EDOC configuration dictionary
        backend_name: Computational backend name ("jax", "numpy", "torch")
        device: Device name ("cpu", "gpu", "webgpu")
    
    Returns:
        Dictionary with results
    """
    # Set up runtime backend
    RuntimeBackendManager.set_backend(backend_name, device=device)
    backend = RuntimeBackendManager.get_backend()
    print(f"  Using backend: {backend_name} on {device}")
    print(f"\n{'=' * 60}")
    print(f"Starting Experiment: Level {level}, Seed {seed}")
    print(f"{'=' * 60}")
    experiment_start_time = time.time()

    env = SingleIntegratorBox2DEnv(
        dt=edoc_config.get("dt", 0.1),
        horizon=edoc_config.get("horizon", 80),
        p_max=2.0,
        control_limit=edoc_config.get("u_max", 1.0),
    )

    target = np.asarray(env.target, dtype=np.float32)
    start_pos = generate_optimized_start_position(level, seed, target, env.p_max)
    print(f"  Start position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}]")
    print(f"  Target position: [{target[0]:.3f}, {target[1]:.3f}]")
    print(f"  Distance: {np.linalg.norm(start_pos - target):.3f}")

    print("  Generating obstacles...")
    obstacles = generate_obstacle_config(level, seed, start_pos, target)
    num_union_obstacles = 0
    num_primitives_total = 0
    for obs in list(obstacles):
        if isinstance(obs, UnionObstacle):
            num_union_obstacles += 1
            num_primitives_total += int(len(obs.obstacles))
        else:
            num_primitives_total += 1
    if num_union_obstacles > 0:
        print(
            f"  Generated {len(obstacles)} obstacle(s) "
            f"({num_union_obstacles} union(s), {num_primitives_total} primitives total)"
        )
    else:
        print(f"  Generated {len(obstacles)} obstacle(s) ({num_primitives_total} primitives total)")

    if level > 0 and len(obstacles) > 0:
        obstacles.build_sdf_texture_2d(
            x_min=float(MAP_X_MIN),
            x_max=float(MAP_X_MAX),
            y_min=float(MAP_Y_MIN),
            y_max=float(MAP_Y_MAX),
            res=0.01,
            force_rebuild=True,
        )

    energy = make_energy("single_integrator_box_2d")

    soft_constraint = None
    hard_constraint = None
    if level > 0:
        soft_constraint = ObstacleSoftConstraint(obstacles=obstacles, alpha=1.0, beta=10.0)
        hard_constraint = ObstacleHardConstraint(obstacles=obstacles, clearance=0.1)

    speed_constraint = SpeedConstraint(u_max=float(env.control_limit))

    schedule_manager = None
    if level > 0:
        schedule_manager = ConstraintScheduleManager.create_soft_to_hard(
            soft_alpha_start=1.0,
            soft_alpha_end=0.0,
            hard_clearance_start=0.5,
            hard_clearance_end=0.1,
            schedule_type="linear",
            reverse_mode=True,
        )
        if soft_constraint is not None:
            soft_constraint.schedule_manager = schedule_manager
        if hard_constraint is not None:
            hard_constraint.schedule_manager = schedule_manager

    feasibility_op = None
    cfs_enabled = False
    if level > 0:
        feasibility_op = CFSProjection(
            obstacles=obstacles,
            schedule_manager=schedule_manager,
            use_late_stage_only=True,
            late_stage_ratio=0.2,
            use_trajectory_qp=False,
            smoothness_weight=0.0,
            reconstruct_velocity=False,
            velocity_dt=None,
            max_iterations=5,
        )
        cfs_enabled = True

    soft_constraints_list = [soft_constraint] if soft_constraint is not None else []
    hard_constraints_list = []
    if hard_constraint is not None:
        hard_constraints_list.append(hard_constraint)
    hard_constraints_list.append(speed_constraint)

    constraint_manager = ConstraintManager(
        soft_constraints=soft_constraints_list,
        hard_constraints=hard_constraints_list,
        feasibility_operator=feasibility_op if level > 0 else None,
        action_filter_operator=None,
        schedule_manager=schedule_manager if level > 0 else None,
    )

    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=env.horizon,
        dt=env.dt,
        action_space=True,
        diffusion_mode="reverse",
        action_diffuse_steps=edoc_config.get("action_diffuse_steps", 100),
        action_nsample=edoc_config.get("action_nsample", 256),
        use_antithetic=edoc_config.get("use_antithetic", True),
        action_score_mode="energy",
        constraint_manager=constraint_manager,
        use_constraint_in_scoring=True,
        lambda_energy=1.0,
        terminal_energy_weight=edoc_config.get("terminal_energy_weight", 50.0),
    )

    initial_state = start_pos.astype(np.float32)

    print("  Starting EDOC planning...")
    rng = backend.create_rng(seed)
    original_reset = env.reset

    def custom_reset(rng=None):
        return initial_state.copy(), {}

    env.reset = custom_reset
    start_time = time.time()
    result = planner.plan(rng)
    planning_time = time.time() - start_time
    env.reset = original_reset
    print(f"  Planning completed in {planning_time:.2f} seconds")

    states = result.get("states", [])
    actions = result.get("actions", None)
    if actions is None or len(actions) == 0:
        actions = [np.zeros(env.act_dim, dtype=np.float32) for _ in range(len(states) - 1)]

    states_list = [np.asarray(s, dtype=np.float32) for s in np.asarray(states)]
    actions_list = [np.asarray(a, dtype=np.float32) for a in np.asarray(actions)]
    trajectory = Trajectory(states=states_list, actions=actions_list)

    ssr_metrics = compute_ssr(trajectory, env, obstacles, speed_constraint, robot_radius=ROBOT_RADIUS)
    obstacle_density = compute_obstacle_density(
        obstacles,
        p_max=float(env.p_max),
        robot_radius=float(ROBOT_RADIUS),
        seed=int(seed),
        x_min=float(MAP_X_MIN),
        x_max=float(MAP_X_MAX),
        y_min=float(MAP_Y_MIN),
        y_max=float(MAP_Y_MAX),
    )
    nonconvexity = compute_nonconvexity_metrics(obstacles)
    nonconvexity_sdf = compute_nonconvexity_score_sdf(
        obstacles,
        p_max=float(env.p_max),
        robot_radius=float(ROBOT_RADIUS),
        seed=int(seed),
        x_min=float(MAP_X_MIN),
        x_max=float(MAP_X_MAX),
        y_min=float(MAP_Y_MIN),
        y_max=float(MAP_Y_MAX),
    )
    print("  Results:")
    print(f"    SSR: {ssr_metrics['ssr']:.3f}")
    print(f"    Safe: {ssr_metrics['safe']}")
    print(f"    Task Success: {ssr_metrics['task_success']}")
    print(f"    Distance to Target: {ssr_metrics['distance_to_target']:.4f}")
    print(f"    Speed Feasible: {ssr_metrics['speed_feasible']}")
    print(f"    Obstacle Density (inflated): {obstacle_density:.4f}")
    print(f"    Nonconvexity Score: {nonconvexity['nonconvexity_score']:.4f}")
    print(f"    Nonconvexity Score (SDF raw): {nonconvexity_sdf['score_raw']:.6f}")

    results = {
        "level": level,
        "seed": seed,
        "num_obstacles": int(len(obstacles)),
        "num_union_obstacles": int(num_union_obstacles),
        "num_primitives_total": int(num_primitives_total),
        "ssr": ssr_metrics["ssr"],
        "safe": ssr_metrics["safe"],
        "speed_feasible": ssr_metrics["speed_feasible"],
        "task_success": ssr_metrics["task_success"],
        "distance_to_target": float(ssr_metrics["distance_to_target"]),
        "planning_time": float(planning_time),
        "cfs_enabled": bool(cfs_enabled),
        "obstacle_density": float(obstacle_density),
        "nonconvexity": convert_to_json_serializable(nonconvexity),
        "nonconvexity_sdf": convert_to_json_serializable(nonconvexity_sdf),
        "trajectory": {
            "states": [s.tolist() for s in states_list],
            "actions": [a.tolist() for a in actions_list],
        },
        "energies": result.get("energies", []).tolist() if hasattr(result.get("energies", []), "tolist") else [],
        "rewards": result.get("rewards", []).tolist() if hasattr(result.get("rewards", []), "tolist") else [],
    }

    level_dir = output_dir / f"level_{level}" / f"seed_{seed}"
    level_dir.mkdir(parents=True, exist_ok=True)

    print("  Generating visualizations...")

    fig_diff, axes_diff = plt.subplots(1, 3, figsize=(18, 6))
    diffusion_actions = result.get("diffusion_actions_traj", None)
    diffusion_samples = result.get("diffusion_sampled_actions", None)

    if diffusion_actions is not None and len(diffusion_actions) > 0:
        diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
        Ndiffuse = diffusion_actions.shape[0]
        for ax, frac in zip(axes_diff, DIFFUSION_FRACTIONS):
            step_idx = int((1.0 - frac) * (Ndiffuse))
            step_idx = max(0, min(step_idx, Ndiffuse - 1))
            action_seq = diffusion_actions[step_idx]

            sample_acts = None
            if diffusion_samples is not None and len(diffusion_samples) > 0:
                diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                    sample_acts = diffusion_samples_arr[step_idx]

            visualize_diffusion_step(
                ax,
                env,
                obstacles,
                initial_state,
                action_seq,
                sample_acts,
                robot_radius=ROBOT_RADIUS,
                title=f"Diffusion {int((1-frac) * 100)}%",
            )
    else:
        for ax, frac in zip(axes_diff, DIFFUSION_FRACTIONS):
            visualize_diffusion_step(
                ax,
                env,
                obstacles,
                initial_state,
                np.array(actions_list, dtype=np.float32) if actions_list else None,
                None,
                robot_radius=ROBOT_RADIUS,
                title=f"Diffusion {int((1-frac) * 100)}% (Final)",
            )

    plt.tight_layout()
    plt.savefig(level_dir / "diffusion_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig_diff)

    fig_traj, ax_traj = plt.subplots(1, 1, figsize=(8, 8))
    visualize_trajectory(ax_traj, env, obstacles, trajectory)
    plt.savefig(level_dir / "trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig_traj)

    fig_er, (ax_energy, ax_reward) = plt.subplots(1, 2, figsize=(12, 5))
    energies_array = np.array(results["energies"]) if len(results["energies"]) > 0 else np.array([0.0])
    rewards_array = np.array(results["rewards"]) if len(results["rewards"]) > 0 else np.array([0.0])
    plot_energy_reward(ax_energy, ax_reward, energies_array, rewards_array, dt=env.dt)
    plt.savefig(level_dir / "energy_reward.png", dpi=150, bbox_inches="tight")
    plt.close(fig_er)

    fig_states, axes_states = plt.subplots(1, 2, figsize=(12, 4))
    plot_states(list(axes_states), trajectory, dt=env.dt)
    plt.tight_layout()
    plt.savefig(level_dir / "states.png", dpi=150, bbox_inches="tight")
    plt.close(fig_states)

    results_serializable = convert_to_json_serializable(results)
    with open(level_dir / "results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    total_experiment_time = time.time() - experiment_start_time
    print(f"  Results saved to {level_dir}")
    print(f"  Total experiment time: {total_experiment_time:.2f}s")
    print(f"{'=' * 60}\n")

    return results


def run_all_experiments(
    output_dir: Path = Path("results/single2d"),
    num_seeds: int = 10,
    edoc_config: Optional[Dict[str, Any]] = None,
    backend_name: str = "jax",
    device: str = "cpu",
):
    """
    Run all experiments across all levels and seeds.
    
    Args:
        output_dir: Output directory for results
        num_seeds: Number of random seeds per level
        edoc_config: EDOC configuration
        backend_name: Computational backend name ("jax", "numpy", "torch")
        device: Device name ("cpu", "gpu", "webgpu")
    """
    if edoc_config is None:
        edoc_config = {
            "dt": 0.05,
            "horizon": 80,
            "u_max": 1.0,
            "action_diffuse_steps": 100,
            "action_nsample": 256,
            "use_antithetic": True,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print("Starting All Experiments (single2d)")
    print(f"  Output directory: {output_dir}")
    print("  Levels: 0-10 (11 levels)")
    print(f"  Seeds per level: {num_seeds}")
    print(f"  Total experiments: {11 * num_seeds}")
    print(f"  Backend: {backend_name} on {device}")
    print(f"  EDOC config: {edoc_config}")
    print(f"{'#' * 70}\n")

    all_results = []
    for level in range(0, 11):
        print(f"\n>>> Processing Level {level} <<<")
        level_results = []
        for seed in range(num_seeds):
            try:
                print(f"  Seed {seed}/{num_seeds - 1}...")
                result = run_single_experiment(level, seed, output_dir, edoc_config, backend_name, device)
                level_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR in Seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        if level_results:
            ssr_values = [r["ssr"] for r in level_results]
            level_ssr = float(np.mean(ssr_values))
            planning_times = [r.get("planning_time", 0.0) for r in level_results]
            avg_time = float(np.mean(planning_times))

            print(f"\n  Level {level} Summary:")
            print(f"    SSR: {level_ssr:.3f} ({int(np.sum(ssr_values))}/{len(ssr_values)} successes)")
            print(f"    Avg Planning Time: {avg_time:.2f}s")

            level_summary = {
                "level": level,
                "ssr": float(level_ssr),
                "num_success": int(np.sum(ssr_values)),
                "num_total": int(len(level_results)),
                "results": level_results,
            }
            with open(output_dir / f"level_{level}" / "summary.json", "w") as f:
                json.dump(convert_to_json_serializable(level_summary), f, indent=2)

    # ------------------ per-seed monotone post-processing ------------------
    # Ensure obstacle density and SDF-based nonconvexity are non-decreasing with level for each seed.
    if all_results:
        by_seed: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for r in all_results:
            by_seed.setdefault(int(r["seed"]), {})[int(r["level"])] = r

        for seed, level_map in by_seed.items():
            levels_sorted = sorted(level_map.keys())
            dens_raw = [float(level_map[l].get("obstacle_density", 0.0)) for l in levels_sorted]
            nc_raw = [float(level_map[l].get("nonconvexity_sdf", {}).get("score_raw", 0.0)) for l in levels_sorted]

            dens_mono = []
            nc_mono = []
            d_max = -1e9
            n_max = -1e9
            for d, ncv in zip(dens_raw, nc_raw):
                d_max = max(d_max, d)
                n_max = max(n_max, ncv)
                dens_mono.append(d_max)
                nc_mono.append(n_max)

            # Write back into each result dict and persist to per-seed results.json
            for l, d_adj, n_adj in zip(levels_sorted, dens_mono, nc_mono):
                rr = level_map[l]
                rr["obstacle_density_raw"] = float(rr.get("obstacle_density", 0.0))
                rr["obstacle_density"] = float(d_adj)
                rr.setdefault("nonconvexity_sdf", {})
                rr["nonconvexity_sdf"]["score_raw"] = float(rr["nonconvexity_sdf"].get("score_raw", 0.0))
                rr["nonconvexity_sdf"]["score_monotone"] = float(n_adj)

                # Persist back to the per-experiment results.json so downstream analysis sees it.
                level_dir = output_dir / f"level_{l}" / f"seed_{seed}"
                try:
                    with open(level_dir / "results.json", "w") as f:
                        json.dump(convert_to_json_serializable(rr), f, indent=2)
                except Exception:
                    pass

    if all_results:
        overall_ssr = float(np.mean([r["ssr"] for r in all_results]))
        total_planning_time = float(sum([r.get("planning_time", 0.0) for r in all_results]))
        print(f"\n{'#' * 70}")
        print("Overall Results:")
        print(f"  Overall SSR: {overall_ssr:.3f}")
        print(f"  Total Planning Time: {total_planning_time:.2f}s")
        print(f"  Average Planning Time: {total_planning_time / len(all_results):.2f}s per experiment")
        print(f"{'#' * 70}\n")

        overall_summary = {
            "overall_ssr": float(overall_ssr),
            "total_experiments": int(len(all_results)),
            "level_ssrs": {},
        }
        for level in range(0, 11):
            level_results = [r for r in all_results if r["level"] == level]
            if level_results:
                overall_summary["level_ssrs"][f"level_{level}"] = float(np.mean([r["ssr"] for r in level_results]))

        with open(output_dir / "overall_summary.json", "w") as f:
            json.dump(convert_to_json_serializable(overall_summary), f, indent=2)

        fig_ssr, ax_ssr = plt.subplots(1, 1, figsize=(10, 6))
        level_numbers = sorted([int(k.split("_")[1]) for k in overall_summary["level_ssrs"].keys()])
        ssrs = [overall_summary["level_ssrs"][f"level_{l}"] for l in level_numbers]
        ax_ssr.bar(level_numbers, ssrs, alpha=0.7, color=EDOC_COLOR)
        ax_ssr.set_xlabel("Level")
        ax_ssr.set_ylabel("SSR")
        ax_ssr.set_title("Safety Success Rate by Level (single2d)")
        ax_ssr.set_xticks(level_numbers)
        ax_ssr.grid(True, alpha=0.3, axis="y")
        plt.savefig(output_dir / "ssr_by_level.png", dpi=150, bbox_inches="tight")
        plt.close(fig_ssr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDOC experiments on Single Integrator 2D")
    parser.add_argument("--output_dir", type=str, default="results/single2d", help="Output directory")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of seeds per level")
    parser.add_argument("--level", type=int, default=None, help="Run single level only")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed only")
    parser.add_argument("--backend", type=str, default="numpy", choices=["jax", "numpy", "torch"],
                       help="Computational backend (jax, numpy, torch)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu", "webgpu"],
                       help="Device to use (cpu, gpu, webgpu)")

    args = parser.parse_args()

    if args.level is not None and args.seed is not None:
        output_dir = Path(args.output_dir)
        edoc_config = {
            "dt": 0.05,
            "horizon": 80,
            "u_max": 1.0,
            "action_diffuse_steps": 100,
            "action_nsample": 64,
            "use_antithetic": True,
        }
        run_single_experiment(args.level, args.seed, output_dir, edoc_config, args.backend, args.device)
    else:
        run_all_experiments(
            output_dir=Path(args.output_dir),
            num_seeds=args.num_seeds,
            backend_name=args.backend,
            device=args.device
        )