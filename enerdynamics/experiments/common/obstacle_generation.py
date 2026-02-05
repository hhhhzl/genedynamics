"""
Common obstacle generation utilities for 2D box environments.

This module provides unified obstacle generation functions that can be used
across different experiment types. The code is extracted from the existing
experiment scripts to eliminate duplication.
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import numpy as np

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from enerdynamics.envs.obstacles.nonconvex import UnionObstacle

# Alias for 2D: SphereObstacle is CircleObstacle in 2D
CircleObstacle = SphereObstacle

# Default constants for obstacle generation
DEFAULT_ROBOT_RADIUS = 0.05
DEFAULT_MIN_OBSTACLE_MARGIN = 2.4 * DEFAULT_ROBOT_RADIUS
DEFAULT_OBSTACLE_RADIUS_SCALE = 1.0
# Extra clearance for start/goal so obstacles stay clearly away (not tangent)
START_GOAL_CLEARANCE_MARGIN = 0.03


def check_obstacle_spacing(
    new_center: np.ndarray,
    new_radius: float,
    existing_obstacles: List[Tuple[np.ndarray, float]],
    margin: float
) -> bool:
    """
    Check if new obstacle has enough margin from existing obstacles.
    
    Args:
        new_center: Center of new obstacle
        new_radius: Radius of new obstacle
        existing_obstacles: List of (center, radius) tuples for existing obstacles
        margin: Minimum margin required
        
    Returns:
        True if spacing is sufficient, False otherwise
    """
    for existing_center, existing_radius in existing_obstacles:
        distance = np.linalg.norm(new_center - existing_center)
        min_distance = new_radius + existing_radius + margin
        if distance < min_distance:
            return False
    return True


def get_obstacle_radius(obstacle) -> float:
    """
    Get obstacle radius (for circles) or max half-extent (for boxes).
    
    Args:
        obstacle: Obstacle instance
        
    Returns:
        Effective radius for spacing calculations
    """
    if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
        return float(obstacle.radius)
    elif isinstance(obstacle, BoxObstacle):
        # Use circumscribed radius (center -> corner) so spacing is conservative
        he = np.asarray(obstacle.half_extents, dtype=np.float32).reshape(-1)
        he = he[:2] if he.size >= 2 else np.array([float(he[0]), float(he[0])], dtype=np.float32)
        return float(np.linalg.norm(he))
    elif isinstance(obstacle, UnionObstacle):
        # For union, get max radius of all primitives
        max_radius = 0.0
        for prim in obstacle.obstacles:
            prim_radius = get_obstacle_radius(prim)
            max_radius = max(max_radius, prim_radius)
        return max_radius
    return 0.0


def check_start_target_clearance(
    obstacle,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    buffer: float = DEFAULT_ROBOT_RADIUS
) -> bool:
    """
    Ensure start/target are not inside (or too close to) the obstacle.
    So the circle of radius `buffer` centered at start (or goal) does not overlap the obstacle.
    
    Uses obstacle SDF semantics:
    - sdf < 0  : inside
    - sdf == 0 : on boundary
    - sdf > 0  : outside (distance)
    
    Args:
        obstacle: Obstacle instance
        start_pos: Start position
        target_pos: Target position
        buffer: Minimum clearance (center-to-obstacle distance); use robot_radius so the
            robot-radius disk at start/goal stays clear.
        
    Returns:
        True if start/target have sufficient clearance, False otherwise
    """
    buf = float(buffer)
    for p in (start_pos, target_pos):
        p = np.asarray(p, dtype=np.float32)
        sdf = obstacle.sdf(p)
        sdf = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
        if sdf < buf:
            return False
        # Extra guard if obstacle implements an exact contains() check
        try:
            if hasattr(obstacle, "contains") and obstacle.contains(p):
                return False
        except Exception:
            # If contains() fails, rely on sdf threshold above
            pass
    return True


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance from point p to segment a-b (2D).
    
    Args:
        p: Point
        a: Segment start
        b: Segment end
        
    Returns:
        Distance from point to segment
    """
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


def place_union_obstacle(
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
    robot_radius: float,
    obstacle_radius_scale: float,
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
    
    Fixes a key failure mode: samples primitive offsets/sizes first, computes
    feasible base_center bounds, then samples base_center within those bounds
    (no post-hoc clipping).
    
    Args:
        union_idx: Index of union obstacle
        primitives_per_union: Number of primitives in union
        base_radius_scale: Scale factor for base radius
        center_region: Center of region for placement
        region_size: Size of region
        p_max: Maximum position bound
        margin: Minimum margin between obstacles
        start_pos: Start position
        target_pos: Target position
        existing: List of existing obstacles (center, radius)
        robot_radius: Robot radius
        obstacle_radius_scale: Scale factor for obstacle radius
        max_attempts: Maximum placement attempts
        offset_range: Range for primitive offsets
        corridor_width: Corridor keep-out width
        region_low_override: Override for region lower bound
        region_high_override: Override for region upper bound
        bias_to_path: Whether to bias placement near path
        path_bias_sigma: Standard deviation for path bias
        path_perp_range: Perpendicular range for path bias
        sep_min_mult: Minimum separation multiplier
        sep_max_mult: Maximum separation multiplier
        
    Returns:
        Tuple of (union_obstacle, base_center, union_radius) or None if failed
    """
    base_radius = robot_radius * obstacle_radius_scale * float(base_radius_scale)
    center_region = np.asarray(center_region, dtype=np.float32)
    region_half = float(region_size) / 2.0
    region_low = center_region - region_half
    region_high = center_region + region_half

    for _ in range(int(max_attempts)):
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

        extents = np.asarray(extents, dtype=np.float32)

        if primitives_per_union == 2:
            theta = float(np.random.uniform(0.0, 2.0 * np.pi))
            dvec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            rmax = float(np.max(extents))
            lo = float(max(1.01, float(sep_min_mult))) * rmax
            hi = float(max(lo + 1e-6, float(sep_max_mult))) * rmax
            sep = float(np.random.uniform(lo, hi))
            offsets = np.stack([-0.5 * sep * dvec, 0.5 * sep * dvec], axis=0).astype(np.float32)
        else:
            offsets = np.random.uniform(-offset_range, offset_range, size=(primitives_per_union, 2)).astype(np.float32)

        union_radius = float(np.max(np.linalg.norm(offsets, axis=1) + extents))

        # Require: |base + offset| <= p_max - (extent + margin)
        ext2 = (extents + float(margin))[:, None]
        per_prim_low = -float(p_max) + ext2 - offsets
        per_prim_high = float(p_max) - ext2 - offsets
        low = np.max(per_prim_low, axis=0)
        high = np.min(per_prim_high, axis=0)

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

        if bias_to_path:
            a = np.asarray(start_pos, dtype=np.float32).reshape(2)
            b = np.asarray(target_pos, dtype=np.float32).reshape(2)
            ab = b - a
            dn2 = float(np.dot(ab, ab))
            if dn2 > 1e-12:
                dn = float(np.sqrt(dn2))
                dir_u = ab / dn
                perp = np.array([-dir_u[1], dir_u[0]], dtype=np.float32)
                t = float(np.random.uniform(0.05, 0.98))
                p_line = a + t * ab
                p_pref = p_line + perp * float(np.random.uniform(-path_perp_range, path_perp_range))

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

        if corridor_width is not None and corridor_width > 0.0:
            d_seg = point_segment_distance(base_center, start_pos, target_pos)
            if d_seg < float(corridor_width) + float(union_radius) * 0.35:
                continue

        if not check_obstacle_spacing(base_center, union_radius, existing, margin):
            continue

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
        if not check_start_target_clearance(union, start_pos, target_pos, buffer=robot_radius + START_GOAL_CLEARANCE_MARGIN):
            continue

        return union, base_center, union_radius

    return None


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
    Check if there is a free space path from start to target using grid BFS.
    
    Args:
        obstacles: Obstacle manager
        start: Start position
        target: Target position
        p_max: Maximum position bound
        robot_radius: Robot radius for inflation
        grid_res: Grid resolution
        
    Returns:
        True if path exists, False otherwise
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
    
    Uses inflated obstacle SDF: d(p) = obstacles.sdf(p) - robot_radius.
    Measures violations of: d(mid) >= min(d(x), d(y)).
    
    Args:
        obstacles: Obstacle manager
        p_max: Maximum position bound
        robot_radius: Robot radius
        seed: Random seed
        n_pairs: Number of point pairs to sample
        x_min, x_max, y_min, y_max: Optional bounds override
        
    Returns:
        Dictionary with nonconvexity metrics
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


def generate_box2d_obstacles(
    level: int,
    seed: int,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    config: Dict[str, Any],
) -> ObstacleManager:
    """
    Generate obstacle configuration for 2D box environments.
    
    Levels progress in both density and non-convexity:
    0: No obstacles
    1-3: Convex only (3, 6, 10 obstacles)
    4-6: Mixed convex (14, 18, 22 obstacles)
    7-10: Non-convex unions (increasing numbers)
    
    Args:
        level: Obstacle difficulty level (0-10)
        seed: Random seed for reproducibility
        start_pos: Start position (2D array)
        target_pos: Target position (2D array)
        config: Configuration dictionary with keys:
            - robot_radius: Robot radius (default: 0.05)
            - obstacle_radius_scale: Scale factor (default: 1.1)
            - min_obstacle_margin: Minimum margin (default: 2.4 * robot_radius)
            - p_max: Position bounds (default: 2.0)
            - map_bounds: Dict with x_min, x_max, y_min, y_max
            - enable_connectivity_check: Whether to check connectivity (default: True)
            - enable_nonconvexity_check: Whether to check nonconvexity (default: True for levels 7-9)
            
    Returns:
        ObstacleManager with generated obstacles
    """
    np.random.seed(seed)
    manager = ObstacleManager()

    if level == 0:
        return manager

    robot_radius = float(config.get('robot_radius', DEFAULT_ROBOT_RADIUS))
    obstacle_radius_scale = float(config.get('obstacle_radius_scale', DEFAULT_OBSTACLE_RADIUS_SCALE))
    min_obstacle_margin = float(config.get('min_obstacle_margin', 2.4 * robot_radius))
    p_max = float(config.get('p_max', 2.0))
    
    map_bounds = config.get('map_bounds', {})
    pref_low = np.array([
        float(map_bounds.get('x_min', -1.5)),
        float(map_bounds.get('y_min', -2.0))
    ], dtype=np.float32)
    pref_high = np.array([
        float(map_bounds.get('x_max', 1.0)),
        float(map_bounds.get('y_max', 0.5))
    ], dtype=np.float32)
    pref_low = np.maximum(pref_low, -p_max)
    pref_high = np.minimum(pref_high, p_max)

    # Compute region between start and target
    direction = target_pos - start_pos
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        center_region = np.array([0.0, 0.0], dtype=np.float32)
        region_size = p_max * 0.8
    else:
        center_region = (start_pos + target_pos) / 2.0
        region_size = direction_norm * 0.8 + 0.5

    existing = []

    if level <= 3:
        # Convex only (simple primitives)
        num_obstacles = {1: 3, 2: 6, 3: 10}[level]
        max_attempts = 100
        
        for i in range(num_obstacles):
            base_radius = robot_radius * obstacle_radius_scale
            radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
            placed = False
            
            for attempt in range(max_attempts):
                offset = np.array([
                    np.random.uniform(-region_size / 2, region_size / 2),
                    np.random.uniform(-region_size / 2, region_size / 2)
                ], dtype=np.float32)
                center = center_region + offset
                center = np.clip(
                    center,
                    np.maximum(-p_max, pref_low) + radius + min_obstacle_margin + robot_radius,
                    np.minimum(p_max, pref_high) - radius - min_obstacle_margin - robot_radius,
                )

                if check_obstacle_spacing(center, radius, existing, min_obstacle_margin):
                    if np.random.rand() < 0.5:
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    else:
                        size = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=size, name=f"box_{i}")

                    if not check_start_target_clearance(obstacle, start_pos, target_pos, buffer=robot_radius + START_GOAL_CLEARANCE_MARGIN):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius + robot_radius))
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    elif level <= 6:
        # Mixed convex (more diverse types and sizes)
        num_obstacles = {4: 18, 5: 24, 6: 28}[level]
        max_attempts = 100
        
        for i in range(num_obstacles):
            obstacle_type = np.random.choice(['circle', 'box', 'small_circle'])
            base_radius = robot_radius * obstacle_radius_scale
            
            if obstacle_type == 'circle':
                radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
            elif obstacle_type == 'box':
                size = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
                radius = size
            else:
                radius = np.random.uniform(base_radius * 0.5, base_radius * 1.0)
            
            placed = False
            for attempt in range(max_attempts):
                offset = np.array([
                    np.random.uniform(-region_size / 2, region_size / 2),
                    np.random.uniform(-region_size / 2, region_size / 2)
                ], dtype=np.float32)
                center = center_region + offset
                center = np.clip(
                    center,
                    np.maximum(-p_max, pref_low) + radius + min_obstacle_margin + robot_radius,
                    np.minimum(p_max, pref_high) - radius - min_obstacle_margin - robot_radius,
                )

                if check_obstacle_spacing(center, radius, existing, min_obstacle_margin):
                    if obstacle_type == 'circle':
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    elif obstacle_type == 'box':
                        half_extents = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=half_extents, name=f"box_{i}")
                    else:
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"small_circle_{i}")

                    if not check_start_target_clearance(obstacle, start_pos, target_pos, buffer=robot_radius + START_GOAL_CLEARANCE_MARGIN):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius + robot_radius))
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    else:
        # Non-convex unions (level 7-10)
        num_unions = {7: 18, 8: 24, 9: 30, 10: 36}[level]
        primitives_per_union = 2

        size_scale0 = 1.0
        offset0 = 0.16
        region_size_union = float(min(2.0 * p_max, float(region_size)))
        
        dir_vec = np.asarray(target_pos - start_pos, dtype=np.float32)
        dn = float(np.linalg.norm(dir_vec))
        mid = ((np.asarray(start_pos, dtype=np.float32) + np.asarray(target_pos, dtype=np.float32)) / 2.0).astype(np.float32)
        center_region_union = mid if dn > 1e-6 else ((pref_low + pref_high) / 2.0).astype(np.float32)
        center_region_union = np.clip(center_region_union, pref_low + 0.25, pref_high - 0.25).astype(np.float32)

        base_region_size_union = float(min(2.0 * p_max, float(pref_high[0] - pref_low[0])))
        region_size_union = float(min(region_size_union, base_region_size_union))
        margin_union = float(max(0.00, float(min_obstacle_margin) * 0.30))
        spacing_shrink = 0.75

        existing_before = list(existing)
        added_unions: List[UnionObstacle] = []
        placed_all = False
        last_connect_ok: Optional[bool] = None
        last_nonconv_sdf_raw: Optional[float] = None

        enable_connectivity_check = config.get('enable_connectivity_check', True)
        enable_nonconvexity_check = config.get('enable_nonconvexity_check', level in [7, 8, 9])
        nonconv_targets = {7: 0.11, 8: 0.125, 9: 0.14}
        target_nonconv = nonconv_targets.get(int(level), None) if enable_nonconvexity_check else None

        cal_max = 16 if (target_nonconv is not None) else 10
        for cal in range(int(cal_max)):
            np.random.seed(int(seed) + 10000 * (cal + 1))
            region_mult = float(min(1.0, 0.55 + 0.06 * float(cal)))
            region_size_eff = float(max(0.9, float(region_size_union) * region_mult))
            
            for _round in range(12):
                for u in added_unions:
                    try:
                        manager.remove(u)
                    except Exception:
                        pass
                added_unions = []
                existing = list(existing_before)

                size_scale = float(size_scale0)
                offset_range = float(np.clip(offset0 * (0.95 ** _round), 0.08, 0.22))

                failed = False
                for union_idx in range(num_unions):
                    lvl = int(level)
                    base_sep_min = {7: 1.10, 8: 1.22, 9: 1.35, 10: 1.35}.get(lvl, 1.10)
                    base_sep_max = {7: 1.60, 8: 1.80, 9: 2.00, 10: 2.10}.get(lvl, 1.60)
                    sep_min = float(base_sep_min + 0.03 * float(cal))
                    sep_max = float(base_sep_max + 0.05 * float(cal))
                    
                    placed = place_union_obstacle(
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
                        robot_radius=robot_radius,
                        obstacle_radius_scale=obstacle_radius_scale,
                        max_attempts=900,
                        offset_range=offset_range,
                        corridor_width=0.0,
                        region_low_override=pref_low,
                        region_high_override=pref_high,
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
                    existing.append((base_center, float(union_radius) * float(spacing_shrink) + robot_radius))

                if not failed and enable_connectivity_check:
                    last_connect_ok = has_free_space_path(
                        manager,
                        start=start_pos,
                        target=target_pos,
                        p_max=float(p_max),
                        robot_radius=float(robot_radius),
                        grid_res=0.06 if level >= 9 else 0.05,
                    )
                    if not last_connect_ok:
                        failed = True

                if not failed and target_nonconv is not None:
                    x_min = float(map_bounds.get('x_min', -1.5))
                    x_max = float(map_bounds.get('x_max', 1.0))
                    y_min = float(map_bounds.get('y_min', -2.0))
                    y_max = float(map_bounds.get('y_max', 0.5))
                    last_nonconv_sdf_raw = float(
                        compute_nonconvexity_score_sdf(
                            manager,
                            p_max=float(p_max),
                            robot_radius=float(robot_radius),
                            seed=int(seed),
                            n_pairs=6000,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
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
            missing = num_unions - len(added_unions)
            if missing > 0:
                print(f"Warning: Could not place {missing}/{num_unions} union(s) with proper spacing after retries")
            else:
                if target_nonconv is not None:
                    n_msg = "n/a" if last_nonconv_sdf_raw is None else f"{last_nonconv_sdf_raw:.4f}"
                    print(
                        f"Warning: placed all unions but gates failed in retries "
                        f"(connect_ok={last_connect_ok}, nonconv_target={target_nonconv:.3f}, got={n_msg})."
                    )
                else:
                    print(f"Warning: placed all unions but connectivity gate failed in retries (last_connect_ok={last_connect_ok}).")

    min_clearance = robot_radius + START_GOAL_CLEARANCE_MARGIN
    if len(manager) > 0:
        start_pt = np.asarray(start_pos, dtype=np.float32).reshape(-1)[:2]
        target_pt = np.asarray(target_pos, dtype=np.float32).reshape(-1)[:2]
        sdf_start = manager.sdf(start_pt)
        sdf_start = float(np.asarray(sdf_start).item() if hasattr(sdf_start, "item") else sdf_start)
        if sdf_start < min_clearance or manager.contains(start_pt):
            raise RuntimeError(
                f"Start position is in/too close to an obstacle (sdf={sdf_start:.6f}, need >= {min_clearance:.4f}). "
                "Obstacle generation should prevent this."
            )
        sdf_goal = manager.sdf(target_pt)
        sdf_goal = float(np.asarray(sdf_goal).item() if hasattr(sdf_goal, "item") else sdf_goal)
        if sdf_goal < min_clearance or manager.contains(target_pt):
            raise RuntimeError(
                f"Goal/target position is in/too close to an obstacle (sdf={sdf_goal:.6f}, need >= {min_clearance:.4f}). "
                "Obstacle generation should prevent this."
            )

    return manager
