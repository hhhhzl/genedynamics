"""
Obstacle factory for deploy pipeline.

Builds ObstacleManager from TaskConfig.obstacles for obstacle_avoid task.
Supports box2d (2D), box3d (3D), and quadruped_3d (3D ground-level) generators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from genedynamics.deploy.task_config import ObstacleConfig, TaskConfig
from genedynamics.envs.obstacles.base import ObstacleManager


def make_obstacles_from_task(
    task: Optional[TaskConfig],
    *,
    seed: int = 0,
    start_pos: Optional[np.ndarray] = None,
    target_pos: Optional[np.ndarray] = None,
    robot_radius_override: Optional[float] = None,
) -> Optional[ObstacleManager]:
    """
    Build ObstacleManager from TaskConfig.obstacles.

    Returns None when task is None, task.obstacles is None, or level==0.

    Args:
        task: TaskConfig (may be None)
        seed: Random seed for obstacle generation
        start_pos: Start position for clearance (2D or 3D)
        target_pos: Target position for clearance (2D or 3D)
        robot_radius_override: Override robot_radius from config

    Returns:
        ObstacleManager or None
    """
    if task is None or task.obstacles is None:
        return None
    obs_cfg = task.obstacles
    if obs_cfg.level <= 0 and obs_cfg.num_obstacles <= 0:
        return None

    generator = obs_cfg.generator or _infer_generator(obs_cfg)
    level = obs_cfg.level
    num_obs = obs_cfg.num_obstacles
    if num_obs > 0:
        level = max(level, (num_obs + 1) // 2)

    robot_radius = robot_radius_override if robot_radius_override is not None else obs_cfg.robot_radius
    config: Dict[str, Any] = {
        "robot_radius": robot_radius,
        "obstacle_radius_scale": obs_cfg.obstacle_radius_scale,
        "min_obstacle_margin": obs_cfg.min_obstacle_margin,
        "obstacle_type": obs_cfg.type if obs_cfg.type in ("sphere", "box") else "sphere",
        "obstacle_radius_by_level": obs_cfg.obstacle_radius_by_level,
    }
    if obs_cfg.map_bounds:
        config["map_bounds"] = obs_cfg.map_bounds

    start = np.asarray(start_pos, dtype=np.float32).reshape(-1) if start_pos is not None else np.zeros(3)
    target = np.asarray(target_pos, dtype=np.float32).reshape(-1) if target_pos is not None else np.array([2.0, 0.0, 0.5])
    if start.size < 3:
        start = np.pad(start, (0, 3 - start.size), constant_values=0.0)
    if target.size < 3:
        target = np.pad(target, (0, 3 - target.size), constant_values=0.0)

    if generator == "box2d":
        return _generate_box2d(level, seed, start, target, config)
    if generator in ("box3d", "quadruped_3d"):
        return _generate_box3d(level, seed, start, target, config)
    return _generate_box3d(level, seed, start, target, config)


def _infer_generator(obs_cfg: ObstacleConfig) -> str:
    """Infer generator from obstacle config."""
    if obs_cfg.generator:
        return obs_cfg.generator
    if obs_cfg.type == "box" and getattr(obs_cfg, "map_bounds", None):
        bounds = obs_cfg.map_bounds
        if isinstance(bounds, dict) and "z_min" in bounds:
            return "box3d"
    return "box3d"


def _generate_box2d(
    level: int,
    seed: int,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    config: Dict[str, Any],
) -> ObstacleManager:
    """Generate 2D obstacles via common obstacle_generation."""
    try:
        from genedynamics.experiments.common.obstacle_generation import generate_box2d_obstacles
    except ImportError:
        return ObstacleManager()

    robot_radius = config.get("robot_radius", 0.05)
    gen_config = {
        "robot_radius": robot_radius,
        "obstacle_radius_scale": config.get("obstacle_radius_scale", 1.1),
        "min_obstacle_margin": config.get("min_obstacle_margin", 2.4 * robot_radius),
    }
    return generate_box2d_obstacles(level, seed, start_pos[:2], target_pos[:2], gen_config)


def _generate_box3d(
    level: int,
    seed: int,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    config: Dict[str, Any],
) -> ObstacleManager:
    """Generate 3D obstacles (sphere/box) for quadruped/drone."""
    try:
        from genedynamics.experiments.plugins.obstacles.box3d import Box3DObstacleGeneratorPlugin
        plugin = Box3DObstacleGeneratorPlugin()
        return plugin.generate(level, seed, start_pos, target_pos, config)
    except ImportError:
        from genedynamics.envs.obstacles.base import ObstacleManager
        from genedynamics.envs.obstacles.convex import SphereObstacle

        np.random.seed(seed)
        obstacles = ObstacleManager()
        if level <= 0:
            return obstacles
        robot_radius = config.get("robot_radius", 0.1)
        scale = config.get("obstacle_radius_scale", 1.1)
        margin = config.get("min_obstacle_margin", 2.4 * robot_radius)
        map_bounds = config.get("map_bounds", {})
        x_min = map_bounds.get("x_min", -1.5)
        x_max = map_bounds.get("x_max", 1.5)
        y_min = map_bounds.get("y_min", -1.5)
        y_max = map_bounds.get("y_max", 1.5)
        z_min = map_bounds.get("z_min", 0.0)
        z_max = map_bounds.get("z_max", 1.0)
        n = min(level * 2, 20)
        for i in range(n):
            center = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(z_min, z_max),
            ], dtype=np.float32)
            r = robot_radius * scale * (0.8 + 0.4 * np.random.rand())
            if np.linalg.norm(center - start_pos[:3]) < r + robot_radius + margin:
                continue
            if np.linalg.norm(center - target_pos[:3]) < r + robot_radius + margin:
                continue
            obstacles.add(SphereObstacle(center=center, radius=float(r), name=f"sphere3d_{i}"))
        return obstacles
