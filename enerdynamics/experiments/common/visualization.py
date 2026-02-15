"""
Common visualization utilities.

This module provides shared visualization functions used across
different visualization plugins.
"""

from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from enerdynamics.envs.obstacles.nonconvex import UnionObstacle

# Alias for 2D: SphereObstacle is CircleObstacle in 2D
CircleObstacle = SphereObstacle

# Default visualization constants
OBSTACLE_COLOR = "gray"
OBSTACLE_ALPHA = 0.5
EDOC_COLOR = "#1f77b4"  # Blue
MAX_SAMPLE_TRAJ_PLOT = 80

# D3IL-only palette (trajectory/diffusion): (R,G,B) or (R,G,B,A) in [0,1]
D3IL_BG_YELLOW = "#FFFFE6"
D3IL_OBSTACLE_RED = "#F21B1B"
D3IL_TARGET_GREEN = "#88F26B"


def is_d3il_experiment(exp_cfg: Any) -> bool:
    """True if this experiment is D3IL (method==d3il_unified or env_name contains 'd3il'). Only D3IL uses the yellow/red/green palette."""
    if exp_cfg is None:
        return False
    method = exp_cfg.get("method") if isinstance(exp_cfg, dict) else getattr(exp_cfg, "method", None)
    env_name = exp_cfg.get("env_name", "") if isinstance(exp_cfg, dict) else getattr(exp_cfg, "env_name", "")
    return method == "d3il_unified" or ("d3il" in str(env_name))


def draw_obstacles(
    ax: plt.Axes,
    obstacles: ObstacleManager,
    obstacle_color: Optional[str] = None,
    obstacle_alpha: Optional[float] = None,
) -> None:
    """
    Draw obstacles on matplotlib axes.

    Args:
        ax: Matplotlib axes
        obstacles: Obstacle manager instance
        obstacle_color: Optional face color for obstacles (default: OBSTACLE_COLOR). Used by D3IL for red.
        obstacle_alpha: Optional alpha for obstacles (default: OBSTACLE_ALPHA).
    """
    face = obstacle_color if obstacle_color is not None else OBSTACLE_COLOR
    alpha = obstacle_alpha if obstacle_alpha is not None else OBSTACLE_ALPHA
    for obstacle in obstacles:
        if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
            center = np.asarray(obstacle.center, dtype=np.float32)
            # Ensure 2D
            if center.ndim == 0:
                center = np.array([center, 0.0], dtype=np.float32)
            elif len(center) > 2:
                center = center[:2]
            elif len(center) == 1:
                center = np.concatenate([center, [0.0]])
            
            radius = float(obstacle.radius)
            circle = plt.Circle(
                tuple(center),
                radius,
                facecolor=face,
                alpha=alpha,
                edgecolor='darkgray',
                linewidth=1.5
            )
            ax.add_patch(circle)
        
        elif isinstance(obstacle, BoxObstacle):
            center = np.asarray(obstacle.center, dtype=np.float32)
            half_extents = np.asarray(obstacle.half_extents, dtype=np.float32)
            
            # Ensure 2D
            if center.ndim == 0:
                center = np.array([center, 0.0], dtype=np.float32)
            elif len(center) > 2:
                center = center[:2]
            elif len(center) == 1:
                center = np.concatenate([center, [0.0]])
            
            if half_extents.ndim == 0:
                half_extents = np.array([half_extents, half_extents], dtype=np.float32)
            elif len(half_extents) > 2:
                half_extents = half_extents[:2]
            elif len(half_extents) == 1:
                half_extents = np.array([half_extents[0], half_extents[0]], dtype=np.float32)
            
            rect = mpatches.Rectangle(
                tuple(center - half_extents),
                2 * float(half_extents[0]),
                2 * float(half_extents[1]),
                facecolor=face,
                alpha=alpha,
                edgecolor='darkgray',
                linewidth=1.5
            )
            ax.add_patch(rect)
        
        elif isinstance(obstacle, UnionObstacle):
            # Draw all primitives in union
            for prim in obstacle.obstacles:
                if isinstance(prim, (SphereObstacle, CircleObstacle)):
                    center = np.asarray(prim.center, dtype=np.float32)
                    if center.ndim == 0:
                        center = np.array([center, 0.0], dtype=np.float32)
                    elif len(center) > 2:
                        center = center[:2]
                    elif len(center) == 1:
                        center = np.concatenate([center, [0.0]])
                    
                    radius = float(prim.radius)
                    circle = plt.Circle(
                        tuple(center),
                        radius,
                        facecolor=face,
                        alpha=alpha * 0.8,
                        edgecolor='darkgray',
                        linewidth=1.2
                    )
                    ax.add_patch(circle)
                
                elif isinstance(prim, BoxObstacle):
                    center = np.asarray(prim.center, dtype=np.float32)
                    half_extents = np.asarray(prim.half_extents, dtype=np.float32)
                    
                    if center.ndim == 0:
                        center = np.array([center, 0.0], dtype=np.float32)
                    elif len(center) > 2:
                        center = center[:2]
                    elif len(center) == 1:
                        center = np.concatenate([center, [0.0]])
                    
                    if half_extents.ndim == 0:
                        half_extents = np.array([half_extents, half_extents], dtype=np.float32)
                    elif len(half_extents) > 2:
                        half_extents = half_extents[:2]
                    elif len(half_extents) == 1:
                        half_extents = np.array([half_extents[0], half_extents[0]], dtype=np.float32)
                    
                    rect = mpatches.Rectangle(
                        tuple(center - half_extents),
                        2 * float(half_extents[0]),
                        2 * float(half_extents[1]),
                        facecolor=face,
                        alpha=alpha * 0.8,
                        edgecolor='darkgray',
                        linewidth=1.2
                    )
                    ax.add_patch(rect)

