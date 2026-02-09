"""
Trajectory visualization plugin.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from enerdynamics.core.types import Trajectory
from ...framework.base import VisualizationPlugin
from ...common.visualization import (
    draw_obstacles,
    EDOC_COLOR,
    is_d3il_experiment,
    D3IL_BG_YELLOW,
    D3IL_OBSTACLE_RED,
    D3IL_TARGET_GREEN,
)


class TrajectoryVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing final trajectories.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "trajectory"
    
    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create trajectory visualization.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes
            data: Dictionary containing:
                - trajectory: Trajectory object
                - env: Environment instance
                - obstacles: Obstacle manager
                - env_plugin: Environment plugin
            config: Visualization configuration dictionary
        """
        trajectory = data['trajectory']
        env = data['env']
        obstacles = data['obstacles']
        env_plugin = data['env_plugin']
        
        # Get map bounds from config
        obstacle_config = config.get('config', {}).obstacle_config or {}
        map_bounds = obstacle_config.get('map_bounds', {})
        x_min = map_bounds.get('x_min', -2.0)
        x_max = map_bounds.get('x_max', 2.0)
        y_min = map_bounds.get('y_min', -2.0)
        y_max = map_bounds.get('y_max', 2.0)
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        exp_cfg = config.get("config")
        is_d3il = is_d3il_experiment(exp_cfg)
        if is_d3il:
            ax.set_facecolor(D3IL_BG_YELLOW)
        # Draw obstacles (D3IL: red; single2d/default: gray)
        if is_d3il:
            draw_obstacles(ax, obstacles, obstacle_color=D3IL_OBSTACLE_RED, obstacle_alpha=1.0)
        else:
            draw_obstacles(ax, obstacles)
        
        partial_until_step = data.get('partial_until_step', None)
        gif_style = data.get('gif_style', False)
        robot_radius = float(obstacle_config.get('robot_radius', 0.05))
        color = config.get('color', EDOC_COLOR)

        # Full trajectory positions (for full line or midpoint / moving robot)
        all_positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in trajectory.states])
        if len(all_positions) < 1:
            all_positions = np.zeros((0, 2))

        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)

        def draw_tail(ax, positions, t, color, robot_radius, zorder=7, alpha_min=0.25, alpha_max=0.75):
            """Draw tail as small circles behind robot at step t (progressively smaller and more transparent)."""
            tail_length = min(5, t)
            if tail_length <= 0:
                return
            # i=1: closest to robot (t-1), i=tail_length: oldest (t-tail_length)
            for i in range(1, tail_length + 1):
                idx = t - i
                if idx < 0:
                    continue
                # Closest to robot (i=1) = largest; oldest (i=tail_length) = smallest
                scale = 0.85 - 0.15 * (i - 1)  # 0.85, 0.70, 0.55, 0.40, 0.25 for i=1..5
                r = robot_radius * scale
                # Closest = less transparent, oldest = more transparent
                div = max(1, tail_length - 1)
                alpha = alpha_max - (alpha_max - alpha_min) * (i - 1) / div
                circle = plt.Circle(
                    tuple(positions[idx]),
                    r,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.3,
                    alpha=alpha,
                    zorder=zorder,
                )
                ax.add_patch(circle)

        # For GIF style: full trajectory line; robot moves; tail from third-way (highly visible).
        if gif_style and partial_until_step is not None:
            if len(all_positions) > 1:
                ax.plot(all_positions[:, 0], all_positions[:, 1], color=color, linewidth=2.0, zorder=4)
            if len(all_positions) > 0:
                ax.scatter(
                    all_positions[0, 0], all_positions[0, 1],
                    marker='o', s=30, facecolors='white', edgecolors=color,
                    linewidths=1.0, zorder=6
                )
            t = min(partial_until_step, len(all_positions) - 1)
            if t >= 0 and len(all_positions) > 0:
                # Tail from first step so it's visible in the early segment (t>=1 gives at least one tail circle)
                if t >= 1:
                    draw_tail(ax, all_positions, t, color, robot_radius, zorder=7, alpha_min=0.25, alpha_max=0.75)
                robot_circle = plt.Circle(
                    tuple(all_positions[t]),
                    robot_radius,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=8
                )
                ax.add_patch(robot_circle)
        else:
            # Static or legacy partial
            states_to_use = trajectory.states
            if partial_until_step is not None and not gif_style:
                end_idx = min(partial_until_step + 1, len(trajectory.states))
                states_to_use = trajectory.states[:end_idx]
            if len(states_to_use) > 1:
                positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in states_to_use])
                ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2.0, zorder=4)
                ax.scatter(
                    positions[0, 0], positions[0, 1],
                    marker='o', s=30, facecolors='white', edgecolors=color,
                    linewidths=1.0, zorder=6
                )
                if partial_until_step is None:
                    # PNG: robot_radius at t closest to midpoint between start and target; add tail
                    start_xy = positions[0]
                    mid_xy = 0.5 * (start_xy + target_pos)
                    dists = np.linalg.norm(positions - mid_xy, axis=1)
                    t_mid = int(np.argmin(dists))
                    t_mid = min(t_mid, len(positions) - 1)
                    tail_length = min(5, t_mid)
                    if t_mid >= 1 and tail_length >= 1:
                        draw_tail(ax, positions, t_mid, color, robot_radius, zorder=7, alpha_min=0.25, alpha_max=0.75)
                    robot_circle = plt.Circle(
                        tuple(positions[t_mid]),
                        robot_radius,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=0.5,
                        zorder=8
                    )
                    ax.add_patch(robot_circle)
                else:
                    # Legacy GIF frame (growing trajectory): head as dot or circle
                    is_partial = partial_until_step < len(trajectory.states) - 1
                    if is_partial:
                        ax.scatter(
                            positions[-1, 0], positions[-1, 1],
                            marker='o', s=40, facecolors=color, edgecolors='black',
                            linewidths=0.5, zorder=6
                        )
                    else:
                        end_circle = plt.Circle(
                            tuple(positions[-1]),
                            robot_radius,
                            facecolor=color,
                            edgecolor='black',
                            linewidth=0.5
                        )
                        ax.add_patch(end_circle)
            elif len(states_to_use) == 1:
                positions = np.array([env_plugin.extract_position(np.asarray(states_to_use[0]))])
                ax.scatter(
                    positions[0, 0], positions[0, 1],
                    marker='o', s=30, facecolors='white', edgecolors=color,
                    linewidths=1.0, zorder=6
                )
        
        # Draw target (no margin circle). D3IL: green; single2d: red.
        target_color = D3IL_TARGET_GREEN if is_d3il else 'r'
        ax.plot(target_pos[0], target_pos[1], color=target_color, marker='*', markersize=15, linestyle='')
        
        title = config.get('title', 'Final Trajectory')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('0.4')
            spine.set_linewidth(0.8)
        ax.grid(True, alpha=0.3)
        if not is_d3il:
            ax.set_facecolor('white')
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Path to save figure
            fig: Matplotlib figure
            **kwargs: Additional save options
        """
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

