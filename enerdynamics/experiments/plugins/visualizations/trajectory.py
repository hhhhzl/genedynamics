"""
Trajectory visualization plugin.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from enerdynamics.core.types import Trajectory
from ...framework.base import VisualizationPlugin
from ...common.visualization import draw_obstacles, EDOC_COLOR


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
        
        # Draw obstacles
        draw_obstacles(ax, obstacles)
        
        # Draw trajectory
        if len(trajectory.states) > 1:
            positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in trajectory.states])
            color = config.get('color', EDOC_COLOR)
            ax.plot(positions[:, 0], positions[:, 1], color=color, linewidth=2.0, label='Trajectory')
            
            # Draw start with robot radius
            robot_radius = float(obstacle_config.get('robot_radius', 0.05))
            start_circle = plt.Circle(
                tuple(positions[0]),
                robot_radius,
                facecolor='white',
                edgecolor=color,
                linewidth=1.5,
                label='Start'
            )
            ax.add_patch(start_circle)
            
            # Draw end with robot radius
            end_circle = plt.Circle(
                tuple(positions[-1]),
                robot_radius,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(end_circle)
        
        # Draw target with margin
        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)
        robot_radius = float(obstacle_config.get('robot_radius', 0.05))
        target_circle = plt.Circle(
            tuple(target_pos),
            robot_radius * 2,  # Robot diameter
            facecolor='none',
            edgecolor='red',
            linewidth=1.5,
            linestyle='--',
            label='Target (margin)'
        )
        ax.add_patch(target_circle)
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
        
        title = config.get('title', 'Final Trajectory')
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
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

