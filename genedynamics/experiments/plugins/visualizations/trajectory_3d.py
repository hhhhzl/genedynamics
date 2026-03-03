"""
3D Trajectory visualization plugin.
"""

from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from genedynamics.core.types import Trajectory
from ...framework.base import VisualizationPlugin
from genedynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from genedynamics.envs.obstacles.nonconvex import UnionObstacle

EDOC_COLOR = "#1f77b4"
OBSTACLE_COLOR = "gray"
OBSTACLE_ALPHA = 0.5


class Trajectory3DVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing 3D trajectories.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "trajectory_3d"
    
    def visualize(self, fig: Any, ax: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create 3D trajectory visualization.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes (if not 3D, will create new 3D axes)
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
        z_min = map_bounds.get('z_min', 0.0)
        z_max = map_bounds.get('z_max', 2.0)
        
        # Ensure we have a 3D axes
        if not isinstance(ax, Axes3D):
            # Clear existing axes and create 3D
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Draw obstacles
        self._draw_3d_obstacles(ax, obstacles)
        
        # Draw trajectory
        if len(trajectory.states) > 1:
            positions = np.array([env_plugin.extract_position(np.asarray(s)) for s in trajectory.states])
            color = config.get('color', EDOC_COLOR)
            
            # Plot trajectory line
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                linewidth=2.0,
                label='Trajectory'
            )
            
            # Draw start point
            robot_radius = float(obstacle_config.get('robot_radius', 0.1))
            ax.scatter(
                positions[0, 0],
                positions[0, 1],
                positions[0, 2],
                color='green',
                s=100,
                marker='o',
                label='Start',
                edgecolors='darkgreen',
                linewidths=1.5
            )
            
            # Draw end point
            ax.scatter(
                positions[-1, 0],
                positions[-1, 1],
                positions[-1, 2],
                color=color,
                s=150,
                marker='*',
                label='End',
                edgecolors='black',
                linewidths=0.5
            )
        
        # Draw target
        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)
        robot_radius = float(obstacle_config.get('robot_radius', 0.1))
        
        ax.scatter(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            color='red',
            s=200,
            marker='*',
            label='Target',
            edgecolors='darkred',
            linewidths=1.5
        )
        
        title = config.get('title', '')
        if title:
            ax.set_title(title, fontsize=12)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _draw_3d_obstacles(self, ax: Axes3D, obstacles) -> None:
        """
        Draw 3D obstacles on axes.
        
        Args:
            ax: 3D matplotlib axes
            obstacles: ObstacleManager instance
        """
        for obstacle in obstacles:
            if isinstance(obstacle, SphereObstacle):
                self._draw_sphere(ax, obstacle.center, obstacle.radius)
            
            elif isinstance(obstacle, BoxObstacle):
                self._draw_box(ax, obstacle.center, obstacle.half_extents)
            
            elif isinstance(obstacle, UnionObstacle):
                # Draw all primitives in union
                for prim in obstacle.obstacles:
                    if isinstance(prim, SphereObstacle):
                        self._draw_sphere(ax, prim.center, prim.radius)
                    elif isinstance(prim, BoxObstacle):
                        self._draw_box(ax, prim.center, prim.half_extents)
    
    def _draw_sphere(self, ax: Axes3D, center: np.ndarray, radius: float) -> None:
        """
        Draw a sphere obstacle.
        
        Args:
            ax: 3D matplotlib axes
            center: Center point [x, y, z]
            radius: Sphere radius
        """
        center = np.asarray(center, dtype=np.float32)
        if center.size < 3:
            # Pad to 3D
            center = np.concatenate([center.flatten(), [0.0]])[:3]
        center = center[:3]
        radius = float(radius)
        
        # Create sphere using wireframe
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=OBSTACLE_ALPHA, color=OBSTACLE_COLOR, edgecolor='darkgray', linewidth=0.5)
    
    def _draw_box(self, ax: Axes3D, center: np.ndarray, half_extents: np.ndarray) -> None:
        """
        Draw a box obstacle.
        
        Args:
            ax: 3D matplotlib axes
            center: Center point [x, y, z]
            half_extents: Half-extents [hx, hy, hz]
        """
        center = np.asarray(center, dtype=np.float32)
        half_extents = np.asarray(half_extents, dtype=np.float32)
        
        if center.size < 3:
            center = np.concatenate([center.flatten(), [0.0]])[:3]
        center = center[:3]
        
        if half_extents.size < 3:
            if half_extents.size == 1:
                half_extents = np.array([half_extents[0], half_extents[0], half_extents[0]], dtype=np.float32)
            elif half_extents.size == 2:
                half_extents = np.concatenate([half_extents.flatten(), [half_extents[0]]])[:3]
        half_extents = half_extents[:3]
        
        # Create box vertices
        hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        
        # 8 vertices of the box
        vertices = np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ])
        
        # Define the 6 faces of the box
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        ]
        
        # Draw box
        collection = Poly3DCollection(faces, alpha=OBSTACLE_ALPHA, facecolor=OBSTACLE_COLOR, 
                                     edgecolor='darkgray', linewidth=0.5)
        ax.add_collection3d(collection)
    
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

