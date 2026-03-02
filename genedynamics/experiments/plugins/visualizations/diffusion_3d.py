"""
3D Diffusion steps visualization plugin.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

from ...framework.base import VisualizationPlugin
from genedynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from genedynamics.envs.obstacles.nonconvex import UnionObstacle

EDOC_COLOR = "#1f77b4"
OBSTACLE_COLOR = "gray"
OBSTACLE_ALPHA = 0.5
MAX_SAMPLE_TRAJ_PLOT = 80


class Diffusion3DVisualizationPlugin(VisualizationPlugin):
    """
    Plugin for visualizing 3D diffusion steps during planning.
    """
    
    @property
    def name(self) -> str:
        """Visualization name identifier."""
        return "diffusion_3d"
    
    def visualize(self, fig: Any, axes: Any, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Create 3D diffusion steps visualization.
        
        Args:
            fig: Matplotlib figure
            axes: Array of matplotlib axes (one for each diffusion step)
            data: Dictionary containing:
                - result: Planning result dictionary
                - env: Environment instance
                - obstacles: Obstacle manager
                - initial_state: Initial state array
                - env_plugin: Environment plugin
            config: Visualization configuration dictionary
        """
        result = data['result']
        env = data['env']
        obstacles = data['obstacles']
        initial_state = data['initial_state']
        env_plugin = data['env_plugin']
        
        # Get map bounds
        obstacle_config = config.get('config', {}).obstacle_config or {}
        map_bounds = obstacle_config.get('map_bounds', {})
        x_min = map_bounds.get('x_min', -2.0)
        x_max = map_bounds.get('x_max', 2.0)
        y_min = map_bounds.get('y_min', -2.0)
        y_max = map_bounds.get('y_max', 2.0)
        z_min = map_bounds.get('z_min', 0.0)
        z_max = map_bounds.get('z_max', 2.0)
        
        # Get diffusion data
        diffusion_actions = result.get('diffusion_actions_traj', None)
        diffusion_samples = result.get('diffusion_sampled_actions', None)
        diffusion_fractions = config.get('fractions', (0.1, 0.5, 0.9))  # 90%, 50%, 10%
        
        if diffusion_actions is not None and len(diffusion_actions) > 0:
            diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
            Ndiffuse = diffusion_actions.shape[0]
            
            for ax, frac in zip(axes, diffusion_fractions):
                step_idx = int((1.0 - frac) * Ndiffuse)
                step_idx = max(0, min(step_idx, Ndiffuse - 1))
                
                action_seq = diffusion_actions[step_idx]
                sample_acts = None
                if diffusion_samples is not None and len(diffusion_samples) > 0:
                    diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                    if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                        sample_acts = diffusion_samples_arr[step_idx]
                
                self._visualize_single_step_3d(
                    ax, env, obstacles, initial_state, action_seq, sample_acts,
                    env_plugin, x_min, x_max, y_min, y_max, z_min, z_max,
                    title=f"Diffusion {int((1-frac) * 100)}%"
                )
        else:
            # Fallback: show final trajectory
            final_actions = result.get('actions', None)
            for ax, frac in zip(axes, diffusion_fractions):
                self._visualize_single_step_3d(
                    ax, env, obstacles, initial_state, final_actions, None,
                    env_plugin, x_min, x_max, y_min, y_max, z_min, z_max,
                    title=f"Diffusion {int((1-frac) * 100)}% (Final)"
                )
    
    def _visualize_single_step_3d(
        self, ax: Axes3D, env: Any, obstacles: Any, initial_state: np.ndarray,
        action_sequence: np.ndarray, sample_actions: np.ndarray,
        env_plugin: Any, x_min: float, x_max: float, y_min: float, y_max: float,
        z_min: float, z_max: float, title: str = None
    ) -> None:
        """Visualize a single 3D diffusion step."""
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Draw obstacles
        self._draw_3d_obstacles(ax, obstacles)
        
        # Draw sample trajectories
        if sample_actions is not None and len(sample_actions) > 0:
            sample_actions = np.asarray(sample_actions, dtype=np.float32)
            num_samples = min(len(sample_actions), MAX_SAMPLE_TRAJ_PLOT)
            indices = np.linspace(0, len(sample_actions) - 1, num_samples, dtype=int)
            
            light_rgba = mcolors.to_rgba(EDOC_COLOR, alpha=0.2)
            for idx in indices:
                acts = sample_actions[idx]
                sample_traj = env.rollout_actions(initial_state, acts)
                positions = np.array([
                    env_plugin.extract_position(s) for s in sample_traj
                ])
                ax.plot(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    color=light_rgba,
                    linewidth=0.5,
                    alpha=0.3
                )
        
        # Draw main trajectory
        if action_sequence is not None and len(action_sequence) > 0:
            action_sequence = np.asarray(action_sequence, dtype=np.float32)
            traj = env.rollout_actions(initial_state, action_sequence)
            positions = np.array([
                env_plugin.extract_position(s) for s in traj
            ])
            
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=EDOC_COLOR,
                linewidth=2.0,
                label='Trajectory'
            )
            
            # Draw start and end
            ax.scatter(
                positions[0, 0], positions[0, 1], positions[0, 2],
                color='green', s=50, marker='o', label='Start'
            )
            ax.scatter(
                positions[-1, 0], positions[-1, 1], positions[-1, 2],
                color=EDOC_COLOR, s=100, marker='*', label='End'
            )
        
        # Draw target
        target = np.asarray(env.target, dtype=np.float32)
        target_pos = env_plugin.extract_position(target)
        ax.scatter(
            target_pos[0], target_pos[1], target_pos[2],
            color='red', s=150, marker='*', label='Target'
        )
        
        if title:
            ax.set_title(title, fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_3d_obstacles(self, ax: Axes3D, obstacles) -> None:
        """Draw 3D obstacles on axes."""
        for obstacle in obstacles:
            if isinstance(obstacle, SphereObstacle):
                self._draw_sphere(ax, obstacle.center, obstacle.radius)
            elif isinstance(obstacle, BoxObstacle):
                self._draw_box(ax, obstacle.center, obstacle.half_extents)
            elif isinstance(obstacle, UnionObstacle):
                for prim in obstacle.obstacles:
                    if isinstance(prim, SphereObstacle):
                        self._draw_sphere(ax, prim.center, prim.radius)
                    elif isinstance(prim, BoxObstacle):
                        self._draw_box(ax, prim.center, prim.half_extents)
    
    def _draw_sphere(self, ax: Axes3D, center: np.ndarray, radius: float) -> None:
        """Draw a sphere obstacle."""
        center = np.asarray(center, dtype=np.float32)
        if center.size < 3:
            center = np.concatenate([center.flatten(), [0.0]])[:3]
        center = center[:3]
        radius = float(radius)
        
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=OBSTACLE_ALPHA, color=OBSTACLE_COLOR, 
                       edgecolor='darkgray', linewidth=0.3)
    
    def _draw_box(self, ax: Axes3D, center: np.ndarray, half_extents: np.ndarray) -> None:
        """Draw a box obstacle."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
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
        
        hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        
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
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]
        
        collection = Poly3DCollection(faces, alpha=OBSTACLE_ALPHA, facecolor=OBSTACLE_COLOR,
                                     edgecolor='darkgray', linewidth=0.3)
        ax.add_collection3d(collection)
    
    def save(self, output_path: Any, fig: Any, **kwargs: Any) -> None:
        """Save visualization to file."""
        dpi = kwargs.get('dpi', 150)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

