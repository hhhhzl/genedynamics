"""
Obstacle density metrics plugin.
"""

from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.types import Trajectory
from enerdynamics.envs.obstacles.base import ObstacleManager
from ...framework.base import MetricsPlugin


class ObstacleDensityMetricsPlugin(MetricsPlugin):
    """
    Plugin for computing obstacle density metrics.
    
    Computes the fraction of the workspace occupied by obstacles.
    """
    
    @property
    def name(self) -> str:
        """Metric name identifier."""
        return "obstacle_density"
    
    def compute(self, trajectory: Trajectory, env: Any, obstacles: ObstacleManager,
                constraints: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Compute obstacle density metrics.
        
        Args:
            trajectory: Computed trajectory (not used, but required by interface)
            env: Environment instance
            obstacles: Obstacle manager instance
            constraints: Constraint manager (not used)
            **kwargs: Additional parameters:
                - robot_radius: Robot radius for collision checking (default: 0.05)
                - level: Obstacle level (for seed offset, default: 0)
                - obstacle_config: Obstacle configuration dict (for map bounds)
                - n_samples: Number of samples for density estimation (default: 50000)
                
        Returns:
            Dictionary containing obstacle density metric
        """
        if len(obstacles) == 0:
            return {
                'obstacle_density': 0.0,
                'n_samples': 0,
            }
        
        robot_radius = float(kwargs.get('robot_radius', 0.05))
        level = int(kwargs.get('level', 0))
        n_samples = int(kwargs.get('n_samples', 50000))
        obstacle_config = kwargs.get('obstacle_config', {})
        
        # Get map bounds
        map_bounds = obstacle_config.get('map_bounds', {})
        p_max = float(getattr(env, 'p_max', 2.0))
        x_min = map_bounds.get('x_min', -p_max)
        x_max = map_bounds.get('x_max', p_max)
        y_min = map_bounds.get('y_min', -p_max)
        y_max = map_bounds.get('y_max', p_max)
        
        # Sample points uniformly in the workspace
        # Use deterministic seed based on level for reproducibility
        seed = int(level) + 12345
        rng = np.random.RandomState(seed)
        
        xlo = float(x_min)
        xhi = float(x_max)
        ylo = float(y_min)
        yhi = float(y_max)
        
        pts = np.stack(
            [
                rng.uniform(xlo, xhi, size=(n_samples,)).astype(np.float32),
                rng.uniform(ylo, yhi, size=(n_samples,)).astype(np.float32),
            ],
            axis=-1,
        )
        
        # Check which points are in collision (sdf < robot_radius)
        sdf = obstacles.sdf(pts)
        sdf = np.asarray(sdf, dtype=np.float32).reshape(-1)
        occupied = sdf < robot_radius
        
        density = float(np.mean(occupied))
        
        return {
            'obstacle_density': density,
            'n_samples': n_samples,
            'map_bounds': {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
            },
        }

