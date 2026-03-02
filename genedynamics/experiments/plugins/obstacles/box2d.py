"""
Box2D obstacle generator plugin.

This plugin generates 2D obstacle configurations for box environments
(single/double integrator 2D). It uses the unified obstacle generation
module for code reuse.
"""

from typing import Dict, Any
import numpy as np

from genedynamics.envs.obstacles.base import ObstacleManager
from ...framework.base import ObstacleGeneratorPlugin
from ...common.obstacle_generation import generate_box2d_obstacles


class Box2DObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    """
    Plugin for generating 2D obstacle configurations for box environments.
    
    This plugin uses the unified obstacle generation module to generate
    obstacle configurations for single and double integrator 2D environments.
    """
    
    @property
    def name(self) -> str:
        """Generator name identifier."""
        return "box2d"
    
    def generate(self, level: int, seed: int, start_pos: np.ndarray,
                 target_pos: np.ndarray, config: Dict[str, Any]) -> ObstacleManager:
        """
        Generate obstacles for given level and configuration.
        
        Args:
            level: Obstacle difficulty level (0-10)
            seed: Random seed for reproducibility
            start_pos: Start position (2D array)
            target_pos: Target position (2D array)
            config: Obstacle generation configuration:
                - robot_radius: Robot radius (default: 0.05)
                - map_bounds: Dictionary with x_min, x_max, y_min, y_max
                - p_max: Position bounds (default: 2.0)
                - obstacle_radius_scale: Scale factor (default: 1.1)
                - min_obstacle_margin: Minimum margin (default: 2.4 * robot_radius)
                - enable_connectivity_check: Whether to check connectivity (default: True)
                - enable_nonconvexity_check: Whether to check nonconvexity (default: True for levels 7-9)
                
        Returns:
            ObstacleManager instance with generated obstacles
        """
        # Prepare configuration for generate_box2d_obstacles
        gen_config = {
            'robot_radius': config.get('robot_radius', 0.05),
            'obstacle_radius_scale': config.get('obstacle_radius_scale', 1.1),
            'min_obstacle_margin': config.get('min_obstacle_margin', None),  # Will be computed from robot_radius
            'p_max': config.get('p_max', 2.0),
            'map_bounds': config.get('map_bounds', {
                'x_min': -1.5,
                'x_max': 1.0,
                'y_min': -2.0,
                'y_max': 0.5,
            }),
            'enable_connectivity_check': config.get('enable_connectivity_check', True),
            'enable_nonconvexity_check': config.get('enable_nonconvexity_check', level in [7, 8, 9]),
        }
        
        # Compute min_obstacle_margin if not provided
        if gen_config['min_obstacle_margin'] is None:
            gen_config['min_obstacle_margin'] = 2.4 * gen_config['robot_radius']
        
        return generate_box2d_obstacles(level, seed, start_pos, target_pos, gen_config)

