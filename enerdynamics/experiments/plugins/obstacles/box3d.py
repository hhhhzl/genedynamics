"""
Box3D obstacle generator plugin.

This plugin generates 3D obstacle configurations for drone environments.
It generates spherical or box obstacles in 3D space.
"""

from typing import Dict, Any
import numpy as np

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from ...framework.base import ObstacleGeneratorPlugin


class Box3DObstacleGeneratorPlugin(ObstacleGeneratorPlugin):
    """
    Plugin for generating 3D obstacle configurations for drone environments.
    
    This plugin generates spherical or box obstacles in 3D space with
    difficulty levels similar to the 2D version.
    """
    
    @property
    def name(self) -> str:
        """Generator name identifier."""
        return "box3d"
    
    def generate(self, level: int, seed: int, start_pos: np.ndarray,
                 target_pos: np.ndarray, config: Dict[str, Any]) -> ObstacleManager:
        """
        Generate 3D obstacles for given level and configuration.
        
        Args:
            level: Obstacle difficulty level (0-10)
            seed: Random seed for reproducibility
            start_pos: Start position (3D array: [x, y, z])
            target_pos: Target position (3D array: [x, y, z])
            config: Obstacle generation configuration:
                - robot_radius: Robot radius (default: 0.1)
                - map_bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
                - p_max: Position bounds (default: 2.0)
                - obstacle_radius_scale: Scale factor (default: 1.3)
                - min_obstacle_margin: Minimum margin (default: 2.4 * robot_radius)
                - obstacle_type: "sphere" or "box" (default: "sphere")
                
        Returns:
            ObstacleManager instance with generated obstacles
        """
        np.random.seed(seed)
        
        # Extract configuration
        robot_radius = config.get('robot_radius', 0.1)
        obstacle_radius_scale = config.get('obstacle_radius_scale', 1.3)
        min_obstacle_margin = config.get('min_obstacle_margin', 2.4 * robot_radius)
        p_max = config.get('p_max', 2.0)
        obstacle_type = config.get('obstacle_type', 'sphere')
        
        map_bounds = config.get('map_bounds', {
            'x_min': -1.5,
            'x_max': 1.5,
            'y_min': -1.5,
            'y_max': 1.5,
            'z_min': 0.0,
            'z_max': 2.0,
        })
        
        x_min = map_bounds.get('x_min', -1.5)
        x_max = map_bounds.get('x_max', 1.5)
        y_min = map_bounds.get('y_min', -1.5)
        y_max = map_bounds.get('y_max', 1.5)
        z_min = map_bounds.get('z_min', 0.0)
        z_max = map_bounds.get('z_max', 2.0)
        
        # Extract 3D position from start_pos and target_pos
        # start_pos and target_pos might be full state vectors (e.g., 6D for drone: x,y,z,vx,vy,vz)
        # or just positions (3D: x,y,z)
        start_pos = np.asarray(start_pos, dtype=np.float32).flatten()
        target_pos = np.asarray(target_pos, dtype=np.float32).flatten()
        
        # Extract first 3 elements as position (x, y, z)
        if start_pos.size >= 3:
            start_pos = start_pos[:3]
        elif start_pos.size == 2:
            # 2D position, pad with z=0
            start_pos = np.concatenate([start_pos, [0.0]])
        elif start_pos.size == 1:
            # 1D position, pad with y=0, z=0
            start_pos = np.concatenate([start_pos, [0.0, 0.0]])
        
        if target_pos.size >= 3:
            target_pos = target_pos[:3]
        elif target_pos.size == 2:
            # 2D position, pad with z=0
            target_pos = np.concatenate([target_pos, [0.0]])
        elif target_pos.size == 1:
            # 1D position, pad with y=0, z=0
            target_pos = np.concatenate([target_pos, [0.0, 0.0]])
        
        obstacles = ObstacleManager()
        
        # Level 0: no obstacles
        if level == 0:
            return obstacles
        
        # Calculate number of obstacles based on level
        # Similar progression to 2D version
        base_obstacles = max(1, level)
        num_obstacles = min(base_obstacles * 2, 30)  # Cap at reasonable number
        
        # Base radius
        base_radius = robot_radius * obstacle_radius_scale
        
        existing_centers = []
        existing_radii = []
        max_attempts = num_obstacles * 50
        
        for i in range(num_obstacles):
            attempts = 0
            placed = False
            
            while attempts < max_attempts and not placed:
                attempts += 1
                
                # Random position within bounds
                center = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max)
                ], dtype=np.float32)
                
                # Vary radius based on level
                radius_variation = 1.0 + 0.3 * np.random.uniform(-1, 1)
                radius = base_radius * radius_variation
                
                # Check clearance from start and target
                start_dist = np.linalg.norm(center - start_pos)
                target_dist = np.linalg.norm(center - target_pos)
                min_clearance = radius + robot_radius + min_obstacle_margin
                
                if start_dist < min_clearance or target_dist < min_clearance:
                    continue
                
                # Check spacing from existing obstacles
                valid_spacing = True
                for existing_center, existing_radius in zip(existing_centers, existing_radii):
                    dist = np.linalg.norm(center - existing_center)
                    if dist < radius + existing_radius + min_obstacle_margin:
                        valid_spacing = False
                        break
                
                if not valid_spacing:
                    continue
                
                # Create obstacle
                if obstacle_type == 'box':
                    # Create box obstacle
                    half_extent = radius / np.sqrt(3)  # Approximate box size
                    half_extents = np.array([half_extent, half_extent, half_extent], dtype=np.float32)
                    obstacle = BoxObstacle(center=center, half_extents=half_extents, name=f"box3d_{i}")
                else:
                    # Create sphere obstacle (default)
                    obstacle = SphereObstacle(center=center, radius=radius, name=f"sphere3d_{i}")
                
                obstacles.add(obstacle)
                existing_centers.append(center)
                existing_radii.append(radius)
                placed = True
        
        return obstacles

