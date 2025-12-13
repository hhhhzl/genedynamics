"""
Collision detection utilities.

This module provides utility functions for collision detection,
including batch operations and JAX support.
"""

import numpy as np
from typing import List, Union

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from enerdynamics.envs.obstacles.base import Obstacle, ObstacleManager


def check_collision_point(
    point: np.ndarray,
    obstacles: Union[List[Obstacle], ObstacleManager]
) -> bool:
    """
    Check if a single point collides with any obstacle.
    
    Args:
        point: Point to check, shape (dim,)
        obstacles: List of obstacles or ObstacleManager
        
    Returns:
        True if collision detected, False otherwise
    """
    point = np.asarray(point, dtype=np.float32)
    
    if isinstance(obstacles, ObstacleManager):
        return obstacles.contains(point)
    
    # Check each obstacle
    for obstacle in obstacles:
        if obstacle.contains(point):
            return True
    
    return False


def check_collision_batch(
    points: np.ndarray,
    obstacles: Union[List[Obstacle], ObstacleManager]
) -> np.ndarray:
    """
    Check collision for a batch of points.
    
    Args:
        points: Points to check, shape (N, dim)
        obstacles: List of obstacles or ObstacleManager
        
    Returns:
        Boolean array, shape (N,), True if collision detected
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    collisions = np.zeros(len(points), dtype=bool)
    
    if isinstance(obstacles, ObstacleManager):
        for i, point in enumerate(points):
            collisions[i] = obstacles.contains(point)
    else:
        for i, point in enumerate(points):
            for obstacle in obstacles:
                if obstacle.contains(point):
                    collisions[i] = True
                    break
    
    return collisions


def compute_distances(
    points: np.ndarray,
    obstacles: Union[List[Obstacle], ObstacleManager]
) -> np.ndarray:
    """
    Compute distances from points to obstacles.
    
    Args:
        points: Points to evaluate, shape (N, dim) or (dim,)
        obstacles: List of obstacles or ObstacleManager
        
    Returns:
        Distance array, shape (N,) or scalar
    """
    points = np.asarray(points, dtype=np.float32)
    single_point = points.ndim == 1
    if single_point:
        points = points.reshape(1, -1)
    
    if isinstance(obstacles, ObstacleManager):
        distances = np.array([obstacles.distance(p) for p in points])
    else:
        # For each point, find minimum distance to any obstacle
        distances = []
        for point in points:
            min_dist = min(obs.distance(point) for obs in obstacles)
            distances.append(min_dist)
        distances = np.array(distances)
    
    return distances[0] if single_point else distances


def compute_sdf_batch(
    points: np.ndarray,
    obstacles: Union[List[Obstacle], ObstacleManager]
) -> np.ndarray:
    """
    Compute SDF for a batch of points.
    
    Args:
        points: Points to evaluate, shape (N, dim)
        obstacles: List of obstacles or ObstacleManager
        
    Returns:
        SDF array, shape (N,)
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if isinstance(obstacles, ObstacleManager):
        return obstacles.sdf(points)
    
    # Compute SDF for each obstacle and take minimum (union)
    sdfs = []
    for obstacle in obstacles:
        sdf_vals = obstacle.sdf(points)
        sdfs.append(sdf_vals)
    
    sdf_array = np.stack(sdfs, axis=0)
    min_sdf = np.min(sdf_array, axis=0)
    
    return min_sdf


# ============================================================================
# JAX-compatible functions (if JAX is available)
# ============================================================================

if jnp is not None:
    
    def jax_check_collision_batch(
        points: jnp.ndarray,
        obstacles: List[Obstacle]
    ) -> jnp.ndarray:
        """
        JAX-compatible batch collision checking.
        
        Note: This requires obstacles with JAX-compatible contains() method.
        Most obstacles will need to be converted to JAX arrays first.
        
        Args:
            points: Points to check, shape (N, dim) (JAX array)
            obstacles: List of obstacles (must support JAX)
            
        Returns:
            Boolean array, shape (N,)
        """
        # For now, convert to numpy and back
        # In future, can implement fully JAX-native version
        points_np = np.asarray(points)
        collisions_np = check_collision_batch(points_np, obstacles)
        return jnp.asarray(collisions_np)
    
    def jax_compute_sdf_batch(
        points: jnp.ndarray,
        obstacles: List[Obstacle]
    ) -> jnp.ndarray:
        """
        JAX-compatible batch SDF computation.
        
        Args:
            points: Points to evaluate, shape (N, dim) (JAX array)
            obstacles: List of obstacles (must support JAX)
            
        Returns:
            SDF array, shape (N,)
        """
        # For now, convert to numpy and back
        # In future, can implement fully JAX-native version
        points_np = np.asarray(points)
        sdf_np = compute_sdf_batch(points_np, obstacles)
        return jnp.asarray(sdf_np)
    
    def jax_vmap_collision_check(obstacles: List[Obstacle]):
        """
        Create a JAX vmap'd collision checking function.
        
        Args:
            obstacles: List of obstacles
            
        Returns:
            Vectorized collision checking function
        """
        def check_single(point):
            # Convert to numpy for now
            point_np = np.asarray(point)
            return jnp.asarray(check_collision_point(point_np, obstacles))
        
        return jax.vmap(check_single)
