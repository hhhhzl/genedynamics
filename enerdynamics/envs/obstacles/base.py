"""
Base obstacle interface and manager.

This module defines the Obstacle protocol that all obstacles should implement,
and provides an ObstacleManager for managing multiple obstacles.
"""

from typing import Protocol, Optional, List, Any, runtime_checkable
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


@runtime_checkable
class Obstacle(Protocol):
    """
    Protocol for obstacle interface.
    
    Obstacles represent physical or virtual barriers in the environment.
    They support:
    - Signed Distance Function (SDF) computation
    - Point-in-obstacle checking
    - Distance computation
    - Gradient computation (for optimization)
    - Backend conversion (for Rust/GPU acceleration)
    
    Attributes:
        center: Center point of obstacle (np.ndarray, optional)
        bounds: Bounding box (min, max) (tuple, optional)
    """
    
    center: Optional[np.ndarray] = None
    bounds: Optional[tuple] = None
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute Signed Distance Function (SDF) for given points.
        
        SDF returns:
        - Negative values: point is inside obstacle
        - Zero: point is on obstacle boundary
        - Positive values: point is outside obstacle (distance to boundary)
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        ...
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside obstacle.
        
        Args:
            point: Point to check, shape (dim,)
            
        Returns:
            True if point is inside obstacle, False otherwise
        """
        ...
    
    def distance(self, point: np.ndarray) -> float:
        """
        Compute shortest distance from point to obstacle boundary.
        
        Args:
            point: Point to evaluate, shape (dim,)
            
        Returns:
            Distance (always non-negative)
        """
        ...
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute gradient of SDF at point (for optimization).
        
        Args:
            point: Point to evaluate, shape (dim,)
            
        Returns:
            Gradient vector, shape (dim,)
        """
        ...
    
    def to_backend(self, backend: str) -> Any:
        """
        Convert obstacle to backend-specific representation.
        
        This allows obstacles to be used with different backends:
        - "rust": Convert to Rust obstacle struct
        - "jax": Convert to JAX-compatible representation
        - "torch": Convert to PyTorch representation
        - "mujoco": Convert to MuJoCo geom
        - "isaac": Convert to Isaac Sim prim
        
        Args:
            backend: Backend name
            
        Returns:
            Backend-specific obstacle representation
        """
        ...


class ObstacleManager:
    """
    Manager for multiple obstacles.
    
    Provides efficient batch operations for collision detection,
    distance computation, and SDF evaluation across multiple obstacles.
    """
    
    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        """
        Initialize obstacle manager.
        
        Args:
            obstacles: Initial list of obstacles
        """
        self.obstacles: List[Obstacle] = obstacles or []
    
    def add(self, obstacle: Obstacle) -> None:
        """
        Add an obstacle to the manager.
        
        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)
    
    def remove(self, obstacle: Obstacle) -> None:
        """
        Remove an obstacle from the manager.
        
        Args:
            obstacle: Obstacle to remove
        """
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
    
    def clear(self) -> None:
        """Remove all obstacles."""
        self.obstacles.clear()
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside any obstacle.
        
        Args:
            point: Point to check, shape (dim,)
            
        Returns:
            True if point is inside any obstacle
        """
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True
        return False
    
    def distance(self, point: np.ndarray) -> float:
        """
        Compute shortest distance from point to any obstacle.
        
        Args:
            point: Point to evaluate, shape (dim,)
            
        Returns:
            Minimum distance to any obstacle
        """
        if not self.obstacles:
            return float('inf')
        
        distances = [obstacle.distance(point) for obstacle in self.obstacles]
        return min(distances)
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for points (union of all obstacles).
        
        For union, we take the minimum SDF (closest obstacle).
        Negative values indicate point is inside at least one obstacle.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        if not self.obstacles:
            # No obstacles: all points are outside
            if points.ndim == 1:
                return float('inf')
            else:
                return np.full(len(points), float('inf'))
        
        # Compute SDF for each obstacle
        sdfs = []
        for obstacle in self.obstacles:
            sdf_vals = obstacle.sdf(points)
            sdfs.append(sdf_vals)
        
        # Union: take minimum (closest obstacle)
        sdf_array = np.stack(sdfs, axis=0)
        min_sdf = np.min(sdf_array, axis=0)
        
        return min_sdf
    
    def collision_check(self, state: np.ndarray) -> bool:
        """
        Check if state collides with any obstacle.
        
        Extracts position from state (assumes first 3 dims are position)
        and checks collision.
        
        Args:
            state: State to check
            
        Returns:
            True if collision detected
        """
        state_np = np.asarray(state, dtype=np.float32)
        
        # Extract position (assume first 3 dimensions)
        if len(state_np) >= 3:
            position = state_np[:3]
        else:
            position = state_np
        
        return self.contains(position)
    
    def __len__(self) -> int:
        """Return number of obstacles."""
        return len(self.obstacles)
    
    def __iter__(self):
        """Iterate over obstacles."""
        return iter(self.obstacles)
