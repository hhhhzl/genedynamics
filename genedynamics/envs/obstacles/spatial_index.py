"""
Spatial indexing for accelerated collision detection.

This module provides spatial data structures (e.g., KD-tree, octree)
for accelerating collision detection and distance queries with obstacles.
"""

from typing import Optional, List, Tuple, Any
import numpy as np

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None


class SpatialIndex:
    """
    Spatial index for accelerating obstacle queries.
    
    Uses KD-tree or similar structure to quickly find nearby obstacles.
    """
    
    def __init__(
        self,
        obstacles: List[Any],
        method: str = "kdtree"
    ):
        """
        Initialize spatial index.
        
        Args:
            obstacles: List of obstacles to index
            method: Indexing method ("kdtree", "octree", etc.)
        """
        self.obstacles = obstacles
        self.method = method
        
        if method == "kdtree":
            self._build_kdtree()
        else:
            self.tree = None
    
    def _build_kdtree(self) -> None:
        """Build KD-tree from obstacle centers."""
        if not SCIPY_AVAILABLE:
            self.tree = None
            return
        
        # Extract obstacle centers
        centers = []
        for obs in self.obstacles:
            if hasattr(obs, 'center') and obs.center is not None:
                center = np.asarray(obs.center, dtype=np.float32)
                if len(center) >= 3:
                    centers.append(center[:3])
                else:
                    centers.append(np.zeros(3, dtype=np.float32))
            else:
                # Use bounding box center
                if hasattr(obs, 'bounds') and obs.bounds is not None:
                    bounds_min, bounds_max = obs.bounds
                    center = (bounds_min + bounds_max) / 2.0
                    centers.append(center[:3] if len(center) >= 3 else np.zeros(3, dtype=np.float32))
                else:
                    centers.append(np.zeros(3, dtype=np.float32))
        
        if centers:
            centers_array = np.array(centers, dtype=np.float32)
            self.tree = cKDTree(centers_array)
        else:
            self.tree = None
    
    def query_nearby(
        self,
        point: np.ndarray,
        radius: float = 1.0,
        k: Optional[int] = None
    ) -> List[int]:
        """
        Find obstacles near a point.
        
        Args:
            point: Query point, shape (3,)
            radius: Search radius
            k: Maximum number of neighbors (if None, use radius)
            
        Returns:
            List of obstacle indices
        """
        point = np.asarray(point, dtype=np.float32)
        if len(point) >= 3:
            point_3d = point[:3]
        else:
            point_3d = np.pad(point, (0, 3 - len(point)), mode='constant')
        
        if self.tree is None:
            # Fallback: return all obstacles
            return list(range(len(self.obstacles)))
        
        if k is not None:
            # Query k nearest
            distances, indices = self.tree.query(point_3d, k=min(k, len(self.obstacles)))
            if np.isscalar(indices):
                return [int(indices)]
            return [int(idx) for idx in indices if distances[idx] <= radius]
        else:
            # Query within radius
            indices = self.tree.query_ball_point(point_3d, radius)
            return [int(idx) for idx in indices]
    
    def query_nearest(
        self,
        point: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find nearest obstacle to point.
        
        Args:
            point: Query point, shape (3,)
            
        Returns:
            Tuple of (obstacle_index, distance)
        """
        point = np.asarray(point, dtype=np.float32)
        if len(point) >= 3:
            point_3d = point[:3]
        else:
            point_3d = np.pad(point, (0, 3 - len(point)), mode='constant')
        
        if self.tree is None:
            # Fallback: return first obstacle
            return 0, float('inf')
        
        distance, index = self.tree.query(point_3d, k=1)
        return int(index), float(distance)
    
    def update(self) -> None:
        """Update spatial index (rebuild if needed)."""
        if self.method == "kdtree":
            self._build_kdtree()


class AcceleratedObstacleManager:
    """
    Obstacle manager with spatial indexing for faster queries.
    
    Wraps ObstacleManager and adds spatial indexing for performance.
    """
    
    def __init__(
        self,
        obstacles: Optional[List[Any]] = None,
        use_spatial_index: bool = True
    ):
        """
        Initialize accelerated obstacle manager.
        
        Args:
            obstacles: Initial list of obstacles
            use_spatial_index: Whether to use spatial indexing
        """
        from genedynamics.envs.obstacles.base import ObstacleManager
        
        self.manager = ObstacleManager(obstacles)
        self.use_spatial_index = use_spatial_index
        
        if use_spatial_index:
            self.spatial_index = SpatialIndex(self.manager.obstacles)
        else:
            self.spatial_index = None
    
    def add(self, obstacle: Any) -> None:
        """Add obstacle and update spatial index."""
        self.manager.add(obstacle)
        if self.use_spatial_index:
            self.spatial_index = SpatialIndex(self.manager.obstacles)
    
    def remove(self, obstacle: Any) -> None:
        """Remove obstacle and update spatial index."""
        self.manager.remove(obstacle)
        if self.use_spatial_index:
            self.spatial_index = SpatialIndex(self.manager.obstacles)
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside any obstacle (accelerated).
        
        Args:
            point: Point to check
            
        Returns:
            True if point is inside any obstacle
        """
        if self.spatial_index is not None:
            # Query nearby obstacles first
            nearby_indices = self.spatial_index.query_nearby(point, radius=10.0)
            for idx in nearby_indices:
                if self.manager.obstacles[idx].contains(point):
                    return True
            return False
        else:
            return self.manager.contains(point)
    
    def distance(self, point: np.ndarray) -> float:
        """
        Compute minimum distance to any obstacle (accelerated).
        
        Args:
            point: Point to evaluate
            
        Returns:
            Minimum distance
        """
        if self.spatial_index is not None:
            # Query nearest obstacle first
            nearest_idx, _ = self.spatial_index.query_nearest(point)
            if nearest_idx < len(self.manager.obstacles):
                return self.manager.obstacles[nearest_idx].distance(point)
            return float('inf')
        else:
            return self.manager.distance(point)
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for points (accelerated).
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values
        """
        # For batch SDF, spatial index helps less, but we can still use it
        # for filtering obstacles
        return self.manager.sdf(points)
    
    def __len__(self) -> int:
        """Return number of obstacles."""
        return len(self.manager)
    
    def __iter__(self):
        """Iterate over obstacles."""
        return iter(self.manager.obstacles)
