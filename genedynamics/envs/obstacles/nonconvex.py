"""
Non-convex obstacle implementations.

This module provides implementations of non-convex obstacles:
- MeshObstacle: Mesh-based obstacles (from STL, OBJ, etc.)
- UnionObstacle: Union of multiple obstacles
- DifferenceObstacle: Difference of obstacles (A - B)
- IntersectionObstacle: Intersection of obstacles (A ∩ B)

These obstacles support CSG (Constructive Solid Geometry) operations.
"""

from typing import Optional, List, Any
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from genedynamics.envs.obstacles.base import Obstacle

# Import convex obstacles for CSG operations
from genedynamics.envs.obstacles.convex import (
    BoxObstacle,
    SphereObstacle,
)


class MeshObstacle(Obstacle):
    """
    Mesh-based obstacle from 3D mesh file (STL, OBJ, PLY, etc.).
    
    Uses trimesh or similar library to load and compute SDF.
    """
    
    def __init__(
        self,
        mesh_path: Optional[str] = None,
        mesh: Optional[Any] = None,
        name: Optional[str] = None,
        scale: float = 1.0,
        translation: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
    ):
        """
        Initialize mesh obstacle.
        
        Args:
            mesh_path: Path to mesh file (STL, OBJ, PLY, etc.)
            mesh: Mesh object (trimesh.Trimesh or similar)
            name: Optional name for the obstacle
            scale: Scale factor for mesh
            translation: Translation vector, shape (3,)
            rotation: Rotation matrix or quaternion
        """
        self.name = name or "mesh"
        self.scale = float(scale)
        self.translation = np.asarray(translation, dtype=np.float32) if translation is not None else np.zeros(3, dtype=np.float32)
        self.rotation = np.asarray(rotation, dtype=np.float32) if rotation is not None else None
        
        # Load mesh
        if mesh is not None:
            self.mesh = mesh
        elif mesh_path is not None:
            self.mesh = self._load_mesh(mesh_path)
        else:
            raise ValueError("Either mesh_path or mesh must be provided")
        
        # Compute bounding box
        self.bounds = self._compute_bounds()
        
        # Center is centroid
        self.center = np.mean(self.mesh.vertices, axis=0).astype(np.float32)
    
    def _load_mesh(self, mesh_path: str) -> Any:
        """Load mesh from file."""
        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                # If scene, get first mesh
                mesh = list(mesh.geometry.values())[0]
            
            # Apply transformations
            if self.scale != 1.0:
                mesh.apply_scale(self.scale)
            if np.any(self.translation != 0):
                mesh.apply_translation(self.translation)
            if self.rotation is not None:
                if self.rotation.shape == (3, 3):
                    mesh.apply_transform(np.eye(4))
                    mesh.vertices = (self.rotation @ mesh.vertices.T).T
                # TODO: Handle quaternion rotation
            
            return mesh
        except ImportError:
            raise ImportError(
                "MeshObstacle requires trimesh. Install with: pip install trimesh"
            )
    
    def _compute_bounds(self) -> tuple:
        """Compute bounding box."""
        if hasattr(self.mesh, 'bounds'):
            bounds = self.mesh.bounds
            return (
                np.asarray(bounds[0], dtype=np.float32),
                np.asarray(bounds[1], dtype=np.float32)
            )
        elif hasattr(self.mesh, 'vertices'):
            vertices = np.asarray(self.mesh.vertices)
            return (
                np.min(vertices, axis=0).astype(np.float32),
                np.max(vertices, axis=0).astype(np.float32)
            )
        else:
            return (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for mesh.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Use trimesh's proximity query
        try:
            import trimesh
            if hasattr(self.mesh, 'nearest'):
                # Trimesh has nearest point query
                distances = []
                for point in points:
                    closest, distance, _ = self.mesh.nearest.on_surface([point])
                    # Check if point is inside
                    if hasattr(self.mesh, 'contains'):
                        inside = self.mesh.contains([point])[0]
                        sdf_val = -distance if inside else distance
                    else:
                        # Approximate: use distance (positive = outside)
                        sdf_val = distance
                    distances.append(sdf_val)
                
                sdf_vals = np.array(distances, dtype=np.float32)
            else:
                # Fallback: use bounding box approximation
                sdf_vals = self._approximate_sdf(points)
            
            return sdf_vals[0] if single_point else sdf_vals
        except ImportError:
            # Fallback to bounding box
            return self._approximate_sdf(points)
    
    def _approximate_sdf(self, points: np.ndarray) -> np.ndarray:
        """Approximate SDF using bounding box."""
        # Simple approximation: distance to bounding box
        from genedynamics.envs.obstacles.convex import BoxObstacle
        bounds_min, bounds_max = self.bounds
        center = (bounds_min + bounds_max) / 2.0
        half_extents = (bounds_max - bounds_min) / 2.0
        box = BoxObstacle(center, half_extents)
        return box.sdf(points)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside mesh."""
        point = np.asarray(point, dtype=np.float32)
        
        try:
            if hasattr(self.mesh, 'contains'):
                return bool(self.mesh.contains([point])[0])
            else:
                # Fallback: use bounding box
                bounds_min, bounds_max = self.bounds
                return np.all(point >= bounds_min) and np.all(point <= bounds_max)
        except Exception:
            # Fallback
            bounds_min, bounds_max = self.bounds
            return np.all(point >= bounds_min) and np.all(point <= bounds_max)
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to mesh surface."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        Uses finite differences for approximation.
        """
        point = np.asarray(point, dtype=np.float32)
        eps = 1e-5
        
        grad = np.zeros_like(point)
        for i in range(len(point)):
            point_plus = point.copy()
            point_plus[i] += eps
            point_minus = point.copy()
            point_minus[i] -= eps
            
            sdf_plus = self.sdf(point_plus)
            sdf_minus = self.sdf(point_minus)
            
            grad[i] = (sdf_plus - sdf_minus) / (2 * eps)
        
        # Normalize
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            grad = grad / norm
        
        return grad
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "mesh",
            "mesh_path": getattr(self, 'mesh_path', None),
            "scale": self.scale,
            "translation": self.translation.tolist(),
            "backend": backend
        }


class UnionObstacle(Obstacle):
    """
    Union of multiple obstacles (A ∪ B ∪ ...).
    
    Point is inside if inside ANY obstacle.
    SDF is minimum of all obstacle SDFs.
    """
    
    def __init__(
        self,
        obstacles: List[Obstacle],
        name: Optional[str] = None
    ):
        """
        Initialize union obstacle.
        
        Args:
            obstacles: List of obstacles to union
            name: Optional name
        """
        if not obstacles:
            raise ValueError("UnionObstacle requires at least one obstacle")
        
        self.obstacles = obstacles
        self.name = name or "union"
        
        # Compute bounding box (union of all bounds)
        # Determine dimension from first obstacle with bounds
        dim = None
        for obs in obstacles:
            if hasattr(obs, 'bounds') and obs.bounds is not None:
                obs_min, obs_max = obs.bounds
                obs_min = np.asarray(obs_min)
                dim = len(obs_min)
                break
        
        # Fallback: try to get dimension from center
        if dim is None:
            for obs in obstacles:
                if hasattr(obs, 'center') and obs.center is not None:
                    center = np.asarray(obs.center)
                    dim = len(center)
                    break
        
        # Final fallback: default to 2D (for double integrator 2D)
        if dim is None:
            dim = 2
        
        # Initialize bounds with correct dimension
        bounds_min = np.full(dim, np.inf, dtype=np.float32)
        bounds_max = np.full(dim, -np.inf, dtype=np.float32)
        
        for obs in obstacles:
            if hasattr(obs, 'bounds') and obs.bounds is not None:
                obs_min, obs_max = obs.bounds
                obs_min = np.asarray(obs_min, dtype=np.float32)
                obs_max = np.asarray(obs_max, dtype=np.float32)
                
                # Ensure same dimension (pad if necessary)
                if len(obs_min) < dim:
                    obs_min = np.pad(obs_min, (0, dim - len(obs_min)), constant_values=-np.inf)
                    obs_max = np.pad(obs_max, (0, dim - len(obs_max)), constant_values=np.inf)
                elif len(obs_min) > dim:
                    obs_min = obs_min[:dim]
                    obs_max = obs_max[:dim]
                
                bounds_min = np.minimum(bounds_min, obs_min)
                bounds_max = np.maximum(bounds_max, obs_max)
        
        self.bounds = (bounds_min, bounds_max) if np.all(np.isfinite(bounds_min)) else None
        self.center = (bounds_min + bounds_max) / 2.0 if self.bounds else np.zeros(dim, dtype=np.float32)
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for union (minimum of all SDFs).
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Compute SDF for each obstacle
        sdfs = []
        for obstacle in self.obstacles:
            sdf_vals = obstacle.sdf(points)
            sdfs.append(sdf_vals)
        
        # Union: take minimum (closest obstacle)
        sdf_array = np.stack(sdfs, axis=0)
        min_sdf = np.min(sdf_array, axis=0)
        
        return min_sdf[0] if single_point else min_sdf
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside ANY obstacle."""
        point = np.asarray(point, dtype=np.float32)
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True
        return False
    
    def distance(self, point: np.ndarray) -> float:
        """Compute minimum distance to any obstacle."""
        point = np.asarray(point, dtype=np.float32)
        distances = [obs.distance(point) for obs in self.obstacles]
        return min(distances)
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient (from closest obstacle).
        
        Args:
            point: Point to evaluate
            
        Returns:
            Gradient vector
        """
        point = np.asarray(point, dtype=np.float32)
        
        # Find closest obstacle
        distances = [obs.distance(point) for obs in self.obstacles]
        closest_idx = np.argmin(distances)
        
        # Use gradient from closest obstacle
        return self.obstacles[closest_idx].gradient(point)
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "union",
            "obstacles": [obs.to_backend(backend) for obs in self.obstacles],
            "backend": backend
        }


class DifferenceObstacle(Obstacle):
    """
    Difference of obstacles (A - B).
    
    Point is inside if inside A but NOT inside B.
    SDF = max(SDF_A, -SDF_B)
    """
    
    def __init__(
        self,
        obstacle_a: Obstacle,
        obstacle_b: Obstacle,
        name: Optional[str] = None
    ):
        """
        Initialize difference obstacle.
        
        Args:
            obstacle_a: First obstacle (minuend)
            obstacle_b: Second obstacle (subtrahend)
            name: Optional name
        """
        self.obstacle_a = obstacle_a
        self.obstacle_b = obstacle_b
        self.name = name or "difference"
        
        # Bounding box is from obstacle_a
        if hasattr(obstacle_a, 'bounds') and obstacle_a.bounds is not None:
            self.bounds = obstacle_a.bounds
            self.center = obstacle_a.center if hasattr(obstacle_a, 'center') else None
        else:
            self.bounds = None
            self.center = None
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for difference.
        
        SDF = max(SDF_A, -SDF_B)
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        
        sdf_a = self.obstacle_a.sdf(points)
        sdf_b = self.obstacle_b.sdf(points)
        
        # Difference: max(SDF_A, -SDF_B)
        sdf_diff = np.maximum(sdf_a, -sdf_b)
        
        return sdf_diff[0] if single_point else sdf_diff
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside A but not inside B."""
        point = np.asarray(point, dtype=np.float32)
        return self.obstacle_a.contains(point) and not self.obstacle_b.contains(point)
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to difference boundary."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        Uses finite differences for approximation.
        """
        point = np.asarray(point, dtype=np.float32)
        eps = 1e-5
        
        grad = np.zeros_like(point)
        for i in range(len(point)):
            point_plus = point.copy()
            point_plus[i] += eps
            point_minus = point.copy()
            point_minus[i] -= eps
            
            sdf_plus = self.sdf(point_plus)
            sdf_minus = self.sdf(point_minus)
            
            grad[i] = (sdf_plus - sdf_minus) / (2 * eps)
        
        # Normalize
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            grad = grad / norm
        
        return grad
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "difference",
            "obstacle_a": self.obstacle_a.to_backend(backend),
            "obstacle_b": self.obstacle_b.to_backend(backend),
            "backend": backend
        }


class IntersectionObstacle(Obstacle):
    """
    Intersection of obstacles (A ∩ B ∩ ...).
    
    Point is inside if inside ALL obstacles.
    SDF = max(SDF_A, SDF_B, ...)
    """
    
    def __init__(
        self,
        obstacles: List[Obstacle],
        name: Optional[str] = None
    ):
        """
        Initialize intersection obstacle.
        
        Args:
            obstacles: List of obstacles to intersect
            name: Optional name
        """
        if not obstacles:
            raise ValueError("IntersectionObstacle requires at least one obstacle")
        
        self.obstacles = obstacles
        self.name = name or "intersection"
        
        # Compute bounding box (intersection of all bounds)
        if all(hasattr(obs, 'bounds') and obs.bounds is not None for obs in obstacles):
            bounds_min = obstacles[0].bounds[0].copy()
            bounds_max = obstacles[0].bounds[1].copy()
            
            for obs in obstacles[1:]:
                obs_min, obs_max = obs.bounds
                bounds_min = np.maximum(bounds_min, obs_min)
                bounds_max = np.minimum(bounds_max, obs_max)
            
            if np.all(bounds_min <= bounds_max):
                self.bounds = (bounds_min, bounds_max)
                self.center = (bounds_min + bounds_max) / 2.0
            else:
                self.bounds = None
                self.center = None
        else:
            self.bounds = None
            self.center = None
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for intersection (maximum of all SDFs).
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Compute SDF for each obstacle
        sdfs = []
        for obstacle in self.obstacles:
            sdf_vals = obstacle.sdf(points)
            sdfs.append(sdf_vals)
        
        # Intersection: take maximum (furthest from any boundary)
        sdf_array = np.stack(sdfs, axis=0)
        max_sdf = np.max(sdf_array, axis=0)
        
        return max_sdf[0] if single_point else max_sdf
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside ALL obstacles."""
        point = np.asarray(point, dtype=np.float32)
        for obstacle in self.obstacles:
            if not obstacle.contains(point):
                return False
        return True
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to intersection boundary."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient (from furthest obstacle).
        
        Args:
            point: Point to evaluate
            
        Returns:
            Gradient vector
        """
        point = np.asarray(point, dtype=np.float32)
        
        # Find furthest obstacle (largest SDF)
        sdfs = [obs.sdf(point) for obs in self.obstacles]
        furthest_idx = np.argmax(sdfs)
        
        # Use gradient from furthest obstacle
        return self.obstacles[furthest_idx].gradient(point)
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "intersection",
            "obstacles": [obs.to_backend(backend) for obs in self.obstacles],
            "backend": backend
        }
