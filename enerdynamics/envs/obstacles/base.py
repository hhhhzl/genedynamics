"""
Base obstacle interface and manager.

This module defines the Obstacle protocol that all obstacles should implement,
and provides an ObstacleManager for managing multiple obstacles.
"""

from typing import Protocol, Optional, List, Any, runtime_checkable
import numpy as np
from typing import Literal, Union

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    import torch
except Exception:
    torch = None

try:
    import jax
except Exception:
    jax = None

from enerdynamics.envs.obstacles.sdf_texture import SDFTexture2D


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
    
    def __init__(self, obstacles: Optional[List[Obstacle]] = None, use_spatial_index: bool = False):
        """
        Initialize obstacle manager.
        
        Args:
            obstacles: Initial list of obstacles
            use_spatial_index: Whether to use spatial indexing for faster queries
                              (requires scipy and obstacles with center/bounds attributes)
        """
        self.obstacles: List[Obstacle] = obstacles or []
        self.use_spatial_index = use_spatial_index
        self._spatial_index = None
        self._sdf_texture_2d: Optional[SDFTexture2D] = None
        
        if use_spatial_index:
            try:
                from enerdynamics.envs.obstacles.spatial_index import SpatialIndex
                self._spatial_index = SpatialIndex(self.obstacles, method="kdtree")
            except (ImportError, Exception):
                # Fallback to no spatial index if scipy unavailable or setup fails
                self.use_spatial_index = False
                self._spatial_index = None
    
    def add(self, obstacle: Obstacle) -> None:
        """
        Add an obstacle to the manager.
        
        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)
        # Invalidate any cached SDF texture (environment changed)
        self._sdf_texture_2d = None
        # Rebuild spatial index if enabled
        if self.use_spatial_index and self._spatial_index is not None:
            self._spatial_index.update()
    
    def remove(self, obstacle: Obstacle) -> None:
        """
        Remove an obstacle from the manager.
        
        Args:
            obstacle: Obstacle to remove
        """
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            self._sdf_texture_2d = None
            # Rebuild spatial index if enabled
            if self.use_spatial_index and self._spatial_index is not None:
                self._spatial_index.update()
    
    def clear(self) -> None:
        """Remove all obstacles."""
        self.obstacles.clear()
        self._sdf_texture_2d = None
        # Rebuild spatial index if enabled
        if self.use_spatial_index and self._spatial_index is not None:
            self._spatial_index.update()

    # --------------------------------------------------------------------- fast SDF
    def build_sdf_texture_2d(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        res: float = 0.01,
        force_rebuild: bool = False,
    ) -> Optional[SDFTexture2D]:
        """
        Build a MDOC-style SDF texture for fast SDF/gradient sampling.

        Notes:
        - This is an *approximation* of the exact geometric SDF.
        - It is intended for fast inner-loop queries (diffusion rollouts / CBF filters).
        """
        if self._sdf_texture_2d is not None and not force_rebuild:
            return self._sdf_texture_2d
        if not self.obstacles:
            self._sdf_texture_2d = None
            return None

        def sdf_fn(p: np.ndarray) -> float:
            return float(self.sdf(np.asarray(p, dtype=np.float32)))

        self._sdf_texture_2d = SDFTexture2D.build_from_sdf_fn(
            sdf_fn,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            res=float(res),
        )
        return self._sdf_texture_2d

    def get_sdf_texture_2d(self) -> Optional[SDFTexture2D]:
        return self._sdf_texture_2d

    def sample_sdf_and_grad_2d(
        self,
        points: Union[np.ndarray, "torch.Tensor", "jax.Array"],
        *,
        backend: Literal["numpy", "torch", "jax"] = "numpy",
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        """
        Sample (sdf, grad) using the cached 2D SDF texture.
        Call build_sdf_texture_2d() once before using this for 2D environments with obstacles.
        When no texture is built (e.g. level=0, no obstacles), returns free-space values
        (large sdf, zero grad) so CBF-style filters effectively no-op.
        """
        if self._sdf_texture_2d is not None:
            return self._sdf_texture_2d.sample(points, backend=backend, device=device)
        # No obstacles or texture not built: return free space (large sdf, zero grad)
        # so that CBF conditions (e.g. h < tau) do not trigger projection
        large_sdf = 1e6
        shape = getattr(points, "shape", (2,))
        if len(shape) > 1:
            out_shape = shape[:-1]
            grad_shape = shape
        else:
            out_shape = ()
            grad_shape = (2,)
        if backend == "jax" and jnp is not None:
            sdf = jnp.full(out_shape, large_sdf, dtype=jnp.float32)
            grad = jnp.zeros(grad_shape, dtype=jnp.float32)
            return sdf, grad
        if backend == "torch" and torch is not None:
            dev = device or (points.device if hasattr(points, "device") else None)
            sdf = torch.full((*out_shape,) if out_shape else (1,), large_sdf, dtype=torch.float32, device=dev)
            if not out_shape:
                sdf = sdf.squeeze(0)
            grad = torch.zeros(grad_shape, dtype=torch.float32, device=dev)
            return sdf, grad
        sdf = np.full(out_shape, large_sdf, dtype=np.float32)
        grad = np.zeros(grad_shape, dtype=np.float32)
        return sdf, grad
    
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
        
        Optimized with spatial indexing when enabled: only queries obstacles
        near each point, reducing computation for sparse obstacle distributions.
        
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
        
        points_array = np.asarray(points, dtype=np.float32)
        is_single_point = points_array.ndim == 1
        if is_single_point:
            points_array = points_array[None, :]
        
        num_points = len(points_array)
        
        # For batch queries (many points), use standard batch processing
        # Obstacles' sdf methods can efficiently handle batch inputs
        # Spatial index is more beneficial for single-point or small-batch queries
        if num_points > 1 or not self.use_spatial_index or self._spatial_index is None:
            # Standard batch approach: compute SDF for all obstacles
            # This is efficient when obstacles support batch SDF computation
            sdfs = []
            for obstacle in self.obstacles:
                sdf_vals = obstacle.sdf(points_array)
                sdfs.append(sdf_vals)
            
            # Union: take minimum (closest obstacle)
            sdf_array = np.stack(sdfs, axis=0)
            result = np.min(sdf_array, axis=0)
        else:
            # Single point query with spatial index: only query nearby obstacles
            # This is beneficial when there are many obstacles
            point = points_array[0]
            nearby_indices = self._spatial_index.query_nearby(point, radius=5.0)
            
            if not nearby_indices or len(nearby_indices) == len(self.obstacles):
                # No spatial filtering benefit, use all obstacles
                sdfs = []
                for obstacle in self.obstacles:
                    sdf_val = obstacle.sdf(point)
                    sdfs.append(sdf_val)
                result = np.array([np.min(sdfs)])
            else:
                # Only compute SDF for nearby obstacles
                point_sdfs = []
                for idx in nearby_indices:
                    if idx < len(self.obstacles):
                        sdf_val = self.obstacles[idx].sdf(point)
                        point_sdfs.append(sdf_val)
                
                if point_sdfs:
                    result = np.array([np.min(point_sdfs)])
                else:
                    result = np.array([float('inf')])
        
        # Return scalar for single point, array otherwise
        if is_single_point:
            return float(result[0]) if result.size == 1 else result[0]
        return result
    
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
