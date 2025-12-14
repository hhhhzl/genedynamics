"""
SDF (Signed Distance Function) computation utilities.

This module provides utilities for computing SDFs for complex shapes,
including mesh-based SDFs, approximate SDFs, and gradient computation.
"""

from typing import Optional, Tuple, List, Any
import numpy as np

try:
    import jax.numpy as jnp
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    jax = None


def compute_mesh_sdf(
    mesh: Any,
    points: np.ndarray,
    approximate: bool = False
) -> np.ndarray:
    """
    Compute SDF for mesh using trimesh.
    
    Args:
        mesh: Trimesh mesh object
        points: Points to evaluate, shape (N, 3)
        approximate: If True, use faster approximation
        
    Returns:
        SDF values, shape (N,)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError(
            "Mesh SDF computation requires trimesh. Install with: pip install trimesh"
        )
    
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    if approximate:
        # Fast approximation using bounding box
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        half_extents = (bounds[1] - bounds[0]) / 2.0
        
        # Distance to bounding box
        q = points - center
        d = np.abs(q) - half_extents
        max_d = np.maximum(d, 0.0)
        length = np.linalg.norm(max_d, axis=-1)
        
        # If inside, add max negative distance
        inside = np.all(d < 0, axis=-1)
        if np.any(inside):
            min_d = np.minimum(np.maximum(d, -np.inf), 0.0)
            length[inside] = np.max(min_d[inside], axis=-1)
        
        return length
    
    # Exact SDF using trimesh
    sdf_vals = []
    for point in points:
        if hasattr(mesh, 'nearest'):
            closest, distance, _ = mesh.nearest.on_surface([point])
            # Check if inside
            if hasattr(mesh, 'contains'):
                inside = mesh.contains([point])[0]
                sdf_val = -distance if inside else distance
            else:
                sdf_val = distance
            sdf_vals.append(sdf_val)
        else:
            # Fallback to approximation
            sdf_vals.append(compute_mesh_sdf(mesh, point.reshape(1, -1), approximate=True)[0])
    
    return np.array(sdf_vals, dtype=np.float32)


def compute_sdf_gradient(
    sdf_fn: callable,
    point: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute SDF gradient using finite differences.
    
    Args:
        sdf_fn: Function that computes SDF: sdf(point) -> float
        point: Point to evaluate, shape (dim,)
        eps: Finite difference step size
        
    Returns:
        Gradient vector, shape (dim,)
    """
    point = np.asarray(point, dtype=np.float32)
    dim = len(point)
    grad = np.zeros(dim, dtype=np.float32)
    
    for i in range(dim):
        point_plus = point.copy()
        point_plus[i] += eps
        point_minus = point.copy()
        point_minus[i] -= eps
        
        sdf_plus = sdf_fn(point_plus)
        sdf_minus = sdf_fn(point_minus)
        
        grad[i] = (sdf_plus - sdf_minus) / (2 * eps)
    
    # Normalize
    norm = np.linalg.norm(grad)
    if norm > 1e-6:
        grad = grad / norm
    
    return grad


def approximate_sdf_grid(
    sdf_fn: callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute SDF on a grid for fast lookup.
    
    Args:
        sdf_fn: Function that computes SDF: sdf(point) -> float
        bounds: Bounding box (min, max), each shape (3,)
        resolution: Grid resolution per dimension
        
    Returns:
        Tuple of (grid_points, sdf_values)
        - grid_points: Grid points, shape (resolution^3, 3)
        - sdf_values: SDF values, shape (resolution^3,)
    """
    bounds_min, bounds_max = np.asarray(bounds[0]), np.asarray(bounds[1])
    
    # Create grid
    x = np.linspace(bounds_min[0], bounds_max[0], resolution)
    y = np.linspace(bounds_min[1], bounds_max[1], resolution)
    z = np.linspace(bounds_min[2], bounds_max[2], resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # Compute SDF for all points
    sdf_values = np.array([sdf_fn(p) for p in grid_points], dtype=np.float32)
    
    return grid_points, sdf_values


def interpolate_sdf_grid(
    point: np.ndarray,
    grid_points: np.ndarray,
    sdf_values: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    resolution: int
) -> float:
    """
    Interpolate SDF from precomputed grid.
    
    Args:
        point: Point to evaluate, shape (3,)
        grid_points: Grid points, shape (N, 3)
        sdf_values: SDF values, shape (N,)
        bounds: Bounding box (min, max)
        resolution: Grid resolution per dimension
        
    Returns:
        Interpolated SDF value
    """
    from scipy.interpolate import RegularGridInterpolator
    
    bounds_min, bounds_max = np.asarray(bounds[0]), np.asarray(bounds[1])
    
    # Create grid for interpolation
    x = np.linspace(bounds_min[0], bounds_max[0], resolution)
    y = np.linspace(bounds_min[1], bounds_max[1], resolution)
    z = np.linspace(bounds_min[2], bounds_max[2], resolution)
    
    # Reshape SDF values to grid
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    # Create interpolator
    interp = RegularGridInterpolator((x, y, z), sdf_grid, method='linear', bounds_error=False, fill_value=np.inf)
    
    return float(interp(point))


# ============================================================================
# JAX-compatible SDF functions
# ============================================================================

if JAX_AVAILABLE:
    
    def jax_sdf_gradient(
        sdf_fn: callable,
        point: jnp.ndarray,
        eps: float = 1e-5
    ) -> jnp.ndarray:
        """
        Compute SDF gradient using JAX automatic differentiation.
        
        Args:
            sdf_fn: JAX-compatible SDF function
            point: Point to evaluate, shape (dim,)
            eps: Not used (JAX uses AD)
            
        Returns:
            Gradient vector, shape (dim,)
        """
        # Use JAX gradient
        grad_fn = jax.grad(sdf_fn)
        grad = grad_fn(point)
        
        # Normalize
        norm = jnp.linalg.norm(grad)
        grad = jnp.where(norm > 1e-6, grad / norm, grad)
        
        return grad
    
    def jax_vmap_sdf(
        sdf_fn: callable
    ) -> callable:
        """
        Create vectorized SDF function using JAX vmap.
        
        Args:
            sdf_fn: JAX-compatible SDF function (single point)
            
        Returns:
            Vectorized SDF function (batch of points)
        """
        return jax.vmap(sdf_fn)
    
    def jax_jit_sdf(
        sdf_fn: callable
    ) -> callable:
        """
        JIT compile SDF function.
        
        Args:
            sdf_fn: JAX-compatible SDF function
            
        Returns:
            JIT-compiled SDF function
        """
        return jax.jit(sdf_fn)
