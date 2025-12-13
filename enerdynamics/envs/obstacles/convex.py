"""
Convex obstacle implementations.

This module provides concrete implementations of convex obstacles:
- BoxObstacle: Axis-aligned box
- SphereObstacle: Sphere
- CylinderObstacle: Cylinder (axis-aligned)
- CapsuleObstacle: Capsule (line segment with radius)

All obstacles implement the Obstacle protocol with:
- SDF (Signed Distance Function)
- Point containment checking
- Distance computation
- Gradient computation (for optimization)
"""

import numpy as np
from typing import Optional, Tuple

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from enerdynamics.envs.obstacles.base import Obstacle


class BoxObstacle:
    """
    Axis-aligned box obstacle.
    
    Defined by center and half-extents (size / 2).
    """
    
    def __init__(
        self,
        center: np.ndarray,
        half_extents: np.ndarray,
        name: Optional[str] = None
    ):
        """
        Initialize box obstacle.
        
        Args:
            center: Center point, shape (dim,)
            half_extents: Half-extents (size / 2), shape (dim,)
            name: Optional name for the obstacle
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.half_extents = np.asarray(half_extents, dtype=np.float32)
        self.name = name or "box"
        
        # Bounding box
        self.bounds = (
            self.center - self.half_extents,
            self.center + self.half_extents
        )
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for box.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Relative position from center
        q = points - self.center
        
        # Distance to box boundary
        # For each dimension, compute distance to nearest face
        d = np.abs(q) - self.half_extents
        
        # Inside box: use max of negative distances
        # Outside box: use length of positive distances
        max_d = np.maximum(d, 0.0)
        length = np.linalg.norm(max_d, axis=-1)
        
        # If inside, add the max negative distance
        inside = np.all(d < 0, axis=-1)
        if np.any(inside):
            min_d = np.minimum(np.maximum(d, -np.inf), 0.0)
            length[inside] = np.max(min_d[inside], axis=-1)
        
        return length[0] if single_point else length
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside box."""
        point = np.asarray(point, dtype=np.float32)
        q = point - self.center
        return np.all(np.abs(q) < self.half_extents)
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to box boundary."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        For box, gradient points to nearest face.
        """
        point = np.asarray(point, dtype=np.float32)
        q = point - self.center
        
        # Distance to each face
        d = np.abs(q) - self.half_extents
        
        # Find nearest face
        max_idx = np.argmax(np.abs(d))
        sign = np.sign(q[max_idx])
        
        # Gradient points to nearest face
        grad = np.zeros_like(q)
        grad[max_idx] = sign
        
        return grad
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "box",
            "center": self.center.tolist(),
            "half_extents": self.half_extents.tolist(),
            "backend": backend
        }


class SphereObstacle:
    """
    Sphere obstacle.
    
    Defined by center and radius.
    """
    
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        name: Optional[str] = None
    ):
        """
        Initialize sphere obstacle.
        
        Args:
            center: Center point, shape (dim,)
            radius: Sphere radius
            name: Optional name for the obstacle
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.name = name or "sphere"
        
        # Bounding box
        self.bounds = (
            self.center - self.radius,
            self.center + self.radius
        )
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for sphere.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Distance from center
        dists = np.linalg.norm(points - self.center, axis=-1)
        
        # SDF = distance - radius
        sdf_vals = dists - self.radius
        
        return sdf_vals[0] if single_point else sdf_vals
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside sphere."""
        point = np.asarray(point, dtype=np.float32)
        dist = np.linalg.norm(point - self.center)
        return dist < self.radius
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to sphere boundary."""
        point = np.asarray(point, dtype=np.float32)
        dist = np.linalg.norm(point - self.center)
        return max(0.0, dist - self.radius)
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        For sphere, gradient points from center to point (normalized).
        """
        point = np.asarray(point, dtype=np.float32)
        vec = point - self.center
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            # At center, return zero gradient
            return np.zeros_like(point)
        
        return vec / dist
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "sphere",
            "center": self.center.tolist(),
            "radius": self.radius,
            "backend": backend
        }


class CylinderObstacle:
    """
    Axis-aligned cylinder obstacle.
    
    Cylinder extends along the z-axis (or specified axis).
    Defined by center, radius, and height.
    """
    
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        height: float,
        axis: int = 2,
        name: Optional[str] = None
    ):
        """
        Initialize cylinder obstacle.
        
        Args:
            center: Center point, shape (dim,)
            radius: Cylinder radius
            height: Cylinder height
            axis: Axis along which cylinder extends (0=x, 1=y, 2=z)
            name: Optional name for the obstacle
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.height = float(height)
        self.axis = int(axis)
        self.name = name or "cylinder"
        
        # Bounding box
        half_height = self.height / 2.0
        bounds_min = self.center.copy()
        bounds_max = self.center.copy()
        bounds_min[self.axis] -= half_height
        bounds_max[self.axis] += half_height
        for i in range(len(self.center)):
            if i != self.axis:
                bounds_min[i] -= self.radius
                bounds_max[i] += self.radius
        
        self.bounds = (bounds_min, bounds_max)
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for cylinder.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Relative position from center
        q = points - self.center
        
        # Project to plane perpendicular to axis
        axis_coord = q[:, self.axis] if q.ndim > 1 else q[self.axis]
        plane_coords = np.delete(q, self.axis, axis=-1)
        
        # Distance in plane
        plane_dist = np.linalg.norm(plane_coords, axis=-1)
        
        # Distance along axis
        half_height = self.height / 2.0
        axis_dist = np.abs(axis_coord) - half_height
        
        # SDF computation
        # Outside cylinder: max of radial and axial distances
        # Inside cylinder: min of negative distances
        radial_d = plane_dist - self.radius
        axial_d = axis_dist
        
        # Combine radial and axial
        d = np.maximum(radial_d, axial_d)
        
        # If inside, use minimum of negative distances
        inside = (radial_d < 0) & (axial_d < 0)
        if np.any(inside):
            d[inside] = np.maximum(radial_d[inside], axial_d[inside])
        
        return d[0] if single_point else d
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside cylinder."""
        point = np.asarray(point, dtype=np.float32)
        q = point - self.center
        
        # Check axis coordinate
        half_height = self.height / 2.0
        if abs(q[self.axis]) >= half_height:
            return False
        
        # Check radial distance
        plane_coords = np.delete(q, self.axis)
        plane_dist = np.linalg.norm(plane_coords)
        return plane_dist < self.radius
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to cylinder boundary."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        Gradient points to nearest point on cylinder surface.
        """
        point = np.asarray(point, dtype=np.float32)
        q = point - self.center
        
        # Project to plane
        axis_coord = q[self.axis]
        plane_coords = np.delete(q, self.axis)
        plane_dist = np.linalg.norm(plane_coords)
        
        half_height = self.height / 2.0
        radial_d = plane_dist - self.radius
        axial_d = abs(axis_coord) - half_height
        
        grad = np.zeros_like(q)
        
        if radial_d > axial_d:
            # Closer to radial boundary
            if plane_dist > 1e-6:
                grad_plane = plane_coords / plane_dist
                grad[:self.axis] = grad_plane[:self.axis]
                grad[self.axis+1:] = grad_plane[self.axis:]
            else:
                # At center, point along axis
                grad[self.axis] = 1.0
        else:
            # Closer to axial boundary
            grad[self.axis] = np.sign(axis_coord)
        
        return grad
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "cylinder",
            "center": self.center.tolist(),
            "radius": self.radius,
            "height": self.height,
            "axis": self.axis,
            "backend": backend
        }


class CapsuleObstacle:
    """
    Capsule obstacle (line segment with radius).
    
    Defined by two endpoints and radius.
    """
    
    def __init__(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        radius: float,
        name: Optional[str] = None
    ):
        """
        Initialize capsule obstacle.
        
        Args:
            point_a: First endpoint, shape (dim,)
            point_b: Second endpoint, shape (dim,)
            radius: Capsule radius
            name: Optional name for the obstacle
        """
        self.point_a = np.asarray(point_a, dtype=np.float32)
        self.point_b = np.asarray(point_b, dtype=np.float32)
        self.radius = float(radius)
        self.name = name or "capsule"
        
        # Center is midpoint
        self.center = (self.point_a + self.point_b) / 2.0
        
        # Bounding box
        min_point = np.minimum(self.point_a, self.point_b) - self.radius
        max_point = np.maximum(self.point_a, self.point_b) + self.radius
        self.bounds = (min_point, max_point)
    
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Compute SDF for capsule.
        
        Args:
            points: Points to evaluate, shape (N, dim) or (dim,)
            
        Returns:
            SDF values, shape (N,) or scalar
        """
        points = np.asarray(points, dtype=np.float32)
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        
        # Vector from A to B
        ab = self.point_b - self.point_a
        ab_len = np.linalg.norm(ab)
        
        if ab_len < 1e-6:
            # Degenerate capsule (points are same) -> treat as sphere
            dists = np.linalg.norm(points - self.point_a, axis=-1)
            sdf_vals = dists - self.radius
            return sdf_vals[0] if single_point else sdf_vals
        
        # For each point, find closest point on line segment
        sdf_vals = []
        for point in points:
            ap = point - self.point_a
            t = np.clip(np.dot(ap, ab) / (ab_len ** 2), 0.0, 1.0)
            closest = self.point_a + t * ab
            dist = np.linalg.norm(point - closest)
            sdf_vals.append(dist - self.radius)
        
        sdf_vals = np.array(sdf_vals)
        return sdf_vals[0] if single_point else sdf_vals
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside capsule."""
        point = np.asarray(point, dtype=np.float32)
        
        # Vector from A to B
        ab = self.point_b - self.point_a
        ab_len = np.linalg.norm(ab)
        
        if ab_len < 1e-6:
            # Degenerate -> sphere
            dist = np.linalg.norm(point - self.point_a)
            return dist < self.radius
        
        # Find closest point on line segment
        ap = point - self.point_a
        t = np.clip(np.dot(ap, ab) / (ab_len ** 2), 0.0, 1.0)
        closest = self.point_a + t * ab
        dist = np.linalg.norm(point - closest)
        
        return dist < self.radius
    
    def distance(self, point: np.ndarray) -> float:
        """Compute distance to capsule boundary."""
        sdf_val = self.sdf(point)
        return max(0.0, float(sdf_val))
    
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute SDF gradient.
        
        Gradient points from closest point on line segment to point.
        """
        point = np.asarray(point, dtype=np.float32)
        
        # Vector from A to B
        ab = self.point_b - self.point_a
        ab_len = np.linalg.norm(ab)
        
        if ab_len < 1e-6:
            # Degenerate -> sphere gradient
            vec = point - self.point_a
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                return np.zeros_like(point)
            return vec / dist
        
        # Find closest point on line segment
        ap = point - self.point_a
        t = np.clip(np.dot(ap, ab) / (ab_len ** 2), 0.0, 1.0)
        closest = self.point_a + t * ab
        
        # Gradient points from closest to point
        vec = point - closest
        dist = np.linalg.norm(vec)
        
        if dist < 1e-6:
            # On line segment, gradient points perpendicular to line
            if ab_len > 1e-6:
                perp = ab / ab_len
                # Return any perpendicular vector
                return perp
            return np.zeros_like(point)
        
        return vec / dist
    
    def to_backend(self, backend: str) -> dict:
        """Convert to backend representation."""
        return {
            "type": "capsule",
            "point_a": self.point_a.tolist(),
            "point_b": self.point_b.tolist(),
            "radius": self.radius,
            "backend": backend
        }
