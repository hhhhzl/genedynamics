"""
Unit tests for obstacle system.

Tests cover:
- Convex obstacles (Box, Sphere, Cylinder, Capsule)
- Non-convex obstacles (Mesh, Union, Difference, Intersection)
- Obstacle manager
- Collision detection
- SDF computation
- Spatial indexing
"""

import pytest
import numpy as np
from typing import List

from genedynamics.envs.obstacles.convex import (
    BoxObstacle,
    SphereObstacle,
    CylinderObstacle,
    CapsuleObstacle,
)
from genedynamics.envs.obstacles.base import ObstacleManager


@pytest.mark.unit
class TestConvexObstacles:
    """Test convex obstacle implementations."""
    
    def test_box_obstacle_creation(self):
        """Test BoxObstacle creation."""
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        box = BoxObstacle(center, half_extents)
        
        assert np.allclose(box.center, center)
        assert box.bounds is not None
        print("✓ BoxObstacle creation works")
    
    def test_box_obstacle_contains(self):
        """Test BoxObstacle.contains()."""
        box = BoxObstacle(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            half_extents=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
        
        # Point inside
        assert box.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert box.contains(np.array([0.5, 0.5, 0.5], dtype=np.float32))
        
        # Point outside
        assert not box.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        print("✓ BoxObstacle.contains() works")
    
    def test_box_obstacle_sdf(self):
        """Test BoxObstacle.sdf()."""
        box = BoxObstacle(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            half_extents=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
        
        # Point inside (negative SDF)
        sdf_inside = box.sdf(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert sdf_inside < 0, "Point inside should have negative SDF"
        
        # Point outside (positive SDF)
        sdf_outside = box.sdf(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        assert sdf_outside > 0, "Point outside should have positive SDF"
        print("✓ BoxObstacle.sdf() works")
    
    def test_sphere_obstacle(self):
        """Test SphereObstacle."""
        sphere = SphereObstacle(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0
        )
        
        # Point inside
        assert sphere.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert sphere.contains(np.array([0.5, 0.0, 0.0], dtype=np.float32))
        
        # Point outside
        assert not sphere.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        
        # SDF
        sdf_inside = sphere.sdf(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert sdf_inside < 0
        
        sdf_outside = sphere.sdf(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        assert sdf_outside > 0
        print("✓ SphereObstacle works")
    
    def test_cylinder_obstacle(self):
        """Test CylinderObstacle."""
        cylinder = CylinderObstacle(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            radius=1.0,
            height=2.0
        )
        
        # Point inside
        assert cylinder.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        
        # Point outside
        assert not cylinder.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        print("✓ CylinderObstacle works")
    
    def test_capsule_obstacle(self):
        """Test CapsuleObstacle."""
        capsule = CapsuleObstacle(
            point_a=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            point_b=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=0.5
        )
        
        # Point inside
        assert capsule.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        
        # Point outside
        assert not capsule.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        print("✓ CapsuleObstacle works")


@pytest.mark.unit
class TestNonConvexObstacles:
    """Test non-convex obstacle implementations."""
    
    @pytest.mark.requires_trimesh
    def test_mesh_obstacle(self):
        """Test MeshObstacle (requires trimesh)."""
        try:
            from genedynamics.envs.obstacles.nonconvex import MeshObstacle
            import trimesh
            
            # Create simple mesh
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
            mesh_obs = MeshObstacle(mesh=mesh)
            
            # Point inside
            assert mesh_obs.contains(np.array([0.1, 0.1, 0.1], dtype=np.float32))
            
            # Point outside
            assert not mesh_obs.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
            print("✓ MeshObstacle works")
        except ImportError:
            pytest.skip("trimesh not available")
    
    def test_union_obstacle(self):
        """Test UnionObstacle."""
        from genedynamics.envs.obstacles.nonconvex import UnionObstacle
        
        box1 = BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        box2 = BoxObstacle(np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        
        union = UnionObstacle([box1, box2])
        
        # Point in box1
        assert union.contains(np.array([1.1, 0.0, 0.0], dtype=np.float32))
        
        # Point in box2
        assert union.contains(np.array([-1.1, 0.0, 0.0], dtype=np.float32))
        
        # Point outside both
        assert not union.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        print("✓ UnionObstacle works")
    
    def test_difference_obstacle(self):
        """Test DifferenceObstacle."""
        from genedynamics.envs.obstacles.nonconvex import DifferenceObstacle
        
        large_box = BoxObstacle(np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32))
        small_sphere = SphereObstacle(np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.3)
        
        difference = DifferenceObstacle(large_box, small_sphere)
        
        # Point in box but outside sphere
        assert difference.contains(np.array([0.5, 0.0, 0.0], dtype=np.float32))
        
        # Point in sphere
        assert not difference.contains(np.array([0.1, 0.0, 0.0], dtype=np.float32))
        print("✓ DifferenceObstacle works")
    
    def test_intersection_obstacle(self):
        """Test IntersectionObstacle."""
        from genedynamics.envs.obstacles.nonconvex import IntersectionObstacle
        
        box = BoxObstacle(np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.5, 0.5, 0.5], dtype=np.float32))
        sphere = SphereObstacle(np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.4)
        
        intersection = IntersectionObstacle([box, sphere])
        
        # Point in both
        assert intersection.contains(np.array([0.1, 0.1, 0.1], dtype=np.float32))
        
        # Point only in box
        assert not intersection.contains(np.array([0.45, 0.0, 0.0], dtype=np.float32))
        print("✓ IntersectionObstacle works")


@pytest.mark.unit
class TestObstacleManager:
    """Test ObstacleManager."""
    
    def test_obstacle_manager_creation(self):
        """Test ObstacleManager creation."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)),
            SphereObstacle(np.array([-1.0, 0.0, 0.0], dtype=np.float32), 0.3),
        ]
        manager = ObstacleManager(obstacles)
        
        assert len(manager) == 2
        print("✓ ObstacleManager creation works")
    
    def test_obstacle_manager_add_remove(self):
        """Test adding and removing obstacles."""
        manager = ObstacleManager()
        
        box = BoxObstacle(np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.1, 0.1, 0.1], dtype=np.float32))
        manager.add(box)
        assert len(manager) == 1
        
        manager.remove(box)
        assert len(manager) == 0
        print("✓ ObstacleManager add/remove works")
    
    def test_obstacle_manager_contains(self):
        """Test ObstacleManager.contains()."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)),
        ]
        manager = ObstacleManager(obstacles)
        
        # Point in obstacle
        assert manager.contains(np.array([1.1, 0.0, 0.0], dtype=np.float32))
        
        # Point outside
        assert not manager.contains(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        print("✓ ObstacleManager.contains() works")
    
    def test_obstacle_manager_sdf(self):
        """Test ObstacleManager.sdf()."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)),
        ]
        manager = ObstacleManager(obstacles)
        
        points = np.array([
            [1.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        sdf_vals = manager.sdf(points)
        assert sdf_vals.shape == (2,)
        assert sdf_vals[0] < 0  # Inside
        assert sdf_vals[1] > 0  # Outside
        print("✓ ObstacleManager.sdf() works")


@pytest.mark.unit
@pytest.mark.requires_scipy
class TestSpatialIndexing:
    """Test spatial indexing for acceleration."""
    
    def test_spatial_index(self):
        """Test SpatialIndex."""
        try:
            from genedynamics.envs.obstacles.spatial_index import SpatialIndex
            
            obstacles = [
                BoxObstacle(np.array([i, 0.0, 0.0], dtype=np.float32), np.array([0.1, 0.1, 0.1], dtype=np.float32))
                for i in range(10)
            ]
            
            spatial_index = SpatialIndex(obstacles)
            
            # Query nearby
            nearby = spatial_index.query_nearby(np.array([5.0, 0.0, 0.0], dtype=np.float32), radius=1.0)
            assert len(nearby) > 0
            
            # Query nearest
            nearest_idx, dist = spatial_index.query_nearest(np.array([5.0, 0.0, 0.0], dtype=np.float32))
            assert nearest_idx < len(obstacles)
            print("✓ SpatialIndex works")
        except ImportError:
            pytest.skip("SciPy not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
