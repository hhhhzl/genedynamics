"""
Obstacle configuration example.

This example demonstrates:
- Creating various obstacle types
- Using CSG operations
- Configuring obstacle managers
"""

import numpy as np
from enerdynamics.envs.obstacles.convex import BoxObstacle, SphereObstacle, CylinderObstacle
from enerdynamics.envs.obstacles.nonconvex import UnionObstacle, DifferenceObstacle, IntersectionObstacle
from enerdynamics.envs.obstacles.base import ObstacleManager


def main():
    """Run obstacle configuration example."""
    print("="*60)
    print("Obstacle Configuration Example")
    print("="*60)
    
    # Create convex obstacles
    box = BoxObstacle(
        center=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        half_extents=np.array([0.2, 0.2, 0.2], dtype=np.float32)
    )
    
    sphere = SphereObstacle(
        center=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        radius=0.3
    )
    
    cylinder = CylinderObstacle(
        center=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        radius=0.2,
        height=0.5
    )
    
    print("✓ Created convex obstacles: Box, Sphere, Cylinder")
    
    # Create CSG obstacles
    union = UnionObstacle([box, sphere])
    print("✓ Created UnionObstacle (box ∪ sphere)")
    
    large_box = BoxObstacle(
        center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        half_extents=np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    small_sphere = SphereObstacle(
        center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        radius=0.3
    )
    difference = DifferenceObstacle(large_box, small_sphere)
    print("✓ Created DifferenceObstacle (box - sphere)")
    
    intersection = IntersectionObstacle([box, sphere])
    print("✓ Created IntersectionObstacle (box ∩ sphere)")
    
    # Create obstacle manager
    obstacles = [box, sphere, cylinder, union, difference, intersection]
    manager = ObstacleManager(obstacles)
    
    print(f"✓ Created ObstacleManager with {len(manager)} obstacles")
    
    # Test queries
    test_points = np.array([
        [1.1, 0.0, 0.0],  # In box
        [-1.1, 0.0, 0.0],  # In sphere
        [0.0, 0.0, 0.0],  # Outside
    ], dtype=np.float32)
    
    for i, point in enumerate(test_points):
        contains = manager.contains(point)
        distance = manager.distance(point)
        sdf = manager.sdf(point)
        
        print(f"Point {i}: contains={contains}, distance={distance:.4f}, sdf={sdf:.4f}")
    
    print("="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
