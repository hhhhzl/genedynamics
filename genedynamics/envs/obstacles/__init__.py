"""
Obstacle system for environments.

This module provides obstacle abstractions for collision detection,
distance computation, and constraint handling in environments.
"""

from genedynamics.envs.obstacles.base import Obstacle, ObstacleManager
from genedynamics.envs.obstacles.convex import (
    BoxObstacle,
    SphereObstacle,
    CylinderObstacle,
    CapsuleObstacle,
)
from genedynamics.envs.obstacles.collision import (
    check_collision_point,
    check_collision_batch,
    compute_distances,
    compute_sdf_batch,
)
from genedynamics.envs.obstacles.sdf_texture import SDFTexture2D

# Non-convex obstacles (optional, may require trimesh)
try:
    from genedynamics.envs.obstacles.nonconvex import (
        MeshObstacle,
        UnionObstacle,
        DifferenceObstacle,
        IntersectionObstacle,
    )
    __all__ = [
        "Obstacle",
        "ObstacleManager",
        "BoxObstacle",
        "SphereObstacle",
        "CylinderObstacle",
        "CapsuleObstacle",
        "MeshObstacle",
        "UnionObstacle",
        "DifferenceObstacle",
        "IntersectionObstacle",
        "check_collision_point",
        "check_collision_batch",
        "compute_distances",
        "compute_sdf_batch",
        "SDFTexture2D",
    ]
except ImportError:
    __all__ = [
        "Obstacle",
        "ObstacleManager",
        "BoxObstacle",
        "SphereObstacle",
        "CylinderObstacle",
        "CapsuleObstacle",
        "check_collision_point",
        "check_collision_batch",
        "compute_distances",
        "compute_sdf_batch",
        "SDFTexture2D",
    ]

# SDF tools (optional)
try:
    from genedynamics.envs.obstacles.sdf import (
        compute_mesh_sdf,
        compute_sdf_gradient,
        approximate_sdf_grid,
        interpolate_sdf_grid,
    )
    __all__.extend([
        "compute_mesh_sdf",
        "compute_sdf_gradient",
        "approximate_sdf_grid",
        "interpolate_sdf_grid",
    ])
except ImportError:
    pass

# Spatial indexing (optional)
try:
    from genedynamics.envs.obstacles.spatial_index import (
        SpatialIndex,
        AcceleratedObstacleManager,
    )
    __all__.extend([
        "SpatialIndex",
        "AcceleratedObstacleManager",
    ])
except ImportError:
    pass
