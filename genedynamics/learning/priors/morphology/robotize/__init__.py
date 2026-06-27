"""Robotization pipeline: geometry asset -> actuatable ``SoftBodySpec``.

Three voxelization fronts share a common tail (``common.finalize_from_voxelization``):
- ``mesh``        — watertight mesh -> SDF carve -> voxels   (the validated Phase-3 path)
- ``point_cloud`` — point cloud    -> voxel occupancy
- ``gaussian``    — Gaussian-splat density -> voxel occupancy

``base`` holds the legacy hand-authored partition (``default_robotize``) and the
shared back-lift fiber pattern; ``sdf`` holds the signed-distance helpers.
"""

from .base import default_robotize, lift_push_fiber_pattern
from .mesh import (
    MeshRobotizeConfig,
    RobotizeReport,
    robotize_mesh,
    partition_actuators,
    assign_voxel_ids,
    assign_fibers,
)
from .point_cloud import robotize_point_cloud
from .gaussian import robotize_gaussians
from .common import (
    finalize_from_voxelization,
    voxelize_points,
    gaussians_to_voxelization,
    normalize_points_to_box,
)

__all__ = [
    "default_robotize",
    "lift_push_fiber_pattern",
    "MeshRobotizeConfig",
    "RobotizeReport",
    "robotize_mesh",
    "partition_actuators",
    "assign_voxel_ids",
    "assign_fibers",
    "robotize_point_cloud",
    "robotize_gaussians",
    "finalize_from_voxelization",
    "voxelize_points",
    "gaussians_to_voxelization",
    "normalize_points_to_box",
]
