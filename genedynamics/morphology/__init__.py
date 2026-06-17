"""Morphology layer: convert geometry assets into actuatable soft-body specs.

Public surface:
- SoftBodySpec   — pure-numpy description of a robotized soft body
- default_robotize(cfg) -> SoftBodySpec — current crawling_ground partition
- check_spec(spec, cfg) -> ValidityReport — connectivity / support / mass checks

Phase 3 additions (mesh-based pipeline):
- robotize_mesh(mesh, cfg) -> (SoftBodySpec | None, RobotizeReport)
- MeshRobotizeConfig — knobs for the mesh pipeline
- AssetBank — on-disk manifest + cache for generated meshes + their specs
- save_spec_npz / load_spec_npz — single-asset round trip

Sub-packages:
- genedynamics.morphology.priors — registry of 3D generative priors
                                    (random_shapes, triposg, ...)
"""

from .protocols import SoftBodySpec
from .robotize import default_robotize, lift_push_fiber_pattern
from .validity import ValidityReport, check_spec

# Phase 3 — keep these imports lightweight (pure numpy + lazy-loaded heavy deps).
from .mesh_robotize import (
    MeshRobotizeConfig,
    RobotizeReport,
    robotize_mesh,
    partition_actuators,
    assign_voxel_ids,
    assign_fibers,
)
from .asset_bank import (
    AssetBank,
    AssetEntry,
    BankManifest,
    PromptEntry,
    SCHEMA_VERSION,
    asset_id,
    prompt_id,
    save_spec_npz,
    load_spec_npz,
)

# Stage 6 — unified robotization fronts (point cloud + Gaussian splat).
from .robotize_common import (
    finalize_from_voxelization,
    voxelize_points,
    gaussians_to_voxelization,
    normalize_points_to_box,
)
from .pc_robotize import robotize_point_cloud
from .gs_robotize import robotize_gaussians

__all__ = [
    # Phase 0
    "SoftBodySpec",
    "default_robotize",
    "lift_push_fiber_pattern",
    "ValidityReport",
    "check_spec",
    # Phase 3 — mesh pipeline
    "MeshRobotizeConfig",
    "RobotizeReport",
    "robotize_mesh",
    "partition_actuators",
    "assign_voxel_ids",
    "assign_fibers",
    # Phase 3 — asset bank
    "AssetBank",
    "AssetEntry",
    "BankManifest",
    "PromptEntry",
    "SCHEMA_VERSION",
    "asset_id",
    "prompt_id",
    "save_spec_npz",
    "load_spec_npz",
    # Stage 6 — unified robotization
    "finalize_from_voxelization",
    "voxelize_points",
    "gaussians_to_voxelization",
    "normalize_points_to_box",
    "robotize_point_cloud",
    "robotize_gaussians",
]
