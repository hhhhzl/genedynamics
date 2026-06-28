"""Point-cloud → SoftBodySpec robotization (Point-E / DiffuseBot prior path).

A point cloud (P, 3) — e.g. from a Point-E style prior, or the DiffuseBot
diffusion output — is normalized into the MPM body box, bucketed into a solid
occupancy grid, and run through the shared robotization tail. Pure numpy: no
trimesh, no neural SDF, so it works on CPU and is unit-testable with synthetic
points. (The DiffuseBot-faithful neural-SDF solidify is a separate GPU path;
this is the gradient-free / non-neural front used by the gradient-free pipeline.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .protocols import SoftBodySpec
from .mesh_robotize import MeshRobotizeConfig, RobotizeReport
from .robotize_common import (
    normalize_points_to_box,
    voxelize_points,
    finalize_from_voxelization,
)


def robotize_point_cloud(
    points: np.ndarray,
    cfg: Optional[MeshRobotizeConfig] = None,
) -> Tuple[Optional[SoftBodySpec], RobotizeReport]:
    """Convert a (P, 3) point cloud into a SoftBodySpec.

    Returns ``(spec, report)``; on a soft failure (too few cells, disconnected,
    no ground support) ``spec`` is None and ``report.failure_reasons`` explains.
    Raises only on malformed input.
    """
    cfg = cfg or MeshRobotizeConfig()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (P, 3), got {pts.shape}")
    if pts.shape[0] < cfg.min_filled_cells:
        return None, RobotizeReport(
            success=False,
            failure_reasons=[f"only {pts.shape[0]} points (< min_filled_cells={cfg.min_filled_cells})"],
        )

    pts_n = normalize_points_to_box(pts, cfg.box_origin, cfg.box_size, margin=cfg.box_margin)
    diameter = float(np.linalg.norm(pts_n.max(axis=0) - pts_n.min(axis=0)))
    vox = voxelize_points(pts_n, cfg.particle_spacing)
    return finalize_from_voxelization(vox, cfg, diameter=diameter)
