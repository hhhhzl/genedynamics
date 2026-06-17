"""3D Gaussian-splat → SoftBodySpec robotization (DiffGS / 3DGS prior path).

A Gaussian splat set (centers μ, scales σ, opacities) — e.g. the output of a
DiffGS-style 3DGS generative prior — is normalized into the MPM body box, its
accumulated density is thresholded into a solid occupancy grid, and run through
the shared robotization tail. Pure numpy: works on CPU and is unit-testable
with hand-made Gaussians. This is the "unified robotization" extension that
lets a 3DGS prior feed the same SoftBodySpec the simulator already consumes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .protocols import SoftBodySpec
from .mesh_robotize import MeshRobotizeConfig, RobotizeReport
from .robotize_common import (
    normalize_points_to_box,
    gaussians_to_voxelization,
    finalize_from_voxelization,
)


def robotize_gaussians(
    centers: np.ndarray,
    scales: np.ndarray,
    opacities: Optional[np.ndarray] = None,
    cfg: Optional[MeshRobotizeConfig] = None,
    *,
    density_threshold: float = 0.5,
) -> Tuple[Optional[SoftBodySpec], RobotizeReport]:
    """Convert a Gaussian splat set into a SoftBodySpec.

    Args
    ----
    centers   : (G, 3) Gaussian means.
    scales    : (G,) isotropic or (G, 3) per-axis σ.
    opacities : (G,) or None (→ ones).
    density_threshold : occupancy = accumulated density > this.

    Returns ``(spec, report)`` with the same failure semantics as the mesh /
    point-cloud paths.
    """
    cfg = cfg or MeshRobotizeConfig()
    mu = np.asarray(centers, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError(f"centers must be (G, 3), got {mu.shape}")
    sig = np.asarray(scales, dtype=np.float64)

    # Normalize centers into the box; rescale σ by the same isotropic factor.
    mu_n, s = normalize_points_to_box(
        mu, cfg.box_origin, cfg.box_size, margin=cfg.box_margin, return_scale=True
    )
    sig_n = sig * float(s)
    diameter = float(np.linalg.norm(np.asarray(mu_n).max(axis=0) - np.asarray(mu_n).min(axis=0)))

    vox = gaussians_to_voxelization(
        np.asarray(mu_n, dtype=np.float64), sig_n, opacities,
        cfg.particle_spacing, threshold=float(density_threshold),
    )
    return finalize_from_voxelization(vox, cfg, diameter=diameter)
