"""Unified robotization back-end: {voxelization} → SoftBodySpec.

The mesh pipeline (`mesh_robotize.robotize_mesh`) turns a mesh into a
`VoxelizationResult`, then runs a fixed tail (connectivity prune → actuator
partition → validity gates → voxel ids → fibers → SoftBodySpec). This module
factors that tail out as `finalize_from_voxelization` and adds two more front
ends that produce a `VoxelizationResult` WITHOUT trimesh:

    - `voxelize_points`         : point cloud (e.g. Point-E / DiffuseBot prior)
    - `gaussians_to_voxelization`: 3D Gaussian splats (e.g. DiffGS prior)

so point-cloud and Gaussian-splat priors robotize through the SAME validity
gates and produce the SAME SoftBodySpec the simulator already consumes. The
mesh path is left untouched (it has its own equivalent inline tail); this keeps
the verified mesh robotization byte-identical while unifying the new fronts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..spec import SoftBodySpec
from .mesh import (
    MeshRobotizeConfig,
    RobotizeReport,
    partition_actuators,
    assign_voxel_ids,
    assign_fibers,
)
from .sdf import (
    VoxelizationResult,
    voxel_centers,
    keep_largest_component,
    has_ground_support,
)


# ---------------------------------------------------------------------------
# Normalization (point-cloud analogue of sdf.normalize_mesh_to_box + floor snap)
# ---------------------------------------------------------------------------


def normalize_points_to_box(
    points: np.ndarray,
    box_origin: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    *,
    margin: float = 0.0,
    return_scale: bool = False,
):
    """Isotropic-fit a point set into the MPM body box, then snap to the floor.

    Mirrors mesh_robotize's normalize+floor-snap so point/Gaussian bodies land
    in the same world frame as mesh bodies. With ``return_scale`` also returns
    the isotropic scale factor (Gaussian splats need it to rescale their σ).
    """
    pts = np.asarray(points, dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extents = np.maximum(hi - lo, 1e-9)
    bs = np.asarray(box_size, dtype=np.float64)
    if margin > 0:
        bs = bs - 2.0 * float(margin)
    if np.any(bs <= 0):
        raise ValueError(f"box_size after margin is non-positive: {bs}")
    s = float(np.min(bs / extents))
    pts = (pts - 0.5 * (lo + hi)) * s            # center at origin, isotropic scale
    box_center = np.asarray(box_origin, dtype=np.float64) + (float(margin) if margin > 0 else 0.0) + 0.5 * bs
    pts = pts + box_center
    # Floor snap: min-y to box_origin.y + margin (rest on the floor).
    target_y = float(box_origin[1]) + float(margin)
    pts[:, 1] += target_y - float(pts[:, 1].min())
    pts = pts.astype(np.float32)
    return (pts, s) if return_scale else pts


# ---------------------------------------------------------------------------
# Front ends → VoxelizationResult
# ---------------------------------------------------------------------------


def voxelize_points(points: np.ndarray, pitch: float) -> VoxelizationResult:
    """Bucket a (P, 3) point cloud (already in world coords) into a solid-ish
    occupancy grid at ``pitch``. Occupied = any point in the cell.

    origin = grid min corner, matching `voxel_centers`' (idx+0.5)·pitch.
    """
    if pitch <= 0:
        raise ValueError(f"pitch must be > 0, got {pitch}")
    pts = np.asarray(points, dtype=np.float64)
    lo = pts.min(axis=0)
    idx = np.floor((pts - lo[None, :]) / float(pitch)).astype(np.int64)
    dims = tuple(int(d) for d in (idx.max(axis=0) + 1))
    occ = np.zeros(dims, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return VoxelizationResult(occupancy=occ, origin=lo.astype(np.float64), pitch=float(pitch))


def gaussians_to_voxelization(
    centers: np.ndarray,
    scales: np.ndarray,
    opacities: Optional[np.ndarray],
    pitch: float,
    *,
    threshold: float = 0.5,
    pad_scales: float = 2.0,
    max_cells: int = 400_000,
) -> VoxelizationResult:
    """3D Gaussian splats → occupancy by thresholding the accumulated density

        ρ(x) = Σ_g opacity_g · exp(−½ Σ_d ((x_d − μ_gd) / σ_gd)²).

    centers: (G, 3). scales: (G,) isotropic or (G, 3) per-axis σ.
    opacities: (G,) or None (→ ones). Grid spans the centers' bbox padded by
    ``pad_scales``·σ. Occupied = ρ > threshold.
    """
    if pitch <= 0:
        raise ValueError(f"pitch must be > 0, got {pitch}")
    mu = np.asarray(centers, dtype=np.float64)
    G = mu.shape[0]
    sig = np.asarray(scales, dtype=np.float64)
    if sig.ndim == 1:
        sig = np.repeat(sig[:, None], 3, axis=1)
    sig = np.maximum(sig, 1e-9)
    op = np.ones(G) if opacities is None else np.asarray(opacities, dtype=np.float64)

    pad = float(pad_scales) * float(sig.max())
    lo = mu.min(axis=0) - pad
    hi = mu.max(axis=0) + pad
    dims = np.maximum(np.ceil((hi - lo) / float(pitch)).astype(np.int64), 1)
    if int(np.prod(dims)) > int(max_cells):
        raise ValueError(
            f"gaussian grid {tuple(dims)} exceeds max_cells={max_cells}; "
            f"increase pitch or reduce padding"
        )
    axes = [lo[d] + (np.arange(dims[d]) + 0.5) * float(pitch) for d in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (Ncell, 3)

    dens = np.zeros(grid.shape[0], dtype=np.float64)
    for g in range(G):  # G is small (a generated splat set); loop keeps memory flat
        d2 = np.sum(((grid - mu[g][None, :]) / sig[g][None, :]) ** 2, axis=1)
        dens += op[g] * np.exp(-0.5 * d2)
    occ = (dens > float(threshold)).reshape(tuple(int(d) for d in dims))
    return VoxelizationResult(occupancy=occ, origin=lo.astype(np.float64), pitch=float(pitch))


# ---------------------------------------------------------------------------
# Shared tail: VoxelizationResult → (SoftBodySpec | None, RobotizeReport)
# ---------------------------------------------------------------------------


def finalize_from_voxelization(
    vox: VoxelizationResult,
    cfg: MeshRobotizeConfig,
    *,
    diameter: float = 0.0,
) -> Tuple[Optional[SoftBodySpec], RobotizeReport]:
    """Connectivity prune → actuator partition → validity → voxel ids → fibers
    → SoftBodySpec. Mirrors mesh_robotize.robotize_mesh stages 4–7 so all three
    fronts (mesh / point-cloud / Gaussian) share identical validity semantics.
    """
    fails: List[str] = []

    if vox.n_filled < cfg.min_filled_cells:
        fails.append(f"voxelization yielded only {vox.n_filled} cells "
                     f"(< min_filled_cells={cfg.min_filled_cells})")
        return None, RobotizeReport(success=False, failure_reasons=fails,
                                    n_filled_cells=vox.n_filled, body_diameter=diameter)

    occ_kept, n_dropped = keep_largest_component(vox.occupancy)
    vox_kept = VoxelizationResult(occupancy=occ_kept, origin=vox.origin, pitch=vox.pitch)
    centers = voxel_centers(vox_kept)
    if centers.shape[0] < cfg.min_filled_cells:
        fails.append(f"after connectivity prune only {centers.shape[0]} cells survive "
                     f"(< min_filled_cells={cfg.min_filled_cells})")
        return None, RobotizeReport(success=False, failure_reasons=fails,
                                    n_filled_cells=int(centers.shape[0]),
                                    n_dropped_for_connectivity=n_dropped,
                                    body_diameter=diameter)

    actuator_id = partition_actuators(centers, cfg.n_actuators, cfg.passive_top_quantile)
    n_active = int((actuator_id >= 0).sum())
    active_frac = n_active / max(actuator_id.size, 1)
    if active_frac < cfg.min_active_frac:
        fails.append(f"too few active particles ({active_frac:.2%} < {cfg.min_active_frac:.2%})")
    n_actuators_used = int(np.unique(actuator_id[actuator_id >= 0]).size)

    floor_y = float(cfg.box_origin[1])
    if cfg.require_ground_support and not has_ground_support(centers, floor_y=floor_y):
        fails.append(f"no particle within tol of floor (y≈{floor_y:.3f})")

    if fails:
        return None, RobotizeReport(success=False, failure_reasons=fails,
                                    n_filled_cells=int(centers.shape[0]),
                                    n_dropped_for_connectivity=n_dropped,
                                    actuator_coverage=active_frac,
                                    n_actuators_used=n_actuators_used,
                                    body_diameter=diameter)

    voxel_id = assign_voxel_ids(centers, cfg.box_origin, cfg.box_size, cfg.voxel_dims)
    fiber_dirs = assign_fibers(cfg.n_actuators, actuator_id, centers)
    spec = SoftBodySpec(
        particles_x0=centers,
        actuator_id=actuator_id,
        voxel_id=voxel_id,
        fiber_dirs=fiber_dirs,
        n_actuators=int(cfg.n_actuators),
        n_voxels=int(np.prod(cfg.voxel_dims)),
        voxel_dims=tuple(int(v) for v in cfg.voxel_dims),
        E_per_particle=None,
    )
    report = RobotizeReport(success=True, failure_reasons=[],
                            n_filled_cells=int(centers.shape[0]),
                            n_dropped_for_connectivity=n_dropped,
                            actuator_coverage=active_frac,
                            n_actuators_used=n_actuators_used,
                            body_diameter=diameter)
    return spec, report
