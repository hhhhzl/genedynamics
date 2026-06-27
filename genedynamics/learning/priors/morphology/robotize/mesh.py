"""Mesh → SoftBodySpec robotization (writeup §3 main pipeline).

Pipeline (matches writeup §3 stages 1-7):

    mesh
    │  1. repair_mesh         — cleanup, drop fragments, fix normals
    │  2. normalize_mesh_to_box — fit inside MPM body box, isotropic scale
    │  3. voxelize_mesh        — solid voxel grid at MPM particle spacing
    │  4. keep_largest_component — connectivity check (writeup step 3)
    │                            (validity filter rejects if too few survive)
    │  5. partition_actuators  — body-axis PCA → K bins along longest axis
    │                            (writeup step 4)
    │  6. assign_fibers        — per-actuator local-axis direction
    │                            (writeup step 5)
    │  7. assign_voxel_ids     — bucket particles into the coarse voxel grid
    │                            used by the MPM occupancy parameter
    │
    SoftBodySpec  (consumed by jax_mpm.scene.build_scene_from_spec)

The actuator partition matches the geometry of `default_robotize` (back→
front along +x, top 20 % of y reserved as a passive dorsal strip) so a
mesh-derived body and the legacy default body are simulator-interchangeable.

Per-axis material field (writeup §3 step 6) is not yet learned — Phase 1's
SoftBodySpec.E_per_particle stays None and the simulator falls back to a
uniform E0. Phase 5+ will plug in `e_latent → E(p)` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..spec import SoftBodySpec
from .base import lift_push_fiber_pattern  # reuse the back-lift / front-press pattern
from .sdf import (
    repair_mesh,
    normalize_mesh_to_box,
    voxelize_mesh,
    voxel_centers,
    keep_largest_component,
    count_components,
    has_ground_support,
    VoxelizationResult,
)


# ---------------------------------------------------------------------------
# Public dataclass: per-asset robotization knobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshRobotizeConfig:
    """Per-asset overrides for the mesh-based robotization pipeline.

    Defaults are tuned for the writeup's locomotion regime: 10 actuators,
    27-cell coarse voxel grid, 20 % passive dorsal strip — matching
    default_robotize so trained controllers transfer.
    """

    # MPM body box (must match the simulator config).
    box_origin: Tuple[float, float, float] = (0.30, 0.05, 0.40)
    box_size: Tuple[float, float, float] = (0.10, 0.06, 0.10)
    box_margin: float = 0.005

    # Voxelization pitch (1/128 = 0.0078) — same density as default_robotize.
    particle_spacing: float = 1.0 / 128.0
    fill_interior: bool = True

    # Coarse voxel grid for the MPM occupancy parameter.
    voxel_dims: Tuple[int, int, int] = (3, 3, 3)

    # Actuators
    n_actuators: int = 10
    passive_top_quantile: float = 0.80

    # Validity gates
    min_filled_cells: int = 64
    min_active_frac: float = 0.05
    require_ground_support: bool = True
    # DiffuseBot-strict connectivity: reject bodies that voxelize into more than
    # one 6-connected component (DiffuseBot resamples until `geometry_is_cc`).
    # Default False → our lenient "keep largest component" behavior (unchanged).
    require_single_component: bool = False


# ---------------------------------------------------------------------------
# Robotization-result diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotizeReport:
    """Per-asset robotization outcome, written to the asset bank manifest."""

    success: bool
    failure_reasons: List[str] = field(default_factory=list)
    n_filled_cells: int = 0
    n_dropped_for_connectivity: int = 0
    # Pre-prune 6-connected component count (1 = DiffuseBot `geometry_is_cc`).
    n_components: int = 1
    actuator_coverage: float = 0.0
    n_actuators_used: int = 0
    body_diameter: float = 0.0


# ---------------------------------------------------------------------------
# Stage helpers (kept top-level so they're independently testable)
# ---------------------------------------------------------------------------


def partition_actuators(
    centers: np.ndarray,
    n_actuators: int,
    passive_top_quantile: float,
) -> np.ndarray:
    """Assign actuator group per particle.

    Algorithm
    ---------
    1. Find the body's principal x-axis via PCA on particle centers; align
       a "body-x" coordinate so back is min, front is max.
    2. Bin body-x into ``n_actuators`` equal-width bins (back → front).
    3. Mark the top ``passive_top_quantile`` of body-y particles passive (-1).

    Output: (N,) int32 in [-1, n_actuators-1].
    """
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    # PCA: principal direction = eigenvector of largest eigenvalue.
    mu = centers.mean(axis=0)
    cov = np.cov(centers - mu, rowvar=False)
    w, V = np.linalg.eigh(cov)              # ascending eigenvalues
    body_x_axis = V[:, -1]                  # largest variance = body length
    # Force "front = +x in world" so the gait is consistent across assets.
    if body_x_axis[0] < 0:
        body_x_axis = -body_x_axis
    body_x = (centers - mu) @ body_x_axis   # (N,) projected coord

    bx_min, bx_max = float(body_x.min()), float(body_x.max())
    n = int(n_actuators)
    bins = np.floor((body_x - bx_min) / max(bx_max - bx_min, 1e-6) * n).astype(np.int32)
    bins = np.clip(bins, 0, n - 1)

    # Passive dorsal strip — top quantile of WORLD y (not body-y) so the
    # spine corresponds to the gravity-up direction of the simulator.
    ys = centers[:, 1]
    if 0.0 < passive_top_quantile < 1.0:
        y_thresh = float(np.quantile(ys, passive_top_quantile))
        actuator_id = np.where(ys >= y_thresh, -1, bins).astype(np.int32)
    else:
        actuator_id = bins.astype(np.int32)
    return actuator_id


def assign_voxel_ids(
    centers: np.ndarray,
    box_origin: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    voxel_dims: Tuple[int, int, int],
) -> np.ndarray:
    """Bucket particle centers into the MPM coarse voxel grid.

    The grid tiles ``box_origin + box_size`` uniformly with ``voxel_dims``
    cells. Returns (N,) int32 in [0, vx*vy*vz).
    """
    vx, vy, vz = (int(v) for v in voxel_dims)
    n_voxels = vx * vy * vz
    if centers.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    box_origin = np.asarray(box_origin, dtype=np.float32)
    box_size = np.asarray(box_size, dtype=np.float32)
    rel = (centers - box_origin) / box_size  # in [0, 1]
    rel = np.clip(rel, 0.0, 0.999999)
    vi = (rel[:, 0] * vx).astype(np.int32)
    vj = (rel[:, 1] * vy).astype(np.int32)
    vk = (rel[:, 2] * vz).astype(np.int32)
    vid = (vi * vy * vz + vj * vz + vk).astype(np.int32)
    return np.clip(vid, 0, n_voxels - 1)


def assign_fibers(
    n_actuators: int,
    actuator_id: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Per-actuator muscle direction.

    Phase 3 ships the same lift-push pattern as `default_robotize` so
    optimizer behavior is comparable across mesh and default morphologies.
    A future learned variant will replace this with PCA-per-bin direction.
    """
    return lift_push_fiber_pattern(int(n_actuators))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def robotize_mesh(
    mesh,
    cfg: Optional[MeshRobotizeConfig] = None,
) -> Tuple[Optional[SoftBodySpec], RobotizeReport]:
    """Convert a single trimesh.Trimesh into a SoftBodySpec.

    Returns ``(spec, report)``. On failure ``spec`` is None and
    ``report.failure_reasons`` carries one or more human-readable messages
    (used by the asset bank to compute RobotizationSuccess).

    The function never raises for "soft" failures (disconnected body, too
    thin, no ground support). It DOES raise for missing dependencies and
    config errors — those are programmer bugs, not asset-bank statistics.
    """
    cfg = cfg or MeshRobotizeConfig()
    fails: List[str] = []

    # 1-2. Repair + normalize + snap to floor.
    m = repair_mesh(mesh)
    m = normalize_mesh_to_box(m, cfg.box_origin, cfg.box_size, margin=cfg.box_margin)
    # Snap min-y to box_origin.y so the body rests on the floor (writeup §3
    # "align to canonical support direction"). normalize_mesh_to_box centers
    # the mesh, which can leave it suspended for shapes shorter than box_size[1].
    min_y = float(m.bounds[0, 1])
    target_y = float(cfg.box_origin[1]) + float(cfg.box_margin)
    m.apply_translation(np.array([0.0, target_y - min_y, 0.0], dtype=np.float64))
    diameter = float(np.linalg.norm(m.extents))

    # 3. Voxelize (solid).
    vox = voxelize_mesh(m, pitch=cfg.particle_spacing, fill_interior=cfg.fill_interior)
    if vox.n_filled < cfg.min_filled_cells:
        fails.append(f"voxelization yielded only {vox.n_filled} cells "
                     f"(< min_filled_cells={cfg.min_filled_cells})")
        return None, RobotizeReport(
            success=False,
            failure_reasons=fails,
            n_filled_cells=vox.n_filled,
            body_diameter=diameter,
        )

    # 4. Connectivity — keep largest 6-connected component.
    n_comp = count_components(vox.occupancy)
    # DiffuseBot-strict: reject anything that is not a single component upfront.
    if cfg.require_single_component and n_comp > 1:
        fails.append(f"not single-connected (n_components={n_comp}); "
                     f"require_single_component=True (DiffuseBot geometry_is_cc)")
        return None, RobotizeReport(
            success=False, failure_reasons=fails,
            n_filled_cells=int(vox.n_filled), n_components=n_comp,
            body_diameter=diameter,
        )
    occ_kept, n_dropped = keep_largest_component(vox.occupancy)
    vox_kept = VoxelizationResult(occupancy=occ_kept, origin=vox.origin, pitch=vox.pitch)
    centers = voxel_centers(vox_kept)
    if centers.shape[0] < cfg.min_filled_cells:
        fails.append(f"after connectivity prune only {centers.shape[0]} cells survive "
                     f"(< min_filled_cells={cfg.min_filled_cells})")
        return None, RobotizeReport(
            success=False,
            failure_reasons=fails,
            n_filled_cells=int(centers.shape[0]),
            n_dropped_for_connectivity=n_dropped,
            n_components=n_comp,
            body_diameter=diameter,
        )

    # 5. Actuator partition.
    actuator_id = partition_actuators(centers, cfg.n_actuators, cfg.passive_top_quantile)
    n_active = int((actuator_id >= 0).sum())
    active_frac = n_active / max(actuator_id.size, 1)
    if active_frac < cfg.min_active_frac:
        fails.append(f"too few active particles ({active_frac:.2%} < "
                     f"{cfg.min_active_frac:.2%})")
    n_actuators_used = int(np.unique(actuator_id[actuator_id >= 0]).size)

    # Optional: ground support gate.
    floor_y = float(cfg.box_origin[1])
    if cfg.require_ground_support and not has_ground_support(centers, floor_y=floor_y):
        fails.append(f"no particle within tol of floor (y≈{floor_y:.3f})")

    if fails:
        return None, RobotizeReport(
            success=False,
            failure_reasons=fails,
            n_filled_cells=int(centers.shape[0]),
            n_dropped_for_connectivity=n_dropped,
            actuator_coverage=active_frac,
            n_actuators_used=n_actuators_used,
            body_diameter=diameter,
        )

    # 6-7. Voxel ids + fiber field.
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
    report = RobotizeReport(
        success=True,
        failure_reasons=[],
        n_filled_cells=int(centers.shape[0]),
        n_dropped_for_connectivity=n_dropped,
        n_components=n_comp,
        actuator_coverage=active_frac,
        n_actuators_used=n_actuators_used,
        body_diameter=diameter,
    )
    return spec, report
