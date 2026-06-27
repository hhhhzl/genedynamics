"""Robotization: geometry → SoftBodySpec.

Phase 0 only ships the default partition used by the existing crawling_ground
task (X-bin actuators, passive dorsal strip, lift-push fiber pattern,
uniform voxel grid). Mesh-based robotization (TripoSG/TRELLIS pipeline) will
land in Phase 3 as a sibling function `mesh_robotize(mesh, a_latent, e_latent)`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..spec import SoftBodySpec


def lift_push_fiber_pattern(n_actuators: int) -> np.ndarray:
    """Per-actuator muscle direction d_i in (X, Y, Z), unit-normalized.

    Convention (matches scene._muscle_directions):
        - All actuators push along +X (forward).
        - Back half rotates toward +Y (lift trailing edge).
        - Front half rotates toward -Y (press leading edge into ground).
        - Z component is zero.

    Returns (n_actuators, 3) float32.
    """
    half = n_actuators // 2
    dx = np.ones((n_actuators,), dtype=np.float32)
    idx = np.arange(n_actuators)
    dy = np.where(idx < half, 1.0, -1.0).astype(np.float32)
    dz = np.zeros((n_actuators,), dtype=np.float32)
    raw = np.stack([dx, dy, dz], axis=-1)
    return (raw / np.linalg.norm(raw, axis=-1, keepdims=True)).astype(np.float32)


def _sample_box_particles(
    box_origin: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    particle_spacing: float,
) -> np.ndarray:
    """Regular grid of particles centred inside each sub-cell of an AABB."""
    sx, sy, sz = box_size
    ox, oy, oz = box_origin
    nx = max(1, int(sx / particle_spacing))
    ny = max(1, int(sy / particle_spacing))
    nz = max(1, int(sz / particle_spacing))
    rx, ry, rz = sx / nx, sy / ny, sz / nz
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    xs = ox + (ii + 0.5) * rx
    ys = oy + (jj + 0.5) * ry
    zs = oz + (kk + 0.5) * rz
    return np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=-1).astype(np.float32)


def default_robotize(cfg) -> SoftBodySpec:
    """Default crawling_ground robotization (extracted from scene.build_scene).

    Pipeline:
      1. Sample particles uniformly inside cfg.box_origin + cfg.box_size.
      2. Actuator label by X-bin: bin 0 at x_min (back), bin n-1 at x_max (front).
      3. Top 20 % of Y becomes passive (actuator_id = -1) — inert dorsal spine.
      4. Voxel id by uniform partition of the bounding box (cfg.voxel_dims).
      5. Fiber field = lift_push_fiber_pattern(cfg.n_actuators).

    Args
    ----
    cfg : MPMConfig (envs.external.jax_mpm.scene.MPMConfig)
        Used as a duck-typed bag of parameters; we read box_origin,
        box_size, particle_spacing, n_actuators, voxel_dims.
    """
    pts = _sample_box_particles(cfg.box_origin, cfg.box_size, cfg.particle_spacing)

    xs_flat = pts[:, 0]
    ys_flat = pts[:, 1]
    zs_flat = pts[:, 2]
    x_min, x_max = float(xs_flat.min()), float(xs_flat.max())
    y_min, y_max = float(ys_flat.min()), float(ys_flat.max())
    z_min, z_max = float(zs_flat.min()), float(zs_flat.max())

    n_act = int(cfg.n_actuators)
    bin_idx = np.floor(
        (xs_flat - x_min) / max(x_max - x_min, 1e-6) * n_act
    ).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_act - 1)

    y_top_thresh = float(np.quantile(ys_flat, 0.80))
    actuator_id = np.where(ys_flat >= y_top_thresh, -1, bin_idx).astype(np.int32)

    vx, vy, vz = cfg.voxel_dims
    n_voxels = int(vx) * int(vy) * int(vz)
    vi = np.clip(np.floor((xs_flat - x_min) / max(x_max - x_min, 1e-6) * vx).astype(np.int32), 0, vx - 1)
    vj = np.clip(np.floor((ys_flat - y_min) / max(y_max - y_min, 1e-6) * vy).astype(np.int32), 0, vy - 1)
    vk = np.clip(np.floor((zs_flat - z_min) / max(z_max - z_min, 1e-6) * vz).astype(np.int32), 0, vz - 1)
    voxel_id = (vi * vy * vz + vj * vz + vk).astype(np.int32)

    fiber_dirs = lift_push_fiber_pattern(n_act)

    return SoftBodySpec(
        particles_x0=pts,
        actuator_id=actuator_id,
        voxel_id=voxel_id,
        fiber_dirs=fiber_dirs,
        n_actuators=n_act,
        n_voxels=n_voxels,
        voxel_dims=(int(vx), int(vy), int(vz)),
        E_per_particle=None,
    )
