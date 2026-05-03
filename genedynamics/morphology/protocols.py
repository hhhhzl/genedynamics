"""SoftBodySpec — robotization output consumed by jax_mpm.scene.

The robotization layer (default_robotize, mesh_robotize, ...) produces a
SoftBodySpec that fully describes a simulatable, actuatable soft body:
particle layout, actuator partitioning, fiber field, voxel id mapping.

The simulator (scene.build_scene_from_spec) turns this into a SceneData
holding JAX arrays for the rollout.

Keeping this layer in pure numpy means:
- it is JAX-version independent
- it can be cached / serialized to disk (asset bank for Phase 3)
- mesh-based robotization (TripoSG/TRELLIS in Phase 3) plugs in here
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SoftBodySpec:
    """Pure-numpy description of a robotized soft body.

    Fields
    ------
    particles_x0 : (N, 3) float32
        Initial particle positions in world coordinates.
    actuator_id : (N,) int32
        Group label per particle, in [-1, n_actuators-1]. -1 = passive.
    voxel_id : (N,) int32
        Index into a voxel grid of size n_voxels, used by the simulator's
        occupancy-based mass field (see scene._voxel_mass_field).
    fiber_dirs : (n_actuators, 3) float32
        Unit-norm muscle direction d_i per actuator group. The MPM stress
        adds act * (d_i ⊗ d_i) for particles in group i.
    n_actuators : int
    n_voxels : int
    voxel_dims : (vx, vy, vz)
        Informational: shape of the voxel grid that defines voxel_id.
    E_per_particle : optional (N,) float32
        Per-particle Young's modulus override. None → simulator uses
        a uniform E0 from cfg.

    Notes
    -----
    Per-particle mass is NOT stored here: it is computed by the simulator
    from a voxel occupancy vector (the morphology parameter being optimized)
    via voxel_id. So the same SoftBodySpec can be re-used across many
    candidate morphology parameter values.
    """

    particles_x0: np.ndarray
    actuator_id: np.ndarray
    voxel_id: np.ndarray
    fiber_dirs: np.ndarray
    n_actuators: int
    n_voxels: int
    voxel_dims: Tuple[int, int, int] = (1, 1, 1)
    E_per_particle: Optional[np.ndarray] = None

    @property
    def n_particles(self) -> int:
        return int(self.particles_x0.shape[0])
