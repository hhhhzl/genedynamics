"""MorphDecoderConfig — A2 shape-latent → occupancy decoder config.

Frozen dataclass + factory, mirroring fidelity_system/specs.py and
mode_system/specs.py so the morphology-latent decoder plugs into the MBD
solver the same way the fidelity ladder and mode marginalizer do.

The decoder g(w) maps a low-dim shape latent w → a per-voxel occupancy field
on the FIXED MBD voxel grid, in [x_lo, x_hi] — exactly the x_morph that the
jax_mpm rollout consumes via _voxel_mass_field. Trained as a VAE on a bank of
robotized occupancy vectors so the latent prior is N(0, I); the joint
model-free/model-based sampler then diffuses w under a standard-normal prior
and decodes to a *plausible* body (learned morphology prior) instead of a flat
Gaussian over raw voxel occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class MorphDecoderConfig:
    """Architecture + output-range config for the A2 morphology decoder.

    Fields
    ------
    latent_dim : int   shape-latent w dimension (the model-free morphology var).
    hidden_dim : int   MLP hidden width (encoder + decoder share the width).
    n_voxels   : int   output occupancy dimension = prod(voxel_dims) of the
                       FIXED MBD grid (e.g. 27 for a 3x3x3 mesh-pipeline body).
    x_lo, x_hi : float physical occupancy range fed to the rollout as x_morph.
    beta_kl    : float VAE KL weight (β-VAE); keeps the latent ≈ N(0, I) so the
                       MBD prior term log p(w) is a clean standard-normal.

    Task-2 (DiffuseBot-aligned continuous co-design fields): when enabled, the
    SAME latent w also decodes the morphology Ψ = {geometry, actuator, stiffness}
    (DiffuseBot Eq 1) — a multi-head decoder so a single gradient-free MBD sample
    of w jointly controls geometry occupancy, the continuous per-voxel actuator
    placement field (softmax over n_actuators), and per-voxel stiffness.
    decode_actuator/decode_stiffness : bool   add the actuator / stiffness head.
    n_actuators : int   K = #actuator groups (actuator head output is n_voxels×K).
    e_lo, e_hi  : float physical Young's-modulus range for the stiffness head.
    """

    latent_dim: int = 8
    hidden_dim: int = 32
    n_voxels: int = 27
    x_lo: float = 0.2
    x_hi: float = 1.0
    beta_kl: float = 1e-3
    # Task-2 multi-head co-design fields (off by default → legacy occ-only decoder).
    decode_actuator: bool = False
    decode_stiffness: bool = False
    n_actuators: int = 0
    e_lo: float = 0.5
    e_hi: float = 3.0
    extra: Dict[str, Any] = field(default_factory=dict)


def default_morph_decoder_config(n_voxels: int = 27) -> MorphDecoderConfig:
    """Factory matching the mesh-pipeline default body (3x3x3 grid, x∈[0.2,1.0])."""
    return MorphDecoderConfig(n_voxels=int(n_voxels))
