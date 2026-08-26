"""Winkler elastic-foundation surface medium (spatially-varying compliance).

A pluggable contact medium for the surface-scan env: the surface yields locally under
the probe like a bed of independent springs (Winkler foundation) with a spatially
varying normal stiffness ``k(ξ,η)`` over the surface coordinates ``(ξ,η)∈[0,1]²``. The
normal contact force is the elastic reaction

    F_n(ξ,η) = k(ξ,η) · max(0, δ)        (δ = probe penetration below the rest surface)

closed-form and differentiable (no mjx contact) — cheap in a many-sample planner rollout
and exact for the clean-state geometry. ``k`` is a fixed map: ``uniform`` or
``stripes`` / ``center_hard`` / ``center_soft`` spots — used for the HYBRID medium, where
mjx's per-geom-only ``solref`` cannot express a spatial stiffness map.

Upstream (``core/contact``) — reusable by any deformable-surface env. Pure JAX.
"""

from __future__ import annotations

import jax.numpy as jnp

_SPOT_R = 0.25          # central-spot radius in (ξ,η)
_N_STRIPES = 4          # hard/soft band count along ξ


def stiffness_field(
    kind: str, xi, eta, k_hard, k_soft, transition_width: float = 0.0
) -> jnp.ndarray:
    """Local surface stiffness ``k(ξ,η)`` for a fixed map ``kind`` (a STATIC config
    string, traced once; ``xi,eta`` are traced). ``stripes`` alternates hard/soft bands
    along ξ; ``center_hard`` / ``center_soft`` is a hard / soft central disk in the
    opposite field; anything else (``uniform``) returns ``k_soft``."""
    kind = str(kind).lower()
    k_hard, k_soft = jnp.asarray(k_hard, jnp.float32), jnp.asarray(k_soft, jnp.float32)
    width = float(transition_width)
    if kind == "stripes":
        if width > 0.0:
            # Smooth square wave with the same four alternating bands.  Scale
            # tanh by the sinusoid slope so ``width`` is approximately the
            # transition width in normalized surface coordinates.
            phase = jnp.sin(jnp.pi * _N_STRIPES * jnp.asarray(xi))
            scale = jnp.pi * _N_STRIPES * max(width, 1.0e-6)
            hard_weight = 0.5 * (1.0 + jnp.tanh(phase / scale))
            return k_soft + (k_hard - k_soft) * hard_weight
        band = (jnp.floor(jnp.asarray(xi) * _N_STRIPES).astype(jnp.int32) % 2) == 0
        return jnp.where(band, k_hard, k_soft)
    r2 = (jnp.asarray(xi) - 0.5) ** 2 + (jnp.asarray(eta) - 0.5) ** 2
    if width > 0.0:
        radius = jnp.sqrt(r2 + 1.0e-12)
        inside_weight = 0.5 * (
            1.0 + jnp.tanh((_SPOT_R - radius) / max(width, 1.0e-6))
        )
        if kind == "center_hard":
            return k_soft + (k_hard - k_soft) * inside_weight
        if kind == "center_soft":
            return k_hard - (k_hard - k_soft) * inside_weight
    inside = r2 < (_SPOT_R ** 2)
    if kind == "center_hard":
        return jnp.where(inside, k_hard, k_soft)
    if kind == "center_soft":
        return jnp.where(inside, k_soft, k_hard)
    return k_soft           # uniform


def winkler_force(k, delta) -> jnp.ndarray:
    """Elastic-foundation normal reaction ``k · max(0, δ)`` (≥ 0)."""
    return jnp.asarray(k) * jnp.maximum(jnp.asarray(delta), 0.0)


__all__ = ["stiffness_field", "winkler_force"]
