"""
Overlay configuration and hardness/constraint/diffusion utilities.

.. deprecated::
    This module re-exports from ``genedynamics.genemetry.schedule``
    for backward compatibility.  New code should import from
    ``genedynamics.genemetry`` directly.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import jax.numpy as jnp

from genedynamics.genemetry.schedule.config import (
    OverlayConfig,
    resolve_overlay_config,
)


# ======================================================================
# Backward-compatible top-level functions
# ======================================================================

def resolve_twogo_overlay_config(
    config: Dict[str, Any] | None, *, rho_ref_default: float
) -> Dict[str, float]:
    """Parse overlay config.  Returns flat dict for backward compat."""
    oc = resolve_overlay_config(config, rho_ref_default=rho_ref_default)
    return oc.to_dict()


def hardness_from_rho_jax(rho: jnp.ndarray, rho_ref: float) -> jnp.ndarray:
    """Log-scale hardness from augmented Lagrangian penalty (JAX)."""
    rho_pos = jnp.maximum(rho.astype(jnp.float32), 0.0)
    denom = jnp.maximum(jnp.log1p(jnp.asarray(rho_ref, dtype=jnp.float32)), 1e-6)
    return jnp.clip(jnp.log1p(rho_pos) / denom, 0.0, 1.0)


def hardness_from_rho_numpy(rho: np.ndarray, rho_ref: float) -> np.ndarray:
    """Log-scale hardness from augmented Lagrangian penalty (NumPy)."""
    rho_arr = np.maximum(np.asarray(rho, dtype=np.float32), 0.0)
    denom = max(float(np.log1p(float(rho_ref))), 1e-6)
    return np.clip(np.log1p(rho_arr) / denom, 0.0, 1.0).astype(np.float32)


def constraint_overlay_jax(
    margin: jnp.ndarray, rho: jnp.ndarray, cfg: Dict[str, float]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Constraint overlay: (margin, rho) -> (kappa, delta) (JAX)."""
    oc = OverlayConfig.from_dict(cfg)
    from genedynamics.genemetry.schedule.backends.overlay_jax import ScheduleOverlayJax
    return ScheduleOverlayJax(oc).constraint_overlay(margin, rho)


def diffusion_overlay_jax(
    hardness: jnp.ndarray, eta_base: float, cfg: Dict[str, float]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Diffusion overlay: (hardness, eta_base) -> (sigma, theta, eta) (JAX)."""
    oc = OverlayConfig.from_dict(cfg)
    from genedynamics.genemetry.schedule.backends.overlay_jax import ScheduleOverlayJax
    return ScheduleOverlayJax(oc).diffusion_overlay(hardness, eta_base)


def constraint_overlay_numpy(
    margin: np.ndarray, rho: np.ndarray, cfg: Dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Constraint overlay: (margin, rho) -> (kappa, delta) (NumPy)."""
    oc = OverlayConfig.from_dict(cfg)
    from genedynamics.genemetry.schedule.backends.overlay_numpy import ScheduleOverlayNumpy
    return ScheduleOverlayNumpy(oc).constraint_overlay(margin, rho)


def diffusion_overlay_numpy(
    hardness: np.ndarray, eta_base: float, cfg: Dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diffusion overlay: (hardness, eta_base) -> (sigma, theta, eta) (NumPy)."""
    oc = OverlayConfig.from_dict(cfg)
    from genedynamics.genemetry.schedule.backends.overlay_numpy import ScheduleOverlayNumpy
    return ScheduleOverlayNumpy(oc).diffusion_overlay(hardness, eta_base)
