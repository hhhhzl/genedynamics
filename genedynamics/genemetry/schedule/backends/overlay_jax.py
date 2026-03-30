"""
JAX schedule overlay backend.

Implements hardness, constraint, and diffusion overlays using JAX
operations suitable for use inside ``jax.lax.scan`` traced code.

Registered as ``("schedule", "overlay", "jax")``.
"""

from typing import Any, Tuple

import jax.numpy as jnp

from genedynamics.genemetry.base import ScheduleOverlayBase
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.schedule.config import OverlayConfig


@register_genemetry("schedule", "overlay", "jax")
class ScheduleOverlayJax(ScheduleOverlayBase):
    """JAX implementation of the schedule overlay chain.

    All operations use JAX arrays and are safe for tracing.
    """

    def __init__(self, config: OverlayConfig) -> None:
        self._cfg = config
        # Pre-compute log denominator as a JAX constant.
        self._log_denom = jnp.maximum(
            jnp.log1p(jnp.asarray(config.rho_ref, dtype=jnp.float32)),
            jnp.asarray(1e-6, dtype=jnp.float32),
        )

    def hardness(self, rho: Any) -> Any:
        rho_pos = jnp.maximum(rho.astype(jnp.float32), 0.0)
        return jnp.clip(jnp.log1p(rho_pos) / self._log_denom, 0.0, 1.0)

    def constraint_overlay(self, margin: Any, rho: Any) -> Tuple[Any, Any]:
        c = self._cfg
        rho_pos = jnp.maximum(rho.astype(jnp.float32), 0.0)
        margin_pos = jnp.maximum(margin.astype(jnp.float32), 0.0)

        kappa = (
            jnp.asarray(c.kappa0, dtype=jnp.float32)
            + jnp.asarray(c.kappa_rho_gain, dtype=jnp.float32) * rho_pos
        )
        delta0 = jnp.asarray(c.delta_margin_scale, dtype=jnp.float32) * margin_pos
        delta = delta0 / (
            1.0 + jnp.asarray(c.delta_rho_gain, dtype=jnp.float32) * rho_pos
        )
        delta = jnp.maximum(delta, jnp.asarray(c.delta_min, dtype=jnp.float32))

        return kappa.astype(jnp.float32), delta.astype(jnp.float32)

    def diffusion_overlay(
        self, hardness: Any, eta_base: float
    ) -> Tuple[Any, Any, Any]:
        c = self._cfg
        hard = jnp.clip(hardness.astype(jnp.float32), 0.0, 1.0)
        one_minus_h = 1.0 - hard

        sigma = jnp.asarray(c.sigma_max, dtype=jnp.float32) * (
            one_minus_h ** jnp.asarray(c.sigma_hardness_power, dtype=jnp.float32)
        )
        theta = jnp.asarray(c.theta_max, dtype=jnp.float32) - (
            jnp.asarray(c.theta_max - c.theta_min, dtype=jnp.float32) * hard
        )
        eta_scale = jnp.asarray(c.eta_scale_min, dtype=jnp.float32) + (
            jnp.asarray(c.eta_scale_max - c.eta_scale_min, dtype=jnp.float32)
            * (one_minus_h ** jnp.asarray(c.eta_hardness_power, dtype=jnp.float32))
        )
        eta = jnp.asarray(eta_base, dtype=jnp.float32) * eta_scale

        return (
            sigma.astype(jnp.float32),
            theta.astype(jnp.float32),
            eta.astype(jnp.float32),
        )
