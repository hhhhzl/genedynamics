"""
JAX implementation of the multimodality gate.

Extracted from ``twogo_jax.py`` body() scan function.  The gate
computes a spread-based multimodality proxy and thresholds it
against the current diffusion noise level.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from genedynamics.genemetry.base import GatePolicy
from genedynamics.genemetry.types import GateDecision
from genedynamics.genemetry.registry import register_genemetry


@register_genemetry("gate", "multimodal", "jax")
class MultimodalGateJax(GatePolicy):
    """Multimodality gate (JAX).

    Computes::

        spread   = mean(std(samples - reference, axis=0) * window_mask)
        pi_multi = spread / max(multi_scale * sigma_scale, eps)
        gamma    = float(pi_multi > theta)   if local gating enabled
                   1.0                        otherwise

    Parameters
    ----------
    multi_scale : float
        Normalization scale for the spread (default 0.25).
    enable_local_gating : bool
        If ``False``, gamma is always 1.0.
    """

    def __init__(
        self,
        *,
        multi_scale: float = 0.25,
        enable_local_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        self._multi_scale = float(max(multi_scale, 1e-6))
        self._enable_local_gating = bool(enable_local_gating)

    def evaluate(
        self,
        samples: jnp.ndarray,
        reference: jnp.ndarray,
        window_mask: jnp.ndarray,
        sigma_scale: jnp.ndarray,
        theta: jnp.ndarray,
    ) -> GateDecision:
        spread = jnp.mean(
            jnp.std(samples - reference, axis=0) * window_mask[:, None]
        )
        denom = jnp.maximum(
            jnp.asarray(self._multi_scale, dtype=jnp.float32) * sigma_scale,
            jnp.asarray(1e-6, dtype=jnp.float32),
        )
        pi_multi = spread / denom
        gamma_geo = (pi_multi > theta).astype(jnp.float32)
        gamma = jax.lax.cond(
            jnp.asarray(self._enable_local_gating),
            lambda _: gamma_geo,
            lambda _: jnp.asarray(1.0, dtype=jnp.float32),
            operand=None,
        )
        return GateDecision(
            gamma=gamma,
            meta={"pi_multi": pi_multi, "spread": spread},
        )
