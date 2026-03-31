"""
JAX implementation of the multimodality gate.

Uses a bimodality coefficient on a 1-D principal projection of
windowed sample endpoints to detect genuine route ambiguity rather
than raw noise spread.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from genedynamics.genemetry.base import GatePolicy
from genedynamics.genemetry.types import GateDecision
from genedynamics.genemetry.registry import register_genemetry


def _sorted_gap_proxy(proj: jnp.ndarray) -> jnp.ndarray:
    """Sorted-gap bimodality proxy for a 1-D array.

    Sorts the projections, finds the largest gap between consecutive
    values, and returns gap / range.  A large gap indicates a split
    between two clusters.  More robust than moment-based BC with
    small N.

    Returns a value in [0, 1].  High → likely bimodal.
    """
    eps = jnp.asarray(1e-8, dtype=jnp.float32)
    s = jnp.sort(proj)                                       # (N,)
    gaps = s[1:] - s[:-1]                                     # (N-1,)
    max_gap = jnp.max(gaps)
    total_range = s[-1] - s[0] + eps
    return jnp.clip(max_gap / total_range, 0.0, 1.0)


@register_genemetry("gate", "multimodal", "jax")
class MultimodalGateJax(GatePolicy):
    """Multimodality gate (JAX) — bimodality-coefficient proxy.

    Extracts the last quarter of windowed sample actions as a "route
    signature", projects them onto the principal axis via one power-
    iteration step, and checks for bimodality using the BC statistic.

    Parameters
    ----------
    multi_scale : float
        Kept for interface compat (unused in bimodality mode).
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
        # samples: (N, H, D)
        N, H, D = samples.shape
        eps = jnp.asarray(1e-8, dtype=jnp.float32)

        # -- Route embedding: last quarter of windowed actions ----------
        quarter = max(H // 4, 1)  # Python int — H is static at trace time
        # Mask by window to focus on the active temporal region.
        masked = samples * window_mask[None, :, None]           # (N, H, D)
        endpoints = masked[:, H - quarter:, :].reshape(N, -1)   # (N, quarter*D)

        # -- 1-D principal projection via one power-iteration step ------
        mu = jnp.mean(endpoints, axis=0)
        centered = endpoints - mu[None, :]                       # (N, F)
        # Initialise with direction of maximum per-feature variance.
        v0 = jnp.std(centered, axis=0)                           # (F,)
        v0 = v0 / jnp.maximum(jnp.linalg.norm(v0), eps)
        # One power iteration:  v <- C^T C v  then normalise.
        Cv = centered @ v0                                       # (N,)
        v1 = centered.T @ Cv                                     # (F,)
        v1 = v1 / jnp.maximum(jnp.linalg.norm(v1), eps)
        proj = centered @ v1                                     # (N,)

        # -- Sorted-gap bimodality proxy --------------------------------
        pi_route = _sorted_gap_proxy(proj)

        gamma_geo = (pi_route > theta).astype(jnp.float32)
        gamma = jax.lax.cond(
            jnp.asarray(self._enable_local_gating),
            lambda _: gamma_geo,
            lambda _: jnp.asarray(1.0, dtype=jnp.float32),
            operand=None,
        )

        # Legacy spread kept for diagnostics / backward compat.
        spread = jnp.mean(
            jnp.std(samples - reference, axis=0) * window_mask[:, None]
        )
        return GateDecision(
            gamma=gamma,
            meta={"pi_multi": pi_route, "spread": spread},
        )
