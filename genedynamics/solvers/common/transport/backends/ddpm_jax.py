"""
JAX DDPM reverse-transport backend.

Reproduces the *verbatim* inline DDPM score-form update used by every MBD-family
solver (canonical: ``genedynamics/solvers/single/mbd/backends/mbd_jax.py``,
lines ~269-271):

    score      = (-Yi + sqrt(abar_k) * Ybar_weighted) / (1 - abar_k)
    Yim1       = (Yi + (1 - abar_k) * score) / sqrt(alpha_k)
    Ybar_next  = Yim1 / sqrt(abar_{k-1})

This is the DDPM family special case with the stochastic term off
(``sigma_eff = 0``); the solver applies its separate annealed ``extra_sigma``
diversity term on top, outside the transport.

The arithmetic here is intentionally identical (same operations, same order) to
the inline default so that running a solver with ``transport=DDPMTransport()``
matches the ``transport=None`` baseline.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax.numpy as jnp

from genedynamics.solvers.common.transport.base import ReverseTransport


class DDPMTransport(ReverseTransport):
    """DDPM score-form reverse transport (deterministic, ``sigma_eff = 0``)."""

    family = "DDPM"

    def step(
        self,
        tau_k: Any,
        tau1_k: Any,
        eps_k: Optional[Any],
        score_g: Optional[Any],
        sched: Mapping[str, Any],
        sigma: Any = 0.0,
        z: Optional[Any] = None,
    ) -> Any:
        # tau_k     = Yi
        # tau1_k    = Ybar_weighted
        # abar_k    = alphas_bar[idx]
        # alpha_k   = alphas[idx]
        # abar_km1  = alphas_bar[idx - 1]
        #
        # ``sigma``/``z`` are accepted for interface parity with DDIM/FM but are
        # NOT consumed here: the verbatim inline default carries no reverse-noise
        # term (the solver adds its own annealed ``extra_sigma`` kick on top), so
        # adding ``sigma * z`` here would break the byte-identical baseline.
        Yi = tau_k
        Ybar_weighted = tau1_k
        abar_k = sched["abar_k"]
        alpha_k = sched["alpha_k"]
        abar_km1 = sched["abar_km1"]

        score = (-Yi + jnp.sqrt(abar_k) * Ybar_weighted) / (1.0 - abar_k)
        Yim1 = (Yi + (1.0 - abar_k) * score) / jnp.sqrt(alpha_k)
        Ybar_next = Yim1 / jnp.sqrt(abar_km1)
        return Ybar_next
