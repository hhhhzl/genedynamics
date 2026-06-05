"""
JAX flow-matching (FM) reverse-transport backend.

Implements the flow-matching-style reverse update (see
latex_2go/sections/prelims.tex, eq:fm_update_expanded_2go):

    tau_{k-1} = sqrt(abar_{k-1}) * tau1
              + sqrt(1 - abar_{k-1}) * eps
              + sigma_k * z_k

where ``tau1`` is the (reward-weighted) clean estimate ``tau-hat_{1|k}`` and
``eps`` is the residual / predicted-noise direction ``eps-hat_k``.

With ``sigma = 0`` (the default) this reduces to the deterministic discrete flow
update

    tau_{k-1} = sqrt(abar_{k-1}) * tau1 + sqrt(1 - abar_{k-1}) * eps,

which is *identical* to the DDIM deterministic update. The two families diverge
only when ``sigma > 0``: FM keeps the full ``sqrt(1 - abar_{k-1})`` eps
coefficient, whereas DDIM (eq:ddim_update_2go) subtracts ``sigma^2`` under that
root.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax.numpy as jnp

from genedynamics.solvers.common.transport.base import ReverseTransport


class FMTransport(ReverseTransport):
    """Flow-matching-style reverse transport (eq:fm_update_expanded_2go)."""

    family = "FM"
    #: FM applies the ``sigma * z`` reverse-noise kick inside ``step``; the MBD
    #: solver hands it the per-step diversity sigma/z and must not re-add them.
    consumes_diversity_noise = True

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
        # tau1_k    = tau-hat_{1|k}   (reward-weighted clean estimate)
        # eps_k     = eps-hat_k       (residual / predicted-noise direction)
        # abar_km1  = alphas_bar[idx - 1]
        if eps_k is None:
            raise ValueError("FMTransport.step requires eps_k (got None).")
        tau1 = tau1_k
        abar_km1 = sched["abar_km1"]

        # FM eps coefficient: full sqrt(1 - abar_{k-1}), independent of sigma.
        eps_coef = jnp.sqrt(jnp.maximum(1.0 - abar_km1, 0.0))
        tau_km1 = jnp.sqrt(abar_km1) * tau1 + eps_coef * eps_k
        if z is not None:
            tau_km1 = tau_km1 + sigma * z
        # Returns the NOISY-space iterate tau_{k-1} (latex eq:fm_update_expanded_2go);
        # see the note in ddim_jax.py — requires a noisy-carry solver, not the
        # clean-estimate resample scheme.
        return tau_km1
