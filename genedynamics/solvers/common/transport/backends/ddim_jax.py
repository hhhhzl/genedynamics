"""
JAX DDIM reverse-transport backend.

Implements the DDIM-style reverse update (see latex_2go/sections/prelims.tex,
eq:ddim_update_2go):

    tau_{k-1} = sqrt(abar_{k-1}) * tau1
              + sqrt(1 - abar_{k-1} - sigma_k^2) * eps
              + sigma_k * z_k

where ``tau1`` is the (reward-weighted) clean estimate ``tau-hat_{1|k}`` and
``eps`` is the residual / predicted-noise direction ``eps-hat_k``.

With ``sigma = 0`` (the default) this reduces to the deterministic DDIM update

    tau_{k-1} = sqrt(abar_{k-1}) * tau1 + sqrt(1 - abar_{k-1}) * eps,

which is *identical* to the FM deterministic update. The two families diverge
only when ``sigma > 0``: DDIM subtracts ``sigma^2`` under the eps coefficient's
square root (eq:fm_update_expanded_2go does not).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax.numpy as jnp

from genedynamics.solvers.common.transport.base import ReverseTransport


class DDIMTransport(ReverseTransport):
    """DDIM-style reverse transport (eq:ddim_update_2go)."""

    family = "DDIM"
    #: DDIM applies the ``sigma * z`` reverse-noise kick inside ``step``; the MBD
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
            raise ValueError("DDIMTransport.step requires eps_k (got None).")
        tau1 = tau1_k
        abar_km1 = sched["abar_km1"]

        sigma2 = sigma * sigma
        # DDIM eps coefficient: subtract sigma^2 under the root.
        eps_coef = jnp.sqrt(jnp.maximum(1.0 - abar_km1 - sigma2, 0.0))
        tau_km1 = jnp.sqrt(abar_km1) * tau1 + eps_coef * eps_k
        if z is not None:
            tau_km1 = tau_km1 + sigma * z
        # Returns the NOISY-space iterate tau_{k-1} (latex eq:ddim_update_2go).
        # A solver that carries the clean estimate Ybar must NOT use this output
        # as-is (dividing by sqrt(abar_{k-1}) collapses it to ~Ybar_curr and does
        # not denoise — see 2GO note). DDIM/FM are deterministic noisy-carry
        # transports; the solver must carry tau, not resample around the clean
        # estimate, for this to be meaningful.
        return tau_km1
