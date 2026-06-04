"""
JAX adaptive reverse-transport: per-step family selection (DDPM / DDIM / FM).

The unified reverse update (latex_2go/sections/method.tex,
eq:unified_reverse_update_k) allows the transport family ``rho_k`` to vary along
the reverse process. ``AdaptiveTransport`` holds one backend per family and, at
each diffusion step, selects the one indexed by ``sched["transport_family_idx"]``
via a traced ``lax.switch``:

    idx 0 -> DDPM   (reward-weighted clean estimate, polish)
    idx 1 -> DDIM   (deterministic eps recombination)
    idx 2 -> FM     (flow eps recombination)

All three backends now return the *renormalised clean-space* iterate Ybar_{k-1}
(DDIM/FM divide their noisy-space tau_{k-1} by sqrt(abar_{k-1}); DDPM is
clean-space by construction), so the switched output is interchangeable.

This is only constructed when an adaptive schedule is configured (e.g.
``twogo_transport_schedule``); the byte-identical default (``transport=None``)
path is untouched.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax

from genedynamics.solvers.common.transport.base import ReverseTransport
from genedynamics.solvers.common.transport.backends.ddpm_jax import DDPMTransport
from genedynamics.solvers.common.transport.backends.ddim_jax import DDIMTransport
from genedynamics.solvers.common.transport.backends.fm_jax import FMTransport

#: family tag -> switch index (the order the backends are laid out below).
FAMILY_TO_IDX = {"DDPM": 0, "DDIM": 1, "FM": 2}


class AdaptiveTransport(ReverseTransport):
    """Per-step family switch over DDPM / DDIM / FM (``lax.switch``)."""

    family = "ADAPTIVE"
    #: 2GO passes ``sigma=0`` and composes its own ``sigma_eff * P * z`` on top,
    #: so the adaptive transport does not consume the diversity kick itself.
    consumes_diversity_noise = False

    def __init__(self) -> None:
        # Order MUST match FAMILY_TO_IDX.
        self._backends = (DDPMTransport(), DDIMTransport(), FMTransport())

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
        idx = sched["transport_family_idx"]  # int32 per-step family selector
        b = self._backends
        return jax.lax.switch(
            idx,
            [
                lambda _: b[0].step(tau_k, tau1_k, eps_k, score_g, sched, 0.0, None),
                lambda _: b[1].step(tau_k, tau1_k, eps_k, score_g, sched, 0.0, None),
                lambda _: b[2].step(tau_k, tau1_k, eps_k, score_g, sched, 0.0, None),
            ],
            operand=None,
        )
