"""Shared prior protocols (`genedynamics/learning/priors`).

A *prior* is reusable infrastructure any solver can consume through a single
additive `prior=` seam (mirrors the `core/prob.NoiseSampler` seam: `None` =>
byte-identical old behaviour). It is NOT MDAC-private — MDAC is just the first
heavy consumer.

Two protocols:

* :class:`Prior` — a model-free policy prior `p_psi(U | s_0, c)`. A solver uses
  whatever it needs:
    - warm-start:        `U^rl = prior.warm_start(state)`            (MPPI/MBD/MDAC)
    - log-prior weight:  `+ lam_psi * prior.logp_of_sequence(...)`    (CFSMBD/MDAC)
* :class:`DiffusionPrior` (a `Prior`) — adds `score(x, t)` for a learned
  diffusion prior `s_theta`, fed to the transport `score_g` seam (MDAC `omega_mf`).

`logp_of_sequence` must be jit-safe: it uses an info-FREE observation (no
`state.info` dependency) so it can run inside the reverse-diffusion scan.

Backend-agnostic: arrays are whatever the backend uses (jax for the brax
backend). Concrete priors live under `rl/` and `diffusion/` as multi-backend
packages (orchestrator + `backends/`), exactly like the solvers.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

ArrayLike = Any
PRNGKey = Any


@runtime_checkable
class Prior(Protocol):
    """Model-free policy prior `p_psi`."""

    #: action / control dimension the prior emits.
    output_dim: int

    def act(self, obs: ArrayLike, *, key: Optional[PRNGKey] = None,
            deterministic: bool = True) -> ArrayLike:
        """Single-step action `a ~ pi_psi(.|obs)` (mode if `deterministic`)."""
        ...

    def logp_of_sequence(self, obs_seq: ArrayLike, act_seq: ArrayLike) -> ArrayLike:
        """`sum_h log pi_psi(a_h | o_h)` for one (obs_seq, act_seq) — jit-safe
        (info-free obs). vmap over a candidate batch at the call site."""
        ...

    def warm_start(self, state: Any) -> ArrayLike:
        """A control sequence `U^rl` (shape the solver expects, e.g.
        `(Hnode+1, nu)`) to mix into the receding-horizon warm start
        (eq:rl_warm_start)."""
        ...


@runtime_checkable
class DiffusionPrior(Prior, Protocol):
    """Learned diffusion prior `s_theta` (adds the score for transport)."""

    def score(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        """`nabla_x log p_theta(x | t)` — fed to `transport` `score_g`
        (fused as `S_hat = omega_mb S_mb + omega_mf s_theta`)."""
        ...


__all__ = ["Prior", "DiffusionPrior", "ArrayLike", "PRNGKey"]
