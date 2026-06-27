"""Backend protocol for the multi-backend RL policy prior.

Mirrors `solvers/single/<name>/backend_impl.py`: the RL prior keeps the
genedynamics multi-backend architecture (orchestrator `rl_prior.py` +
`backends/`); the jax backend integrates brax's training networks.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class RLPriorBackend(Protocol):
    """A concrete RL-prior backend (jax/brax, numpy/torch later)."""

    output_dim: int

    def act(self, obs: Any, *, key: Optional[Any] = None, deterministic: bool = True) -> Any: ...
    def logp_of_sequence(self, obs_seq: Any, act_seq: Any) -> Any: ...
    def warm_start(self, state: Any) -> Any: ...
