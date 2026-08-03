"""RLPrior — multi-backend model-free policy prior (orchestrator).

Keeps the genedynamics multi-backend architecture (like the solvers): a thin
orchestrator that selects a backend by name and delegates the `Prior` surface.
The ``"jax"`` backend integrates brax's training networks (`backends/brax_jax.py`);
numpy/torch backends can be added later under `backends/` without touching
consumers.
"""

from __future__ import annotations

from typing import Any, Optional


def _get_rl_backend(name: str):
    if name in ("jax", "brax"):
        from genedynamics.learning.priors.rl.backends.brax_jax import BraxRLPrior
        return BraxRLPrior
    return None


class RLPrior:
    """Backend-dispatching RL policy prior (conforms to `priors.base.Prior`)."""

    def __init__(self, *, backend: str = "jax", **kwargs: Any) -> None:
        backend_cls = _get_rl_backend(backend)
        if backend_cls is None:
            raise ValueError(f"RLPrior backend '{backend}' not found (have: jax)")
        self.backend_name = str(backend)
        self._impl = backend_cls(**kwargs)
        self.output_dim = int(self._impl.output_dim)

    def act(self, obs: Any, *, key: Optional[Any] = None, deterministic: bool = True) -> Any:
        return self._impl.act(obs, key=key, deterministic=deterministic)

    def logp_of_sequence(self, obs_seq: Any, act_seq: Any) -> Any:
        return self._impl.logp_of_sequence(obs_seq, act_seq)

    def warm_start(self, state: Any) -> Any:
        return self._impl.warm_start(state)

    def sample_horizons(
        self, state: Any, *, key: Any, n_samples: int
    ) -> Any:
        sample = getattr(self._impl, "sample_horizons", None)
        if sample is None:
            raise NotImplementedError(
                f"RLPrior backend '{self.backend_name}' has no horizon sampler"
            )
        return sample(state, key=key, n_samples=int(n_samples))


__all__ = ["RLPrior"]
