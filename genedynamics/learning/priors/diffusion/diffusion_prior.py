"""LearnedDiffusionPrior — multi-backend learned diffusion prior `s_theta`.

Orchestrator mirroring the RL prior / solver pattern; the jax backend is a DDPM
score MLP (`backends/jax.py`). Conforms to `priors.base.DiffusionPrior`: provides
`score(x,t)` for the transport `score_g` seam plus the `Prior` surface
(warm_start via ancestral sampling).
"""

from __future__ import annotations

from typing import Any, Optional


def _get_diffusion_backend(name: str):
    if name == "jax":
        from genedynamics.learning.priors.diffusion.backends.jax import JaxDiffusionPrior
        return JaxDiffusionPrior
    return None


class LearnedDiffusionPrior:
    """Backend-dispatching diffusion prior (conforms to `priors.base.DiffusionPrior`)."""

    def __init__(self, *, backend: str = "jax", **kwargs: Any) -> None:
        backend_cls = _get_diffusion_backend(backend)
        if backend_cls is None:
            raise ValueError(f"LearnedDiffusionPrior backend '{backend}' not found (have: jax)")
        self.backend_name = str(backend)
        self._impl = backend_cls(**kwargs)
        self.output_dim = int(self._impl.output_dim)

    def score(self, x: Any, t: Any) -> Any:
        return self._impl.score(x, t)

    def train(self, data: Any, **kw: Any) -> "LearnedDiffusionPrior":
        self._impl.train(data, **kw)
        return self

    def act(self, obs: Any, *, key: Optional[Any] = None, deterministic: bool = True) -> Any:
        return self._impl.act(obs, key=key, deterministic=deterministic)

    def logp_of_sequence(self, obs_seq: Any, act_seq: Any) -> Any:
        return self._impl.logp_of_sequence(obs_seq, act_seq)

    def warm_start(self, state: Any) -> Any:
        return self._impl.warm_start(state)


__all__ = ["LearnedDiffusionPrior"]
