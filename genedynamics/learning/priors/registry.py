"""Prior registry: `register_prior` / `make_prior(name, **kw)`.

Mirrors the solver registry pattern. Built-in names: ``"rl"`` (multi-backend RL
policy prior, jax=brax integration), ``"diffusion"`` (learned `s_theta`).
``"codesign"`` is reserved for the lifted mrmfmbd ThetaPrior (deferred).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_PRIOR_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register_prior(name: str, factory: Callable[..., Any]) -> None:
    _PRIOR_FACTORIES[str(name)] = factory


def make_prior(name: str, **kwargs: Any) -> Any:
    if name not in _PRIOR_FACTORIES:
        raise KeyError(f"prior '{name}' not registered (have: {sorted(_PRIOR_FACTORIES)})")
    return _PRIOR_FACTORIES[name](**kwargs)


def list_priors() -> list:
    return sorted(_PRIOR_FACTORIES)


def _register_builtins() -> None:
    """Lazy registration (avoids importing backends — brax — at import time)."""
    def _rl(**kw):
        from genedynamics.learning.priors.rl.rl_prior import RLPrior
        return RLPrior(**kw)

    def _diffusion(**kw):
        from genedynamics.learning.priors.diffusion.diffusion_prior import LearnedDiffusionPrior
        return LearnedDiffusionPrior(**kw)

    register_prior("rl", _rl)
    register_prior("diffusion", _diffusion)


_register_builtins()

__all__ = ["register_prior", "make_prior", "list_priors"]
