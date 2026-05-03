"""Morphology-prior registry.

Concrete priors call `register(name, factory)` at import time so the rest of
the pipeline (build_asset_bank CLI, evaluator config) can ask for them by
short name. The registry is intentionally tiny — no plugin discovery, no
entry points — to keep the dependency graph obvious.

Usage
-----
    from genedynamics.morphology.priors import get_prior
    prior = get_prior("random_shapes", n_actuators_hint=10)
    meshes = prior.sample("worm-like crawling robot", n=4, seed=0)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .base import MorphologyPrior, PriorMetadata, MissingDependencyError


_REGISTRY: Dict[str, Callable[..., MorphologyPrior]] = {}


def register(name: str, factory: Callable[..., MorphologyPrior]) -> None:
    """Register a prior factory under ``name``. Re-registration overwrites."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"prior name must be a non-empty string, got {name!r}")
    _REGISTRY[name] = factory


def get_prior(name: str, **kwargs: Any) -> MorphologyPrior:
    """Construct a prior by registry name.

    Raises
    ------
    KeyError
        If no prior is registered under ``name``.
    MissingDependencyError
        If the prior's required dependencies (model weights, third_party
        clone) are not available locally.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown morphology prior: {name!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_priors() -> List[str]:
    """All registered prior names in insertion order."""
    return list(_REGISTRY)


# Eager import so registration side-effects fire. Failed imports are caught
# and converted to "missing dep" markers so the caller can still list what's
# unavailable without crashing the whole package.
def _safe_import(modname: str) -> None:
    try:
        __import__(modname)
    except MissingDependencyError:
        # Module's own __init__ noticed something missing and wrote the entry
        # via register(...) → MissingPlaceholder, so just continue.
        pass
    except Exception:  # pragma: no cover — only reached if a prior file is buggy
        # Don't mask real bugs in the prior modules; let them surface.
        raise


_safe_import("genedynamics.morphology.priors.random_shapes")
_safe_import("genedynamics.morphology.priors.triposg")


__all__ = [
    "MorphologyPrior",
    "PriorMetadata",
    "MissingDependencyError",
    "register",
    "get_prior",
    "list_priors",
]
