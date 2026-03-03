"""
Unified backend protocol and adapters for model-based diffusion wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from genedynamics.core.types import Trajectory


@runtime_checkable
class UnifiedModelBasedDiffusionBackend(Protocol):
    """Common backend interface expected by BaseModelBasedDiffusionSolver."""

    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Any | None = None
    ) -> List[Trajectory]:
        ...

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        ...


class ReverseDiffuseBackendAdapter:
    """
    Adapter for EB-MBD style backends exposing reverse_diffuse(_batch) instead of plan(_batch).
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        return self._backend.reverse_diffuse(rng_key, x0)

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        if hasattr(self._backend, "reverse_diffuse_batch"):
            return list(self._backend.reverse_diffuse_batch(keys, x0))
        return [self._backend.reverse_diffuse(k, x0) for k in keys]

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Any | None = None):
        if hasattr(self._backend, "sample_trajectories"):
            return self._backend.sample_trajectories(x0, n_samples, rng_key=rng_key)
        return [self.plan(x0, rng_key=rng_key) for _ in range(max(1, int(n_samples)))]


def ensure_unified_backend(backend: Any) -> Any:
    """
    Return a backend exposing plan/plan_batch without changing backend math logic.
    """
    if hasattr(backend, "plan"):
        return backend
    if hasattr(backend, "reverse_diffuse"):
        return ReverseDiffuseBackendAdapter(backend)
    raise TypeError("Backend must implement either plan() or reverse_diffuse().")

