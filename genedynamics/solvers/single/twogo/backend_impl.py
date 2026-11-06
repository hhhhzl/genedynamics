"""Backend protocol/base for the 2GO geometry-aware gated solver."""

from __future__ import annotations

from typing import Protocol, Any, Dict, List

from genedynamics.core.types import Trajectory
from genedynamics.solvers.common.model_based_diffusion.backend_protocol import ensure_unified_backend


class TwoGOBackend(Protocol):
    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        ...

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Any | None = None) -> List[Trajectory]:
        ...


def to_unified_backend(backend_impl: Any) -> Any:
    """Expose the unified plan/plan_batch backend interface."""
    return ensure_unified_backend(backend_impl)

