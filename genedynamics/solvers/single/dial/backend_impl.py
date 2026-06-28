"""Backend protocol + unified-backend bridge for the DIAL solver.

Mirrors mbd/backend_impl.py: a lightweight Protocol describing the DIAL backend
surface and ``to_unified_backend`` (= ``ensure_unified_backend``) so the solver
wrapper exposes a uniform ``plan``/``plan_batch`` regardless of backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from genedynamics.solvers.common.model_based_diffusion.backend_protocol import (
    ensure_unified_backend,
)


class DialBackend(Protocol):
    """Protocol for DIAL backend implementations (jax, numpy, ...)."""

    # unified single-shot plan (BaseModelBasedDiffusionSolver path)
    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]: ...

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]: ...

    # WarmStartPlanner capability (receding-horizon bridge path)
    def init_plan_var(self) -> Any: ...
    def make_schedule(self, n_diffuse: int) -> Any: ...
    def replan(self, state: Any, warm_start: Any, schedule: Any, rng: Any) -> Any: ...
    def first_action(self, plan_var: Any) -> Any: ...
    def shift(self, plan_var: Any) -> Any: ...


def to_unified_backend(backend_impl: Any) -> Any:
    """Expose a unified plan/plan_batch backend interface (no math change)."""
    return ensure_unified_backend(backend_impl)
