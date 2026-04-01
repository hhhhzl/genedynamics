"""
CFS retraction operator -- orchestrator with backend dispatch.

Retracts trajectories to the feasible set via a Convex Feasible Set
(CFS) QP filter.
"""

from typing import Any, Optional

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.types import RetractionResult
from genedynamics.genemetry.registry import get_genemetry_registry


class CfsRetraction(RetractionOperator):
    """CFS retraction (backend-dispatched).

    Parameters
    ----------
    backend : str
        Backend identifier (default ``"jax"``).
    filter_fn : callable
        The underlying CFS filter function.  For JAX this is typically
        ``inner._filter_actions_single_jit``.
    **kwargs
        Forwarded to the backend implementation constructor.
    """

    def __init__(
        self,
        *,
        backend: str = "jax",
        filter_fn: Any = None,
        **kwargs: Any,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("retraction", "cfs", backend)
        if impl_class is None:
            available = registry.list_backends("retraction", "cfs")
            raise ValueError(
                f"No '{backend}' backend registered for CfsRetraction. "
                f"Available: {available}"
            )
        self._impl: RetractionOperator = impl_class(
            filter_fn=filter_fn, **kwargs
        )

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        return self._impl.retract(state, trajectory, params)
