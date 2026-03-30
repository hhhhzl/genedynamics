"""
Stepping-stone retraction operator -- orchestrator with backend dispatch.

Retracts trajectories to the feasible set by projecting each foot
position onto the nearest stone and clipping step length.
"""

from typing import Any, Optional

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.types import RetractionResult
from genedynamics.genemetry.registry import get_genemetry_registry


class SteppingRetraction(RetractionOperator):
    """Stepping-stone retraction (backend-dispatched).

    Parameters
    ----------
    backend : str
        Backend identifier (default ``"jax"``).
    stone_centers : (N, 2)
        Stepping-stone center positions.
    stone_radii : (N,)
        Stepping-stone radii.
    l_max : scalar
        Maximum per-step displacement.
    action_limit : float
        Action clipping bound.
    **kwargs
        Forwarded to the backend implementation constructor.
    """

    def __init__(
        self,
        *,
        backend: str = "jax",
        stone_centers: Any = None,
        stone_radii: Any = None,
        l_max: Any = None,
        action_limit: float = 1.0,
        **kwargs: Any,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("retraction", "stepping", backend)
        if impl_class is None:
            available = registry.list_backends("retraction", "stepping")
            raise ValueError(
                f"No '{backend}' backend registered for "
                f"SteppingRetraction. Available: {available}"
            )
        self._impl: RetractionOperator = impl_class(
            stone_centers=stone_centers,
            stone_radii=stone_radii,
            l_max=l_max,
            action_limit=action_limit,
            **kwargs,
        )

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        return self._impl.retract(state, trajectory, params)
