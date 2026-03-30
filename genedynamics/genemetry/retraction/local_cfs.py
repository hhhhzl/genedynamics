"""
Local CFS retraction orchestrator.

Window-local gradient-based retraction that pushes actions away from
constraint violations using the SDF gradient.  Operates per-timestep
within a window, making it lightweight and suitable for post-hoc
refinement.

Dispatches to a backend implementation via the genemetry registry.
"""

from typing import Any, Optional

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.registry import get_genemetry_registry
from genedynamics.genemetry.types import RetractionResult


class LocalCfsRetraction(RetractionOperator):
    """Backend-dispatching local CFS retraction orchestrator.

    Parameters
    ----------
    backend : str
        Backend name (e.g. ``"numpy"``).
    gain : float
        CFS gradient gain (scales the correction magnitude).
    dt : float
        Integration timestep.
    action_limit : float
        Symmetric action clipping bound.
    """

    def __init__(
        self,
        backend: str = "numpy",
        gain: float = 0.35,
        dt: float = 0.1,
        action_limit: float = 1.0,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("retraction", "local_cfs", backend)
        if impl_class is None:
            available = registry.list_backends("retraction", "local_cfs")
            raise ValueError(
                f"No '{backend}' backend for local CFS retraction. "
                f"Available: {available}"
            )
        self._impl = impl_class(
            gain=gain, dt=dt, action_limit=action_limit,
        )

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        """Retract using local gradient-based CFS.

        Parameters
        ----------
        state : unused (kept for interface conformance).
        trajectory : (H, U) actions to retract.
        params : dict with ``violations`` (H,), ``gradients`` (H, dim),
            and ``window`` (start, end).

        Returns
        -------
        RetractionResult
        """
        return self._impl.retract(state, trajectory, params)
