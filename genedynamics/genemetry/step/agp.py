"""
AGP (Augmented Geometric Projection) step orchestrator.

Computes a single constrained update via Woodbury solve + tangent noise
projection.  Dispatches to a backend implementation via the registry.
"""

from typing import Any, Optional, Tuple

from genedynamics.genemetry.base import ConstrainedStep
from genedynamics.genemetry.registry import get_genemetry_registry
from genedynamics.genemetry.types import StepResult


class AgpStep(ConstrainedStep):
    """Backend-dispatching AGP step orchestrator.

    Parameters
    ----------
    backend : str
        Backend name (e.g. ``"numpy"``).
    active_topk : int
        Number of top-k active constraints to retain.
    dt : float
        Integration timestep (for constraint Jacobian scaling).
    action_limit : float
        Symmetric clipping bound for actions.
    rng : optional
        Random state for tangent noise (NumPy ``Generator`` or int seed).
    """

    def __init__(
        self,
        backend: str = "numpy",
        active_topk: int = 8,
        dt: float = 0.1,
        action_limit: float = 1.0,
        rng: Any = None,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("step", "agp", backend)
        if impl_class is None:
            available = registry.list_backends("step", "agp")
            raise ValueError(
                f"No '{backend}' backend for AGP step. "
                f"Available: {available}"
            )
        self._impl = impl_class(
            active_topk=active_topk,
            dt=dt,
            action_limit=action_limit,
            rng=rng,
        )

    def step(
        self,
        actions: Any,
        violations: Any,
        gradients: Any,
        window: Tuple[int, int],
        *,
        sigma: float = 0.0,
        kappa: float = 1.0,
        eta: float = 0.08,
    ) -> StepResult:
        return self._impl.step(
            actions, violations, gradients, window,
            sigma=sigma, kappa=kappa, eta=eta,
        )
