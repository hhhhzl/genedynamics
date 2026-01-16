"""
Fixed emerging-barrier constraint scheduler.

Provides barrier parameters (mu/alpha/bound etc.) as constraint params.
Currently fixed per run; can be extended with adaptive update() later.
"""

from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import get_registry


class EmergingBarrierConstraintScheduler(ConstraintScheduler):
    """
    Fixed emerging-barrier scheduler.

    Parameters are constant for now, but kept as attributes for potential
    adaptive update() in the future.
    """

    def __init__(
        self,
        mu: float = 10.0,
        alpha: float = 1.0,
        bound: float = 0.8,
        use_min_over_time: bool = True,
        terminal_energy_weight: float = 0.0,
        margin: float = 0.0,
        backend: str = "numpy",
        _skip_backend_lookup: bool = False,
        **kwargs: Any,
    ):
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.bound = float(bound)
        self.use_min_over_time = bool(use_min_over_time)
        self.terminal_energy_weight = float(terminal_energy_weight)
        self.margin = float(margin)
        self.extra = kwargs
        self.backend = backend

        self._cached_params = {
            "mu": self.mu,
            "alpha": self.alpha,
            "bound": self.bound,
            "use_min_over_time": self.use_min_over_time,
            "terminal_energy_weight": self.terminal_energy_weight,
            "margin": self.margin,
            "_extra": self.extra,
        }

        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "emerging_barrier", backend)
            if impl_class is not None and impl_class is not EmergingBarrierConstraintScheduler:
                self._backend_impl = impl_class(
                    mu=mu,
                    alpha=alpha,
                    bound=bound,
                    use_min_over_time=use_min_over_time,
                    terminal_energy_weight=terminal_energy_weight,
                    margin=margin,
                    backend=backend,
                    _skip_backend_lookup=True,
                    **kwargs,
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None

    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """Return fixed barrier parameters."""
        if self._backend_impl is not None:
            return self._backend_impl.constraint_params(state)
        return self._cached_params.copy()

    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Placeholder for adaptive variants. Currently no-op (fixed schedule).
        For JAX backends, dynamic updates must be implemented in a JAX-compatible
        scheduler/backend.
        """
        return

