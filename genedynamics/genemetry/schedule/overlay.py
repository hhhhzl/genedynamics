"""
Schedule overlay orchestrator.

Dispatches to a backend implementation (JAX or NumPy) via the genemetry
registry.  Provides the unified ``compute()`` and ``compute_series()``
interface used by the solver.
"""

from typing import Any, Optional, Tuple

from genedynamics.genemetry.base import ScheduleOverlayBase
from genedynamics.genemetry.registry import get_genemetry_registry
from genedynamics.genemetry.schedule.config import OverlayConfig
from genedynamics.genemetry.types import OverlayParams


class ScheduleOverlay(ScheduleOverlayBase):
    """Backend-dispatching schedule overlay orchestrator.

    Parameters
    ----------
    config : OverlayConfig
        Overlay configuration (immutable dataclass).
    backend : str
        Backend name (``"jax"`` or ``"numpy"``).
    """

    def __init__(
        self,
        config: OverlayConfig,
        backend: str = "jax",
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("schedule", "overlay", backend)
        if impl_class is None:
            available = registry.list_backends("schedule", "overlay")
            raise ValueError(
                f"No '{backend}' backend for schedule overlay. "
                f"Available: {available}"
            )
        self._impl: ScheduleOverlayBase = impl_class(config=config)
        self._config = config

    @property
    def config(self) -> OverlayConfig:
        return self._config

    def hardness(self, rho: Any) -> Any:
        return self._impl.hardness(rho)

    def constraint_overlay(self, margin: Any, rho: Any) -> Tuple[Any, Any]:
        return self._impl.constraint_overlay(margin, rho)

    def diffusion_overlay(
        self, hardness: Any, eta_base: float
    ) -> Tuple[Any, Any, Any]:
        return self._impl.diffusion_overlay(hardness, eta_base)
