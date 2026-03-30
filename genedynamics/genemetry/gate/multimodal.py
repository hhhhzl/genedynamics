"""
Multimodality-based gate policy -- orchestrator with backend dispatch.

Activates geometric operations when the sample spread (multimodality
proxy) exceeds a threshold relative to the current diffusion noise.
"""

from typing import Any

from genedynamics.genemetry.base import GatePolicy
from genedynamics.genemetry.types import GateDecision
from genedynamics.genemetry.registry import get_genemetry_registry


class MultimodalGate(GatePolicy):
    """Multimodality gate (backend-dispatched).

    Parameters
    ----------
    backend : str
        Backend identifier (default ``"jax"``).
    multi_scale : float
        Normalization scale for the multimodality proxy.
    enable_local_gating : bool
        If ``False``, gamma is always 1.0 (gate inactive).
    **kwargs
        Forwarded to the backend implementation constructor.
    """

    def __init__(
        self,
        *,
        backend: str = "jax",
        multi_scale: float = 0.25,
        enable_local_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("gate", "multimodal", backend)
        if impl_class is None:
            available = registry.list_backends("gate", "multimodal")
            raise ValueError(
                f"No '{backend}' backend registered for "
                f"MultimodalGate. Available: {available}"
            )
        self._impl: GatePolicy = impl_class(
            multi_scale=multi_scale,
            enable_local_gating=enable_local_gating,
            **kwargs,
        )

    def evaluate(
        self,
        samples: Any,
        reference: Any,
        window_mask: Any,
        sigma_scale: Any,
        theta: Any,
    ) -> GateDecision:
        return self._impl.evaluate(
            samples, reference, window_mask, sigma_scale, theta
        )
