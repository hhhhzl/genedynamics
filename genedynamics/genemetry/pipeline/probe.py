"""
Probe pipeline orchestrator.

Selects a mini-batch of trajectories (mixing tail-cost and random
samples), runs retraction on each, and extracts the mean residual
as probe geometry correction.

Dispatches to a backend implementation via the genemetry registry.
"""

from typing import Any, Optional

from genedynamics.genemetry.base import ProbeSampler, RetractionOperator
from genedynamics.genemetry.registry import get_genemetry_registry
from genedynamics.genemetry.types import ProbeResult


class ProbePipeline(ProbeSampler):
    """Backend-dispatching probe pipeline orchestrator.

    Parameters
    ----------
    backend : str
        Backend name (e.g. ``"jax"``).
    tail_mix : float
        Fraction of probe mini-batch drawn from high-cost tail.
    pool_ratio : float
        Fraction of samples to consider in the tail pool.
    max_probes : int
        Maximum number of probes per step (caps mini-batch size).
    """

    def __init__(
        self,
        backend: str = "jax",
        tail_mix: float = 0.5,
        pool_ratio: float = 0.25,
        max_probes: Optional[int] = None,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("pipeline", "probe", backend)
        if impl_class is None:
            available = registry.list_backends("pipeline", "probe")
            raise ValueError(
                f"No '{backend}' backend for probe pipeline. "
                f"Available: {available}"
            )
        self._impl = impl_class(
            tail_mix=tail_mix,
            pool_ratio=pool_ratio,
            max_probes=max_probes,
        )

    def sample_and_retract(
        self,
        trajectories: Any,
        costs: Any,
        retraction_op: RetractionOperator,
        state: Any,
        retract_params: Any,
        rng_keys: Any,
    ) -> ProbeResult:
        return self._impl.sample_and_retract(
            trajectories, costs, retraction_op, state, retract_params, rng_keys,
        )
