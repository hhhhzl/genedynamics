"""
Window-based refinement pipeline orchestrator.

Iterates over sliding windows, evaluates constraint violations, and
applies constrained AGP steps + local CFS retraction to improve
feasibility.

Dispatches to a backend implementation via the genemetry registry.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from genedynamics.genemetry.base import RefinementPipeline
from genedynamics.genemetry.registry import get_genemetry_registry
from genedynamics.genemetry.types import RefinementResult


class WindowRefinement(RefinementPipeline):
    """Backend-dispatching refinement pipeline orchestrator.

    Parameters
    ----------
    backend : str
        Backend name (e.g. ``"numpy"``).
    window_policy : WindowPolicy
        Windowing scheme (provides slices).
    constrained_step : ConstrainedStep
        AGP step implementation.
    local_retraction : RetractionOperator
        Local CFS retraction implementation.
    multimodality_evaluator : optional
        Window-level multimodality proxy evaluator.
    cvar_alpha : float
        CVaR confidence level.
    tail_ratio : float
        Fraction of candidates to treat as tail (worst).
    enable_sample_tail : bool
        Whether to restrict refinement to tail candidates.
    enable_local_gating : bool
        Whether to apply window-level gating.
    rollout_fn : callable
        ``(x0, actions) -> states`` for re-rollout after refinement.
    """

    def __init__(
        self,
        backend: str = "numpy",
        window_policy: Any = None,
        constrained_step: Any = None,
        local_retraction: Any = None,
        multimodality_evaluator: Any = None,
        cvar_alpha: float = 0.9,
        tail_ratio: float = 0.3,
        enable_sample_tail: bool = True,
        enable_local_gating: bool = True,
        rollout_fn: Optional[Callable] = None,
    ) -> None:
        registry = get_genemetry_registry()
        impl_class = registry.get("pipeline", "refine", backend)
        if impl_class is None:
            available = registry.list_backends("pipeline", "refine")
            raise ValueError(
                f"No '{backend}' backend for refinement pipeline. "
                f"Available: {available}"
            )
        self._impl = impl_class(
            window_policy=window_policy,
            constrained_step=constrained_step,
            local_retraction=local_retraction,
            multimodality_evaluator=multimodality_evaluator,
            cvar_alpha=cvar_alpha,
            tail_ratio=tail_ratio,
            enable_sample_tail=enable_sample_tail,
            enable_local_gating=enable_local_gating,
            rollout_fn=rollout_fn,
        )

    def refine(
        self,
        candidate_actions: List[Any],
        candidate_states: List[Any],
        violation_fn: Callable[[Any, float], Tuple[Any, Any]],
        cost_fn: Callable[[Any], float],
        schedule_params: Dict[str, Any],
        *,
        clearance: float = 0.05,
    ) -> RefinementResult:
        return self._impl.refine(
            candidate_actions, candidate_states,
            violation_fn, cost_fn, schedule_params,
            clearance=clearance,
        )
