"""
JAX implementation of CFS retraction.

Wraps the solver's CFS QP filter function as a
:class:`RetractionOperator`.
"""

from __future__ import annotations

from typing import Any, Optional

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.types import RetractionResult
from genedynamics.genemetry.registry import register_genemetry


@register_genemetry("retraction", "cfs", "jax")
class CfsRetractionJax(RetractionOperator):
    """CFS QP-filter retraction (JAX).

    Parameters
    ----------
    filter_fn : callable
        ``(state, trajectory, sched_state, sched_params) -> filtered_trajectory``
        Typically ``inner._filter_actions_single_jit``.
    """

    def __init__(self, *, filter_fn: Any, **kwargs: Any) -> None:
        if filter_fn is None:
            raise ValueError("CfsRetractionJax requires a filter_fn")
        self._filter_fn = filter_fn

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        params = params or {}
        filtered = self._filter_fn(
            state,
            trajectory,
            params.get("sched_state"),
            params.get("sched_params"),
        )
        return RetractionResult(trajectory=filtered)
