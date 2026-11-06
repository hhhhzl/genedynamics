"""Eager Torch wrapper for evaluator-driven trajectory retraction."""

from __future__ import annotations

from typing import Any

import torch

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import RetractionResult


@register_genemetry("retraction", "cfs", "torch")
class CfsRetractionTorch(RetractionOperator):
    """Apply a source-provided Torch filter while preserving tensor ownership."""

    def __init__(self, *, filter_fn: Any, **kwargs):
        del kwargs
        if filter_fn is None or not callable(filter_fn):
            raise ValueError("CfsRetractionTorch requires a callable filter_fn")
        self._filter_fn = filter_fn

    def retract(self, state, trajectory, params: Any | None = None):
        if not isinstance(trajectory, torch.Tensor) or trajectory.ndim != 2:
            raise TypeError("Torch CFS retraction requires [horizon, action] tensor")
        params = params or {}
        filtered = self._filter_fn(
            state,
            trajectory,
            params.get("sched_state"),
            params.get("sched_params"),
        )
        if (
            not isinstance(filtered, torch.Tensor)
            or filtered.shape != trajectory.shape
            or not torch.isfinite(filtered).all()
        ):
            raise ValueError("Torch CFS retraction returned an invalid trajectory")
        return RetractionResult(trajectory=filtered)


__all__ = ["CfsRetractionTorch"]
