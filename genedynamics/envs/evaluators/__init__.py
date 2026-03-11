"""
Rollout evaluators for external simulators (SoftZoo, etc.).

Provides protocol and implementations for batch evaluation of
design+controller pairs with mode and fidelity support.
"""

from __future__ import annotations

from .protocols import (
    RolloutEvaluator,
    RolloutRequest,
    RolloutResult,
    RolloutBatchRequest,
    RolloutBatchResult,
)
from .softzoo_evaluator import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig

__all__ = [
    "RolloutEvaluator",
    "RolloutRequest",
    "RolloutResult",
    "RolloutBatchRequest",
    "RolloutBatchResult",
    "SoftZooRolloutEvaluator",
    "SoftZooEvaluatorConfig",
]
