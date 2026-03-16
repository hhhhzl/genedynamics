"""
Protocols for rollout evaluation.

Defines abstract interfaces for batch evaluation of design+controller
pairs with mode and fidelity support. Extensible for future backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


# -----------------------------------------------------------------------------
# Request / Result types
# -----------------------------------------------------------------------------


@dataclass
class RolloutRequest:
    """
    Single rollout request.

    Attributes:
        morphology_params: Design parameters x (flattened)
        controller_params: Controller parameters phi
        mode_id: Contact/friction mode index
        fidelity_level: 0=coarse, 1=medium, 2=fine
        seed: Random seed
        num_repeats: Repeats per sample for variance reduction
        record: Whether to record trajectory
    """

    morphology_params: np.ndarray
    controller_params: np.ndarray
    mode_id: int = 0
    fidelity_level: int = 2
    seed: int = 0
    num_repeats: int = 1
    record: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """
    Result from a single rollout.

    Attributes:
        return_: Episode return (sum of rewards)
        success: Whether task succeeded
        num_steps: Steps taken
        mean_return: Mean over num_repeats
        std_return: Std over num_repeats
        failure_code: Optional failure code
        wall_time: Wall-clock time in seconds
        trajectory: Optional state/action trajectory (if record=True)
    """

    return_: float
    success: bool
    num_steps: int
    mean_return: float = 0.0
    std_return: float = 0.0
    failure_code: Optional[str] = None
    wall_time: float = 0.0
    trajectory: Optional[Dict[str, np.ndarray]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBatchRequest:
    """
    Batch of rollout requests.

    Attributes:
        task_id: Task identifier
        requests: List of rollout requests
        extra: Extensible metadata
    """

    task_id: str
    requests: List[RolloutRequest]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBatchResult:
    """
    Aggregated batch result.

    Attributes:
        results: Per-request results
        returns: (N,) episode returns (mean over repeats)
        successes: (N,) success flags
        mean_return: Scalar mean over batch
        std_return: Scalar std over batch
        wall_time_total: Total wall-clock time
        failure_codes: Optional failure codes per request
    """

    results: List[RolloutResult]
    returns: np.ndarray
    successes: np.ndarray
    mean_return: float = 0.0
    std_return: float = 0.0
    wall_time_total: float = 0.0
    failure_codes: Optional[List[Optional[str]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Evaluator protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class RolloutEvaluator(Protocol):
    """
    Protocol for batch rollout evaluation.

    Evaluates design+controller pairs at specified mode and fidelity.
    Supports caching, parallelism, and failure handling.
    """

    def evaluate_batch(
        self,
        request: RolloutBatchRequest,
        *,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> RolloutBatchResult:
        """
        Evaluate a batch of rollout requests.

        Args:
            request: Batch request with task_id and requests
            parallel: Whether to parallelize across requests
            max_workers: Max parallel workers (None = auto)
            use_cache: Whether to use result cache
            **kwargs: Backend-specific options

        Returns:
            RolloutBatchResult with aggregated stats
        """
        ...

    def evaluate_single(
        self,
        morphology_params: np.ndarray,
        controller_params: np.ndarray,
        *,
        task_id: str = "crawling_ground",
        mode_id: int = 0,
        fidelity_level: int = 2,
        seed: int = 0,
        num_repeats: int = 1,
        record: bool = False,
        **kwargs: Any,
    ) -> RolloutResult:
        """
        Evaluate a single design+controller pair.

        Convenience wrapper around evaluate_batch.
        """
        ...
