"""
Baseline protocol for co-design experiments.

Any baseline (MRMFMBD, ES, CMA-ES, random search, etc.) implements this
interface to be pluggable into the experiment platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class BaselineConfig:
    """
    Configuration for a baseline run.

    Attributes:
        task_id: Task identifier
        seed: Random seed
        max_wall_time: Optional time budget (seconds)
        max_evaluations: Optional evaluation budget
        extra: Baseline-specific parameters
    """

    task_id: str = "crawling_ground"
    seed: int = 0
    max_wall_time: Optional[float] = None
    max_evaluations: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "max_wall_time": self.max_wall_time,
            "max_evaluations": self.max_evaluations,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineConfig":
        known = {"task_id", "seed", "max_wall_time", "max_evaluations"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)


@dataclass
class BaselineResult:
    """
    Result from a baseline run.

    Attributes:
        theta: Best design+controller (x, phi) concatenated
        x: Morphology parameters
        phi: Controller parameters
        return_: Episode return (or mean over modes/repeats)
        success: Whether task succeeded
        num_evaluations: Total evaluations used
        wall_time: Wall-clock time (seconds)
        metadata: Baseline-specific diagnostics
    """

    theta: np.ndarray
    x: np.ndarray
    phi: np.ndarray
    return_: float
    success: bool
    num_evaluations: int
    wall_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [_serialize(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            return obj

        meta = _serialize(self.metadata)
        return {
            "theta": np.asarray(self.theta).tolist(),
            "x": np.asarray(self.x).tolist(),
            "phi": np.asarray(self.phi).tolist(),
            "return_": float(self.return_),
            "success": bool(self.success),
            "num_evaluations": int(self.num_evaluations),
            "wall_time": float(self.wall_time),
            **meta,
        }


@runtime_checkable
class BaselineProtocol(Protocol):
    """
    Protocol for co-design baselines. Implement to add new baselines.

    run(config, evaluator, task_spec) -> BaselineResult
    """

    @property
    def name(self) -> str:
        """Baseline identifier (e.g. 'mrmfmbd', 'cmaes')."""
        ...

    def run(
        self,
        config: BaselineConfig,
        evaluator: Any,
        task_spec: Any,
        *,
        x_dim: int,
        phi_dim: int,
        **kwargs: Any,
    ) -> BaselineResult:
        """
        Run baseline. Returns best (theta, return, ...).

        Args:
            config: Run configuration
            evaluator: RolloutEvaluator instance
            task_spec: TaskSpec from task extension layer
            x_dim: Morphology dimension
            phi_dim: Controller dimension
            **kwargs: Additional options (checkpoint_dir, etc.)

        Returns:
            BaselineResult with best design
        """
        ...
