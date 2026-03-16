"""
S3 Fidelity Specs: immutable configuration for multi-fidelity levels.

Extensible design for per-level cost, quality, and simulation params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FidelityLevelSpec:
    """
    Specification for a single fidelity level.

    Attributes:
        level: Integer level (0=coarse, 1=medium, 2=fine)
        cost: Relative cost per rollout (e.g. 1.0, 3.0, 10.0)
        quality: Quality factor (0-1 or higher)
        max_substeps: Sim substeps (SoftZoo)
        max_substeps_local: Local substeps
        n_frames: Frames per episode
        extra: Extensible metadata
    """

    level: int
    cost: float = 1.0
    quality: float = 1.0
    max_substeps: int = 3500
    max_substeps_local: int = 20
    n_frames: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "cost": self.cost,
            "quality": self.quality,
            "max_substeps": self.max_substeps,
            "max_substeps_local": self.max_substeps_local,
            "n_frames": self.n_frames,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelityLevelSpec":
        known = {"level", "cost", "quality", "max_substeps", "max_substeps_local", "n_frames"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs["extra"] = extra
        return cls(**kwargs)


@dataclass
class FidelitySystemConfig:
    """
    Configuration for S3 multi-fidelity system.

    Attributes:
        levels: Tuple of FidelityLevelSpec
        ladder_type: "fixed" | "geometric" | "linear" | "cosine"
        step_ratio: For geometric: n_ℓ+1 / n_ℓ
        cost_budget: Optional total cost budget (early termination)
    """

    levels: Tuple[FidelityLevelSpec, ...]
    ladder_type: str = "geometric"
    step_ratio: float = 1.5
    cost_budget: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def get_cost(self, level: int) -> float:
        """Return cost for level."""
        for spec in self.levels:
            if spec.level == level:
                return spec.cost
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": [l.to_dict() for l in self.levels],
            "ladder_type": self.ladder_type,
            "step_ratio": self.step_ratio,
            "cost_budget": self.cost_budget,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelitySystemConfig":
        levels_data = data.get("levels", [])
        levels = tuple(
            FidelityLevelSpec.from_dict(l) if isinstance(l, dict) else l for l in levels_data
        )
        return cls(
            levels=levels,
            ladder_type=data.get("ladder_type", "geometric"),
            step_ratio=data.get("step_ratio", 1.5),
            cost_budget=data.get("cost_budget"),
        )


def default_fidelity_system_config(num_levels: int = 3) -> FidelitySystemConfig:
    """Factory for default fidelity levels (coarse/medium/fine)."""
    levels = (
        FidelityLevelSpec(level=0, cost=1.0, max_substeps=500, max_substeps_local=10, n_frames=100),
        FidelityLevelSpec(level=1, cost=3.0, max_substeps=1500, max_substeps_local=15, n_frames=150),
        FidelityLevelSpec(level=2, cost=7.0, max_substeps=3500, max_substeps_local=20, n_frames=200),
    )[:num_levels]
    return FidelitySystemConfig(levels=levels, ladder_type="geometric", step_ratio=1.5)
