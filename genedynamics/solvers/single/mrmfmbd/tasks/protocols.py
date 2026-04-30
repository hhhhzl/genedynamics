"""
Task protocols for MRMFMBD co-design.

Defines abstract interfaces for task domains. Any domain implements
TaskDomain to be pluggable into the experiment platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class TaskSpec:
    """
    Unified task specification (domain-agnostic view).

    Attributes:
        task_id: Unique identifier
        domain: Domain name (e.g. "jax_mpm")
        x_dim: Morphology parameter dimension
        phi_dim: Controller parameter dimension
        num_modes: Number of contact/friction modes
        num_fidelity_levels: Number of fidelity levels
        max_steps: Max episode steps
        config: Domain-specific config (serializable)
    """

    task_id: str
    domain: str
    x_dim: int
    phi_dim: int
    num_modes: int
    num_fidelity_levels: int
    max_steps: int
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "x_dim": self.x_dim,
            "phi_dim": self.phi_dim,
            "num_modes": self.num_modes,
            "num_fidelity_levels": self.num_fidelity_levels,
            "max_steps": self.max_steps,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@runtime_checkable
class TaskDomain(Protocol):
    """
    Protocol for task domains. Implement to add new co-design task types.
    """

    @property
    def domain_name(self) -> str:
        """Domain identifier (e.g. 'jax_mpm')."""
        ...

    def get_task_spec(self, task_id: str) -> TaskSpec:
        """Resolve task_id to TaskSpec."""
        ...

    def list_tasks(self) -> List[str]:
        """List available task ids in this domain."""
        ...

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate config for this domain. Returns list of error messages.
        Empty list = valid.
        """
        ...
