"""
Placeholder quadruped planner for deploy framework.

Returns zero/stand actions. Replace with real planner (2GO, learned policy, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class QuadrupedStandPlanner:
    """
    Placeholder planner: returns zero torque actions.

    Use for framework testing. Replace with real planner for actual control.
    """

    act_dim: int = 8
    horizon: int = 64

    def solve(
        self,
        x0: np.ndarray,
        horizon: Optional[int] = None,
        rng_key: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Return zero actions for full horizon."""
        H = horizon or self.horizon
        actions = np.zeros((H, self.act_dim), dtype=np.float32)
        # Rollout states (placeholder: just repeat x0)
        states = np.broadcast_to(x0, (H + 1, x0.size)).astype(np.float32).copy()
        return _PlanResult(actions=actions, states=states, info={"planner": "stand"})


class _PlanResult:
    def __init__(self, actions, states, info):
        self.actions = actions
        self.states = states
        self.info = info


@dataclass
class HumanoidStandPlanner:
    """Placeholder planner for humanoid: returns zero torque."""

    act_dim: int = 17
    horizon: int = 64

    def solve(
        self,
        x0: np.ndarray,
        horizon: Optional[int] = None,
        rng_key: Any = None,
        **kwargs: Any,
    ) -> Any:
        H = horizon or self.horizon
        actions = np.zeros((H, self.act_dim), dtype=np.float32)
        states = np.broadcast_to(x0, (H + 1, x0.size)).astype(np.float32).copy()
        return _PlanResult(actions=actions, states=states, info={"planner": "stand"})
