"""
Shared solver interface for IK today and WBC later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from genedynamics.deploy.sim_plan.humanoid_corridor.schema import (
    FollowerTasks,
    JointTargets,
)


class CorridorTaskSolverBase(ABC):
    def reset(self, qpos: Optional[Sequence[float]] = None, qvel: Optional[Sequence[float]] = None) -> None:
        _ = qpos, qvel

    @abstractmethod
    def solve(
        self,
        tasks: FollowerTasks,
        *,
        qpos: Sequence[float],
        qvel: Optional[Sequence[float]] = None,
        dt: float = 0.0,
    ) -> JointTargets:
        raise NotImplementedError

