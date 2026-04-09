"""Corridor following task — measures progress against a 14D plan endpoint.

The task watches the pelvis world-XY position and signals ``done`` when
either:

* the pelvis enters a goal sphere of radius ``goal_tolerance_m`` around
  the plan's final waypoint (``info.success = True``), or
* ``max_steps`` have elapsed (``info.success = False``).

The task does **not** load the plan itself — that responsibility belongs
to the follower. The runner passes the goal in via :meth:`reset` (or via
the ``goal_xy`` constructor arg for runners that don't have a follower).

The per-step :class:`StepInfo` records:

* ``metrics["progress"]`` — fraction of the start→goal distance covered
* ``metrics["distance_to_goal"]`` — meters
* ``metrics["pelvis_xy_x"]`` / ``["pelvis_xy_y"]`` — convenience columns
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.tasks.base import BaseExecutionTask

__all__ = ["CorridorFollowTask"]


class CorridorFollowTask(BaseExecutionTask):
    """Track corridor traversal progress and signal episode termination.

    Args:
        goal_xy: Optional 2-vector world-XY goal. When ``None``, the task
            waits for the runner / follower to call :meth:`set_goal`
            before the first step.
        goal_tolerance_m: Distance threshold (meters) below which the task
            reports success.
        max_steps: Hard cap on episode length. Reaching this without
            reaching the goal flags ``success=False``.
        name: Display name forwarded to observers.
    """

    def __init__(
        self,
        *,
        goal_xy: Optional[Sequence[float]] = None,
        goal_tolerance_m: float = 0.20,
        max_steps: int = 5_000,
        name: str = "corridor_follow",
    ) -> None:
        super().__init__(name=name)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.max_steps = int(max_steps)
        self._goal_xy: Optional[np.ndarray] = (
            np.asarray(goal_xy, dtype=np.float64).reshape(2) if goal_xy is not None else None
        )
        self._start_xy: Optional[np.ndarray] = None
        self._best_distance: float = float("inf")
        self._success_step: Optional[int] = None

    # ------------------------------------------------------------------
    # ExecutionTask
    # ------------------------------------------------------------------

    def reset(self, io: Any) -> None:
        super().reset(io)
        self._start_xy = None
        self._best_distance = float("inf")
        self._success_step = None
        # Pull the initial pelvis position so progress() has a baseline.
        try:
            state = io.get_state()
            self._start_xy = self._pelvis_xy(state)
        except Exception:
            self._start_xy = None

    def set_goal(self, goal_xy: Sequence[float]) -> None:
        """Late-bind the goal — used when the follower owns the plan."""
        self._goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    def step(
        self,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
    ) -> StepInfo:
        self._step_count += 1
        pelvis_xy = self._pelvis_xy(state)
        if pelvis_xy is None:
            return StepInfo(done=False, metrics={"progress": 0.0})
        if self._start_xy is None:
            self._start_xy = pelvis_xy.copy()

        metrics: dict[str, float] = {
            "pelvis_xy_x": float(pelvis_xy[0]),
            "pelvis_xy_y": float(pelvis_xy[1]),
        }

        if self._goal_xy is not None:
            distance = float(np.linalg.norm(pelvis_xy - self._goal_xy))
            metrics["distance_to_goal"] = distance
            self._best_distance = min(self._best_distance, distance)
            if self._start_xy is not None:
                start_distance = float(np.linalg.norm(self._goal_xy - self._start_xy))
                if start_distance > 1e-6:
                    progress = max(0.0, 1.0 - distance / start_distance)
                else:
                    progress = 1.0
                metrics["progress"] = progress
            if distance <= self.goal_tolerance_m and self._success_step is None:
                self._success_step = self._step_count
                return StepInfo(done=True, success=True, metrics=metrics)
        else:
            metrics["progress"] = 0.0

        if self._step_count >= self.max_steps:
            return StepInfo(done=True, success=False, metrics=metrics)
        return StepInfo(done=False, metrics=metrics)

    def summary(self) -> Mapping[str, Any]:
        return {
            "steps": self._step_count,
            "best_distance_m": self._best_distance if self._best_distance != float("inf") else None,
            "goal_xy": None if self._goal_xy is None else self._goal_xy.tolist(),
            "success_step": self._success_step,
            "succeeded": self._success_step is not None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _pelvis_xy(state: RobotState) -> Optional[np.ndarray]:
        if state.base_pose is not None and len(state.base_pose) >= 2:
            return np.asarray(state.base_pose[:2], dtype=np.float64).copy()
        if state.qpos is not None and state.qpos.shape[0] >= 2:
            return np.asarray(state.qpos[:2], dtype=np.float64).copy()
        return None
