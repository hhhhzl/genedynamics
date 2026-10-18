"""Common outcome metrics for the four transfer tasks."""

from typing import Any, Dict

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.base import MetricsPlugin


class TransferTaskMetricsPlugin(MetricsPlugin):
    @property
    def name(self) -> str:
        return "transfer_task"

    def compute(
        self,
        trajectory: Trajectory,
        env: Any,
        obstacles: Any,
        constraints: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del env, obstacles, constraints, kwargs
        report = dict((trajectory.info or {}).get("task_evaluation", {}))
        return {
            "success": bool(report.get("successes", 0)),
            "attempts": int(report.get("attempts", 1)),
            "control_steps": int(report.get("control_steps", len(trajectory.actions))),
            "forbidden_collision": bool(report.get("forbidden_collisions", 0)),
            "force_limit_violation": bool(report.get("force_limit_violations", 0)),
            "task_cost": float(report.get("mean_task_cost", 0.0)),
        }


__all__ = ["TransferTaskMetricsPlugin"]
