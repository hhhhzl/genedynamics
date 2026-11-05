"""Common outcome metrics for the four transfer tasks."""

from typing import Any, Dict

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.base import MetricsPlugin


class TransferTaskMetricsPlugin(MetricsPlugin):
    execution_failure_metrics = {
        "constrained_success_rate": 0.0,
        "constraint_violation_rate": 1.0,
        "capped_normalized_completion_time": 1.0,
        "worst_group_constrained_success": 0.0,
        "predeployment_rejection_rate": 0.0,
    }

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
        del env, obstacles, constraints
        report = dict((trajectory.info or {}).get("task_evaluation", {}))
        accounting = dict(report.get("accounting", {}))
        runtime = dict((trajectory.info or {}).get("runtime", {}))
        return {
            "evaluation_version": str(report.get("evaluation_version", "missing")),
            "constrained_success_rate": float(report.get("constrained_success_rate", 0.0)),
            "constraint_violation_rate": float(report.get("constraint_violation_rate", 0.0)),
            "capped_normalized_completion_time": float(
                report.get("capped_normalized_completion_time", 1.0)
            ),
            "worst_group_constrained_success": float(
                report.get("worst_group_constrained_success", 0.0)
            ),
            "predeployment_rejection_rate": float(
                report.get("predeployment_rejection_rate", 0.0)
            ),
            "requested_episodes": int(report.get("requested_episodes", 0)),
            "task_cost": float(report.get("mean_task_cost", 0.0)),
            "planning_time_seconds": float(kwargs.get("planning_time", 0.0)),
            "physical_control_steps": int(accounting.get("control_steps", len(trajectory.actions))),
            "preparation_control_steps": int(accounting.get("preparation_control_steps", 0)),
            "model_calls": int(runtime.get("model_calls", 0)),
            "policy_network_calls": int(runtime.get("policy_network_calls", 0)),
            "candidate_rollouts": int(runtime.get("candidate_rollouts", 0)),
        }


__all__ = ["TransferTaskMetricsPlugin"]
