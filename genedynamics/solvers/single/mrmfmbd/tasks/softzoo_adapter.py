"""
SoftZoo Task Adapter: bridges SoftZoo task registry to TaskDomain protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .protocols import TaskDomain, TaskSpec


class SoftZooTaskAdapter(TaskDomain):
    """
    TaskDomain implementation for SoftZoo tasks.

    Maps SoftZooTaskSpec to unified TaskSpec.
    """

    @property
    def domain_name(self) -> str:
        return "softzoo"

    def get_task_spec(self, task_id: str) -> TaskSpec:
        from genedynamics.envs.external.softzoo.task_registry import get_task_spec as _get

        sz = _get(task_id)
        x_dim = 3  # default: geometry, softness, actuator
        phi_dim = 4  # default: omega params for sin wave
        if sz.robot and sz.robot.controller:
            phi_dim = max(phi_dim, sz.robot.controller.n_actuators * 2)
        num_modes = len(sz.modes) if sz.modes else 4
        num_fidelity = len(sz.fidelity_levels) if sz.fidelity_levels else 3
        return TaskSpec(
            task_id=task_id,
            domain=self.domain_name,
            x_dim=x_dim,
            phi_dim=phi_dim,
            num_modes=num_modes,
            num_fidelity_levels=num_fidelity,
            max_steps=sz.max_steps,
            config={
                "env_config_file": sz.env_config_file,
                "env_type": str(sz.env_type.value) if hasattr(sz.env_type, "value") else str(sz.env_type),
                "objective": str(sz.objective.value) if hasattr(sz.objective, "value") else str(sz.objective),
            },
        )

    def list_tasks(self) -> List[str]:
        from genedynamics.envs.external.softzoo.task_registry import list_tasks
        return list_tasks()

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        task_id = config.get("task_id")
        if not task_id:
            errors.append("task_id required for softzoo domain")
        elif task_id not in self.list_tasks():
            errors.append(f"Unknown task_id: {task_id}")
        return errors
