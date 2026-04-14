"""
SoftZoo task domain provider.

Provides task_spec and evaluator for SoftZoo soft robot co-design tasks.
"""

from __future__ import annotations

from typing import Any, List


class SoftZooTaskDomainProvider:
    """
    TaskDomainProvider for SoftZoo tasks.

    Bridges solvers/mrmfmbd/tasks (TaskSpec) and envs/evaluators (SoftZooRolloutEvaluator).
    """

    @property
    def domain_name(self) -> str:
        return "softzoo"

    def get_task_spec(self, task_id: str) -> Any:
        from genedynamics.solvers.single.mrmfmbd.tasks import get_task_spec
        return get_task_spec(task_id, domain="softzoo")

    def create_evaluator(
        self,
        project_root: str,
        *,
        max_workers: int = 0,
        cache_size: int = 64,
        **kwargs: Any,
    ) -> Any:
        from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig
        runtime_keys = ("ti_arch", "ti_device_memory_fraction", "device", "use_renderer")
        runtime_config = {k: kwargs[k] for k in runtime_keys if k in kwargs and kwargs[k] is not None}
        return SoftZooRolloutEvaluator(
            config=SoftZooEvaluatorConfig(max_workers=max_workers, cache_size=cache_size),
            project_root=project_root,
            runtime_config=runtime_config,
        )

    def list_tasks(self) -> List[str]:
        from genedynamics.solvers.single.mrmfmbd.tasks import list_task_ids
        return list_task_ids(domain="softzoo")
