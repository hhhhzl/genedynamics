"""Task contracts for obstacle, guide, insertion, and mobile transfer tasks."""

from .runtime import (
    ExecutionSpec,
    MechanismSpec,
    ObstacleTransferConfig,
    RandomRequest,
    SkillSpec,
)

__all__ = [
    "CANONICAL_TASKS",
    "ExecutionSpec",
    "MechanismSpec",
    "ObstacleTransferConfig",
    "RandomRequest",
    "SkillSpec",
    "task_spec",
]


def __getattr__(name):
    if name == "CANONICAL_TASKS":
        from .task_identity import CANONICAL_TASKS

        return CANONICAL_TASKS
    if name == "task_spec":
        from .specs import task_spec

        return task_spec
    raise AttributeError(name)
