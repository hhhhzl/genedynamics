"""Task contracts for obstacle, guide, insertion, and mobile transfer tasks."""

from .runtime import (
    ExecutionSpec,
    MechanismSpec,
    ObstacleTransferConfig,
    RandomRequest,
    SkillSpec,
)

__all__ = [
    "ExecutionSpec",
    "MechanismSpec",
    "ObstacleTransferConfig",
    "RandomRequest",
    "SkillSpec",
    "task_spec",
]


def __getattr__(name):
    if name == "task_spec":
        from .specs import task_spec

        return task_spec
    raise AttributeError(name)
