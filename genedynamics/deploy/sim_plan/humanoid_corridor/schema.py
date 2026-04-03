"""
Compatibility exports for the legacy humanoid corridor package.

New code should import from `genedynamics.deploy.followers.*`.
"""

from genedynamics.deploy.followers.common.controller_output import JointTargets
from genedynamics.deploy.followers.common.plan_schema import (
    DEFAULT_CORRIDOR_FIELDS,
    CorridorPlanFrame,
    CorridorPlanSchema,
    CorridorPlanTrajectory,
)
from genedynamics.deploy.followers.humanoid.task_spec import (
    ArmJointTask,
    ContactPhase,
    FollowerTasks,
    FootTask,
    HumanoidTaskSpec,
    PelvisTask,
    PhaseState,
)

__all__ = [
    "ArmJointTask",
    "ContactPhase",
    "CorridorPlanFrame",
    "CorridorPlanSchema",
    "CorridorPlanTrajectory",
    "DEFAULT_CORRIDOR_FIELDS",
    "FollowerTasks",
    "FootTask",
    "HumanoidTaskSpec",
    "JointTargets",
    "PelvisTask",
    "PhaseState",
]
