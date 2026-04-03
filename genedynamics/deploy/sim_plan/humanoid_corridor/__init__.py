"""
Legacy compatibility package for the original humanoid corridor follower.

New code should migrate toward `genedynamics.deploy.followers.*`.
"""

from genedynamics.deploy.sim_plan.humanoid_corridor.follower import (
    G1CorridorFollower,
    G1CorridorFollowerConfig,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.g1_model import (
    G1ModelSpec,
    resolve_g1_model_path,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.ik_solver import (
    G1CorridorIKSolver,
    G1IKSolverConfig,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.joint_tracker import (
    JointReferenceTracker,
    JointTrackerConfig,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.phase_scheduler import (
    HumanoidPhaseSchedulerConfig,
    QuasiStaticHumanoidPhaseScheduler,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.schema import (
    ArmJointTask,
    ContactPhase,
    CorridorPlanFrame,
    CorridorPlanSchema,
    CorridorPlanTrajectory,
    FollowerTasks,
    FootTask,
    JointTargets,
    PelvisTask,
    PhaseState,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.task_generator import (
    CorridorTaskGenerator,
    CorridorTaskGeneratorConfig,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.trajectory_adapter import (
    CorridorTrajectoryAdapter,
    load_corridor_plan_from_seed_dir,
)
from genedynamics.deploy.sim_plan.humanoid_corridor.wbc_solver import (
    G1WBCTaskStackConfig,
    G1WholeBodySolverSkeleton,
)

__all__ = [
    "ArmJointTask",
    "ContactPhase",
    "CorridorPlanFrame",
    "CorridorPlanSchema",
    "CorridorPlanTrajectory",
    "CorridorTaskGenerator",
    "CorridorTaskGeneratorConfig",
    "CorridorTrajectoryAdapter",
    "FollowerTasks",
    "FootTask",
    "G1CorridorFollower",
    "G1CorridorFollowerConfig",
    "G1CorridorIKSolver",
    "G1IKSolverConfig",
    "G1ModelSpec",
    "G1WBCTaskStackConfig",
    "G1WholeBodySolverSkeleton",
    "HumanoidPhaseSchedulerConfig",
    "JointReferenceTracker",
    "JointTargets",
    "JointTrackerConfig",
    "PelvisTask",
    "PhaseState",
    "QuasiStaticHumanoidPhaseScheduler",
    "load_corridor_plan_from_seed_dir",
    "resolve_g1_model_path",
]
