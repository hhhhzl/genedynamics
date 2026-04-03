"""
Structured follower stack for plan-to-controller adapters.

The new package is organized by:
- common high-level plan semantics
- embodiment-specific adapters
- backend-specific execution paths
"""

from genedynamics.deploy.followers.common.plan_adapter import (
    CorridorTrajectoryAdapter,
    load_corridor_plan_from_seed_dir,
)
from genedynamics.deploy.followers.common.plan_schema import (
    CorridorPlanFrame,
    CorridorPlanSchema,
    CorridorPlanTrajectory,
)
from genedynamics.deploy.followers.common.traversal_intent import (
    TraversalIntent,
)

__all__ = [
    "CorridorPlanFrame",
    "CorridorPlanSchema",
    "CorridorPlanTrajectory",
    "CorridorTrajectoryAdapter",
    "TraversalIntent",
    "load_corridor_plan_from_seed_dir",
]
