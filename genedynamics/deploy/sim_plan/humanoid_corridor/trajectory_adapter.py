"""
Compatibility exports for the legacy humanoid corridor package.

New code should import from `genedynamics.deploy.followers.common.plan_adapter`.
"""

from genedynamics.deploy.followers.common.plan_adapter import (
    CorridorTrajectoryAdapter,
    load_corridor_plan_from_seed_dir,
)

__all__ = [
    "CorridorTrajectoryAdapter",
    "load_corridor_plan_from_seed_dir",
]
