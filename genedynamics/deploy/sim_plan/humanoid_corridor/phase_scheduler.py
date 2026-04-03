"""
Compatibility exports for the legacy humanoid corridor package.

New code should import from `genedynamics.deploy.followers.humanoid.contact_scheduler`.
"""

from genedynamics.deploy.followers.humanoid.contact_scheduler import (
    HumanoidContactScheduler as QuasiStaticHumanoidPhaseScheduler,
)
from genedynamics.deploy.followers.humanoid.contact_scheduler import (
    HumanoidContactSchedulerConfig as HumanoidPhaseSchedulerConfig,
)

__all__ = [
    "HumanoidPhaseSchedulerConfig",
    "QuasiStaticHumanoidPhaseScheduler",
]
