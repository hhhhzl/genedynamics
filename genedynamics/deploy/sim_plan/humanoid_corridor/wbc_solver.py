"""
Compatibility exports for the legacy humanoid corridor package.

New code should import from `genedynamics.deploy.followers.humanoid.mujoco.wbc_solver`.
"""

from genedynamics.deploy.followers.humanoid.mujoco.wbc_solver import (
    G1WBCTaskStackConfig,
    G1WholeBodySolverSkeleton,
)

__all__ = [
    "G1WBCTaskStackConfig",
    "G1WholeBodySolverSkeleton",
]
