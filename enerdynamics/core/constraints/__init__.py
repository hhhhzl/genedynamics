"""
Constraint system for EDOC: Soft and hard constraints with flexible scheduling.

This module provides:
- Base classes: SoftConstraint, HardConstraint, FeasibilityOperator, ConstraintManager
- Obstacle constraints: ObstacleSoftConstraint, ObstacleHardConstraint
- Projection operators: CFSProjection (Convex Feasible Set)
- Utility functions: project_box, soft_box_energy
"""

from enerdynamics.core.constraints.base import (
    SoftConstraint,
    HardConstraint,
    FeasibilityOperator,
    ConstraintManager,
    project_box,
    soft_box_energy,
)
from enerdynamics.core.constraints.obstacle_constraints import (
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
)
from enerdynamics.core.constraints.projections.cfs import (
    CFSProjection,
)

__all__ = [
    # Base classes
    "SoftConstraint",
    "HardConstraint",
    "FeasibilityOperator",
    "ConstraintManager",
    # Obstacle constraints
    "ObstacleSoftConstraint",
    "ObstacleHardConstraint",
    # Projection operators
    "CFSProjection",
    # Utilities
    "project_box",
    "soft_box_energy",
]
