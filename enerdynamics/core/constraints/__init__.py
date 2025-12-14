"""
Constraint system for EDOC: Soft and hard constraints with flexible scheduling.

This module provides:
- Base classes: SoftConstraint, HardConstraint, FeasibilityOperator, ConstraintManager
- Obstacle constraints: ObstacleSoftConstraint, ObstacleHardConstraint
- Projection operators: CFSProjection (Convex Feasible Set)
- Scheduling system: ConstraintSchedule, ConstraintScheduleManager
- Utility functions: project_box, soft_box_energy
"""

from enerdynamics.core.constraints.base import (
    SoftConstraint,
    HardConstraint,
    FeasibilityOperator,
    ActionFilterOperator,
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
from enerdynamics.core.constraints.action_filters import (
    CBFDoubleIntegrator2DActionFilter,
)
from enerdynamics.core.constraints.schedule import (
    ConstraintSchedule,
    ConstantSchedule,
    LinearSchedule,
    ExponentialSchedule,
    CosineSchedule,
    PiecewiseSchedule,
    CustomSchedule,
    ConstraintScheduleManager,
)

__all__ = [
    # Base classes
    "SoftConstraint",
    "HardConstraint",
    "FeasibilityOperator",
    "ActionFilterOperator",
    "ConstraintManager",
    # Obstacle constraints
    "ObstacleSoftConstraint",
    "ObstacleHardConstraint",
    # Projection operators
    "CFSProjection",
    # Action-space hard filters
    "CBFDoubleIntegrator2DActionFilter",
    # Scheduling
    "ConstraintSchedule",
    "ConstantSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "CosineSchedule",
    "PiecewiseSchedule",
    "CustomSchedule",
    "ConstraintScheduleManager",
    # Utilities
    "project_box",
    "soft_box_energy",
]
