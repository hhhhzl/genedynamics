"""
Legacy constraint system implementations.

This module contains the original constraint system implementations,
kept for backward compatibility during migration.

Components:
- action_filters/: Original CBF action filter implementations
- projections/: Original CFS projection implementations
- obstacle_constraints.py: Original obstacle constraint implementations
- schedule.py: Original schedule implementations

Migration status:
- These components are still functional
- New code should use the new architecture (convexify/operators/schedulers)
- Legacy components will be gradually deprecated

⚠️ DEPRECATED: These components are deprecated. Use the new architecture:
- Replace CFSProjection with CFSConvexifier + PerStepQPFilter
- Replace CBFDoubleIntegrator2DActionFilter with CBFConvexifier + PerStepQPFilter
See MIGRATION_GUIDE.md for details.
"""

# Import legacy components for backward compatibility
from genedynamics.core.constraints.legacy.action_filters import (
    CBFDoubleIntegrator2DActionFilter,
)
from genedynamics.core.constraints.legacy.projections import (
    CFSProjection,
)
from genedynamics.core.constraints.legacy.obstacle_constraints import (
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
)
from genedynamics.core.constraints.legacy.schedule import (
    ConstraintSchedule,
    ConstraintScheduleManager,
    ConstantSchedule,
    LinearSchedule,
    ExponentialSchedule,
    CosineSchedule,
    PiecewiseSchedule,
    CustomSchedule,
)
from genedynamics.core.constraints.legacy.base import (
    SoftConstraint,
    HardConstraint,
    FeasibilityOperator,
    ActionFilterOperator,
    ConstraintManager,
    project_box,
    soft_box_energy,
)

__all__ = [
    # Base classes (legacy)
    "SoftConstraint",
    "HardConstraint",
    "FeasibilityOperator",
    "ActionFilterOperator",
    "ConstraintManager",
    "project_box",
    "soft_box_energy",
    # Action filters
    "CBFDoubleIntegrator2DActionFilter",
    # Projections
    "CFSProjection",
    # Obstacle constraints
    "ObstacleSoftConstraint",
    "ObstacleHardConstraint",
    # Scheduling
    "ConstraintSchedule",
    "ConstraintScheduleManager",
    "ConstantSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "CosineSchedule",
    "PiecewiseSchedule",
    "CustomSchedule",
]

