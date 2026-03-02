"""
High-Performance Multi-Backend Constraint System.

This module provides:
- New Architecture (recommended): HighPerformanceConstraintPipeline, Convexifiers, Operators, Schedulers
- Legacy Architecture (deprecated): ConstraintManager, SoftConstraint, HardConstraint, etc.

⚠️ For new code, use the new architecture:
  from genedynamics.core.constraints.core import HighPerformanceConstraintPipeline

⚠️ Legacy components are kept for backward compatibility only.
  See legacy/MIGRATION_GUIDE.md for migration instructions.
"""

# Legacy imports (deprecated - use new architecture instead)
# See legacy/MIGRATION_GUIDE.md for migration instructions
# These are kept for backward compatibility with existing experiments and solvers
from genedynamics.core.constraints.legacy.base import (
    SoftConstraint,
    HardConstraint,
    FeasibilityOperator,
    ActionFilterOperator,
    ConstraintManager,
    project_box,
    soft_box_energy,
)
from genedynamics.core.constraints.legacy.obstacle_constraints import (
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
)
from genedynamics.core.constraints.legacy.projections import (
    CFSProjection,
)
from genedynamics.core.constraints.legacy.action_filters import (
    CBFDoubleIntegrator2DActionFilter,
)
from genedynamics.core.constraints.legacy.schedule import (
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
