"""
Action-space hard constraint filters.

These operators integrate into ConstraintManager via ActionFilterOperator and are
intended to be applied during rollouts (CBF-style safety filters).
"""

from genedynamics.core.constraints.legacy.action_filters.cbf_double_integrator_2d import (
    CBFDoubleIntegrator2DActionFilter,
)

__all__ = [
    "CBFDoubleIntegrator2DActionFilter",
]

