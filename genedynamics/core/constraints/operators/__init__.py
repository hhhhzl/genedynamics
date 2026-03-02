"""
Constraint operators: Execute/enforce constraints.

Operators take convex constraints and apply them to trajectories:
- QP operators: Solve QP to enforce constraints
- Projection operators: Euclidean projection
- Reweight operators: Modify weights (for importance sampling)
"""

from .base import Operator
from .qp import PerStepQPFilter, TrajQPFilter
from .projection import ProjectionOperator
from .reweight import ReweightOperator
from .primal_dual import PrimalDualOperator
from .repair import RepairOperator

__all__ = [
    "Operator",
    "PerStepQPFilter",
    "TrajQPFilter",
    "ProjectionOperator",
    "ReweightOperator",
    "PrimalDualOperator",
    "RepairOperator",
]

