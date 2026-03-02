"""
Constraint terms: What constraints are (energy, feasibility, violation).

This module provides the constraint definition layer:
- ConstraintTerm: Base class for all constraint terms
- ObstacleSDFTerm: SDF-based obstacle constraints
- BoundsTerm: State/control bounds
- BarrierTerm: Log-barrier / tightening barrier
- CBFTerm: Control Barrier Function as a term
- CompositeTerm: Combine multiple terms

Terms define what constraints are, not how they are enforced.
"""

from .base import ConstraintTerm
from .obstacle_sdf import ObstacleSDFTerm
from .bounds import BoundsTerm
from .barrier import BarrierTerm
from .composite import CompositeTerm

__all__ = [
    "ConstraintTerm",
    "ObstacleSDFTerm",
    "BoundsTerm",
    "BarrierTerm",
    "CompositeTerm",
]


