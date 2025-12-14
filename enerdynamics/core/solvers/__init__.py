"""
Solver base classes for trajectory optimization.

This module defines the abstract Solver interface and base classes for different
solver types:
- Solver: Abstract base class for all solvers
- SamplingSolver: Base for sampling-based methods (MPPI, CEM, EDOC)
- OptimizationSolver: Base for gradient-based methods (iLQR, DDP)
"""

from enerdynamics.core.solvers.base import (
    Solver,
    SamplingSolver,
    OptimizationSolver,
)

__all__ = [
    "Solver",
    "SamplingSolver",
    "OptimizationSolver",
]
