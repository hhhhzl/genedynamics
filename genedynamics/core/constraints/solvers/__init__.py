"""
QP solvers: Numerical backends for solving quadratic programs.

This module provides solver implementations that only solve QP problems,
without defining or interpreting constraints. Solvers are backend-agnostic
and can be used by any operator that needs to solve QP.

Solvers:
- QPSolver: Base interface
- JAXOPTOsqpSolver: jaxopt.OSQP backend (JAX)
- OSQPSolver: OSQP backend (CPU)
- CVXOPTSolver: CVXOPT backend (CPU)
- ClosedFormSolver: Special-case fast solvers
"""

from .base import QPSolver
from .jaxopt_osqp_solver import JAXOPTOsqpSolver
from .closed_form import ClosedFormSolver

__all__ = [
    "QPSolver",
    "JAXOPTOsqpSolver",
    "ClosedFormSolver",
]

# OSQP solver is optional
try:
    from .osqp_solver import OSQPSolver
    __all__.append("OSQPSolver")
except ImportError:
    pass

# CVXOPT solver is optional
try:
    from .cvxopt_solver import CVXOPTSolver
    __all__.append("CVXOPTSolver")
except ImportError:
    pass


