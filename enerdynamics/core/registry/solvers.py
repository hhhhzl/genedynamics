"""
Solver registry for solver implementations.

This registry manages solver class registrations. Solvers can be registered
either programmatically or via entry points.
"""

from typing import Type, Callable, Optional, Any
from enerdynamics.core.registry.base import BaseRegistry

# Type hint for solver classes (using Any for flexibility as solvers
# may have different interfaces)
SolverType = Any

# Global solver registry instance
_solver_registry = BaseRegistry[SolverType]("enerdynamics.solvers")


def register_solver(
    name: str,
    solver_class: Type[SolverType],
    factory: Optional[Callable[..., SolverType]] = None
) -> None:
    """
    Register a solver implementation programmatically.
    
    Args:
        name: Solver name (e.g., "edoc", "mppi", "cem")
        solver_class: Solver class
        factory: Optional factory function
    
    Examples:
        >>> from enerdynamics.core.registry.solvers import register_solver
        >>> register_solver("edoc", EDOCPlanner)
    """
    _solver_registry.register(name, solver_class, factory)


def get_solver_registry() -> BaseRegistry[SolverType]:
    """
    Get the global solver registry.
    
    Returns:
        Solver registry instance
    """
    return _solver_registry

