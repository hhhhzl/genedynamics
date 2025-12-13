"""
Energy functional abstraction for the energy-driven control framework.

This module defines the EnergyFunctional interface, which represents the total
energy/cost of a trajectory. The energy is decomposed into:
- Task cost: primary objective (e.g., reach goal, minimize control effort)
- Constraint energy: barrier/penalty terms for constraints
- Entropy term: regularization for exploration/stochasticity

All solvers depend only on this interface, allowing easy composition of
different task objectives and constraints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple, Optional

from enerdynamics.core.types import Trajectory


class EnergyFunctional(ABC):
    """
    Abstract base class for energy functionals.
    
    An energy functional evaluates the total "energy" (cost) of a trajectory.
    This energy is typically decomposed into:
    - Task cost: primary objective
    - Constraint energy: barrier/penalty terms
    - Entropy term: regularization (often negative for exploration)
    
    The total energy is: E = task_cost + constraint_energy - entropy_term
    
    All solvers minimize this energy to find optimal trajectories.
    """
    
    @abstractmethod
    def task_cost(self, traj: Trajectory) -> float:
        """
        Compute task cost (primary objective).
        
        Examples:
        - Distance to goal
        - Control effort
        - Time to completion
        
        Args:
            traj: Trajectory to evaluate
            
        Returns:
            Task cost (non-negative, lower is better)
        """
        pass
    
    @abstractmethod
    def constraint_energy(self, traj: Trajectory) -> float:
        """
        Compute constraint energy (barrier/penalty terms).
        
        Examples:
        - Collision avoidance barriers
        - State/action bounds
        - Safety constraints
        
        Args:
            traj: Trajectory to evaluate
            
        Returns:
            Constraint energy (non-negative, 0 if all constraints satisfied)
        """
        pass
    
    def entropy_term(self, traj: Trajectory) -> float:
        """
        Compute entropy/regularization term (optional).
        
        This term is often subtracted from the total energy to encourage
        exploration or smoothness. Default implementation returns 0.
        
        Examples:
        - Entropy regularization for stochastic policies
        - Smoothness penalties
        
        Args:
            traj: Trajectory to evaluate
            
        Returns:
            Entropy term (non-negative, higher encourages exploration)
        """
        return 0.0
    
    def total_energy(self, traj: Trajectory) -> float:
        """
        Compute total energy: E = task_cost + constraint_energy - entropy_term.
        
        This is the quantity that solvers minimize.
        
        Args:
            traj: Trajectory to evaluate
            
        Returns:
            Total energy (lower is better)
        """
        return (
            self.task_cost(traj)
            + self.constraint_energy(traj)
            - self.entropy_term(traj)
        )
    
    def __call__(self, traj: Trajectory) -> float:
        """Convenience method: call total_energy()."""
        return self.total_energy(traj)


# Legacy compatibility: keep the old EnergyFunctional for backward compatibility
# This can be gradually migrated to the new interface
from typing import Any as _Any, Callable as _Callable, Dict as _Dict, Iterable as _Iterable, Tuple as _Tuple
import numpy as np
Array = np.ndarray


@dataclass
class EnergyTerm:
    """
    Legacy energy term for point-wise energy evaluation.
    
    This is kept for backward compatibility with existing code that evaluates
    energy at individual (state, action) pairs rather than full trajectories.
    """
    fn: _Callable[[Array, Array, _Dict[str, _Any]], Array]
    weight: float = 1.0

    def __call__(self, x: Array, u: Array, info: _Dict[str, _Any] | None):
        return self.weight * self.fn(x, u, info)


class LegacyEnergyFunctional:
    """
    Legacy energy functional for point-wise evaluation.
    
    This class evaluates energy at individual (state, action) pairs.
    It can be wrapped to create a trajectory-based EnergyFunctional.
    """
    def __init__(self, terms: _Dict[str, EnergyTerm]):
        self.terms = dict(terms)
        self._term_items: _Tuple[_Tuple[str, EnergyTerm], ...] = tuple(self.terms.items())

    def __call__(self, x: Array, u: Array, info: _Dict[str, _Any] = None) -> Array:
        return self.compute(x, u, info)

    def compute(self, x: Array, u: Array, info: _Dict[str, _Any] | None = None) -> Array:
        total = 0.0
        for _, term in self._term_items:
            total = total + term(x, u, info)
        return total

    def compute_terms(self, x: Array, u: Array, info: _Dict[str, _Any] | None = None) -> _Iterable[_Tuple[str, Array]]:
        for name, term in self._term_items:
            yield name, term(x, u, info)

    def breakdown(self, x: Array, u: Array, info: _Dict[str, _Any] = None):
        out = {}
        info_local = info or {}
        for name, term in self.terms.items():
            out[name] = float(term(x, u, info_local))
        return out


def trajectory_energy_from_legacy(legacy_energy: LegacyEnergyFunctional) -> EnergyFunctional:
    """
    Convert a legacy point-wise energy functional to a trajectory-based one.
    
    This helper function sums the legacy energy over all (state, action) pairs
    in the trajectory.
    
    Args:
        legacy_energy: Legacy energy functional
        
    Returns:
        Trajectory-based energy functional
    """
    class WrappedEnergy(EnergyFunctional):
        def task_cost(self, traj: Trajectory) -> float:
            total = 0.0
            for i, (x, u) in enumerate(zip(traj.states[:-1], traj.actions)):
                # Extract data from State/Action protocols
                x_data = x.data if hasattr(x, 'data') else x
                u_data = u.data if hasattr(u, 'data') else u
                total += float(legacy_energy.compute(x_data, u_data, traj.info))
            return total
        
        def constraint_energy(self, traj: Trajectory) -> float:
            # For legacy energy, constraints are mixed with task cost
            # This is a simple wrapper - more sophisticated decomposition
            # would require refactoring the legacy energy terms
            return 0.0
    
    return WrappedEnergy()
