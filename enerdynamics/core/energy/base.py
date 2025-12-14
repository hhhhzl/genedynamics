"""
Base energy functional classes.

This module defines the abstract EnergyFunctional interface for trajectory-based
energy evaluation.
"""

from abc import ABC, abstractmethod

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
