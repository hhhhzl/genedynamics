"""
Base class for constraint terms.

A ConstraintTerm defines what a constraint is:
- energy(): Soft constraint energy (for optimization)
- feasible(): Hard constraint feasibility check
- violation(): Constraint violation magnitude

Terms are independent of how constraints are enforced (that's operators' job).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union
import numpy as np

from enerdynamics.core.types import Trajectory, State, Action


class ConstraintTerm(ABC):
    """
    Base class for constraint terms.
    
    A constraint term defines:
    1. Energy: Soft constraint penalty (for differentiable optimization)
    2. Feasibility: Hard constraint satisfaction (boolean)
    3. Violation: Constraint violation magnitude (non-negative, 0 if satisfied)
    
    Terms can be used in multiple ways:
    - Standalone: Direct energy/feasibility evaluation
    - With convexifiers: Converted to convex form (A, b)
    - With operators: Enforced via QP/projection
    """
    
    @abstractmethod
    def energy(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> float:
        """
        Compute soft constraint energy (penalty).
        
        This is a differentiable function that provides gradients for optimization.
        Higher energy = more constraint violation.
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters (e.g., step, total_steps for scheduling)
            
        Returns:
            Energy value (non-negative, 0 if constraint satisfied)
        """
        pass
    
    @abstractmethod
    def feasible(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> bool:
        """
        Check if trajectory satisfies hard constraint.
        
        Args:
            trajectory: Trajectory to check
            **kwargs: Additional parameters
            
        Returns:
            True if feasible, False otherwise
        """
        pass
    
    @abstractmethod
    def violation(
        self,
        trajectory: Trajectory,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Compute constraint violation magnitude.
        
        Returns non-negative values, where 0 means constraint is satisfied.
        Can return scalar (single constraint) or array (multiple constraints).
        
        Args:
            trajectory: Trajectory to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Violation value(s) (non-negative, 0 if satisfied)
        """
        pass
    
    def energy_batch(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> np.ndarray:
        """
        Batch compute energy for multiple trajectories.
        
        Default implementation loops over trajectories.
        Override for optimized batch processing.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional parameters
            
        Returns:
            Array of energy values, shape (len(trajectories),)
        """
        return np.array([self.energy(traj, **kwargs) for traj in trajectories])
    
    def feasible_batch(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> np.ndarray:
        """
        Batch check feasibility for multiple trajectories.
        
        Default implementation loops over trajectories.
        Override for optimized batch processing.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional parameters
            
        Returns:
            Array of feasibility flags, shape (len(trajectories),)
        """
        return np.array([self.feasible(traj, **kwargs) for traj in trajectories])
    
    def violation_batch(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> np.ndarray:
        """
        Batch compute violations for multiple trajectories.
        
        Default implementation loops over trajectories.
        Override for optimized batch processing.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional parameters
            
        Returns:
            Array of violation values, shape (len(trajectories), ...)
        """
        violations = [self.violation(traj, **kwargs) for traj in trajectories]
        # Stack if all violations are arrays of same shape
        try:
            return np.stack(violations)
        except:
            return np.array(violations)


