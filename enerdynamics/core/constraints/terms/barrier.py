"""
Barrier constraint terms: log-barrier and tightening barrier.

This module provides barrier function constraints, useful for:
- Log-barrier: -log(-g(x)) for inequality constraints g(x) <= 0
- Tightening barrier: EB-MBD style barrier with schedule-based tightening
"""

from typing import Optional
import numpy as np

from .base import ConstraintTerm
from enerdynamics.core.types import Trajectory


class BarrierTerm(ConstraintTerm):
    """
    Barrier constraint term.
    
    Provides barrier functions for inequality constraints:
    - Log-barrier: -log(-g(x)) for g(x) <= 0
    - Tightening barrier: Schedule-based barrier with margin tightening
    
    Energy: Barrier energy (high when constraint is violated)
    Feasible: Constraint satisfied
    Violation: Constraint violation magnitude
    """
    
    def __init__(
        self,
        constraint_fn: callable,
        barrier_type: str = "log",
        margin: float = 0.0,
        t: float = 1.0,
    ):
        """
        Initialize barrier constraint term.
        
        Args:
            constraint_fn: Function g(x) that returns constraint values (g(x) <= 0 means feasible)
            barrier_type: "log" for log-barrier, "tightening" for tightening barrier
            margin: Safety margin (constraint: g(x) <= -margin)
            t: Barrier parameter (for log-barrier: -t * log(-g(x)))
        """
        self.constraint_fn = constraint_fn
        self.barrier_type = barrier_type
        self.margin = margin
        self.t = t
    
    def energy(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        t: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Compute barrier energy.
        
        Args:
            trajectory: Trajectory to evaluate
            margin: Override default margin (for scheduling)
            t: Override default t parameter (for scheduling)
            **kwargs: Additional parameters
            
        Returns:
            Total barrier energy
        """
        margin = margin if margin is not None else self.margin
        t = t if t is not None else self.t
        
        total = 0.0
        
        for state in trajectory.states:
            # Evaluate constraint: g(x)
            g = self.constraint_fn(state)
            g = np.asarray(g, dtype=np.float32)
            
            # Constraint: g(x) <= -margin
            # So we need: g(x) + margin <= 0
            g_adjusted = g + margin
            
            if self.barrier_type == "log":
                # Log-barrier: -t * log(-g_adjusted) for g_adjusted < 0
                # High when g_adjusted approaches 0 from below
                if np.any(g_adjusted >= 0):
                    return float('inf')  # Infeasible
                total += -t * np.sum(np.log(-g_adjusted))
            
            elif self.barrier_type == "tightening":
                # Tightening barrier: exp(-beta * (-g_adjusted))
                # High when g_adjusted is close to 0
                beta = 10.0  # Sharpness parameter
                total += np.sum(np.exp(-beta * (-g_adjusted)))
        
        return float(total)
    
    def feasible(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Check if trajectory satisfies barrier constraint: g(x) <= -margin.
        
        Args:
            trajectory: Trajectory to check
            margin: Override default margin
            **kwargs: Additional parameters
            
        Returns:
            True if all states satisfy constraint
        """
        margin = margin if margin is not None else self.margin
        
        for state in trajectory.states:
            g = self.constraint_fn(state)
            g = np.asarray(g, dtype=np.float32)
            
            # Constraint: g(x) <= -margin
            if np.any(g + margin > 0):
                return False
        
        return True
    
    def violation(
        self,
        trajectory: Trajectory,
        margin: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute constraint violations: max(0, g(x) + margin).
        
        Args:
            trajectory: Trajectory to evaluate
            margin: Override default margin
            **kwargs: Additional parameters
            
        Returns:
            Array of violations per state
        """
        margin = margin if margin is not None else self.margin
        violations = []
        
        for state in trajectory.states:
            g = self.constraint_fn(state)
            g = np.asarray(g, dtype=np.float32)
            
            # Violation: max(0, g(x) + margin)
            violation = np.maximum(0.0, g + margin)
            violations.append(float(np.sum(violation)))
        
        return np.array(violations, dtype=np.float32)

