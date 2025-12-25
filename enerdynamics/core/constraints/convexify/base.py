"""
Base convexifier interface.

Convexifiers transform constraints into convex form (typically linear inequalities).
"""

from abc import ABC, abstractmethod
from typing import Any

from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)


class Convexifier(ABC):
    """
    Base class for constraint convexifiers.
    
    A convexifier takes a reference trajectory/state and converts constraints
    into convex form (typically linear inequalities A x >= b).
    
    The key separation: convexifiers only generate constraint representations,
    they do NOT execute/enforce them. Execution is handled by operators.
    """
    
    @abstractmethod
    def build_constraints(
        self,
        ref: Any,  # Reference trajectory or state
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build convex constraints from reference.
        
        Args:
            ref: Reference trajectory or state (for linearization)
            params: Schedule parameters (margin, rho, etc.)
            state: Schedule state (k, K, etc.)
            
        Returns:
            ConvexConstraint representing the constraints
        """
        pass
    
    def build_constraints_batch(
        self,
        refs: list,
        params: ScheduleParams,
        state: ScheduleState
    ) -> list[ConvexConstraint]:
        """
        Build constraints for batch of references (optional optimization).
        
        Default implementation calls build_constraints in a loop.
        Override for batch processing.
        
        Args:
            refs: List of reference trajectories/states
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            List of ConvexConstraint objects
        """
        return [self.build_constraints(ref, params, state) for ref in refs]

