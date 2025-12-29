"""
Base operator interface.

Operators execute/enforce constraints on trajectories.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from enerdynamics.core.types import Trajectory
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)


class Operator(ABC):
    """
    Base class for constraint operators.
    
    An operator takes a nominal trajectory and convex constraints,
    and returns a repaired trajectory that satisfies the constraints.
    
    The key separation: operators only execute/enforce constraints,
    they do NOT generate them. Constraint generation is handled by convexifiers.
    """
    
    @abstractmethod
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply operator to enforce constraints.
        
        Args:
            nominal: Nominal trajectory to repair
            constraints: Convex constraints to enforce
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (repaired trajectory, operator info)
        """
        pass
    
    def apply_batch(
        self,
        nominals: list[Trajectory],
        constraints: list[ConvexConstraint],
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[list[Trajectory], Dict[str, Any]]:
        """
        Apply operator to batch of trajectories (optional optimization).
        
        Default implementation calls apply in a loop.
        Override for batch processing.
        
        Args:
            nominals: List of nominal trajectories
            constraints: List of convex constraints
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            Tuple of (list of repaired trajectories, merged info dict)
        """
        results = [self.apply(n, c, params, state) for n, c in zip(nominals, constraints)]
        repaired = [r[0] for r in results]
        info = self._merge_info([r[1] for r in results])
        return repaired, info
    
    def _merge_info(self, info_list: list[OperatorInfo]) -> Dict[str, Any]:
        """Merge operator info from batch."""
        if not info_list:
            return {}
        
        merged = {
            "success": all(info.success for info in info_list),
            "violation_before": sum(info.violation_before for info in info_list) / len(info_list),
            "violation_after": sum(info.violation_after for info in info_list) / len(info_list),
            "iterations": sum(info.iterations for info in info_list),
            "time": sum(info.time for info in info_list),
        }
        return merged


