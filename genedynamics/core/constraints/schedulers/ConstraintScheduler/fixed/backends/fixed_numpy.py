"""
NumPy backend for fixed constraint scheduler.

This is the reference implementation using pure NumPy.
Optimized for performance with pre-computed parameter cache.
"""

from typing import Dict, Any
from enerdynamics.core.constraints.schedulers.ConstraintScheduler.fixed.fixed import FixedConstraintScheduler
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "fixed_constraint", "numpy")
class FixedConstraintSchedulerNumpy(FixedConstraintScheduler):
    """
    NumPy backend for fixed constraint scheduler.
    
    This is the reference implementation that other backends should match.
    Optimized for performance with constant-time parameter generation.
    """
    
    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate fixed constraint parameters (NumPy implementation).
        
        Performance: O(1) - returns pre-computed parameters.
        
        Args:
            state: Current schedule state (ignored)
            
        Returns:
            Dictionary with fixed constraint parameters
        """
        # Return pre-computed parameters (fastest possible)
        return {
            "rho": self.rho,
            "topK": self.topK,
            "eps": self.eps,
            "I_QP": self.I_QP,
            "qp_gate": self.qp_gate,
            "qp_prob": self.qp_prob,
            "margin": self.margin,
            "_extra": self.extra,
        }

