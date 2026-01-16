"""
NumPy backend for fixed diffusion scheduler.

This is the reference implementation using pure NumPy.
Optimized for performance with pre-computed parameter cache.
"""

from typing import Dict, Any
from enerdynamics.core.constraints.schedulers.DiffusionScheduler.fixed.fixed import FixedDiffusionScheduler
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "fixed_diffusion", "numpy")
class FixedDiffusionSchedulerNumpy(FixedDiffusionScheduler):
    """
    NumPy backend for fixed diffusion scheduler.
    
    This is the reference implementation that other backends should match.
    Optimized for performance with constant-time parameter generation.
    """
    
    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate fixed diffusion parameters (NumPy implementation).
        
        Performance: O(1) - returns pre-computed parameters.
        
        Args:
            state: Current schedule state (ignored)
            
        Returns:
            Dictionary with fixed diffusion parameters
        """
        # Return pre-computed parameters (fastest possible)
        params = {
            "M_k": self.M_k,
            "T_k": self.T_k,
            **self.extra,
        }
        if self.s_k is not None:
            params["s_k"] = self.s_k
        if getattr(self, "beta0", None) is not None:
            params["beta0"] = self.beta0
        if getattr(self, "betaT", None) is not None:
            params["betaT"] = self.betaT
        if getattr(self, "Ndiffuse", None) is not None:
            params["Ndiffuse"] = self.Ndiffuse
        return params

