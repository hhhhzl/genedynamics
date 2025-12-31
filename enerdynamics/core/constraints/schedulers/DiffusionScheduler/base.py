"""
Base class for diffusion schedulers.

Diffusion schedulers are responsible for diffusion-related parameters:
- M_k (sample size)
- T_k (temperature)
- Optional: s_k (beta scaling factor)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)


class DiffusionScheduler(Scheduler):
    """
    Base class for diffusion schedulers.
    
    Diffusion schedulers generate diffusion-related parameters only.
    They don't handle constraint parameters (rho, topK, eps, etc.).
    
    This interface allows diffusion schedulers to be used independently
    or combined with constraint schedulers in CompositeScheduler.
    """
    
    @abstractmethod
    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Generate diffusion parameters only.
        
        This method extracts only diffusion-related parameters, allowing
        the scheduler to be used in CompositeScheduler alongside constraint schedulers.
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with diffusion parameters:
            - M_k: int - Sample size (number of candidates to generate)
            - T_k: float - Temperature for importance weighting
            - s_k: Optional[float] - Beta scaling factor (if enabled)
            - _extra: Dict - Additional parameters
        """
        pass
    
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate full ScheduleParams (for compatibility with base Scheduler).
        
        This wraps diffusion_params() to return ScheduleParams, ensuring
        backward compatibility with existing code that expects ScheduleParams.
        
        Args:
            state: Current schedule state
            
        Returns:
            ScheduleParams with diffusion parameters in _extra
        """
        diffusion_dict = self.diffusion_params(state)
        return ScheduleParams(
            _extra={
                "M_k": diffusion_dict.get("M_k"),
                "T_k": diffusion_dict.get("T_k"),
                **{k: v for k, v in diffusion_dict.items() 
                   if k not in ["M_k", "T_k"]},
            }
        )

