"""
Base scheduler interface.

Schedulers generate ScheduleParams based on ScheduleState, controlling
constraint enforcement behavior throughout optimization.
"""

from abc import ABC, abstractmethod

from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)


class Scheduler(ABC):
    """
    Base class for constraint schedulers.
    
    A scheduler takes the current schedule state (diffusion step, etc.)
    and generates appropriate parameters (margin, rho, topK, etc.) for
    constraint enforcement.
    
    This is the "single source of truth" for all schedule parameters.
    """
    
    @abstractmethod
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate schedule parameters for given state.
        
        Args:
            state: Current schedule state
            
        Returns:
            ScheduleParams with appropriate values
        """
        pass
    
    def update(self, state: ScheduleState, feedback: dict) -> None:
        """
        Update scheduler based on feedback (for adaptive schedulers).
        
        Args:
            state: Current schedule state
            feedback: Feedback dictionary (violation, stats, etc.)
        """
        # Default: no-op (fixed schedulers don't need updates)
        pass

