"""
Base class for constraint schedulers.

Constraint schedulers are responsible for constraint-related parameters:
- rho (slack penalty)
- topK (active constraint count)
- eps (solver tolerance)
- I_QP (QP iterations)
- qp_gate, qp_prob (QP invocation)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)


class ConstraintScheduler(Scheduler):
    """
    Base class for constraint schedulers.
    
    Constraint schedulers generate constraint-related parameters only.
    They don't handle diffusion parameters (M_k, T_k).
    
    This interface allows constraint schedulers to be used independently
    or combined with diffusion schedulers in CompositeScheduler.
    """
    
    @abstractmethod
    def constraint_params(self, state: ScheduleState, record: bool = True) -> Dict[str, Any]:
        """
        Generate constraint parameters only.
        
        This method extracts only constraint-related parameters, allowing
        the scheduler to be used in CompositeScheduler alongside diffusion schedulers.
        
        Args:
            state: Current schedule state
            record: Whether to record history
            
        Returns:
            Dictionary with constraint parameters:
            - rho: float - Slack penalty weight
            - topK: Optional[int] - Top-K active constraints
            - eps: float - Solver tolerance
            - I_QP: int - Maximum QP iterations
            - qp_gate: bool - Whether to apply QP
            - qp_prob: float - QP invocation probability
            - margin: float - Safety margin (optional)
            - _extra: Dict - Additional parameters
        """
        pass
    
    def params(self, state: ScheduleState, record: bool = True) -> ScheduleParams:
        """
        Generate full ScheduleParams (for compatibility with base Scheduler).
        
        This wraps constraint_params() to return ScheduleParams, ensuring
        backward compatibility with existing code that expects ScheduleParams.
        
        Args:
            state: Current schedule state
            record: Whether to record history (passed to constraint_params)
            
        Returns:
            ScheduleParams with constraint parameters
        """
        # Pass record parameter if supported by constraint_params
        import inspect
        sig = inspect.signature(self.constraint_params)
        if 'record' in sig.parameters:
            constraint_dict = self.constraint_params(state, record=record)
        else:
            constraint_dict = self.constraint_params(state)
            
        return ScheduleParams(
            margin=constraint_dict.get("margin", 0.0),
            rho=constraint_dict.get("rho", 1.0),
            topK=constraint_dict.get("topK"),
            topL=constraint_dict.get("topL"),
            qp_gate=constraint_dict.get("qp_gate", True),
            qp_prob=constraint_dict.get("qp_prob", 1.0),
            _extra={
                "eps": constraint_dict.get("eps", 1e-4),
                "I_QP": constraint_dict.get("I_QP", 10),
                **constraint_dict.get("_extra", {}),
            }
        )

