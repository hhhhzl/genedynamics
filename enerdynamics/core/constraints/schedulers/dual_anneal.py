"""
Dual annealing scheduler: Principled dual schedule.

This scheduler generates schedule parameters based on a target feasible-rate curve,
using dual annealing to maintain a desired feasible rate throughout optimization.
"""

from typing import Optional, Dict, Any
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "dual_anneal", "numpy")
class DualAnnealScheduler(Scheduler):
    """
    Dual annealing scheduler.
    
    Generates schedule parameters to maintain a target feasible-rate curve.
    Uses dual annealing to adaptively adjust parameters based on current
    feasible rate vs target feasible rate.
    
    The scheduler adjusts margin and rho to keep feasible rate close to
    a target curve (e.g., gradually increasing from 0.5 to 1.0).
    """
    
    def __init__(
        self,
        target_feasible_rate_start: float = 0.5,
        target_feasible_rate_end: float = 1.0,
        margin_start: float = 0.5,
        margin_end: float = 0.1,
        rho_start: float = 0.1,
        rho_end: float = 10.0,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize dual annealing scheduler.
        
        Args:
            target_feasible_rate_start: Target feasible rate at start (k=0)
            target_feasible_rate_end: Target feasible rate at end (k=K)
            margin_start: Initial margin
            margin_end: Final margin
            rho_start: Initial rho
            rho_end: Final rho
            adaptation_rate: How fast to adapt to target (0-1)
            **kwargs: Additional arguments
        """
        self.target_feasible_rate_start = target_feasible_rate_start
        self.target_feasible_rate_end = target_feasible_rate_end
        self.margin_start = margin_start
        self.margin_end = margin_end
        self.rho_start = rho_start
        self.rho_end = rho_end
        self.adaptation_rate = adaptation_rate
        
        # Track current feasible rate
        self.current_feasible_rate = None
        self.kwargs = kwargs
    
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate schedule parameters based on dual annealing.
        
        Args:
            state: Current schedule state
            
        Returns:
            ScheduleParams with adjusted parameters
        """
        # Compute target feasible rate at current step
        progress = state.progress  # 0 = final, 1 = initial
        target_feasible_rate = (
            self.target_feasible_rate_start * (1 - progress) +
            self.target_feasible_rate_end * progress
        )
        
        # Get base parameters (cosine annealing)
        base_margin = (
            self.margin_start * (1 - progress) +
            self.margin_end * progress
        )
        base_rho = (
            self.rho_start * (1 - progress) +
            self.rho_end * progress
        )
        
        # Adjust parameters based on current vs target feasible rate
        if self.current_feasible_rate is not None:
            # Compute error
            error = target_feasible_rate - self.current_feasible_rate
            
            # Adjust margin: if feasible rate too low, increase margin (easier)
            # if feasible rate too high, decrease margin (harder)
            margin_adjustment = self.adaptation_rate * error
            margin = base_margin + margin_adjustment
            
            # Adjust rho: if feasible rate too low, decrease rho (softer)
            # if feasible rate too high, increase rho (harder)
            rho_adjustment = -self.adaptation_rate * error
            rho = base_rho * (1 + rho_adjustment)
            
            # Clip to reasonable ranges
            margin = np.clip(margin, 0.0, 1.0)
            rho = np.clip(rho, 0.01, 100.0)
        else:
            # No feedback yet - use base parameters
            margin = base_margin
            rho = base_rho
        
        return ScheduleParams(
            margin=float(margin),
            rho=float(rho),
            topK=None,
            topL=None,
            qp_gate=True,
            qp_prob=1.0,
            _extra={
                "target_feasible_rate": target_feasible_rate,
                "current_feasible_rate": self.current_feasible_rate,
            }
        )
    
    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update scheduler based on feedback.
        
        Args:
            state: Current schedule state
            feedback: Feedback dictionary with "feasible_rate" key
        """
        if "feasible_rate" in feedback:
            # Exponential moving average
            if self.current_feasible_rate is None:
                self.current_feasible_rate = feedback["feasible_rate"]
            else:
                alpha = 0.1  # Smoothing factor
                self.current_feasible_rate = (
                    alpha * feedback["feasible_rate"] +
                    (1 - alpha) * self.current_feasible_rate
                )

