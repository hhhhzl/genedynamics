"""
Cosine annealing scheduler for constraint parameters.

Smoothly transitions constraint parameters using cosine interpolation,
similar to diffusion schedule annealing.
"""

import numpy as np

from typing import Optional
from genedynamics.core.constraints.schedulers.base import Scheduler
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)
from genedynamics.core.constraints.core.registry import register


@register("scheduler", "cosine_anneal", "numpy")
class CosineAnnealScheduler(Scheduler):
    """
    Cosine annealing scheduler.
    
    Smoothly transitions parameters using cosine interpolation:
    value(step) = start + (end - start) * (1 - cos(π * progress)) / 2
    
    This provides smooth transitions similar to diffusion schedules.
    """
    
    def __init__(
        self,
        margin_start: float = 0.5,
        margin_end: float = 0.1,
        rho_start: float = 0.1,
        rho_end: float = 10.0,
        topK_start: Optional[int] = None,
        topK_end: Optional[int] = None,
        topL_start: Optional[int] = None,
        topL_end: Optional[int] = None,
        qp_gate_start: bool = False,
        qp_gate_end: bool = True,
        qp_prob_start: float = 0.0,
        qp_prob_end: float = 1.0,
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            margin_start: Initial margin (at step 0)
            margin_end: Final margin (at step K)
            rho_start: Initial slack penalty
            rho_end: Final slack penalty
            topK_start: Initial top-K (None = no limit)
            topK_end: Final top-K
            topL_start: Initial top-L (None = no limit)
            topL_end: Final top-L
            qp_gate_start: Initial QP gate (False = disabled)
            qp_gate_end: Final QP gate (True = enabled)
            qp_prob_start: Initial QP probability
            qp_prob_end: Final QP probability
        """
        self.margin_start = float(margin_start)
        self.margin_end = float(margin_end)
        self.rho_start = float(rho_start)
        self.rho_end = float(rho_end)
        self.topK_start = topK_start
        self.topK_end = topK_end
        self.topL_start = topL_start
        self.topL_end = topL_end
        self.qp_gate_start = bool(qp_gate_start)
        self.qp_gate_end = bool(qp_gate_end)
        self.qp_prob_start = float(qp_prob_start)
        self.qp_prob_end = float(qp_prob_end)
    
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate schedule parameters using cosine annealing.
        
        Args:
            state: Current schedule state
            
        Returns:
            ScheduleParams with annealed values
        """
        progress = state.progress  # 0 = final, 1 = initial
        
        # Cosine interpolation
        smooth_progress = (1.0 - np.cos(np.pi * progress)) / 2.0
        
        # Interpolate parameters
        margin = self.margin_start + smooth_progress * (self.margin_end - self.margin_start)
        rho = self.rho_start + smooth_progress * (self.rho_end - self.rho_start)
        
        # Interpolate topK/topL (if specified)
        topK = None
        if self.topK_start is not None and self.topK_end is not None:
            topK = int(self.topK_start + smooth_progress * (self.topK_end - self.topK_start))
        elif self.topK_end is not None:
            topK = int(smooth_progress * self.topK_end)
        
        topL = None
        if self.topL_start is not None and self.topL_end is not None:
            topL = int(self.topL_start + smooth_progress * (self.topL_end - self.topL_start))
        elif self.topL_end is not None:
            topL = int(smooth_progress * self.topL_end)
        
        # Interpolate QP gate (threshold at 0.5)
        qp_gate = self.qp_gate_start if smooth_progress < 0.5 else self.qp_gate_end
        qp_prob = self.qp_prob_start + smooth_progress * (self.qp_prob_end - self.qp_prob_start)
        
        return ScheduleParams(
            margin=margin,
            rho=rho,
            topK=topK,
            topL=topL,
            qp_gate=qp_gate,
            qp_prob=qp_prob
        )

