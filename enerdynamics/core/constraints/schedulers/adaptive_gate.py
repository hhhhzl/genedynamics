"""
Adaptive gate scheduler: qp_gate/qp_prob from violation/ESS/SNR.

This scheduler adaptively controls when to apply QP based on:
- Constraint violations
- Effective sample size (ESS)
- Signal-to-noise ratio (SNR)
"""

from typing import Optional, Dict, Any
import numpy as np

from enerdynamics.core.constraints.schedulers.base import Scheduler
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
)
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "adaptive_gate", "numpy")
class AdaptiveGateScheduler(Scheduler):
    """
    Adaptive gate scheduler.
    
    Adaptively controls qp_gate and qp_prob based on:
    - Constraint violations: Higher violation → higher qp_prob
    - Effective sample size (ESS): Lower ESS → higher qp_prob
    - Signal-to-noise ratio (SNR): Lower SNR → higher qp_prob
    
    This enables adaptive constraint enforcement that only applies QP
    when needed, improving efficiency.
    """
    
    def __init__(
        self,
        margin_start: float = 0.5,
        margin_end: float = 0.1,
        rho_start: float = 0.1,
        rho_end: float = 10.0,
        violation_threshold: float = 0.1,
        ess_threshold: float = 0.5,
        snr_threshold: float = 1.0,
        qp_prob_min: float = 0.0,
        qp_prob_max: float = 1.0,
        **kwargs
    ):
        """
        Initialize adaptive gate scheduler.
        
        Args:
            margin_start: Initial margin
            margin_end: Final margin
            rho_start: Initial rho
            rho_end: Final rho
            violation_threshold: Violation threshold for QP gating
            ess_threshold: ESS threshold for QP gating
            snr_threshold: SNR threshold for QP gating
            qp_prob_min: Minimum QP probability
            qp_prob_max: Maximum QP probability
            **kwargs: Additional arguments
        """
        self.margin_start = margin_start
        self.margin_end = margin_end
        self.rho_start = rho_start
        self.rho_end = rho_end
        self.violation_threshold = violation_threshold
        self.ess_threshold = ess_threshold
        self.snr_threshold = snr_threshold
        self.qp_prob_min = qp_prob_min
        self.qp_prob_max = qp_prob_max
        
        # Track current statistics
        self.current_violation = None
        self.current_ess = None
        self.current_snr = None
        self.kwargs = kwargs
    
    def params(self, state: ScheduleState) -> ScheduleParams:
        """
        Generate schedule parameters with adaptive gating.
        
        Args:
            state: Current schedule state
            
        Returns:
            ScheduleParams with adaptive qp_gate and qp_prob
        """
        # Base parameters (cosine annealing)
        progress = state.progress
        margin = (
            self.margin_start * (1 - progress) +
            self.margin_end * progress
        )
        rho = (
            self.rho_start * (1 - progress) +
            self.rho_end * progress
        )
        
        # Compute adaptive qp_prob based on feedback
        qp_prob = self._compute_adaptive_qp_prob()
        
        # qp_gate: True if qp_prob > threshold
        qp_gate = qp_prob > 0.5
        
        return ScheduleParams(
            margin=float(margin),
            rho=float(rho),
            topK=None,
            topL=None,
            qp_gate=qp_gate,
            qp_prob=float(qp_prob),
            _extra={
                "current_violation": self.current_violation,
                "current_ess": self.current_ess,
                "current_snr": self.current_snr,
            }
        )
    
    def update(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update scheduler based on feedback.
        
        Args:
            state: Current schedule state
            feedback: Feedback dictionary with "violation", "ess", "snr" keys
        """
        # Update violation
        if "violation" in feedback:
            if self.current_violation is None:
                self.current_violation = feedback["violation"]
            else:
                alpha = 0.1
                self.current_violation = (
                    alpha * feedback["violation"] +
                    (1 - alpha) * self.current_violation
                )
        
        # Update ESS
        if "ess" in feedback:
            if self.current_ess is None:
                self.current_ess = feedback["ess"]
            else:
                alpha = 0.1
                self.current_ess = (
                    alpha * feedback["ess"] +
                    (1 - alpha) * self.current_ess
                )
        
        # Update SNR
        if "snr" in feedback:
            if self.current_snr is None:
                self.current_snr = feedback["snr"]
            else:
                alpha = 0.1
                self.current_snr = (
                    alpha * feedback["snr"] +
                    (1 - alpha) * self.current_snr
                )
    
    def _compute_adaptive_qp_prob(self) -> float:
        """
        Compute adaptive QP probability based on current statistics.
        
        Returns:
            QP probability (0-1)
        """
        if self.current_violation is None:
            # No feedback - use default
            return 1.0
        
        # Compute probability based on violation
        violation_prob = 0.0
        if self.current_violation > self.violation_threshold:
            # High violation → high probability
            violation_prob = min(1.0, self.current_violation / self.violation_threshold)
        else:
            # Low violation → low probability
            violation_prob = self.current_violation / self.violation_threshold
        
        # Compute probability based on ESS
        ess_prob = 0.0
        if self.current_ess is not None:
            if self.current_ess < self.ess_threshold:
                # Low ESS → high probability (need to repair)
                ess_prob = 1.0 - (self.current_ess / self.ess_threshold)
            else:
                # High ESS → low probability
                ess_prob = 0.0
        
        # Compute probability based on SNR
        snr_prob = 0.0
        if self.current_snr is not None:
            if self.current_snr < self.snr_threshold:
                # Low SNR → high probability (need to repair)
                snr_prob = 1.0 - (self.current_snr / self.snr_threshold)
            else:
                # High SNR → low probability
                snr_prob = 0.0
        
        # Combine probabilities (max of all)
        qp_prob = max(violation_prob, ess_prob, snr_prob)
        
        # Clip to range
        qp_prob = np.clip(qp_prob, self.qp_prob_min, self.qp_prob_max)
        
        return float(qp_prob)

