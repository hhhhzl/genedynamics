"""
NumPy backend for dual-control diffusion scheduler.

High-performance NumPy implementation with optimized computations.
Uses pre-computed constants and efficient mathematical operations.
"""

from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.constraints.schedulers.DiffusionScheduler.dualcontrol.dual_control import (
    DualControlDiffusionScheduler
)
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "dual_control_diffusion", "numpy")
class DualControlDiffusionSchedulerNumpy(DualControlDiffusionScheduler):
    """
    NumPy backend for dual-control diffusion scheduler.
    
    High-performance implementation with:
    - Pre-computed constants for fast parameter computation
    - Optimized exponential and sigmoid functions
    - Efficient dual variable updates
    """
    
    def _compute_diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Compute diffusion parameters (NumPy implementation).
        
        Performance optimizations:
        - Pre-computed ranges (T_range, ess_range, etc.)
        - Fast sigmoid and exponential computation
        - Single-pass parameter computation
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with adaptive diffusion parameters
        """
        # Compute diffusion parameters from λ^diff
        M_k = self._compute_sample_size()
        T_k = self._compute_temperature()
        
        params = {
            "M_k": M_k,
            "T_k": T_k,
        }
        
        # Optional beta scaling
        if self.enable_beta_scaling:
            s_k = self._compute_beta_scaling()
            params["s_k"] = s_k
        
        return params
    
    def _compute_sample_size(self) -> int:
        """
        Compute sample size M_k from dual variable.
        
        M_k = clip(M_min * exp(c_M * λ^diff), M_min, M_max)
        
        Performance: O(1) with fast exponential and clipping.
        
        Returns:
            Sample size (integer)
        """
        # Fast exponential computation with clipping
        exp_arg = self.c_M * self.lambda_diff
        exp_arg = np.clip(exp_arg, -500.0, 500.0)  # Avoid overflow
        M_k = self.M_min * np.exp(exp_arg)
        return int(np.clip(M_k, self.M_min, self.M_max))
    
    def _compute_temperature(self) -> float:
        """
        Compute temperature T_k from dual variable.
        
        T_k = T_min + (T_max - T_min) * σ(a_T * λ^diff + b_T)
        
        Performance: O(1) with fast sigmoid and pre-computed range.
        
        Returns:
            Temperature value
        """
        # Fast sigmoid computation
        sigmoid_arg = self.a_T * (self.lambda_diff - self.b_T)
        sigmoid_arg = np.clip(sigmoid_arg, -500.0, 500.0)  # Avoid overflow
        sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_arg))
        return self.T_min + self._T_range * sigmoid
    
    def _compute_beta_scaling(self) -> float:
        """
        Compute beta scaling factor s_k from dual variable.
        
        s_k = clip(exp(c_beta * λ^diff), s_min, s_max)
        
        Performance: O(1) with fast exponential and clipping.
        
        Returns:
            Beta scaling factor
        """
        exp_arg = self.c_beta * self.lambda_diff
        exp_arg = np.clip(exp_arg, -500.0, 500.0)  # Avoid overflow
        s_k = np.exp(exp_arg)
        return float(np.clip(s_k, self.s_min, self.s_max))
    
    def _update_dual_variable(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable λ^diff using Robbins-Monro style update.
        
        λ_{k+1} = clip(λ_k + η_k * (ESS*(t_k) - ESŜ_k), [λ_min, λ_max])
        
        Performance: O(1) with efficient clipping.
        
        Args:
            state: Current schedule state
            feedback: Dictionary with "ess" or "ess_hat" key
        """
        k = state.k
        
        # Compute progress and target ESS
        t_k = self.diffusion_schedule.get_progress(k, self.snr_min, self.snr_max)
        ess_star = self._compute_target_ess(t_k)
        
        # Get estimated ESS
        ess_hat = feedback.get("ess", feedback.get("ess_hat", None))
        if ess_hat is None:
            # Try to compute from weights if provided
            weights = feedback.get("weights", None)
            if weights is not None:
                ess_hat = self._compute_ess(weights)
            else:
                # No feedback available - skip update
                return
        
        # Compute step size with decay (cached computation)
        eta_k = self.eta_diff_base * np.power(self.eta_diff_decay, self.step_count)
        
        # Update dual variable
        error = ess_star - ess_hat
        lambda_new = self.lambda_diff + eta_k * error
        
        # Clip to bounds (fast)
        self.lambda_diff = float(np.clip(lambda_new, self.lambda_diff_min, self.lambda_diff_max))
        self.step_count += 1
        
        # Update beta scaling if enabled (store in diffusion_schedule)
        if self.enable_beta_scaling:
            s_k = self._compute_beta_scaling()
            # Store scaling factor (can be used to scale betas)
            if not hasattr(self.diffusion_schedule, 's_k') or self.diffusion_schedule.s_k is None:
                # Initialize scaling array
                self.diffusion_schedule.s_k = np.ones_like(self.diffusion_schedule.betas)
            # Update scaling for current step (if needed)
            # Note: In practice, scaling might be applied globally or per-step
            # This is a placeholder - actual implementation depends on usage
    
    def _compute_target_ess(self, t_k: float) -> float:
        """
        Compute target ESS ESS*(t_k) using cosine annealing.
        
        ESS*(t) = ESS_min + (ESS_max - ESS_min) * CosAnneal(t; p_ess)
        CosAnneal(t; p_ess) = 0.5 * (1 - cos(π * t^p_ess))
        
        Performance: O(1) with pre-computed range.
        
        Args:
            t_k: Normalized progress in [0, 1]
            
        Returns:
            Target ESS value
        """
        # Fast cosine annealing computation
        t_powered = np.power(t_k, self.p_ess)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * t_powered))
        return self.ess_star_min + self._ess_range * cos_anneal
    
    def _compute_ess(self, weights: np.ndarray) -> float:
        """
        Compute effective sample size from importance weights.
        
        ESS = 1 / (Σ_m w_m^2)
        
        Normalized by sample size: ESS_normalized = ESS / M
        
        Performance: O(M) where M is number of weights.
        
        Args:
            weights: Importance weights array
            
        Returns:
            Normalized ESS in [0, 1]
        """
        weights = np.asarray(weights, dtype=np.float32)
        if weights.size == 0:
            return 0.0
        
        # Normalize weights
        weight_sum = np.sum(weights)
        if weight_sum <= 0.0:
            return 0.0
        
        weights_normalized = weights / weight_sum
        
        # Compute ESS
        ess = 1.0 / (np.sum(weights_normalized ** 2) + 1e-10)
        
        # Normalize by sample size
        ess_normalized = ess / len(weights)
        
        return float(ess_normalized)
    
    def get_scaled_betas(self) -> Optional[np.ndarray]:
        """
        Get scaled beta schedule if scaling is enabled.
        
        Returns:
            Scaled betas array (β'_k = s_k * β_k) or None if scaling disabled
        """
        if not self.enable_beta_scaling:
            return None
        
        # Compute current scaling factor
        s_k = self._compute_beta_scaling()
        
        # Apply scaling to all betas
        # Note: In practice, scaling might be step-dependent
        # This implementation uses a global scaling factor
        return self.diffusion_schedule.betas * s_k

