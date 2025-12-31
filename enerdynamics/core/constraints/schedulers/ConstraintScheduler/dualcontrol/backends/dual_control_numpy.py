"""
NumPy backend for dual-control constraint scheduler.

High-performance NumPy implementation with optimized computations.
Uses pre-computed constants and vectorized operations where possible.
"""

from typing import Dict, Any, Optional
import numpy as np

from enerdynamics.core.constraints.schedulers.ConstraintScheduler.dualcontrol.dual_control import (
    DualControlConstraintScheduler
)
from enerdynamics.core.constraints.core.types import ScheduleState
from enerdynamics.core.constraints.core.registry import register


@register("scheduler", "dual_control_constraint", "numpy")
class DualControlConstraintSchedulerNumpy(DualControlConstraintScheduler):
    """
    NumPy backend for dual-control constraint scheduler.
    
    High-performance implementation with:
    - Pre-computed constants for fast parameter computation
    - Optimized sigmoid and cosine annealing functions
    - Efficient dual variable updates
    """
    
    def _compute_constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """
        Compute constraint parameters (NumPy implementation).
        
        Performance optimizations:
        - Pre-computed ranges (rho_range, K_range, etc.)
        - Fast sigmoid computation
        - Single-pass parameter computation
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with adaptive constraint parameters
        """
        k = state.k
        K = state.K
        
        # Compute progress t_k from SNR (cached in diffusion_schedule)
        t_k = self.diffusion_schedule.get_progress(k, self.snr_min, self.snr_max)
        
        # Check terminal hard region
        is_terminal_hard = t_k >= self.t_hard
        
        # Compute target feasibility q*(t_k)
        q_star = self._compute_target_feasibility(t_k)
        
        # Compute constraint parameters
        if is_terminal_hard:
            # Terminal hard region: use maximum values
            rho_k = self.rho_max
            p_k = 1.0
            K_c_k = self.K_max
            eps_k = self.eps_min
            I_QP_k = self.I_max
        else:
            # Adaptive parameters based on λ^con and t_k
            rho_k = self._compute_rho()
            p_k = self._compute_gate_probability()
            K_c_k = self._compute_active_set_size(t_k)
            eps_k = self._compute_solver_tolerance(t_k)
            I_QP_k = self._compute_qp_iterations(t_k)
        
        # Draw gate (stochastic) - use pre-computed probability
        gate_k = np.random.random() < p_k
        
        return {
            "rho": float(rho_k),
            "topK": int(K_c_k),
            "eps": float(eps_k),
            "I_QP": int(I_QP_k),
            "qp_gate": bool(gate_k),
            "qp_prob": float(p_k),
            "margin": 0.0,
            "_extra": {
                "t_k": float(t_k),
                "is_terminal_hard": is_terminal_hard,
                "lambda_con": float(self.lambda_con),
                "q_star": float(q_star),
            }
        }
    
    def _compute_target_feasibility(self, t_k: float) -> float:
        """
        Compute target feasibility q*(t_k) using cosine annealing.
        
        q*(t) = q_min + (q_max - q_min) * CosAnneal(t; p_q)
        CosAnneal(t; p_q) = 0.5 * (1 - cos(π * t^p_q))
        
        Performance: O(1) with pre-computed range.
        
        Args:
            t_k: Normalized progress in [0, 1]
            
        Returns:
            Target feasibility rate
        """
        # Fast cosine annealing computation
        t_powered = np.power(t_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * t_powered))
        return self.q_star_min + self._q_range * cos_anneal
    
    def _compute_rho(self) -> float:
        """
        Compute slack penalty ρ_k from dual variable.
        
        ρ_k = ρ_min + (ρ_max - ρ_min) * σ(a_ρ * λ^con + b_ρ)
        
        Performance: O(1) with pre-computed range and fast sigmoid.
        
        Returns:
            Slack penalty value
        """
        # Fast sigmoid: 1 / (1 + exp(-scale * (x - offset)))
        sigmoid_arg = self.a_rho * (self.lambda_con - self.b_rho)
        # Clamp to avoid overflow
        sigmoid_arg = np.clip(sigmoid_arg, -500.0, 500.0)
        sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_arg))
        return self.rho_min + self._rho_range * sigmoid
    
    def _compute_gate_probability(self) -> float:
        """
        Compute QP gate probability p_k from dual variable.
        
        p_k = σ(a_g * λ^con + b_g)
        
        Performance: O(1) with fast sigmoid.
        
        Returns:
            Gate probability in [0, 1]
        """
        sigmoid_arg = self.a_g * (self.lambda_con - self.b_g)
        sigmoid_arg = np.clip(sigmoid_arg, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-sigmoid_arg))
    
    def _compute_active_set_size(self, t_k: float) -> int:
        """
        Compute active-set size K_c(k).
        
        K_c(k) = K_min + floor((K_max - K_min) * t_k)
        
        Performance: O(1) with pre-computed range.
        
        Args:
            t_k: Normalized progress
            
        Returns:
            Active constraint count
        """
        return self.K_min + int(np.floor(self._K_range * t_k))
    
    def _compute_solver_tolerance(self, t_k: float) -> float:
        """
        Compute solver tolerance ε_k.
        
        ε_k = ε_min + (ε_max - ε_min) * (1 - t_k)^p_eps
        
        Performance: O(1) with pre-computed range.
        
        Args:
            t_k: Normalized progress
            
        Returns:
            Solver tolerance
        """
        one_minus_t = 1.0 - t_k
        power_term = np.power(one_minus_t, self.p_eps)
        return self.eps_min + self._eps_range * power_term
    
    def _compute_qp_iterations(self, t_k: float) -> int:
        """
        Compute QP iterations I^QP_k.
        
        I^QP_k = I_min + floor((I_max - I_min) * t_k)
        
        Performance: O(1) with pre-computed range.
        
        Args:
            t_k: Normalized progress
            
        Returns:
            Maximum QP iterations
        """
        return self.I_min + int(np.floor(self._I_range * t_k))
    
    def _update_dual_variable(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable λ^con using Robbins-Monro style update.
        
        λ_{k+1} = clip(λ_k + η_k * (q*(t_k) - q̂_k), [λ_min, λ_max])
        
        Performance: O(1) with efficient clipping.
        
        Args:
            state: Current schedule state
            feedback: Dictionary with "feasible_rate" or "q_hat" key
        """
        k = state.k
        
        # Compute progress and target
        t_k = self.diffusion_schedule.get_progress(k, self.snr_min, self.snr_max)
        q_star = self._compute_target_feasibility(t_k)
        
        # Get estimated feasibility rate
        q_hat = feedback.get("feasible_rate", feedback.get("q_hat", None))
        if q_hat is None:
            # Try to compute from trajectories if provided
            trajectories = feedback.get("trajectories", [])
            if trajectories:
                q_hat = self._estimate_feasibility_rate(trajectories, feedback)
            else:
                # No feedback available - skip update
                return
        
        # Compute step size with decay (cached computation)
        eta_k = self.eta_con_base * np.power(self.eta_con_decay, self.step_count)
        
        # Update dual variable
        error = q_star - q_hat
        lambda_new = self.lambda_con + eta_k * error
        
        # Clip to bounds (fast)
        self.lambda_con = float(np.clip(lambda_new, self.lambda_con_min, self.lambda_con_max))
        self.step_count += 1
    
    def _estimate_feasibility_rate(
        self,
        trajectories: list,
        feedback: Dict[str, Any]
    ) -> float:
        """
        Estimate feasibility rate from rollout trajectories.
        
        q̂_k = (1/M) * Σ_m 1{τ̂_m ∈ F}
        
        Performance: O(M) where M is number of trajectories.
        
        Args:
            trajectories: List of rollout trajectories
            feedback: Feedback dictionary (may contain pre-computed feasibility)
            
        Returns:
            Estimated feasibility rate in [0, 1]
        """
        # Check if feasibility is already computed
        if "feasible_rate" in feedback:
            return feedback["feasible_rate"]
        
        if not trajectories:
            return 0.0
        
        # Count feasible trajectories
        feasible_count = 0
        for traj in trajectories:
            if hasattr(traj, 'is_feasible'):
                if traj.is_feasible:
                    feasible_count += 1
            elif isinstance(traj, dict) and 'is_feasible' in traj:
                if traj['is_feasible']:
                    feasible_count += 1
            else:
                # Default: assume feasible if no info (conservative)
                feasible_count += 1
        
        return float(feasible_count) / len(trajectories)

