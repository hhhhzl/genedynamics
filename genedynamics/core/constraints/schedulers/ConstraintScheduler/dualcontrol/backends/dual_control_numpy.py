"""
NumPy backend for dual-control constraint scheduler.

High-performance NumPy implementation with optimized computations.
Uses pre-computed constants and vectorized operations where possible.
"""

from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from genedynamics.core.constraints.schedulers.ConstraintScheduler.dualcontrol.dual_control import (
    DualControlConstraintScheduler
)
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import register


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
        
        Uses unified tightening progress s_k = (K - k) / K to drive all scheduling.
        s_k ∈ [0, 1] where s_k increases as reverse diffusion progresses (k decreases).
        
        Performance optimizations:
        - Pre-computed ranges (rho_range, K_range, etc.)
        - Fast sigmoid and cosine annealing computation
        - Single-pass parameter computation
        
        Args:
            state: Current schedule state
            
        Returns:
            Dictionary with adaptive constraint parameters
        """
        k = state.k
        K = state.K
        
        # Compute progress t_k from SNR (t_k: 1 = initial noise, 0 = final)
        t_k = self.diffusion_schedule.get_progress(k, self.snr_min, self.snr_max)
        
        # Compute unified tightening progress s_k = 1 - t_k
        # s_k: 0 = initial (high noise), 1 = final (low noise)
        # When t_k decreases from 1→0, s_k increases from 0→1
        s_k = 1.0 - t_k
        s_k = np.clip(s_k, 0.0, 1.0)
        
        # Check terminal hard region: last 1-3 steps (force gate ON + final projection)
        # Use last 3 steps or 10% of total steps, whichever is smaller
        k_final = min(3, max(1, int(0.1 * K))) if K > 0 else 1
        is_terminal_hard = k < k_final
        
        # Compute all parameters using unified s_k with continuous cosine annealing interpolation
        # All parameters use continuous interpolation (no "if gate then max else min")
        q_star = self._compute_target_feasibility(s_k)
        
        # Get current q_tilde (synthesized feasibility, from last feedback update)
        # Use q_tilde instead of q_hat for parameter modulation (Step 5)
        # Important: q_tilde is updated in _update_dual_variable AFTER constraint_params is called
        # So we need to use the last updated value, or estimate from current q_hat if not available
        q_tilde = getattr(self, 'q_tilde', None)
        if q_tilde is None:
            # First step or not yet updated: estimate q_tilde from q_hat
            # For first step, use q_hat directly (no improvement term yet)
            q_hat = getattr(self, 'q_hat', self.q_star_min)
            q_tilde = q_hat
        else:
            q_tilde = float(q_tilde)
        
        # Fix A: Remove duplicate blend - only blend once
        # Ensure q_tilde reflects environment difficulty: if q_hat is low, q_tilde should also be low
        # This helps differentiate between different environments
        q_hat_current = getattr(self, 'q_hat', self.q_star_min)
        # Blend q_tilde with current q_hat to ensure it reflects current environment state
        # Use a moderate weight to ensure environment adaptation while avoiding sudden jumps
        # Make env_weight adjustable (can be related to variance/sample size in future)
        env_weight = 0.25  # Weight for current q_hat to reflect environment
        q_tilde = (1.0 - env_weight) * q_tilde + env_weight * q_hat_current
        q_tilde = float(np.clip(q_tilde, 0.0, 1.0))
        
        # Compute parameters with feedback modulation using q_tilde (Solution 2, Step 5)
        # Use error e_k = q*_k - q̃_k to modulate gate/ρ/I_QP
        e_k = q_star - q_tilde  # Tracking error
        rho_k = self._compute_rho(s_k, q_star, q_tilde)
        eps_base = self._compute_solver_tolerance(s_k)
        # Fix 2.1: Apply exp modulation to eps_k: ε_k = ε_base * exp(-c_ε * e_k)
        # Fix A3: Reduce gain (similar to c_rho)
        c_eps = 0.3  # Fix A3: Reduced modulation strength (from 1.0 to 0.3)
        eps_k = eps_base * np.exp(-c_eps * e_k)
        eps_k = float(np.clip(eps_k, self.eps_min, self.eps_max))
        I_QP_k = self._compute_qp_iterations(s_k, q_star, q_tilde)
        K_c_k = self._compute_active_set_size(s_k, q_star, q_tilde)
        
        # Compute gate probability based on feasibility gap with feedback modulation
        p_k = self._compute_gate_probability(s_k, q_star, q_tilde)
        
        # Fix 2.3: Terminal feasibility guarantee - force deterministic final projection
        # For last few steps (k <= k_final), enforce: p_k = 1, I_QP = I_max, eps = eps_min
        if is_terminal_hard:
            p_k = 1.0
            I_QP_k = self.I_max  # Force maximum QP iterations
            eps_k = self.eps_min  # Force minimum tolerance
            # rho_k and K_c_k can remain continuous or be forced to max (keeping continuous for now)
        
        # Draw gate (stochastic) - use deterministic approach for reproducibility
        # Use a hash-based seed combining step k and a fixed offset to ensure
        # same gate decision for same step across runs, while maintaining randomness
        # This ensures reproducibility when same seed is used for the overall experiment
        # Use a simple hash of k to create a deterministic but pseudo-random value
        gate_seed = (int(state.k) * 7919 + 12345) % (2**31)  # Prime number for better distribution
        rng_gate = np.random.default_rng(seed=gate_seed)
        gate_k = rng_gate.random() < p_k
        
        # Fix B3: Compute margin_k that tightens with s_k
        # margin_k = margin_min + (margin_max - margin_min) * CosAnneal(s_k)
        # Use robot_radius as margin_max, and margin_min = 0.5 * robot_radius
        robot_radius = getattr(self, 'robot_radius', 0.05)  # Get from parent class
        margin_max = robot_radius  # Maximum margin is robot_radius
        margin_min = 0.5 * margin_max  # Minimum margin is half of max
        s_powered = np.power(s_k, self.p_q)  # Reuse p_q for margin schedule
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        margin_k = margin_min + (margin_max - margin_min) * cos_anneal
        margin_k = float(np.clip(margin_k, margin_min, margin_max))
        
        # Fix B3: Assert margin is valid
        if margin_k < 1e-4:
            print(f"WARNING: margin_k={margin_k} is too small (< 1e-4)! Using minimum 1e-4.")
            margin_k = 1e-4
        
        return {
            "rho": float(rho_k),
            "topK": int(K_c_k),
            "eps": float(eps_k),
            "I_QP": int(I_QP_k),
            "qp_gate": bool(gate_k),
            "qp_prob": float(p_k),
            "margin": float(margin_k),  # Fix B3: Return computed margin_k
            "_extra": {
                "t_k": float(t_k),
                "s_k": float(s_k),  # Add s_k to extra for visualization
                "is_terminal_hard": is_terminal_hard,
                "lambda_con": float(self.lambda_con),
                "q_star": float(q_star),
                "q_hat": float(getattr(self, 'q_hat', self.q_star_min)),  # Add q_hat to extra for visualization
                "q_tilde": float(getattr(self, 'q_tilde', self.q_star_min)),  # Add q_tilde for visualization
                "V_k": float(getattr(self, 'V_k', 0.0)),  # Add V_k for visualization
            }
        }
    
    def _compute_target_feasibility(self, s_k: float) -> float:
        """
        Compute target feasibility q*(s_k) using cosine annealing.
        
        Fix C: Make q* reachable by using environment-aware reference governor.
        q*(s) = min(q*_sched(s_k), q_max + delta)
        where q_max is the 90th percentile of observed q values.
        
        This prevents q* from being set too high for the current environment,
        avoiding meaningless saturation of lambda, rho, p.
        
        Args:
            s_k: Tightening progress in [0, 1] (0 = initial, 1 = final)
            
        Returns:
            Target feasibility rate (reachable for current environment)
        """
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        q_star_sched = self.q_star_min + self._q_range * cos_anneal
        
        # Fix C: Environment-aware reference governor
        # Get q_max from sliding window (90th percentile)
        q_max = None
        if hasattr(self, 'q_max_history') and len(self.q_max_history) >= 5:
            q_max = float(np.percentile(self.q_max_history, 90))
        
        if q_max is not None:
            # Set q*_k = min(q*_sched(s_k), q_max + delta)
            q_max_upper = q_max + getattr(self, 'q_max_delta', 0.1)
            q_star_sched = min(q_star_sched, q_max_upper)
        
        return float(q_star_sched)
    
    def _compute_rho(self, s_k: float, q_star: float, q_tilde: float) -> float:
        """
        Compute slack penalty ρ_k with environmental adaptation (Solution 2).
        
        Baseline schedule: ρ_base(s_k) = ρ_min + (ρ_max - ρ_min) * CosAnneal(s_k)
        Feedback modulation: ρ_k = ρ_base(s_k) * (1 + c_ρ * clip((q* - q̂) / q*, 0, 1))
        
        If actual feasibility q_hat < q* (difficult environment/current trajectory difficult),
        automatically increase penalty.
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1]
            q_star: Target feasibility q*(s_k)
            q_hat: Current feasibility rate q̂
        
        Returns:
            Slack penalty value
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        rho_base = self.rho_min + self._rho_range * cos_anneal
        
        # Fix E: Change from exponential modulation to log-domain linear + soft saturation
        # Original: rho_k = rho_base * exp(c_rho * e_k) - saturates too early
        # New: log rho_k = log rho_base + softsat(c_rho * e_k)
        # Or: rho_k = rho_base * (1 + kappa * tanh(c_rho * e_k))
        c_rho = 0.3  # Modulation strength
        e_k = q_star - q_tilde  # Tracking error (positive when q_tilde < q_star)
        
        # Fix E: Use tanh for soft saturation (preserves monotonicity without early saturation)
        kappa = 0.5  # Saturation scale (can be tuned)
        rho_k = rho_base * (1.0 + kappa * np.tanh(c_rho * e_k))
        
        # Clip to bounds (soft saturation should prevent this, but keep for safety)
        return float(np.clip(rho_k, self.rho_min, self.rho_max))
    
    def _compute_gate_probability(self, s_k: float, q_star: float, q_tilde: float) -> float:
        """
        Compute QP gate probability p_k with environmental adaptation (Solution 2).
        
        Baseline schedule: p_base(s_k) = p_min + (p_max - p_min) * CosAnneal(s_k)
        Feedback modulation: p_k = clip(p_base(s_k) + c_p * (q* - q̃), 0, 1)
        
        The less the target feasibility is met (q_tilde < q*), the more inclined to enable QP.
        
        Performance: O(1) with fast cosine annealing.
        
        Args:
            s_k: Tightening progress in [0, 1] (0 = initial, 1 = final)
            q_star: Target feasibility q*(s_k)
            q_tilde: Current smoothed feasibility q̃ (from backend EMA)
        
        Returns:
            Gate probability in [0, 1]
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        p_min = 0.3
        p_max = 0.95
        p_base = p_min + (p_max - p_min) * cos_anneal
        
        # Fix D: Late-stage determinism - for last 30% of diffusion steps, set gate=ON
        # Check if we're in late stage (s_k >= 0.7 means last 30% of steps)
        if s_k >= 0.7:
            # Late stage: deterministic gate ON
            return 1.0
        
        # Early stage: use adaptive modulation
        c_p = 0.5  # Modulation strength
        feasibility_gap = q_star - q_tilde
        p_k = p_base + c_p * feasibility_gap
        
        # Clip to [0, 1]
        return float(np.clip(p_k, 0.0, 1.0))
    
    def _compute_active_set_size(self, s_k: float, q_star: float, q_tilde: float) -> int:
        """
        Compute active-set size K_c(k) with environmental adaptation (Solution 2).
        
        Baseline schedule: K_base(s_k) = K_min + (K_max - K_min) * CosAnneal(s_k)
        Feedback modulation: K_c(k) = K_base(s_k) + g_k * K_boost (continuous modulation)
        
        Fix A3: Uses continuous modulation g_k instead of hard switch.
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1]
            q_star: Target feasibility q*(s_k)
            q_tilde: Current smoothed feasibility q̃ (from backend EMA)
            
        Returns:
            Active constraint count
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        K_base = self.K_min + int(np.floor(self._K_range * cos_anneal))
        
        # Fix A3: Replace hard switch with continuous modulation
        # Change indicator = 1 if q_tilde < q_star else 0 to continuous g_k
        delta = 0.2  # Threshold for continuous transition
        g_k = np.clip((q_star - q_tilde) / delta, 0.0, 1.0)  # Continuous modulation signal
        K_boost = max(1, int(0.1 * self.K_max))  # Modulation amount (10% of max, at least 1)
        # Fix A3: Remove hard switch (cK=10), use continuous modulation
        K_c = K_base + int(np.floor(g_k * K_boost))  # Apply continuous modulation
        
        # Clip to bounds
        return int(np.clip(K_c, self.K_min, self.K_max))
    
    def _compute_solver_tolerance(self, s_k: float) -> float:
        """
        Compute solver tolerance ε(s_k) using cosine annealing (decreasing).
        
        ε(s) = ε_max - (ε_max - ε_min) * CosAnneal(s; p_eps)
        
        This makes tolerance decrease as s_k increases (tighter tolerance at later steps).
        When s_k=0 (initial): ε = ε_max (looser)
        When s_k=1 (final): ε = ε_min (tighter)
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1]
            
        Returns:
            Solver tolerance
        """
        s_powered = np.power(s_k, self.p_eps)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        eps_base = self.eps_max - self._eps_range * cos_anneal
        
        # Fix 2.1: Close the loop - modulate by tracking error e_k (log-domain)
        # Use exp form: ε_k = ε_base * exp(-c_ε * e_k)
        # Note: This method doesn't have access to q_tilde directly, so we return base value
        # The modulation will be applied in _compute_constraint_params where we have q_tilde
        return eps_base
    
    def _compute_qp_iterations(self, s_k: float, q_star: float, q_tilde: float) -> int:
        """
        Compute QP iterations I_QP(k) with environmental adaptation (Solution 2).
        
        Baseline schedule: I_base(s_k) = I_min + (I_max - I_min) * CosAnneal(s_k)
        Feedback modulation: I_QP(k) = I_base(s_k) + c_I * 1[q̂ < q*]
        
        If q_hat < q*, add extra QP iterations.
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1]
            q_star: Target feasibility q*(s_k)
            q_hat: Current feasibility rate q̂
            
        Returns:
            Maximum QP iterations
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        I_base = self.I_min + int(np.floor(self._I_range * cos_anneal))
        
        # Fix A3: Replace hard switch with continuous modulation
        # Change indicator = 1 if q_tilde < q_star else 0 to continuous g_k
        delta = 0.2  # Threshold for continuous transition
        g_k = np.clip((q_star - q_tilde) / delta, 0.0, 1.0)  # Continuous modulation signal
        I_boost = max(1, int(0.1 * self.I_max))  # Modulation amount (10% of max, at least 1)
        # Fix A3: Remove hard switch (cI=10), use continuous modulation
        I_QP = I_base + int(np.floor(g_k * I_boost))  # Apply continuous modulation
        
        # Clip to bounds
        return int(np.clip(I_QP, self.I_min, self.I_max))
    
    def _update_dual_variable(self, state: ScheduleState, feedback: Dict[str, Any]) -> None:
        """
        Update dual variable λ^con using Robbins-Monro style update.
        
        λ_{k+1} = clip(λ_k + η_k * (q*(s_k) - q̂_k), [λ_min, λ_max])
        
        Note: λ^con is a signed feasibility tracking variable (not a standard 
        Lagrange multiplier). It can be negative when q_hat > q* (actual feasibility 
        exceeds target), and positive when q_hat < q* (actual feasibility below target).
        This allows adaptive control in both directions.
        
        Performance: O(1) with efficient clipping.
        
        Args:
            state: Current schedule state
            feedback: Dictionary with "feasible_rate" or "q_hat" key
        """
        k = state.k
        K = state.K
        
        # Compute progress t_k from SNR
        t_k = self.diffusion_schedule.get_progress(k, self.snr_min, self.snr_max)
        
        # Compute unified tightening progress s_k = 1 - t_k
        s_k = 1.0 - t_k
        s_k = np.clip(s_k, 0.0, 1.0)
        
        # Compute target feasibility using s_k
        q_star = self._compute_target_feasibility(s_k)
        
        # Fix A1: Get q_hat and q_tilde from feedback (clarified semantics)
        # q_hat: Current step observation (batch aggregation)
        # q_tilde: Smoothed observation (EMA from backend)
        q_hat = feedback.get("q_hat", feedback.get("feasible_rate", None))  # Prefer q_hat over feasible_rate
        q_tilde = feedback.get("q_tilde", None)  # Get q_tilde from backend (EMA)
        V_k = feedback.get("V_k", None)
        
        if q_hat is None:
            # Try to compute from trajectories if provided
            trajectories = feedback.get("trajectories", [])
            if trajectories:
                q_hat = self._estimate_feasibility_rate(trajectories, feedback)
                # If V_k not provided, estimate from q_hat: V_k = -log(q_hat) / alpha
                if V_k is None:
                    alpha = 10.0  # Same as in _compute_feasibility_rate
                    V_k = -np.log(max(q_hat, 1e-10)) / alpha
            else:
                # No feedback available - skip update
                return
        
        # Fix A2: q_tilde is provided by backend (EMA), scheduler should not re-synthesize it
        # If q_tilde is not provided, fallback to q_hat (for backward compatibility)
        if q_tilde is None:
            q_tilde = q_hat  # Fallback: use q_hat if q_tilde not available
        
        # Ensure V_k is available
        if V_k is None:
            # Estimate V_k from q_hat: V_k = -log(q_hat) / alpha
            alpha = 10.0
            V_k = -np.log(max(q_hat, 1e-10)) / alpha
        
        # Store V_k for next iteration (for diagnostics, not for re-synthesis)
        self.V_k_prev = getattr(self, 'V_k', V_k)
        self.V_k = float(V_k)
        
        # Fix B: Two time scales + uncertainty-based step size + true anti-windup
        # 1. Track q variance for uncertainty-based step size (Optimization 3: use deque)
        if not hasattr(self, 'q_history'):
            q_window_size = getattr(self, 'q_window_size', 10)
            self.q_history = deque(maxlen=q_window_size)
        self.q_history.append(float(q_tilde))
        
        # Compute q variance (uncertainty measure)
        q_var = 0.0
        if len(self.q_history) >= 3:
            q_var = float(np.var(self.q_history))
        
        # Fix B: Uncertainty-based step size - reduce eta_k when variance is large
        # Base step size with decay
        eta_k_base = self.eta_con_base * np.power(self.eta_con_decay, self.step_count)
        
        # Reduce step size when uncertainty is high (large variance, small batch, difficult environment)
        # Use sigmoid to smoothly reduce eta_k: eta_k = eta_k_base * (1 - sigmoid(alpha * q_var))
        alpha_var = 10.0  # Scaling factor for variance (can be tuned)
        uncertainty_factor = 1.0 / (1.0 + np.exp(alpha_var * q_var - 5.0))  # Sigmoid: 1 when var=0, ~0 when var large
        eta_k = eta_k_base * (0.3 + 0.7 * uncertainty_factor)  # Scale between 0.3 and 1.0 of base
        
        # Fix B: Two time scales - make lambda update slower (already done via eta_k reduction)
        # Actuators (rho, p) respond faster via direct modulation in _compute_rho/_compute_gate_probability
        
        # Update dual variable using q_tilde from backend (Fix A2: use backend-provided q_tilde)
        error = q_star - q_tilde
        lambda_new = self.lambda_con + eta_k * error
        
        # Fix B: True anti-windup - back-calculation (feed saturated error back into integrator)
        lambda_clipped = float(np.clip(lambda_new, self.lambda_con_min, self.lambda_con_max))
        
        # True anti-windup: if saturated and error pushes in same direction, freeze integral term
        if (lambda_clipped == self.lambda_con_max and error > 0) or \
           (lambda_clipped == self.lambda_con_min and error < 0):
            # Saturated and error pushes further - freeze lambda (don't update)
            # This is true anti-windup: prevent integrator from accumulating error when saturated
            pass  # Keep lambda_con unchanged
        else:
            # Not saturated or error pushes away from saturation - update normally
            self.lambda_con = lambda_clipped
        
        # Fix C: Track q_max for environment-aware reference governor (Optimization 3: use deque)
        if not hasattr(self, 'q_max_history'):
            q_max_window_size = getattr(self, 'q_max_window_size', 20)
            self.q_max_history = deque(maxlen=q_max_window_size)
        self.q_max_history.append(float(q_tilde))
        
        self.step_count += 1
        
        # Store values for next iteration
        self.q_hat = float(q_hat)
        self.q_tilde = float(q_tilde)
        self.V_k_prev = float(V_k)  # Store current V_k as previous for next step
        self.V_k = float(V_k)
        self.q_hat_prev = float(q_hat)
    
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

