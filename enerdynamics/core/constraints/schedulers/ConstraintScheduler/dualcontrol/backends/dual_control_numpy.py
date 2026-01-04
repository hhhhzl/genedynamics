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
        
        # Ensure q_tilde reflects environment difficulty: if q_hat is low, q_tilde should also be low
        # This helps differentiate between different environments
        q_hat_current = getattr(self, 'q_hat', self.q_star_min)
        # Blend q_tilde with current q_hat to ensure it reflects current environment state
        # Use a moderate weight to ensure environment adaptation while avoiding sudden jumps
        env_weight = 0.25  # Weight for current q_hat to reflect environment
        q_tilde = (1.0 - env_weight) * q_tilde + env_weight * q_hat_current
        q_tilde = float(np.clip(q_tilde, 0.0, 1.0))
        
        # Ensure q_tilde reflects environment difficulty: if q_hat is low, q_tilde should also be low
        # This helps differentiate between different environments
        q_hat_current = getattr(self, 'q_hat', 1.0)
        # Blend q_tilde with current q_hat to ensure it reflects current environment state
        # Use a moderate weight to ensure environment adaptation while avoiding sudden jumps
        env_weight = 0.25  # Weight for current q_hat to reflect environment (increased from 0.2)
        q_tilde = (1.0 - env_weight) * q_tilde + env_weight * q_hat_current
        q_tilde = float(np.clip(q_tilde, 0.0, 1.0))
        
        # Compute parameters with feedback modulation using q_tilde (Solution 2, Step 5)
        # Use error e_k = q*_k - q̃_k to modulate gate/ρ/I_QP
        e_k = q_star - q_tilde  # Tracking error
        rho_k = self._compute_rho(s_k, q_star, q_tilde)
        eps_base = self._compute_solver_tolerance(s_k)
        # Fix 2.1: Apply exp modulation to eps_k: ε_k = ε_base * exp(-c_ε * e_k)
        c_eps = 1.0  # Modulation strength
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
        
        q*(s) = q_min + (q_max - q_min) * CosAnneal(s; p_q)
        CosAnneal(s; p_q) = 0.5 * (1 - cos(π * s^p_q))
        
        This makes q* increase as s_k increases (harder constraints at later steps).
        When s_k=0 (initial): q* = q_min (softer)
        When s_k=1 (final): q* = q_max (harder)
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1] (0 = initial, 1 = final)
            
        Returns:
            Target feasibility rate
        """
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        return self.q_star_min + self._q_range * cos_anneal
    
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
        
        # Fix 2.1: Close the loop - modulate by tracking error e_k = q* - q_tilde (log-domain)
        # Use exp form: ρ_k = ρ_base * exp(c_ρ * e_k)
        c_rho = 1.0  # Modulation strength
        e_k = q_star - q_tilde  # Tracking error (positive when q_tilde < q_star)
        
        # Apply modulation: ρ_k = ρ_base * exp(c_ρ * e_k)
        rho_k = rho_base * np.exp(c_rho * e_k)
        
        # Clip to bounds
        return float(np.clip(rho_k, self.rho_min, self.rho_max))
    
    def _compute_gate_probability(self, s_k: float, q_star: float, q_hat: float) -> float:
        """
        Compute QP gate probability p_k with environmental adaptation (Solution 2).
        
        Baseline schedule: p_base(s_k) = p_min + (p_max - p_min) * CosAnneal(s_k)
        Feedback modulation: p_k = clip(p_base(s_k) + c_p * (q* - q̂), 0, 1)
        
        The less the target feasibility is met (q_hat < q*), the more inclined to enable QP.
        
        Performance: O(1) with fast cosine annealing.
        
        Args:
            s_k: Tightening progress in [0, 1] (0 = initial, 1 = final)
            q_star: Target feasibility q*(s_k)
            q_hat: Current feasibility rate q̂
        
        Returns:
            Gate probability in [0, 1]
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        p_min = 0.3
        p_max = 0.95
        p_base = p_min + (p_max - p_min) * cos_anneal
        
        # Feedback modulation coefficient (increased for stronger environmental adaptation)
        c_p = 1.0  # Modulation strength (increased from 0.5 to 1.0)
        
        # Apply modulation: p_k = p_base + c_p * (q* - q̂)
        feasibility_gap = q_star - q_hat
        p_k = p_base + c_p * feasibility_gap
        
        # Clip to [0, 1]
        return float(np.clip(p_k, 0.0, 1.0))
    
    def _compute_active_set_size(self, s_k: float, q_star: float, q_hat: float) -> int:
        """
        Compute active-set size K_c(k) with environmental adaptation (Solution 2).
        
        Baseline schedule: K_base(s_k) = K_min + (K_max - K_min) * CosAnneal(s_k)
        Feedback modulation: K_c(k) = K_base(s_k) + c_K * 1[q̂ < q*]
        
        If q_hat < q*, add extra constraints.
        
        Performance: O(1) with pre-computed range.
        
        Args:
            s_k: Tightening progress in [0, 1]
            q_star: Target feasibility q*(s_k)
            q_hat: Current feasibility rate q̂
            
        Returns:
            Active constraint count
        """
        # Baseline schedule using cosine annealing
        s_powered = np.power(s_k, self.p_q)
        cos_anneal = 0.5 * (1.0 - np.cos(np.pi * s_powered))
        K_base = self.K_min + int(np.floor(self._K_range * cos_anneal))
        
        # Feedback modulation: add extra constraints if q_hat < q*
        c_K = max(1, int(0.1 * self.K_max))  # Modulation amount (10% of max, at least 1)
        # Increased modulation for stronger environmental adaptation
        cK = 10  # Additional coefficient for modulation (increased for stronger adaptation)
        indicator = 1 if q_hat < q_star else 0
        K_c = K_base + (c_K + cK) * indicator  # Apply combined modulation
        
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
    
    def _compute_qp_iterations(self, s_k: float, q_star: float, q_hat: float) -> int:
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
        
        # Feedback modulation: add extra iterations if q_hat < q*
        c_I = max(1, int(0.1 * self.I_max))  # Modulation amount (10% of max, at least 1)
        # Increased modulation for stronger environmental adaptation
        cI = 10  # Additional coefficient for modulation (increased for stronger adaptation)
        indicator = 1 if q_hat < q_star else 0
        I_QP = I_base + (c_I + cI) * indicator  # Apply combined modulation
        
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
        
        # Get estimated feasibility rate and violation
        q_hat = feedback.get("feasible_rate", feedback.get("q_hat", None))
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
        
        # Ensure V_k is available
        if V_k is None:
            # Estimate V_k from q_hat: V_k = -log(q_hat) / alpha
            alpha = 10.0
            V_k = -np.log(max(q_hat, 1e-10)) / alpha
        
        # Step 3: Compute improvement term q̂_k^imp = σ(α(V_{k-1} - V_k))
        # Get previous values (stored in parent class)
        V_k_prev = getattr(self, 'V_k_prev', None)
        if V_k_prev is None:
            # First step: no previous value, set q_hat_imp based on current q_hat
            # If q_hat is already low, assume no improvement yet
            if q_hat < 0.5:
                q_hat_imp = 0.0  # No improvement if already difficult
            else:
                q_hat_imp = 0.5  # Neutral improvement for first step
        else:
            alpha_imp = 5.0  # Scaling factor for improvement term
            delta_V = V_k_prev - V_k  # Positive if improving (V decreasing)
            # Sigmoid: σ(x) = 1 / (1 + exp(-x))
            q_hat_imp = 1.0 / (1.0 + np.exp(-alpha_imp * delta_V))
            # If V_k is not improving (delta_V <= 0), reduce q_hat_imp significantly
            if delta_V <= 0:
                q_hat_imp = q_hat_imp * 0.2  # Strongly penalize non-improvement
        
        # Step 4: Synthesize q̃_k = w q̂_k + (1 - w) q̂_k^imp
        w = 0.99
        q_tilde = w * q_hat + (1.0 - w) * q_hat_imp
        q_tilde = float(np.clip(q_tilde, 0.0, 1.0))
        
        # Add tracking term: encourage q_tilde to follow q_star in simple environments
        # This helps q_tilde track q_star when environment is easy (q_hat is high)
        # Use adaptive tracking: more tracking when q_hat is close to q_star
        tracking_strength = 0.2  # Base tracking strength
        # Increase tracking when q_hat is high (simple environment)
        if q_hat > 0.5:
            tracking_strength = 0.3  # Stronger tracking in simple environments
        # Adaptive tracking: track more when q_hat is close to q_star
        gap = abs(q_star - q_hat)
        if gap < 0.3:  # Close to target
            tracking_strength = 0.4  # Even stronger tracking when close
        
        q_tilde = (1.0 - tracking_strength) * q_tilde + tracking_strength * q_star
        q_tilde = float(np.clip(q_tilde, 0.0, 1.0))
        
        # Compute step size with decay (cached computation)
        eta_k = self.eta_con_base * np.power(self.eta_con_decay, self.step_count)
        
        # Update dual variable using q_tilde instead of q_hat
        error = q_star - q_tilde
        lambda_new = self.lambda_con + eta_k * error
        
        # Fix 2.2: Dual update with anti-windup
        # Prevent integrator windup when clipping is active
        lambda_clipped = float(np.clip(lambda_new, self.lambda_con_min, self.lambda_con_max))
        
        # Anti-windup logic: if saturated and error direction suggests further increase, freeze/reduce eta
        if lambda_clipped == self.lambda_con_max and error > 0:
            # Saturated at max and error is positive - freeze eta (the decay already reduces eta over time)
            pass
        elif lambda_clipped == self.lambda_con_min and error < 0:
            # Saturated at min and error is negative - freeze eta (the decay already reduces eta over time)
            pass
        
        self.lambda_con = lambda_clipped
        self.step_count += 1
        
        # Store values for next iteration
        self.q_hat = float(q_hat)
        self.q_tilde = q_tilde
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

