"""
NumPy backend implementation for EDOC reverse diffusion.

This module contains the pure NumPy implementation of the reverse diffusion
algorithm, with no JAX dependencies. All operations use NumPy for maximum compatibility.
"""

from typing import Optional, Tuple, Any, List, Dict
from collections import deque
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from enerdynamics.core.types import Trajectory
from ..backend_impl import EDOCBackendBase

# Import registry decorator
try:
    from enerdynamics.core.registry.edoc_backends import register_edoc_backend
except ImportError:
    # Fallback if registry not available
    def register_edoc_backend(name):
        def decorator(cls):
            return cls
        return decorator


@register_edoc_backend("numpy")
class EDOCBackendNumpy(EDOCBackendBase):
    """
    Pure NumPy backend implementation of EDOC reverse diffusion.
    
    This implementation uses pure NumPy with no JAX dependencies.
    Optimized for batch operations where possible.
    """
    
    def __init__(self, planner: Any, **config: Any):
        """Initialize NumPy backend implementation."""
        super().__init__(planner, **config)
        # Use NumPy random number generator from planner for reproducibility
        # Directly use planner's _np_rng if available, otherwise create a new one
        planner_rng = getattr(planner, '_np_rng', None)
        if planner_rng is not None:
            # Use planner's random number generator directly to ensure same seed
            self._np_rng = planner_rng
        else:
            # Fallback: create new RNG (should not happen if planner is properly initialized)
            self._np_rng = np.random.default_rng()
        
        # Get robot_radius from config or planner (for violation computation)
        self.robot_radius = 0.0
        if 'robot_radius' in config:
            self.robot_radius = float(config['robot_radius'])
        elif hasattr(planner, 'config') and planner.config is not None:
            # Try to get from planner.config.obstacle_config
            if hasattr(planner.config, 'obstacle_config'):
                obstacle_config = planner.config.obstacle_config
                if obstacle_config and 'robot_radius' in obstacle_config:
                    self.robot_radius = float(obstacle_config['robot_radius'])
            elif isinstance(planner.config, dict) and 'obstacle_config' in planner.config:
                obstacle_config = planner.config['obstacle_config']
                if obstacle_config and 'robot_radius' in obstacle_config:
                    self.robot_radius = float(obstacle_config['robot_radius'])
        
        # Initialize EMA state for q_tilde (Fix 1.3: Batch aggregation + EMA)
        self.q_tilde_prev = None  # Previous q_tilde for EMA
        self.ema_beta = 0.3  # EMA smoothing factor (beta in [0,1], can be tuned)
        
        # Fix B1: Initialize sliding window for tau calibration (Optimization 3: use deque)
        self.V_window_size = 50  # Window size for median calculation (can be tuned)
        self.V_history = deque(maxlen=self.V_window_size)  # Sliding window with automatic size limit
        self.tau = 0.1  # Initial tau (will be calibrated online)
    
    def rollout_states_and_energy(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rollout states and compute energy sequence using pure NumPy."""
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        horizon = actions.shape[0]
        state_dim = state.shape[0]
        
        # Pre-allocate arrays for better performance
        states = np.zeros((horizon + 1, state_dim), dtype=np.float32)
        energies = np.zeros((horizon,), dtype=np.float32)
        states[0] = state
        
        # Use states[0] directly to avoid unnecessary copy
        x = states[0]
        for t in range(horizon):
            act = actions[t]
            # Apply action filter if needed
            act_safe = self._apply_action_filter_numpy(x, act, hard_clearance, hard_enabled)
            
            # Compute energy
            ctx = {"t": t}
            e_val = float(self.energy.compute(x, act_safe, ctx))
            energies[t] = e_val
            
            # State transition
            x = self.env.transition(x, act_safe)
            states[t + 1] = x  # Direct assignment instead of append
        
        return states, energies
    
    def rollout_env_states(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> np.ndarray:
        """Rollout environment states using pure NumPy."""
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        horizon = actions.shape[0]
        state_dim = state.shape[0]
        
        # Pre-allocate arrays for better performance
        states = np.zeros((horizon + 1, state_dim), dtype=np.float32)
        states[0] = state
        
        x = state  # Use reference, no need to copy since we're modifying in-place
        for t, act in enumerate(actions):
            act_safe = self._apply_action_filter_numpy(x, act, hard_clearance, hard_enabled)
            x = self.env.transition(x, act_safe)
            states[t + 1] = x  # Direct assignment instead of append
        
        return states
    
    def rollout_env_states_batch(
        self,
        state_init: np.ndarray,
        batch_actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> np.ndarray:
        """
        Batch rollout environment states (Optimization 2).
        
        Args:
            state_init: Initial state, shape (state_dim,)
            batch_actions: Batch of actions, shape (M, H, act_dim)
            hard_clearance: Hard constraint clearance
            hard_enabled: Whether hard constraints are enabled
            
        Returns:
            Batch of state trajectories, shape (M, H+1, state_dim)
        """
        state_init = np.asarray(state_init, dtype=np.float32)
        batch_actions = np.asarray(batch_actions, dtype=np.float32)
        M, H, act_dim = batch_actions.shape
        state_dim = state_init.shape[0]
        
        # Pre-allocate arrays
        batch_states = np.zeros((M, H + 1, state_dim), dtype=np.float32)
        batch_states[:, 0] = state_init
        
        # Batch rollout (vectorized where possible)
        for t in range(H):
            x_batch = batch_states[:, t]  # (M, state_dim)
            u_batch = batch_actions[:, t]  # (M, act_dim)
            
            # Apply action filter if needed (batch version if available)
            if hard_enabled and self.constraint_manager:
                # Try batch filter, fallback to loop
                if hasattr(self.constraint_manager, 'action_filter_operator'):
                    action_filter = self.constraint_manager.action_filter_operator
                    # Most filters don't support batch, so we loop
                    for m in range(M):
                        try:
                            filtered_act = action_filter.filter(x_batch[m], u_batch[m], hard_clearance)
                            if filtered_act is not None:
                                u_batch[m] = np.asarray(filtered_act, dtype=np.float32)
                        except Exception:
                            pass
            
            # State transition (env.transition doesn't support batch, so we loop)
            for m in range(M):
                batch_states[m, t+1] = self.env.transition(x_batch[m], u_batch[m])
        
        return batch_states
    
    def compute_total_energy(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> float:
        """Compute total energy including terminal and dynamics loss."""
        states_full, energy_seq = self.rollout_states_and_energy(
            state, actions, hard_clearance, hard_enabled
        )
        total = float(np.sum(energy_seq))
        
        # Terminal energy
        if self.terminal_energy_weight > 0.0:
            u0 = np.zeros((self.env.act_dim,), dtype=np.float32)
            ctxT = {"t": int(len(actions))}
            total += float(self.terminal_energy_weight) * float(
                self.energy.compute(states_full[-1], u0, ctxT)
            )
        
        # Dynamics loss
        if self.dyn_loss_coeff > 0.0:
            env_states_full = self.rollout_env_states(state, actions, hard_clearance, hard_enabled)
            if self.dyn_loss_mode == "terminal":
                diff = states_full[-1] - env_states_full[-1]
                dyn_loss = float(np.sum(diff * diff))
            else:
                dyn_loss = float(np.sum((states_full - env_states_full) ** 2))
            total += self.dyn_loss_coeff * dyn_loss
        
        return total
    
    def compute_mean_reward(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> float:
        """Compute mean reward over trajectory."""
        reward_cost_fn = getattr(self.env, "cost", None)
        if reward_cost_fn is None:
            return 0.0
        
        x = np.asarray(state, dtype=np.float32).copy()
        horizon = len(actions)
        # Pre-allocate array for rewards
        rewards = np.empty(horizon, dtype=np.float32)
        
        for i, act in enumerate(actions):
            act_safe = self._apply_action_filter_numpy(x, act, hard_clearance, hard_enabled)
            x = self.env.transition(x, act_safe)
            rewards[i] = -float(reward_cost_fn(x))
        
        return float(np.mean(rewards))
    
    def score_particles(
        self,
        state_init: np.ndarray,
        batch_actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
    ) -> np.ndarray:
        """Score action particles in batch using pure NumPy."""
        mode = self.action_score_mode.lower()
        num = batch_actions.shape[0]
        
        # Get constraint parameters
        hard_enabled = False
        hard_clearance = 0.0
        if self.constraint_manager:
            if self.constraint_manager.action_filter_operator is not None:
                hard_enabled = True
            if self.constraint_manager.schedule_manager is not None:
                hard_enabled = bool(
                    self.constraint_manager.schedule_manager.is_hard_active(step, total_steps)
                )
                hard_clearance = float(
                    self.constraint_manager.schedule_manager.get_hard_clearance(
                        default=0.0, step=step, total_steps=total_steps
                    )
                )
        
        # Batch compute scores
        scores = np.zeros((num,), dtype=np.float32)
        
        if mode == "reward":
            # Batch rollout for rewards
            for idx in range(num):
                scores[idx] = self.compute_mean_reward(
                    state_init, batch_actions[idx], hard_clearance, hard_enabled
                )
        elif mode == "energy":
            # Batch rollout for energy
            for idx in range(num):
                scores[idx] = -self.compute_total_energy(
                    state_init, batch_actions[idx], hard_clearance, hard_enabled
                )
        elif mode == "learned":
            scores[:] = 0.0
        else:
            raise ValueError(f"Unknown action_score_mode {self.action_score_mode}")
        
        # Add soft constraint penalties if enabled
        if self.use_constraint_in_scoring and self.constraint_manager and self.constraint_manager.has_soft():
            scores = scores / self.lambda_energy
            # Batch compute trajectories (Optimization 4: use batch function)
            trajectories = self.actions_to_trajectory_batch(state_init, batch_actions)
            soft_penalties = self.constraint_manager.compute_soft_energy_batch(
                trajectories, step=step, total_steps=total_steps
            )
            scores = scores - soft_penalties
        
        return scores
    
    def actions_to_trajectory(
        self,
        x0: np.ndarray,
        actions: np.ndarray,
    ) -> Trajectory:
        """Convert actions array to Trajectory."""
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            actions_array = actions_array[None, :]
        
        x0_arr = np.asarray(x0, dtype=np.float32)
        state_dim = x0_arr.shape[0]
        horizon = actions_array.shape[0]
        
        # Pre-allocate arrays for better performance
        states = np.zeros((horizon + 1, state_dim), dtype=np.float32)
        states[0] = x0_arr
        actions_list = []
        
        x = x0_arr.copy()
        for t, act in enumerate(actions_array):
            actions_list.append(act.copy())  # Keep copy for Trajectory
            x = self.env.transition(x, act)
            states[t + 1] = x  # Direct assignment instead of append
        
        # Convert to list format for Trajectory (required by interface)
        states_list = [states[i] for i in range(horizon + 1)]
        return Trajectory(states=states_list, actions=actions_list)
    
    def actions_to_trajectory_batch(
        self,
        x0: np.ndarray,
        batch_actions: np.ndarray,
    ) -> List[Trajectory]:
        """
        Batch convert actions to trajectories (Optimization 4).
        
        Args:
            x0: Initial state, shape (state_dim,)
            batch_actions: Batch of actions, shape (M, H, act_dim)
            
        Returns:
            List of M Trajectory objects
        """
        batch_actions = np.asarray(batch_actions, dtype=np.float32)
        if batch_actions.ndim == 2:
            batch_actions = batch_actions[None, :, :]  # Add batch dimension
        
        M, H, act_dim = batch_actions.shape
        x0_arr = np.asarray(x0, dtype=np.float32)
        state_dim = x0_arr.shape[0]
        
        # Use batch rollout for efficiency
        batch_states = self.rollout_env_states_batch(
            x0_arr, batch_actions, hard_clearance=0.0, hard_enabled=False
        )
        
        # Convert to Trajectory list
        trajectories = []
        for m in range(M):
            states_list = [batch_states[m, t] for t in range(H + 1)]
            actions_list = [batch_actions[m, t].copy() for t in range(H)]
            trajectories.append(Trajectory(states=states_list, actions=actions_list))
        
        return trajectories
    
    def extract_actions_from_trajectory(
        self,
        trajectory: Trajectory,
    ) -> np.ndarray:
        """Extract actions array from trajectory."""
        if not trajectory.actions:
            return np.zeros((self.horizon, self.env.act_dim), dtype=np.float32)
        return np.stack([np.asarray(act, dtype=np.float32) for act in trajectory.actions], axis=0)
    
    def track_actions_to_projected_states(
        self,
        x0: np.ndarray,
        actions_init: np.ndarray,
        target_states: List[np.ndarray],
        gn_iters: int = 2,
        reg: float = 1e-3,
    ) -> np.ndarray:
        """Map projected state trajectory back to actions using numerical differentiation."""
        actions = np.asarray(actions_init, dtype=np.float32).copy()
        if actions.ndim != 2:
            return actions
        
        if not target_states or len(target_states) < 2:
            return actions
        
        horizon = actions.shape[0]
        T = min(horizon, len(target_states) - 1)
        if T <= 0:
            return actions
        
        limit = getattr(self.env, "control_limit", None)
        limit_val = float(limit) if limit is not None else None
        
        x = np.asarray(x0, dtype=np.float32).copy()
        eps = 1e-5  # Numerical differentiation step size
        
        for t in range(T):
            u = actions[t].copy()
            x_tgt = np.asarray(target_states[t + 1], dtype=np.float32)
            
            for _ in range(max(1, int(gn_iters))):
                # Predict next state
                x_pred = self.env.transition(x, u)
                
                # Compute Jacobian using numerical differentiation
                act_dim = u.shape[0]
                state_dim = x.shape[0]
                B = np.zeros((state_dim, act_dim), dtype=np.float32)
                
                for i in range(act_dim):
                    u_pert = u.copy()
                    u_pert[i] += eps
                    x_pert = self.env.transition(x, u_pert)
                    B[:, i] = (x_pert - x_pred) / eps
                
                # Use position part only (first 2 dimensions typically)
                if state_dim >= 2:
                    x_pred_pos = x_pred[:2]
                    x_tgt_pos = x_tgt[:2]
                    B_pos = B[:2, :]
                else:
                    x_pred_pos = x_pred
                    x_tgt_pos = x_tgt
                    B_pos = B
                
                # Gauss-Newton step
                pos_err = x_tgt_pos - x_pred_pos
                
                # Check for NaN/inf in position error before solving
                if not np.all(np.isfinite(pos_err)):
                    # If position error contains NaN/inf, skip this update
                    delta_u = np.zeros((actions.shape[1],), dtype=np.float32)
                else:
                    Bt = B_pos.T
                    BtB = Bt @ B_pos
                    act_dim_curr = actions.shape[1]
                    H = BtB + reg * np.eye(act_dim_curr, dtype=np.float32)
                    try:
                        delta_u = np.linalg.solve(H, Bt @ pos_err)
                        # Check if solution is valid
                        if not np.all(np.isfinite(delta_u)):
                            delta_u = np.zeros((act_dim_curr,), dtype=np.float32)
                    except np.linalg.LinAlgError:
                        delta_u = np.zeros((act_dim_curr,), dtype=np.float32)
                
                u = u + delta_u
                if limit_val is not None:
                    u = np.clip(u, -limit_val, limit_val)
            
            actions[t] = u
            x = self.env.transition(x, u)
        
        return actions
    
    def add_extra_noise(
        self,
        actions_array: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Add extra noise to actions."""
        if sigma <= 0.0:
            return actions_array
        noise = self._np_rng.standard_normal(size=actions_array.shape).astype(np.float32)
        return actions_array + sigma * noise
    
    def _apply_action_filter_numpy(
        self,
        s: np.ndarray,
        act: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> np.ndarray:
        """Apply action filter using NumPy (if available from constraint manager)."""
        if not hard_enabled or self.constraint_manager is None:
            return act
        
        # Try to get NumPy action filter
        action_filter = getattr(self.constraint_manager, "action_filter_operator", None)
        if action_filter is not None:
            # If action filter has NumPy interface, use it
            # Otherwise, return action unchanged
            try:
                # Try calling with NumPy arrays
                filtered_act = action_filter.filter(s, act, hard_clearance)
                if filtered_act is not None:
                    return np.asarray(filtered_act, dtype=np.float32)
            except Exception:
                pass
        
        return act
    
    def _get_diffusion_params(self, k: int, K: int) -> Dict[str, Any]:
        """
        Get diffusion parameters from scheduler.
        
        Args:
            k: Current diffusion step (0-indexed, k=0 is final step)
            K: Total diffusion steps (0-indexed, K is max step index)
            
        Returns:
            Dictionary with M_k, T_k, s_k (if available)
        """
        if self.scheduler is None:
            # Fallback to fixed parameters
            return {
                "M_k": self.action_nsample,
                "T_k": self.action_temp,
                "s_k": None,
            }
        
        try:
            from enerdynamics.core.constraints.core.types import ScheduleState
            state = ScheduleState(k=k, K=K)
            params = self.scheduler.params(state)
            
            # Extract diffusion parameters from _extra
            return {
                "M_k": params._extra.get("M_k", self.action_nsample),
                "T_k": params._extra.get("T_k", self.action_temp),
                "s_k": params._extra.get("s_k"),
            }
        except Exception:
            # Fallback to fixed parameters if scheduler fails
            return {
                "M_k": self.action_nsample,
                "T_k": self.action_temp,
                "s_k": None,
            }
    
    def _update_scheduler(
        self,
        k: int,
        K: int,
        feedback: Dict[str, Any]
    ) -> None:
        """
        Update scheduler with feedback.
        
        Args:
            k: Current diffusion step (0-indexed)
            K: Total diffusion steps (0-indexed)
            feedback: Feedback dictionary (feasible_rate, ess, weights, trajectories, etc.)
        """
        if self.scheduler is None or not hasattr(self.scheduler, 'update'):
            return
        
        try:
            from enerdynamics.core.constraints.core.types import ScheduleState
            state = ScheduleState(k=k, K=K)
            self.scheduler.update(state, feedback)
        except Exception:
            # Silently fail if update fails (scheduler might not support update)
            pass
    
    def _compute_feasibility_rate(self, trajectory: Trajectory, margin: Optional[float] = None) -> tuple[float, float]:
        """
        Compute continuous feasibility score q̂ and violation V from violation metric.
        
        Implementation of Solution 1: Replace binary 0/1 feasibility with continuous
        violation-based metric.
        
        Steps:
        1. Compute violations v^h = max(0, margin - sdf) for each time step
        2. Normalize: v_bar = clip(v / (margin + epsilon), 0, v_max)
        3. Aggregate: V(τ) = mean(v_bar) - smooth violation
        4. Map to [0,1]: q̂ = exp(-V/τ) with online-calibrated tau
        
        Fix B1: tau is calibrated online using sliding window median V50.
        Fix B3: margin is passed as parameter (from scheduler) instead of using static robot_radius.
        
        Args:
            trajectory: Trajectory to evaluate
            margin: Safety margin (from scheduler, defaults to robot_radius if None)
            
        Returns:
            Tuple of (q_hat, V_k): Continuous feasibility score q̂ ∈ [0, 1] and violation V_k
        """
        # Parameters for violation-to-feasibility mapping
        alpha = 10.0  # Scaling factor for exponential mapping
        b_safe = 0.0  # Safety threshold (0 for constraints: c ≤ 0)
        use_max = False  # Use mean (smooth violation), not max
        violations = None

        # Try to get violations from constraint_pipeline
        if violations is None and self.constraint_pipeline is not None:
            try:
                # Method 1: Try to get violations from convexifier (obstacles SDF)
                if hasattr(self.constraint_pipeline, 'convexifier'):
                    convexifier = self.constraint_pipeline.convexifier
                    if hasattr(convexifier, 'obstacles') and convexifier.obstacles is not None:
                        # Compute violations from obstacle SDF
                        violations_list = []
                        # Fix B3: Use margin from parameter (from scheduler), fallback to robot_radius
                        if margin is None:
                            margin = self.robot_radius
                        
                        # Fix B3: Assert margin is valid (prevent division by near-zero)
                        if margin < 1e-4:
                            print(f"WARNING: margin={margin} is too small (< 1e-4)! This will cause V to explode. "
                                  f"Using robot_radius={self.robot_radius} as fallback.")
                            margin = max(self.robot_radius, 1e-4)  # Use at least 1e-4
                        
                        # Compute SDF violations for all states in trajectory
                        for state in trajectory.states:
                            pos = np.asarray(state[:2], dtype=np.float32)  # Extract position (first 2 dims)
                            sdf = convexifier.obstacles.sdf(pos)
                            sdf_val = float(sdf) if np.isscalar(sdf) else float(sdf[0])
                            # Violation = max(0, margin - sdf)  (if sdf < margin, trajectory is inside obstacle)
                            violation = max(0.0, margin - sdf_val)
                            violations_list.append(violation)
                        
                        if violations_list:
                            violations = np.asarray(violations_list, dtype=np.float32)
            except Exception as e:
                # Silently continue to next method
                pass
        
        # If we have violations, compute continuous feasibility score
        if violations is not None and len(violations) > 0:
            # Fix B3: Use margin from parameter (from scheduler), fallback to robot_radius
            if margin is None:
                margin = self.robot_radius
            
            # Fix B3: Assert margin is valid (prevent division by near-zero)
            if margin < 1e-4:
                print(f"WARNING: margin={margin} is too small (< 1e-4)! This will cause V to explode. "
                      f"Using robot_radius={self.robot_radius} as fallback.")
                margin = max(self.robot_radius, 1e-4)  # Use at least 1e-4
            
            # Normalize violations to prevent scale explosion (Fix 1.2)
            # v_bar = clip(v / (margin + epsilon), 0, v_max)
            epsilon_0 = 1e-6  # Prevent division by zero
            v_max = 10.0  # Upper bound for normalized violation
            v_h = np.maximum(violations, 0.0)  # Ensure non-negative
            # Normalize: divide by (margin + epsilon) to make scale comparable
            v_bar = np.clip(v_h / (margin + epsilon_0), 0.0, v_max)
            
            # Aggregate: V(τ) = mean(v_bar) or max(v_bar)
            if use_max:
                V_tau = float(np.max(v_bar))
            else:
                V_tau = float(np.mean(v_bar))
            
            # Fix B1: Online calibration of tau using sliding window median (Optimization 3: use deque)
            # Maintain sliding window of V values (deque automatically handles maxlen)
            self.V_history.append(V_tau)
            
            # Compute median V50 for calibration
            if len(self.V_history) >= 5:  # Need at least 5 samples for meaningful median
                V50 = float(np.median(self.V_history))
                # Set q(V50) = 0.5, so tau = V50 / ln(2)
                self.tau = V50 / np.log(2.0)
                self.tau = max(0.01, min(self.tau, 1.0))  # Clip tau to reasonable range [0.01, 1.0]
            # else: use current tau (initialized to 0.1)
            
            # Map to feasibility score: q̂ = exp(-V/τ) with online-calibrated tau
            q_hat = float(np.exp(-V_tau / self.tau))
            q_hat = np.clip(q_hat, 0.0, 1.0)
            
            return q_hat, V_tau  # Return both q_hat and V_k
        
        # Fallback: try binary feasibility check
        if self.constraint_manager is not None:
            try:
                if hasattr(self.constraint_manager, 'check_feasibility'):
                    is_feasible = self.constraint_manager.check_feasibility(trajectory)
                    V_tau = 0.0 if is_feasible else 1.0  # Default violation
                    return (1.0 if is_feasible else 0.0), V_tau
            except Exception:
                pass
        
        # Default: assume feasible if no constraint system available
        return 1.0, 0.0  # q_hat=1.0, V_k=0.0
    
    def _compute_feasibility_rate_batch(
        self,
        trajectories: List[Trajectory],
        margin: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch compute feasibility rates for multiple trajectories (Optimization 1).
        
        This function processes all trajectories in batch, using vectorized SDF queries
        where possible to significantly improve performance.
        
        Args:
            trajectories: List of trajectories to evaluate
            margin: Safety margin (from scheduler, defaults to robot_radius if None)
            
        Returns:
            Tuple of (q_hat_array, V_k_array): Arrays of shape (M,) where M is number of trajectories
        """
        num_trajs = len(trajectories)
        if num_trajs == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        # Fix B3: Use margin from parameter (from scheduler), fallback to robot_radius
        if margin is None:
            margin = self.robot_radius
        
        # Fix B3: Assert margin is valid (prevent division by near-zero)
        if margin < 1e-4:
            margin = max(self.robot_radius, 1e-4)
        
        # Try to get violations from constraint_pipeline
        if self.constraint_pipeline is not None:
            try:
                # Method 1: Try to get violations from convexifier (obstacles SDF)
                if hasattr(self.constraint_pipeline, 'convexifier'):
                    convexifier = self.constraint_pipeline.convexifier
                    if hasattr(convexifier, 'obstacles') and convexifier.obstacles is not None:
                        # Collect all positions from all trajectories
                        all_positions = []
                        traj_lengths = []
                        traj_start_indices = [0]  # Track where each trajectory starts in all_positions
                        
                        for traj in trajectories:
                            positions = np.array([state[:2] for state in traj.states], dtype=np.float32)
                            all_positions.append(positions)
                            traj_lengths.append(len(traj.states))
                            traj_start_indices.append(traj_start_indices[-1] + len(traj.states))
                        
                        if all_positions:
                            # Stack all positions into single array for batch SDF query
                            total_positions = np.vstack(all_positions)  # (total_states, 2)
                            
                            # Batch SDF query (obstacles.sdf supports batch input)
                            try:
                                sdf_vals = convexifier.obstacles.sdf(total_positions)
                                sdf_vals = np.asarray(sdf_vals, dtype=np.float32)
                                if sdf_vals.ndim == 0:
                                    sdf_vals = sdf_vals[None]
                                elif sdf_vals.ndim == 1 and len(sdf_vals) == 1 and total_positions.shape[0] > 1:
                                    # Handle case where sdf returns scalar for each point but shape is wrong
                                    sdf_vals = np.array([convexifier.obstacles.sdf(pos) for pos in total_positions], dtype=np.float32)
                            except (TypeError, ValueError, AttributeError):
                                # Fallback: SDF doesn't support batch, use loop
                                sdf_vals = np.array([
                                    float(convexifier.obstacles.sdf(pos)) if np.isscalar(convexifier.obstacles.sdf(pos)) 
                                    else float(convexifier.obstacles.sdf(pos)[0])
                                    for pos in total_positions
                                ], dtype=np.float32)
                            
                            # Batch compute violations
                            violations_batch = np.maximum(0.0, margin - sdf_vals)
                            
                            # Normalize violations
                            epsilon_0 = 1e-6
                            v_max = 10.0
                            v_bar_batch = np.clip(violations_batch / (margin + epsilon_0), 0.0, v_max)
                            
                            # Process each trajectory
                            q_hat_list = []
                            V_k_list = []
                            for traj_idx in range(num_trajs):
                                start_idx = traj_start_indices[traj_idx]
                                end_idx = traj_start_indices[traj_idx + 1]
                                traj_v_bar = v_bar_batch[start_idx:end_idx]
                                
                                # Aggregate: V(τ) = mean(v_bar)
                                V_tau = float(np.mean(traj_v_bar))
                                
                                # Update tau calibration (shared across all trajectories in this batch)
                                self.V_history.append(V_tau)
                                
                                # Compute median V50 for calibration
                                if len(self.V_history) >= 5:
                                    V50 = float(np.median(self.V_history))
                                    # Set q(V50) = 0.5, so tau = V50 / ln(2)
                                    self.tau = V50 / np.log(2.0)
                                    self.tau = max(0.01, min(self.tau, 1.0))  # Clip tau to reasonable range
                                
                                # Map to feasibility score: q̂ = exp(-V/τ) with online-calibrated tau
                                q_hat = float(np.exp(-V_tau / self.tau))
                                q_hat = np.clip(q_hat, 0.0, 1.0)
                                
                                q_hat_list.append(q_hat)
                                V_k_list.append(V_tau)
                            
                            return np.array(q_hat_list, dtype=np.float32), np.array(V_k_list, dtype=np.float32)
            except Exception:
                # Silently continue to fallback
                pass
        
        # Fallback: use individual trajectory computation
        q_hat_list = []
        V_k_list = []
        for traj in trajectories:
            q, v = self._compute_feasibility_rate(traj, margin)
            q_hat_list.append(q)
            V_k_list.append(v)
        return np.array(q_hat_list, dtype=np.float32), np.array(V_k_list, dtype=np.float32)
    
    def reverse_diffuse(
        self,
        rng_key: Any,
        state_init: np.ndarray,
        hard_clearance_by_idx: Optional[np.ndarray],
        hard_enabled_by_idx: Optional[np.ndarray],
        soft_alpha_by_idx: Optional[np.ndarray],
        soft_beta_by_idx: Optional[np.ndarray],
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run reverse diffusion using pure NumPy.
        
        This is the main reverse diffusion algorithm implementation.
        """
        act_dim = self.env.act_dim
        horizon = self.horizon
        Ndiffuse = self.action_diffuse_steps
        beta0 = self.action_beta0
        betaT = self.action_betaT
        temp = self.action_temp
        control_limit = getattr(self.env, "control_limit", None)
        num_particles = max(1, int(self.action_nsample))
        temp_eps = temp if temp > 1e-6 else 1e-6
        
        # Diffusion schedule
        betas = np.linspace(beta0, betaT, Ndiffuse, dtype=np.float32)
        
        # Get diffusion schedule from scheduler if available (for beta scaling)
        diffusion_schedule = None
        if self.scheduler is not None:
            try:
                from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
                # Try to get diffusion schedule from scheduler
                if hasattr(self.scheduler, 'diffusion_schedulers') and self.scheduler.diffusion_schedulers:
                    # Get from first diffusion scheduler
                    diff_sched = self.scheduler.diffusion_schedulers[0]
                    if hasattr(diff_sched, 'diffusion_schedule'):
                        diffusion_schedule = diff_sched.diffusion_schedule
                    elif hasattr(diff_sched, 'get_scaled_betas'):
                        # Scheduler can provide scaled betas
                        scaled_betas = diff_sched.get_scaled_betas()
                        if scaled_betas is not None:
                            betas = np.asarray(scaled_betas, dtype=np.float32)
                # Create diffusion schedule object for progress calculation
                if diffusion_schedule is None:
                    diffusion_schedule = DiffusionNoiseSchedule.from_betas(betas)
            except Exception:
                # Fallback: create from betas
                from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
                diffusion_schedule = DiffusionNoiseSchedule.from_betas(betas)
        else:
            from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule
            diffusion_schedule = DiffusionNoiseSchedule.from_betas(betas)
        
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        sigmas = np.sqrt(1.0 - alphas_bar)
        extra_sigmas = np.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=np.float32)
        
        K = Ndiffuse - 1  # Total steps (0-indexed, K is max step index)
        
        # Initialize Ybar using NumPy random with deterministic seed for reproducibility
        # If rng_key is an int, use it to create a new RNG for this run
        # Otherwise, use the existing _np_rng from planner (which was initialized with seed)
        if isinstance(rng_key, (int, np.integer)):
            # Create a new RNG with the provided seed for this specific run
            # This ensures reproducibility when same seed is used
            run_rng = np.random.default_rng(int(rng_key))
        else:
            # Use the planner's RNG (which should already be initialized with seed)
            run_rng = self._np_rng
        Ybar = run_rng.standard_normal(size=(horizon, act_dim)).astype(np.float32)
        
        reward_history = []
        Ybar_history = []
        Ysamples_history = []
        
        # Prepare constraint parameters if not provided
        if hard_clearance_by_idx is None or hard_enabled_by_idx is None:
            hard_clearance_by_idx = np.zeros((Ndiffuse,), dtype=np.float32)
            hard_enabled_by_idx = np.zeros((Ndiffuse,), dtype=bool)
            if self.constraint_manager and self.constraint_manager.schedule_manager is not None:
                sched = self.constraint_manager.schedule_manager
                total_steps = max(1, Ndiffuse - 2)
                has_hard = (
                    (self.constraint_manager.action_filter_operator is not None) or
                    (self.constraint_manager.feasibility_operator is not None)
                )
                for idx in range(Ndiffuse):
                    step_k = int((Ndiffuse - 1) - idx)
                    if step_k < 0:
                        step_k = 0
                    if step_k > total_steps:
                        step_k = total_steps
                    if has_hard:
                        hard_enabled_by_idx[idx] = bool(sched.is_hard_active(step_k, total_steps))
                        hard_clearance_by_idx[idx] = float(
                            sched.get_hard_clearance(default=0.0, step=step_k, total_steps=total_steps)
                        )
            elif self.constraint_manager:
                has_hard = (
                    (self.constraint_manager.action_filter_operator is not None) or
                    (self.constraint_manager.feasibility_operator is not None)
                )
                if has_hard:
                    hard_enabled_by_idx[:] = True
        
        # Reverse diffusion loop (100 steps: i=99 to i=0)
        diffusion_iter = range(Ndiffuse - 1, -1, -1)
        if HAS_TQDM and self.show_tqdm:
            diffusion_iter = tqdm(diffusion_iter, desc="EDOC Diffusion", unit="step", 
                                 total=Ndiffuse, leave=False)
        
        for i in diffusion_iter:
            # Convert diffusion index to step index (k=0 is final step, k=K is initial step)
            k = (Ndiffuse - 1) - i  # Current step (0-indexed)
            
            # Get diffusion parameters from scheduler
            diffusion_params = self._get_diffusion_params(k, K)
            M_k = int(diffusion_params.get("M_k", num_particles))
            T_k = float(diffusion_params.get("T_k", temp))
            s_k = diffusion_params.get("s_k")
            
            # Use M_k for current step
            num_particles_current = max(1, M_k)
            temp_eps_current = T_k if T_k > 1e-6 else 1e-6
            
            Yi = Ybar * np.sqrt(alphas_bar[i])
            sigma_i = float(sigmas[i])
            
            # Generate random samples with antithetic sampling if enabled
            if self.use_antithetic and num_particles_current > 1:
                half = num_particles_current // 2
                has_extra = num_particles_current % 2
                sample_count = half + has_extra
                
                eps_core = run_rng.standard_normal(
                    size=(sample_count, horizon, act_dim)
                ).astype(np.float32)
                
                Y_candidates = []
                if half > 0:
                    eps_half = eps_core[:half]
                    Y_candidates.append(Ybar[None, :] + sigma_i * eps_half)
                    Y_candidates.append(Ybar[None, :] - sigma_i * eps_half)
                if has_extra:
                    eps_extra = eps_core[-1:]
                    Y_candidates.append(Ybar[None, :] + sigma_i * eps_extra)
                
                if not Y_candidates:
                    Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()
                elif len(Y_candidates) == 1:
                    Y0s = Y_candidates[0]
                else:
                    Y0s = np.concatenate(Y_candidates, axis=0)
            else:
                eps = run_rng.standard_normal(
                    size=(num_particles_current, horizon, act_dim)
                ).astype(np.float32)
                Y0s = Ybar[None, :] + sigma_i * eps
            
            # Clip actions if control limit exists
            if control_limit is not None:
                limit = float(control_limit)
                Y0s = np.clip(Y0s, -limit, limit)
            
            # Convert diffusion index to constraint step
            constraint_step = (Ndiffuse - 1) - i
            constraint_total_steps = Ndiffuse - 1  # Total steps should match Ndiffuse (0-indexed, so Ndiffuse-1)
            
            # Score particles in batch
            scores = self.score_particles(
                state_init, Y0s, constraint_step, constraint_total_steps
                        )
            
            # Compute weights from scores using T_k (needed for Fix B2: weighted aggregation)
            score_std = float(np.std(scores))
            if score_std < 1e-4:
                weights = np.full((num_particles_current,), 1.0 / num_particles_current, dtype=np.float32)
            else:
                score_mean = float(np.mean(scores))
                denom = score_std * temp_eps_current  # Use T_k instead of temp_eps
                logw = (scores - score_mean) / denom
                logw = logw - np.max(logw)
                weights = np.exp(logw)
                weights = weights / np.sum(weights)
            weights = np.asarray(weights, dtype=np.float32)
            
            # Get margin from scheduler (Fix B3: margin as scheduler parameter)
            margin_k = None
            if self.scheduler is not None:
                try:
                    from enerdynamics.core.constraints.core.types import ScheduleState
                    debug_state = ScheduleState(k=constraint_step, K=constraint_total_steps)
                    if hasattr(self.scheduler, 'constraint_params'):
                        params = self.scheduler.constraint_params(debug_state)
                        if isinstance(params, dict):
                            margin_k = params.get('margin', None)
                        elif hasattr(params, 'margin'):
                            margin_k = params.margin
                    elif hasattr(self.scheduler, 'params'):
                        params_obj = self.scheduler.params(debug_state)
                        margin_k = getattr(params_obj, 'margin', None)
                except Exception:
                    pass
            
            # Fix 1.3: Batch aggregation + EMA for feasibility tracking (Optimization 1, 4)
            # Fix B2: Use importance weights for weighted aggregation
            # Compute q_k^(m) and V_k^(m) for all particles in batch
            # Optimization: Use batch functions for better performance
            trajectories_batch = self.actions_to_trajectory_batch(state_init, Y0s)
            q_k_array, V_k_array = self._compute_feasibility_rate_batch(trajectories_batch, margin=margin_k)
            
            # Fix B2: Weighted aggregation: q_bar_k = sum_m (w_m * q_k^(m))
            if len(q_k_array) > 0 and len(weights) == len(q_k_array):
                q_bar_k = float(np.sum(weights * q_k_array))
                V_bar_k = float(np.sum(weights * V_k_array))
            else:
                # Fallback to unweighted mean if weights don't match
                q_bar_k = float(np.mean(q_k_array)) if len(q_k_array) > 0 else 0.0
                V_bar_k = float(np.mean(V_k_array)) if len(V_k_array) > 0 else 0.0
            
            # EMA smoothing: q_tilde_k = (1-beta) * q_tilde_{k-1} + beta * q_bar_k
            if self.q_tilde_prev is None:
                # First step: initialize with q_bar_k
                q_tilde_k = q_bar_k
            else:
                beta = self.ema_beta
                q_tilde_k = (1.0 - beta) * self.q_tilde_prev + beta * q_bar_k
            q_tilde_k = float(np.clip(q_tilde_k, 0.0, 1.0))
            self.q_tilde_prev = q_tilde_k  # Store for next iteration
            
            reward_history.append(float(np.mean(scores)))
            
            # Compute ESS for feedback
            ess = 1.0 / (np.sum(weights ** 2) + 1e-10) / num_particles_current
            
            # Update Ybar using weighted average
            Ybar_weighted = np.tensordot(weights, Y0s, axes=([0], [0]))
            
            # Reverse diffusion step
            score = (-Yi + np.sqrt(alphas_bar[i]) * Ybar_weighted) / (1.0 - alphas_bar[i])
            Yim1 = (Yi + (1.0 - alphas_bar[i]) * score) / np.sqrt(alphas[i])
            # Handle final step (i=0): Ybar = Yim1 (no division by sqrt(alphas_bar[0]))
            # For i > 0: Ybar = Yim1 / sqrt(alphas_bar[i-1])
            if i > 0:
                Ybar = Yim1 / np.sqrt(alphas_bar[i - 1])
            else:
                # Final step: i=0, Ybar is already Yim1 (alphas_bar[0] = 1.0 typically)
                Ybar = Yim1
            
            # Add extra noise
            Ybar = self.add_extra_noise(Ybar, extra_sigmas[i])
            
            # Create trajectory for constraint projection and feedback
            trajectory = self.actions_to_trajectory(state_init, Ybar)
            
            # Apply hard constraint projection if enabled
            if hard_enabled_by_idx[i]:
                # Use new pipeline if available, otherwise fallback to legacy constraint_manager
                if self.constraint_pipeline is not None:
                    from enerdynamics.core.constraints.core.types import ScheduleState
                    state = ScheduleState(k=constraint_step, K=constraint_total_steps)
                    repaired_trajectory, _ = self.constraint_pipeline.apply(
                        nominal=trajectory,
                        ref=trajectory,
                        state=state
                    )
                    projected_trajectory = repaired_trajectory
                elif (self.constraint_manager and 
                      self.constraint_manager.has_hard()):
                    projected_trajectory = self.constraint_manager.project_hard(
                        trajectory,
                        step=constraint_step,
                        total_steps=constraint_total_steps,
                    )
                else:
                    projected_trajectory = trajectory
                
                # Check if states changed
                states_changed = True
                try:
                    traj_states = np.asarray(trajectory.states, dtype=np.float32)
                    proj_states = np.asarray(projected_trajectory.states, dtype=np.float32)
                    if traj_states.shape == proj_states.shape:
                        max_delta = float(np.max(np.linalg.norm(traj_states - proj_states, axis=-1)))
                        states_changed = max_delta > 1e-6
                except Exception:
                    states_changed = True
                
                if states_changed:
                    Ybar = self.track_actions_to_projected_states(
                        state_init, Ybar, projected_trajectory.states
                    )
                else:
                    Ybar = self.extract_actions_from_trajectory(projected_trajectory)
                
                # Update trajectory after projection for feedback
                trajectory = projected_trajectory
            
            feedback = {
                "feasible_rate": q_bar_k,  # Fix A1: Use q_bar_k (batch aggregation) as feasible_rate
                "q_hat": q_bar_k,  # Batch-aggregated q_bar_k (current observation)
                "q_tilde": q_tilde_k,  # Fix A1: Explicitly pass q_tilde (EMA) for modulation/plotting
                "V_k": V_bar_k,  # Batch-aggregated V_bar_k
                "ess": ess,
                "weights": weights,
                "trajectories": [trajectory],  # Keep single trajectory for backward compatibility
            }
            self._update_scheduler(k, K, feedback)
            
            # Clip and store history
            if control_limit is not None:
                limit = float(control_limit)
                Ybar = np.clip(Ybar, -limit, limit)
                Ybar_history.append(Ybar.copy())
                Ysamples_history.append(np.clip(Y0s, -limit, limit))
            else:
                Ybar_history.append(Ybar.copy())
                Ysamples_history.append(Y0s.copy())
        
        # Final hard constraint projection
        if self.constraint_pipeline is not None or (self.constraint_manager and self.constraint_manager.has_hard()):
            trajectory = self.actions_to_trajectory(state_init, Ybar)
            
            if self.constraint_pipeline is not None:
                from enerdynamics.core.constraints.core.types import ScheduleState
                state = ScheduleState(k=constraint_total_steps, K=constraint_total_steps)
                repaired_trajectory, _ = self.constraint_pipeline.apply(
                    nominal=trajectory,
                    ref=trajectory,
                    state=state
                )
                projected_trajectory = repaired_trajectory
            else:
                projected_trajectory = self.constraint_manager.project_hard(
                    trajectory, step=constraint_total_steps, total_steps=constraint_total_steps
                )
            
            Ybar = self.track_actions_to_projected_states(
                state_init, Ybar, projected_trajectory.states
            )
            
            # Overwrite last stored diffusion frame
            if len(Ybar_history) > 0:
                if control_limit is not None:
                    limit = float(control_limit)
                    Ybar = np.clip(Ybar, -limit, limit)
                Ybar_history[-1] = Ybar.copy()
        
        # Final clipping
        if control_limit is not None:
            limit = float(control_limit)
            actions_final = np.clip(Ybar, -limit, limit)
        else:
            actions_final = Ybar
        
        # Convert to arrays (reverse order to match JAX convention)
        reward_history_arr = np.array(reward_history[::-1], dtype=np.float32)
        diffusion_actions_traj_arr = np.array(Ybar_history[::-1], dtype=np.float32)
        diffusion_samples_traj_arr = np.array(Ysamples_history[::-1], dtype=np.float32)
        
        return (rng_key, actions_final, reward_history_arr,
               diffusion_actions_traj_arr, diffusion_samples_traj_arr)
