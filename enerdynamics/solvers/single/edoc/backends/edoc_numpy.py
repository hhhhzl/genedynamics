"""
NumPy backend implementation for EDOC reverse diffusion.

This module contains the pure NumPy implementation of the reverse diffusion
algorithm, with no JAX dependencies. All operations use NumPy for maximum compatibility.
"""

from typing import Optional, Tuple, Any, List
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
        # Use NumPy random number generator
        self._np_rng = np.random.default_rng(
            getattr(planner, '_np_rng', None) or None
        )
        if self._np_rng is None:
            self._np_rng = np.random.default_rng()
    
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
            # Batch compute trajectories
            trajectories = [
                self.actions_to_trajectory(state_init, batch_actions[idx]) 
                for idx in range(num)
            ]
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
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        sigmas = np.sqrt(1.0 - alphas_bar)
        extra_sigmas = np.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=np.float32)
        
        # Initialize Ybar using NumPy random
        if isinstance(rng_key, (int, np.integer)):
            self._np_rng = np.random.default_rng(int(rng_key))
        Ybar = self._np_rng.standard_normal(size=(horizon, act_dim)).astype(np.float32)
        
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
        
        # Reverse diffusion loop
        diffusion_iter = range(Ndiffuse - 1, 0, -1)
        if HAS_TQDM and self.show_tqdm:
            diffusion_iter = tqdm(diffusion_iter, desc="EDOC Diffusion", unit="step", 
                                 total=Ndiffuse-1, leave=False)
        
        for i in diffusion_iter:
            Yi = Ybar * np.sqrt(alphas_bar[i])
            sigma_i = float(sigmas[i])
            
            # Generate random samples with antithetic sampling if enabled
            if self.use_antithetic and num_particles > 1:
                half = num_particles // 2
                has_extra = num_particles % 2
                sample_count = half + has_extra
                
                eps_core = self._np_rng.standard_normal(
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
                eps = self._np_rng.standard_normal(
                    size=(num_particles, horizon, act_dim)
                ).astype(np.float32)
                Y0s = Ybar[None, :] + sigma_i * eps
            
            # Clip actions if control limit exists
            if control_limit is not None:
                limit = float(control_limit)
                Y0s = np.clip(Y0s, -limit, limit)
            
            # Convert diffusion index to constraint step
            constraint_step = (Ndiffuse - 1) - i
            constraint_total_steps = Ndiffuse - 2
            
            # Score particles in batch
            scores = self.score_particles(
                state_init, Y0s, constraint_step, constraint_total_steps
                        )
            
            # Compute weights from scores
            score_std = float(np.std(scores))
            if score_std < 1e-4:
                weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
            else:
                score_mean = float(np.mean(scores))
                denom = score_std * temp_eps
                logw = (scores - score_mean) / denom
                logw = logw - np.max(logw)
                weights = np.exp(logw)
                weights = weights / np.sum(weights)
            
            reward_history.append(float(np.mean(scores)))
            weights = np.asarray(weights, dtype=np.float32)
            
            # Update Ybar using weighted average
            Ybar_weighted = np.tensordot(weights, Y0s, axes=([0], [0]))
            
            # Reverse diffusion step
            score = (-Yi + np.sqrt(alphas_bar[i]) * Ybar_weighted) / (1.0 - alphas_bar[i])
            Yim1 = (Yi + (1.0 - alphas_bar[i]) * score) / np.sqrt(alphas[i])
            Ybar = Yim1 / np.sqrt(alphas_bar[i - 1])
            
            # Add extra noise
            Ybar = self.add_extra_noise(Ybar, extra_sigmas[i])
            
            # Apply hard constraint projection if enabled
            if hard_enabled_by_idx[i]:
                trajectory = self.actions_to_trajectory(state_init, Ybar)
                
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
