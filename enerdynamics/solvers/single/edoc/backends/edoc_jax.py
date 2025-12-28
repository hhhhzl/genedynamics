"""
JAX backend implementation for EDOC reverse diffusion.

This module contains the JAX-optimized implementation of the reverse diffusion
algorithm, with optional JIT compilation for high performance.
"""

from typing import Optional, Tuple, Any, List
import numpy as np

import jax
import jax.numpy as jnp

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


@register_edoc_backend("jax")
class EDOCBackendJax(EDOCBackendBase):
    """
    JAX backend implementation of EDOC reverse diffusion.
    
    This implementation uses JAX with optional JIT compilation for high performance.
    It requires JAX-compatible environment and constraint functions.
    """
    
    def __init__(self, planner: Any, use_jit: bool = False, **config: Any):
        """
        Initialize JAX backend implementation.
        
        Args:
            planner: EDOCPlanner instance
            use_jit: Whether to use JIT compilation (default: False)
            **config: Additional configuration
        """
        super().__init__(planner, **config)
        self.use_jit = use_jit
        self._jax_transition = None
        self._jax_model_transition = None
        self._jax_env_transition = None
        self._jax_cost = None
        self._jax_action_filter = None
        self._jac_model_u_fn = None
        self._time_index = None
        self._jax_cfs_projector = None
        self._reverse_diffuse_jit = None
        self._reverse_diffuse_chunk_jit = None
        self._initialized = False
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize JAX functions and optionally JIT compile them."""
        if not self.action_space:
            return
        
        # Get JAX transition functions from environment
        if not hasattr(self.env, "jax_transition"):
            raise ValueError("Environment must provide jax_transition when using JAX backend.")
        
        self._jax_transition = self.env.jax_transition
        self._jax_model_transition = (
            self.env.jax_model_transition if hasattr(self.env, "jax_model_transition")
            else self.env.jax_transition
        )
        self._jax_env_transition = (
            self.env.jax_env_transition if hasattr(self.env, "jax_env_transition")
            else self._jax_model_transition
        )
        self._jax_cost = getattr(self.env, "jax_cost", None)
        
        # Get action filter from constraint manager
        if self.constraint_manager is not None:
            self._jax_action_filter = self.constraint_manager.get_jax_action_filter()
        
        # Create time index for scan
        self._time_index = jnp.arange(self.horizon, dtype=jnp.int32)
        
        # Create Jacobian function for tracking
        self._jac_model_u_fn = jax.jacrev(self._jax_model_transition, argnums=1)
        
        # Get JAX CFS projector if available
        if self.constraint_manager is not None:
            feasibility_op = getattr(self.constraint_manager, "feasibility_operator", None)
            if feasibility_op is not None and hasattr(feasibility_op, "make_jax_projector"):
                try:
                    self._jax_cfs_projector = feasibility_op.make_jax_projector()
                except Exception:
                    pass
        
        # Optionally JIT compile functions
        if self.use_jit:
            self._jax_transition = jax.jit(self._jax_transition)
            self._jax_model_transition = jax.jit(self._jax_model_transition)
            self._jax_env_transition = jax.jit(self._jax_env_transition)
            self._jac_model_u_fn = jax.jit(self._jac_model_u_fn)
        
        self._initialized = True
    
    def _apply_action_filter_jax(
        self,
        s: jnp.ndarray,
        act: jnp.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> jnp.ndarray:
        """Apply action filter using JAX if available."""
        if self._jax_action_filter is None or not hard_enabled:
            return act
        return self._jax_action_filter(s, act, hard_clearance, hard_enabled)
    
    def rollout_states_and_energy(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rollout states and compute energy sequence using JAX scan."""
        state_jnp = jnp.asarray(state, dtype=jnp.float32)
        actions_jnp = jnp.asarray(actions, dtype=jnp.float32)
        
        def body(carry, inputs):
            s = carry
            t, act = inputs
            ctx = {"t": int(t)}
            act_safe = self._apply_action_filter_jax(s, act, hard_clearance, hard_enabled)
            # Energy computation requires NumPy arrays
            e_val = float(self.energy.compute(
                np.asarray(s), np.asarray(act_safe), ctx
            ))
            s_next = self._jax_model_transition(s, act_safe)
            return s_next, (s_next, jnp.float32(e_val))
        
        _, (states_seq, energy_seq) = jax.lax.scan(
            body, state_jnp, (self._time_index, actions_jnp)
        )
        states_full = jnp.concatenate([state_jnp[None, :], states_seq], axis=0)
        return np.asarray(states_full), np.asarray(energy_seq)
    
    def rollout_env_states(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        hard_clearance: float,
        hard_enabled: bool,
    ) -> np.ndarray:
        """Rollout environment states using JAX scan."""
        state_jnp = jnp.asarray(state, dtype=jnp.float32)
        actions_jnp = jnp.asarray(actions, dtype=jnp.float32)
        
        def body(carry, inputs):
            s = carry
            _, act = inputs
            act_safe = self._apply_action_filter_jax(s, act, hard_clearance, hard_enabled)
            s_next = self._jax_env_transition(s, act_safe)
            return s_next, s_next
        
        _, states_seq = jax.lax.scan(
            body, state_jnp, (self._time_index, actions_jnp)
        )
        states_full = jnp.concatenate([state_jnp[None, :], states_seq], axis=0)
        return np.asarray(states_full)
    
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
        total = float(jnp.sum(jnp.asarray(energy_seq, dtype=jnp.float32)))
        
        # Terminal energy
        if self.terminal_energy_weight > 0.0:
            u0 = jnp.zeros((self.env.act_dim,), dtype=jnp.float32)
            ctxT = {"t": int(len(actions))}
            terminal_energy = float(self.energy.compute(
                np.asarray(states_full[-1]), np.asarray(u0), ctxT
            ))
            total += float(self.terminal_energy_weight) * terminal_energy
        
        # Dynamics loss
        if self.dyn_loss_coeff > 0.0:
            env_states_full = self.rollout_env_states(state, actions, hard_clearance, hard_enabled)
            states_full_jnp = jnp.asarray(states_full, dtype=jnp.float32)
            env_states_full_jnp = jnp.asarray(env_states_full, dtype=jnp.float32)
            if self.dyn_loss_mode == "terminal":
                diff = states_full_jnp[-1] - env_states_full_jnp[-1]
                dyn_loss = float(jnp.sum(diff * diff))
            else:
                dyn_loss = float(jnp.sum((states_full_jnp - env_states_full_jnp) ** 2))
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
        if self._jax_cost is None:
            return 0.0
        
        state_jnp = jnp.asarray(state, dtype=jnp.float32)
        actions_jnp = jnp.asarray(actions, dtype=jnp.float32)
        
        def body(carry, inputs):
            s = carry
            _, act = inputs
            act_safe = self._apply_action_filter_jax(s, act, hard_clearance, hard_enabled)
            s_next = self._jax_env_transition(s, act_safe)
            reward = -self._jax_cost(s_next)
            return s_next, reward
        
        _, reward_seq = jax.lax.scan(
            body, state_jnp, (self._time_index, actions_jnp)
        )
        return float(jnp.mean(reward_seq))
    
    def score_particles(
        self,
        state_init: np.ndarray,
        batch_actions: np.ndarray,
        step: Optional[int],
        total_steps: Optional[int],
    ) -> np.ndarray:
        """Score action particles in batch using JAX vmap for performance."""
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
        
        state_init_jnp = jnp.asarray(state_init, dtype=jnp.float32)
        batch_actions_jnp = jnp.asarray(batch_actions, dtype=jnp.float32)
        
        # Batch compute scores using vmap
        if mode == "reward":
            if self._jax_cost is not None:
                # Use vmap for batch reward computation
                def compute_reward_one(actions):
                    return self.compute_mean_reward(
                        state_init, np.asarray(actions), hard_clearance, hard_enabled
                    )
                scores = jax.vmap(lambda acts: jnp.float32(compute_reward_one(acts)))(
                    batch_actions_jnp
                )
            else:
                scores = jnp.zeros((num,), dtype=jnp.float32)
        elif mode == "energy":
            # Use vmap for batch energy computation
            def compute_energy_one(actions):
                return self.compute_total_energy(
                    state_init, np.asarray(actions), hard_clearance, hard_enabled
                )
            energy_vals = jax.vmap(lambda acts: jnp.float32(compute_energy_one(acts)))(
                batch_actions_jnp
            )
            scores = -energy_vals
        elif mode == "learned":
            scores = jnp.zeros((num,), dtype=jnp.float32)
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
            scores = scores - jnp.asarray(soft_penalties, dtype=jnp.float32)
        
        return np.asarray(scores, dtype=np.float32)
    
    def actions_to_trajectory(
        self,
        x0: np.ndarray,
        actions: np.ndarray,
    ) -> Trajectory:
        """Convert actions array to Trajectory using JAX rollout."""
        actions_array = np.asarray(actions, dtype=np.float32)
        if actions_array.ndim == 1:
            actions_array = actions_array[None, :]
        
        # Use JAX rollout for efficiency
        states_full = self.rollout_env_states(
            x0, actions_array, 0.0, False
        )
        
        # Convert to lists
        states = [np.asarray(s, dtype=np.float32) for s in states_full]
        actions_list = [np.asarray(act, dtype=np.float32) for act in actions_array]
        
        return Trajectory(states=states, actions=actions_list)
    
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
        """Map projected state trajectory back to actions using JAX automatic differentiation."""
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
        
        x = jnp.asarray(x0, dtype=jnp.float32)
        for t in range(T):
            u = jnp.asarray(actions[t], dtype=jnp.float32)
            x_tgt = jnp.asarray(target_states[t + 1], dtype=jnp.float32)
            
            for _ in range(max(1, int(gn_iters))):
                # Predict next state using JAX
                x_pred = self._jax_model_transition(x, u)
                
                # Compute Jacobian using JAX automatic differentiation
                B = self._jac_model_u_fn(x, u)  # (state_dim, act_dim)
                
                # Use position part only (first 2 dimensions typically)
                state_dim = x.shape[0]
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
                Bt = jnp.swapaxes(B_pos, -2, -1)  # (act_dim, 2)
                act_dim_curr = actions.shape[1]
                BtB = Bt @ B_pos
                H = BtB + reg * jnp.eye(act_dim_curr, dtype=jnp.float32)
                try:
                    delta_u = jnp.linalg.solve(H, Bt @ pos_err)
                except Exception:
                    delta_u = jnp.zeros((act_dim_curr,), dtype=jnp.float32)
                
                u = u + delta_u
                if limit_val is not None:
                    u = jnp.clip(u, -limit_val, limit_val)
            
            actions[t] = np.asarray(u, dtype=np.float32)
            x = self._jax_model_transition(x, u)
        
        return actions
    
    def add_extra_noise(
        self,
        actions_array: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Add extra noise to actions using JAX random."""
        if sigma <= 0.0:
            return actions_array
        rng_key = jax.random.PRNGKey(0)  # Use fixed key for deterministic behavior
        noise = jax.random.normal(
            rng_key, shape=actions_array.shape, dtype=jnp.float32
        )
        return np.asarray(actions_array) + sigma * np.asarray(noise)
    
    def _prepare_constraint_parameters(self, Ndiffuse: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Prepare constraint parameters for each diffusion step.
        
        Returns:
            Tuple of (hard_clearance_by_idx, hard_enabled_by_idx, soft_alpha_by_idx, soft_beta_by_idx)
        """
        hard_clearance_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)
        hard_enabled_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.bool_)
        soft_alpha_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)
        soft_beta_by_idx = jnp.zeros((Ndiffuse,), dtype=jnp.float32)
        
        # Resolve soft defaults if an ObstacleSoftConstraint exists
        soft_alpha_default = 0.0
        soft_beta_default = 10.0
        if self.constraint_manager is not None and self.constraint_manager.has_soft():
            try:
                from enerdynamics.core.constraints.legacy.obstacle_constraints import ObstacleSoftConstraint
                for c in self.constraint_manager.soft_constraints:
                    if isinstance(c, ObstacleSoftConstraint):
                        soft_alpha_default = float(getattr(c, "alpha", 1.0))
                        soft_beta_default = float(getattr(c, "beta", 10.0))
                        break
            except Exception:
                pass
        
        if self.constraint_manager is not None and self.constraint_manager.schedule_manager is not None:
            sched = self.constraint_manager.schedule_manager
            total_steps = max(1, Ndiffuse - 2)
            hard_filter_exists = (
                (self.constraint_manager.action_filter_operator is not None) or
                (self.constraint_manager.feasibility_operator is not None)
            )
            hc = []
            he = []
            sa = []
            sb = []
            for idx in range(Ndiffuse):
                step_k = int((Ndiffuse - 1) - idx)
                if step_k < 0:
                    step_k = 0
                if step_k > total_steps:
                    step_k = total_steps
                if hard_filter_exists:
                    he.append(bool(sched.is_hard_active(step_k, total_steps)))
                    hc.append(float(sched.get_hard_clearance(default=0.0, step=step_k, total_steps=total_steps)))
                else:
                    he.append(False)
                    hc.append(0.0)
                sa.append(float(sched.get_soft_alpha(default=soft_alpha_default, step=step_k, total_steps=total_steps)))
                sb.append(float(sched.get_soft_beta(default=soft_beta_default, step=step_k, total_steps=total_steps)))
            
            hard_clearance_by_idx = jnp.asarray(np.asarray(hc, dtype=np.float32), dtype=jnp.float32)
            hard_enabled_by_idx = jnp.asarray(np.asarray(he, dtype=bool))
            soft_alpha_by_idx = jnp.asarray(np.asarray(sa, dtype=np.float32), dtype=jnp.float32)
            soft_beta_by_idx = jnp.asarray(np.asarray(sb, dtype=np.float32), dtype=jnp.float32)
        else:
            # No schedule: enable hard filter if present
            if self.constraint_manager is not None:
                has_hard = (
                    (self.constraint_manager.action_filter_operator is not None) or
                    (self.constraint_manager.feasibility_operator is not None)
                )
                if has_hard:
                    hard_enabled_by_idx = jnp.ones((Ndiffuse,), dtype=jnp.bool_)
            soft_alpha_by_idx = jnp.full((Ndiffuse,), jnp.float32(soft_alpha_default), dtype=jnp.float32)
            soft_beta_by_idx = jnp.full((Ndiffuse,), jnp.float32(soft_beta_default), dtype=jnp.float32)
        
        return hard_clearance_by_idx, hard_enabled_by_idx, soft_alpha_by_idx, soft_beta_by_idx
    
    def reverse_diffuse(
        self,
        rng_key: Any,
        state_init: np.ndarray,
        hard_clearance_by_idx: Optional[np.ndarray] = None,
        hard_enabled_by_idx: Optional[np.ndarray] = None,
        soft_alpha_by_idx: Optional[np.ndarray] = None,
        soft_beta_by_idx: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run reverse diffusion using JAX (with optional JIT compilation).
        
        This is a complete implementation that doesn't depend on planner's JIT functions.
        """
        horizon = self.horizon
        act_dim = self.env.act_dim
        Ndiffuse = self.action_diffuse_steps
        beta0 = self.action_beta0
        betaT = self.action_betaT
        temp = self.action_temp
        control_limit = getattr(self.env, "control_limit", None)
        num_particles = max(1, int(self.action_nsample))
        temp_eps = temp if temp > 1e-6 else 1e-6
        
        # Diffusion schedule
        betas = jnp.linspace(beta0, betaT, Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas, axis=0)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        extra_sigmas = jnp.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=jnp.float32)
        diffusion_indices = jnp.arange(Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        
        # Prepare constraint parameters if not provided
        if hard_clearance_by_idx is None:
            hard_clearance_by_idx, hard_enabled_by_idx, soft_alpha_by_idx, soft_beta_by_idx = \
                self._prepare_constraint_parameters(Ndiffuse)
        
        x0_jnp = jnp.asarray(state_init, dtype=jnp.float32)
        
        # Convert to JAX arrays if needed
        if not isinstance(hard_clearance_by_idx, jnp.ndarray):
            hard_clearance_by_idx = jnp.asarray(hard_clearance_by_idx, dtype=jnp.float32)
        if not isinstance(hard_enabled_by_idx, jnp.ndarray):
            hard_enabled_by_idx = jnp.asarray(hard_enabled_by_idx, dtype=jnp.bool_)
        if soft_alpha_by_idx is not None and not isinstance(soft_alpha_by_idx, jnp.ndarray):
            soft_alpha_by_idx = jnp.asarray(soft_alpha_by_idx, dtype=jnp.float32)
        if soft_beta_by_idx is not None and not isinstance(soft_beta_by_idx, jnp.ndarray):
            soft_beta_by_idx = jnp.asarray(soft_beta_by_idx, dtype=jnp.float32)
        
        # Initialize Ybar
        rng, init_key = jax.random.split(rng_key)
        Ybar = jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32)
        
        # Run reverse diffusion loop
        def step_one(carry, idx):
            rng_curr, Ybar_curr = carry
            rng_next, noise_key, extra_key = jax.random.split(rng_curr, 3)
            
            sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx])
            sigma_i = sigmas[idx]
            Yi = Ybar_curr * sqrt_alpha_bar_i
            
            hard_clearance = hard_clearance_by_idx[idx]
            hard_enabled = hard_enabled_by_idx[idx]
            
            # Generate samples with antithetic sampling if enabled
            if self.use_antithetic and num_particles > 1:
                half = num_particles // 2
                has_extra = num_particles % 2
                sample_count = half + has_extra
                eps_core = jax.random.normal(
                    noise_key, (sample_count, horizon, act_dim), dtype=jnp.float32
                )
                eps_half = eps_core[:half]
                Y_pos = Ybar_curr[None, :] + sigma_i * eps_half
                Y_neg = Ybar_curr[None, :] - sigma_i * eps_half
                if has_extra:
                    eps_extra = eps_core[-1:]
                    Y_extra = Ybar_curr[None, :] + sigma_i * eps_extra
                    Y0s = jnp.concatenate([Y_pos, Y_neg, Y_extra], axis=0)
                else:
                    Y0s = jnp.concatenate([Y_pos, Y_neg], axis=0)
            else:
                eps = jax.random.normal(
                    noise_key, (num_particles, horizon, act_dim), dtype=jnp.float32
                )
                Y0s = Ybar_curr[None, :] + sigma_i * eps
            
            # Clip actions if control limit exists
            if control_limit is not None:
                limit = float(control_limit)
                Y0s = jnp.clip(Y0s, -limit, limit)
            
            # Score particles (convert to NumPy for scoring, then back to JAX)
            constraint_step = (Ndiffuse - 1) - idx
            constraint_total_steps = Ndiffuse - 2
            scores = jnp.asarray(
                self.score_particles(
                    state_init, np.asarray(Y0s), constraint_step, constraint_total_steps
                ),
                dtype=jnp.float32
            )
            
            # Compute weights
            score_std = jnp.std(scores)
            score_std = jnp.where(score_std < 1e-4, 1.0, score_std)
            score_mean = jnp.mean(scores)
            logw = (scores - score_mean) / (score_std * temp_eps)
            logw = logw - jnp.max(logw)
            weights = jax.nn.softmax(logw)
            
            # Weighted average
            Ybar_weighted = jnp.tensordot(weights, Y0s, axes=([0], [0]))
            
            # Reverse diffusion step
            one_minus_alpha_bar = 1.0 - alphas_bar[idx]
            score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
            Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx])
            sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx - 1])
            Ybar_next = Yim1 / sqrt_alpha_bar_prev
            
            # Add extra noise
            extra_sigma = extra_sigmas[idx]
            if jnp.abs(extra_sigma) > 0.0:
                noise = jax.random.normal(extra_key, (horizon, act_dim), dtype=jnp.float32)
                Ybar_next = Ybar_next + extra_sigma * noise
            
            # Apply CFS projection if available
            if self._jax_cfs_projector is not None and hard_enabled:
                states_full = self.rollout_env_states(
                    state_init, np.asarray(Ybar_next), float(hard_clearance), bool(hard_enabled)
                )
                positions = jnp.asarray(states_full[:, :2], dtype=jnp.float32)
                projected_positions = self._jax_cfs_projector(positions, hard_clearance)
                Ybar_next = self.track_actions_to_projected_states(
                    state_init, np.asarray(Ybar_next), 
                    [np.asarray(p) for p in projected_positions]
                )
                Ybar_next = jnp.asarray(Ybar_next, dtype=jnp.float32)
            
            # Clip final actions
            if control_limit is not None:
                limit = float(control_limit)
                Ybar_next = jnp.clip(Ybar_next, -limit, limit)
            
            reward_val = jnp.mean(scores)
            return (rng_next, Ybar_next), (reward_val, Ybar_next, Y0s)
        
        # Run reverse diffusion
        (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
            step_one, (rng, Ybar), diffusion_indices
        )
        
        # Reverse to match convention (early→late)
        reward_hist = reward_hist[::-1]
        Ybar_hist = Ybar_hist[::-1]
        Ysamples_hist = Ysamples_hist[::-1]
        
        # Convert to NumPy
        actions_np_final = np.asarray(Ybar_final, dtype=np.float32)
        reward_history_arr = np.asarray(reward_hist, dtype=np.float32)
        diffusion_actions_traj_arr = np.asarray(Ybar_hist, dtype=np.float32)
        diffusion_samples_traj_arr = np.asarray(Ysamples_hist, dtype=np.float32)
        
        return (rng_out, actions_np_final, reward_history_arr,
               diffusion_actions_traj_arr, diffusion_samples_traj_arr)

