"""
NumPy backend implementation for EDOC reverse diffusion.

This module contains the NumPy-based implementation of the reverse diffusion
algorithm, using pure Python loops (no JIT compilation).
"""

from typing import Optional, Tuple, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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
    NumPy backend implementation of EDOC reverse diffusion.
    
    This implementation uses pure Python loops (no JIT compilation).
    It's slower than JAX but doesn't require JAX dependencies.
    """
    
    def __init__(self, planner: Any, **config: Any):
        """Initialize NumPy backend implementation."""
        super().__init__(planner, **config)
    
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
        Run reverse diffusion using NumPy (pure Python loops).
        
        This method implements the NumPy branch of _run_action_reverse.
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
        
        betas = np.linspace(beta0, betaT, Ndiffuse, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        sigmas = np.sqrt(1.0 - alphas_bar)
        extra_sigmas = np.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=np.float32)
        
        # Initialize Ybar using JAX random if available (for consistency), then convert to NumPy
        if JAX_AVAILABLE:
            rng, init_key = jax.random.split(rng_key)
            Ybar = np.array(
                jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32),
                dtype=np.float32,
            )
        else:
            # Fallback to NumPy random
            np.random.seed(int(rng_key) if isinstance(rng_key, (int, np.integer)) else 0)
            Ybar = np.random.randn(horizon, act_dim).astype(np.float32)
            rng = rng_key
        
        reward_history = []
        Ybar_history = []
        Ysamples_history = []
        
        # Create progress bar for reverse diffusion
        diffusion_iter = range(Ndiffuse - 1, 0, -1)
        if HAS_TQDM and self.show_tqdm:
            diffusion_iter = tqdm(diffusion_iter, desc="EDOC Diffusion", unit="step", 
                                 total=Ndiffuse-1, leave=False)
        
        for i in diffusion_iter:
            Yi = Ybar * np.sqrt(alphas_bar[i])
            
            # Generate random samples
            if JAX_AVAILABLE:
                rng, sample_key = jax.random.split(rng)
                sigma_i = float(sigmas[i])
                if self.use_antithetic and num_particles > 0:
                    half = num_particles // 2
                    has_extra = num_particles % 2
                    sample_count = half + has_extra
                    if sample_count > 0:
                        eps_core = np.array(
                            jax.random.normal(
                                sample_key, (sample_count, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                    else:
                        eps_core = np.zeros((0, horizon, act_dim), dtype=np.float32)
                    
                    Y_candidates = []
                    if half > 0:
                        eps_half = eps_core[:half]
                        Y_candidates.append(Ybar + sigma_i * eps_half)
                        Y_candidates.append(Ybar - sigma_i * eps_half)
                    if has_extra:
                        eps_extra = eps_core[-1:]
                        Y_candidates.append(Ybar + sigma_i * eps_extra)
                    
                    if not Y_candidates:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()
                    elif len(Y_candidates) == 1:
                        Y0s = Y_candidates[0]
                    else:
                        Y0s = np.concatenate(Y_candidates, axis=0)
                else:
                    if num_particles > 0:
                        eps = np.array(
                            jax.random.normal(
                                sample_key, (num_particles, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                        Y0s = Ybar + sigma_i * eps
                    else:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()
            else:
                # NumPy-only path
                sigma_i = float(sigmas[i])
                eps = np.random.randn(num_particles, horizon, act_dim).astype(np.float32) * sigma_i
                Y0s = Ybar[None, :] + eps
            
            if control_limit is not None:
                limit = float(control_limit)
                Y0s = np.clip(Y0s, -limit, limit)
            
            # Convert diffusion index to constraint step
            constraint_step = (Ndiffuse - 1) - i
            constraint_total_steps = Ndiffuse - 2
            
            # Score particles
            if self.action_score_mode == "energy":
                # Use batch energy rollout if available
                if hasattr(self.planner, '_batch_rollout_energy_fn') and self.planner._batch_rollout_energy_fn is not None:
                    actions_batch = jnp.asarray(Y0s, dtype=jnp.float32) if JAX_AVAILABLE else Y0s
                    hard_enabled = False
                    hard_clearance = 0.0
                    if self.constraint_manager and self.constraint_manager.action_filter_operator is not None:
                        hard_enabled = True
                    if self.constraint_manager and self.constraint_manager.schedule_manager is not None:
                        hard_enabled = bool(self.constraint_manager.schedule_manager.is_hard_active(constraint_step, constraint_total_steps))
                        hard_clearance = float(
                            self.constraint_manager.schedule_manager.get_hard_clearance(
                                default=0.0, step=constraint_step, total_steps=constraint_total_steps
                            )
                        )
                    energy_vals = np.asarray(self.planner._batch_rollout_energy_fn(
                        jnp.asarray(state_init, dtype=jnp.float32) if JAX_AVAILABLE else state_init,
                        actions_batch, hard_clearance, hard_enabled
                    ))
                    
                    # Add soft constraint penalties if enabled
                    if self.use_constraint_in_scoring and self.constraint_manager and self.constraint_manager.has_soft():
                        energy_vals = energy_vals / self.lambda_energy
                        trajectories = [self.planner._actions_to_trajectory(state_init, Y0s[idx]) for idx in range(len(Y0s))]
                        soft_penalties = self.constraint_manager.compute_soft_energy_batch(
                            trajectories, step=constraint_step, total_steps=constraint_total_steps
                        )
                        energy_vals = energy_vals + soft_penalties
                    
                    energy_mean = float(np.mean(energy_vals))
                    energy_std = float(np.std(energy_vals))
                    if energy_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        denom = energy_std * temp_eps
                        logw = -(energy_vals - energy_mean) / denom
                        logw = logw - np.max(logw)
                        weights = np.exp(logw).astype(np.float64)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(-np.mean(energy_vals)))
                else:
                    # Fallback: score using planner's method
                    scores = self.planner._score_particles(
                        state_init, {}, Y0s,
                        step=constraint_step,
                        total_steps=constraint_total_steps
                    )
                    score_std = float(scores.std())
                    if score_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        score_mean = float(scores.mean())
                        denom = score_std * temp_eps
                        logp0 = (scores - score_mean) / denom
                        logp0 = logp0 - np.max(logp0)
                        weights = np.exp(logp0)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(np.mean(scores)))
            else:
                # Use planner's scoring method
                if self.action_score_mode == "learned":
                    scores = np.zeros((num_particles,), dtype=np.float32)
                else:
                    scores = self.planner._score_particles(
                        state_init, {}, Y0s,
                        step=constraint_step,
                        total_steps=constraint_total_steps
                    )
                
                score_std = float(scores.std())
                if score_std < 1e-4:
                    weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                else:
                    score_mean = float(scores.mean())
                    denom = score_std * temp_eps
                    logp0 = (scores - score_mean) / denom
                    logp0 = logp0 - np.max(logp0)
                    weights = np.exp(logp0)
                    weights = weights / np.sum(weights)
                reward_history.append(float(np.mean(scores)))
            
            weights = np.asarray(weights, dtype=np.float32)
            
            Ybar_weighted = np.tensordot(weights, Y0s, axes=([0], [0]))
            
            score = (-Yi + np.sqrt(alphas_bar[i]) * Ybar_weighted) / (1.0 - alphas_bar[i])
            Yim1 = (Yi + (1.0 - alphas_bar[i]) * score) / np.sqrt(alphas[i])
            Ybar = Yim1 / np.sqrt(alphas_bar[i - 1])
            Ybar = self.planner._add_extra_noise(Ybar, extra_sigmas[i])
            
            # Apply hard constraint projection (state-space feasibility operator, e.g., CFS)
            if self.constraint_manager and self.constraint_manager.has_hard():
                trajectory = self.planner._actions_to_trajectory(state_init, Ybar)
                projected_trajectory = self.constraint_manager.project_hard(
                    trajectory,
                    step=constraint_step,
                    total_steps=constraint_total_steps,
                )
                
                # If projection edits states, map back to actions (tracking)
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
                    Ybar = self.planner._track_actions_to_projected_states(
                        state_init, Ybar, projected_trajectory.states
                    )
                else:
                    Ybar = self.planner._extract_actions_from_trajectory(projected_trajectory)
            
            if control_limit is not None:
                limit = float(control_limit)
                Ybar_history.append(np.clip(Ybar, -limit, limit))
                Ysamples_history.append(np.clip(Y0s, -limit, limit))
            else:
                Ybar_history.append(np.array(Ybar, copy=True))
                Ysamples_history.append(np.array(Y0s, copy=True))
        
        # Final hard constraint projection
        if self.constraint_manager and self.constraint_manager.has_hard():
            trajectory = self.planner._actions_to_trajectory(state_init, Ybar)
            projected_trajectory = self.constraint_manager.project_hard(
                trajectory, step=constraint_total_steps, total_steps=constraint_total_steps
            )
            Ybar = self.planner._track_actions_to_projected_states(
                state_init, Ybar, projected_trajectory.states
            )
            
            # Overwrite the last stored diffusion frame
            if len(Ybar_history) > 0:
                if control_limit is not None:
                    limit = float(control_limit)
                    Ybar_history[-1] = np.clip(np.asarray(Ybar, dtype=np.float32), -limit, limit)
                else:
                    Ybar_history[-1] = np.asarray(Ybar, dtype=np.float32)
        
        if control_limit is not None:
            limit = float(control_limit)
            actions_final = np.clip(Ybar, -limit, limit)
        else:
            actions_final = Ybar
        
        actions_np_final = np.asarray(actions_final, dtype=np.float32)
        if JAX_AVAILABLE:
            reward_history_arr = jnp.asarray(np.array(reward_history[::-1], dtype=np.float32))
            diffusion_actions_traj_arr = jnp.asarray(np.array(Ybar_history[::-1], dtype=np.float32))
            diffusion_samples_traj_arr = jnp.asarray(np.array(Ysamples_history[::-1], dtype=np.float32))
        else:
            reward_history_arr = np.array(reward_history[::-1], dtype=np.float32)
            diffusion_actions_traj_arr = np.array(Ybar_history[::-1], dtype=np.float32)
            diffusion_samples_traj_arr = np.array(Ysamples_history[::-1], dtype=np.float32)
        
        return (rng, actions_np_final, reward_history_arr,
               diffusion_actions_traj_arr, diffusion_samples_traj_arr)

