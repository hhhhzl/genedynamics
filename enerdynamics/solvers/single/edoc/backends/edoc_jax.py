"""
JAX backend implementation for EDOC reverse diffusion.

This module contains the JAX-optimized implementation of the reverse diffusion
algorithm, including JIT-compiled functions for high performance.
"""

from typing import Optional, Tuple, Any
import numpy as np

import jax
import jax.numpy as jnp

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


@register_edoc_backend("jax")
class EDOCBackendJax(EDOCBackendBase):
    """
    JAX backend implementation of EDOC reverse diffusion.
    
    This implementation uses JAX JIT compilation for high performance.
    It requires JAX-compatible environment and constraint functions.
    """
    
    def __init__(self, planner: Any, **config: Any):
        """Initialize JAX backend implementation."""
        super().__init__(planner, **config)
        self._reverse_diffuse_jit = None
        self._reverse_diffuse_chunk_jit = None
        self._jax_cfs_projector = None
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize JIT-compiled functions."""
        # This initialization is done in planner._init_reverse_diffuse_jit()
        # We'll set up the JIT functions when reverse_diffuse is first called
        # For now, we mark that initialization is needed
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure JIT functions are initialized."""
        if self._initialized:
            return
        
        # Call the planner's initialization method to set up JIT functions
        # We'll extract the logic here, but for now delegate to planner
        if hasattr(self.planner, '_init_reverse_diffuse_jit'):
            self.planner._init_reverse_diffuse_jit()
            self._reverse_diffuse_jit = self.planner._reverse_diffuse_jit
            self._reverse_diffuse_chunk_jit = self.planner._reverse_diffuse_chunk_jit
            self._jax_cfs_projector = getattr(self.planner, '_jax_cfs_projector', None)
            self._initialized = True
    
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
                from enerdynamics.core.constraints.obstacle_constraints import ObstacleSoftConstraint
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
        Run reverse diffusion using JAX JIT-compiled functions.
        
        This method implements the JAX branch of _run_action_reverse.
        If parameters are not provided, they will be prepared automatically.
        """
        # Ensure JIT functions are initialized by calling planner's initialization
        if not self._initialized:
            if hasattr(self.planner, '_init_reverse_diffuse_jit'):
                self.planner._init_reverse_diffuse_jit()
            self._reverse_diffuse_jit = getattr(self.planner, '_reverse_diffuse_jit', None)
            self._reverse_diffuse_chunk_jit = getattr(self.planner, '_reverse_diffuse_chunk_jit', None)
            self._jax_cfs_projector = getattr(self.planner, '_jax_cfs_projector', None)
            self._initialized = True
        
        # If JIT functions are not available, fall back to NumPy backend behavior
        if self._reverse_diffuse_jit is None:
            # Delegate to NumPy backend as fallback
            from .edoc_numpy import EDOCBackendNumpy
            numpy_backend = EDOCBackendNumpy(self.planner)
            return numpy_backend.reverse_diffuse(
                rng_key, state_init, hard_clearance_by_idx,
                hard_enabled_by_idx, soft_alpha_by_idx, soft_beta_by_idx
            )
        
        horizon = self.horizon
        act_dim = self.env.act_dim
        Ndiffuse = self.action_diffuse_steps
        
        # Prepare constraint parameters if not provided
        if hard_clearance_by_idx is None:
            hard_clearance_by_idx, hard_enabled_by_idx, soft_alpha_by_idx, soft_beta_by_idx = \
                self._prepare_constraint_parameters(Ndiffuse)
        x0_jnp = jnp.asarray(state_init, dtype=jnp.float32)
        
        # Convert NumPy arrays to JAX arrays if needed
        if hard_clearance_by_idx is not None and not isinstance(hard_clearance_by_idx, jnp.ndarray):
            hard_clearance_by_idx = jnp.asarray(hard_clearance_by_idx, dtype=jnp.float32)
        if hard_enabled_by_idx is not None and not isinstance(hard_enabled_by_idx, jnp.ndarray):
            hard_enabled_by_idx = jnp.asarray(hard_enabled_by_idx, dtype=jnp.bool_)
        if soft_alpha_by_idx is not None and not isinstance(soft_alpha_by_idx, jnp.ndarray):
            soft_alpha_by_idx = jnp.asarray(soft_alpha_by_idx, dtype=jnp.float32)
        if soft_beta_by_idx is not None and not isinstance(soft_beta_by_idx, jnp.ndarray):
            soft_beta_by_idx = jnp.asarray(soft_beta_by_idx, dtype=jnp.float32)
        
        # Use chunked version if available and tqdm is enabled
        if self.show_tqdm and HAS_TQDM and self._reverse_diffuse_chunk_jit is not None:
            return self._reverse_diffuse_chunked(
                rng_key, x0_jnp, Ndiffuse, horizon, act_dim,
                hard_clearance_by_idx, hard_enabled_by_idx,
                soft_alpha_by_idx, soft_beta_by_idx
            )
        else:
            # Use full JIT version
            rng, diffuse_key = jax.random.split(rng_key)
            _, actions_jnp, reward_hist_jnp, traj_jnp, samples_jnp = self._reverse_diffuse_jit(
                diffuse_key,
                x0_jnp,
                hard_clearance_by_idx,
                hard_enabled_by_idx,
                soft_alpha_by_idx,
                soft_beta_by_idx,
            )
            actions_np_final = np.asarray(actions_jnp, dtype=np.float32)
            reward_history_arr = np.asarray(reward_hist_jnp, dtype=np.float32)
            diffusion_actions_traj_arr = np.asarray(traj_jnp, dtype=np.float32)
            diffusion_samples_traj_arr = np.asarray(samples_jnp, dtype=np.float32)
            
            return (rng, actions_np_final, reward_history_arr, 
                   diffusion_actions_traj_arr, diffusion_samples_traj_arr)
    
    def _reverse_diffuse_chunked(
        self,
        rng_key: Any,
        state_init: jnp.ndarray,
        Ndiffuse: int,
        horizon: int,
        act_dim: int,
        hard_clearance_by_idx: jnp.ndarray,
        hard_enabled_by_idx: jnp.ndarray,
        soft_alpha_by_idx: jnp.ndarray,
        soft_beta_by_idx: jnp.ndarray,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run reverse diffusion using chunked JIT for progress bar."""
        chunk_len = int(self._reverse_diffuse_chunk_len)
        if chunk_len < 1:
            chunk_len = 1
        total_steps = Ndiffuse - 1
        diffusion_indices = np.arange(Ndiffuse - 1, 0, -1, dtype=np.int32)
        
        rng, init_key = jax.random.split(rng_key)
        Ybar = jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32)
        
        reward_chunks = []
        ybar_chunks = []
        ysamp_chunks = []
        pbar = tqdm(total=total_steps, desc="EDOC Diffusion", unit="step", leave=False)
        try:
            for start in range(0, total_steps, chunk_len):
                sub = diffusion_indices[start:start + chunk_len]
                valid = np.ones((len(sub),), dtype=bool)
                if len(sub) < chunk_len:
                    pad_n = chunk_len - len(sub)
                    sub = np.pad(sub, (0, pad_n), mode="constant", constant_values=1)
                    valid = np.pad(valid, (0, pad_n), mode="constant", constant_values=False)
                idx_chunk = jnp.asarray(sub, dtype=jnp.int32)
                valid_chunk = jnp.asarray(valid, dtype=jnp.bool_)
                rng, Ybar, r_c, y_c, s_c = self._reverse_diffuse_chunk_jit(
                    rng,
                    state_init,
                    Ybar,
                    idx_chunk,
                    valid_chunk,
                    hard_clearance_by_idx,
                    hard_enabled_by_idx,
                    soft_alpha_by_idx,
                    soft_beta_by_idx,
                )
                reward_chunks.append(r_c)
                ybar_chunks.append(y_c)
                ysamp_chunks.append(s_c)
                pbar.update(int(np.sum(valid)))
        finally:
            pbar.close()
        
        reward_hist = jnp.concatenate(reward_chunks, axis=0)[:total_steps]
        Ybar_hist = jnp.concatenate(ybar_chunks, axis=0)[:total_steps]
        Ysamples_hist = jnp.concatenate(ysamp_chunks, axis=0)[:total_steps]
        
        # Match the full-jit function output convention (reverse to early→late)
        reward_hist_jnp = reward_hist[::-1]
        traj_jnp = Ybar_hist[::-1]
        samples_jnp = Ysamples_hist[::-1]
        actions_jnp = Ybar
        
        actions_np_final = np.asarray(actions_jnp, dtype=np.float32)
        reward_history_arr = np.asarray(reward_hist_jnp, dtype=np.float32)
        diffusion_actions_traj_arr = np.asarray(traj_jnp, dtype=np.float32)
        diffusion_samples_traj_arr = np.asarray(samples_jnp, dtype=np.float32)
        
        return (rng, actions_np_final, reward_history_arr,
               diffusion_actions_traj_arr, diffusion_samples_traj_arr)

