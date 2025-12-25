"""
JAX backend implementation of CBF double integrator 2D action filter.
"""

import numpy as np
from typing import Optional, Callable

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from enerdynamics.core.constraints.legacy.action_filters.backend_impl import ActionFilterBase
from enerdynamics.core.registry.action_filters import register_action_filter
from enerdynamics.core.types import State
from enerdynamics.envs.obstacles.base import ObstacleManager


if JAX_AVAILABLE:
    @register_action_filter("cbf_double_integrator_2d", "jax")
    class CBFActionFilterJax(ActionFilterBase):
        """
        JAX backend implementation of CBF double integrator 2D action filter.
        
        This implementation uses JAX operations optimized for GPU acceleration
        and JIT compilation. The filter function is designed to be JIT-compiled
        for performance.
        """
        
        def __init__(self, obstacles: ObstacleManager, **config):
            """
            Initialize JAX backend CBF action filter.
            
            Args:
                obstacles: ObstacleManager containing obstacles
                **config: Configuration parameters:
                    - robot_radius: Robot radius
                    - dt: Time step
                    - tau: Activation threshold
                    - k0: CBF gain k0
                    - k1: CBF gain k1
                    - schedule_manager: Optional ConstraintScheduleManager
            """
            super().__init__(obstacles, **config)
            self.robot_radius = config.get('robot_radius', 0.05)
            self.dt = config.get('dt', 0.1)
            self.tau = config.get('tau', 0.05)
            self.k0 = config.get('k0', 1.0)
            self.k1 = config.get('k1', 4.0)
            self.schedule_manager = config.get('schedule_manager', None)
            
            # Build JAX filter function
            self._jax_filter_fn = None
            self._build_jax_filter()
        
        def _build_jax_filter(self):
            """Build JAX-compatible filter function."""
            if jnp is None or jax is None:
                return
            
            tex = self.obstacles.get_sdf_texture_2d()
            if tex is None:
                return
            
            tex_j = tex.to_jax()  # (3, H, W)
            x_min = float(tex.x_min)
            y_min = float(tex.y_min)
            res = float(tex.res)
            H = int(tex.H)
            W = int(tex.W)
            tau = float(self.tau)
            k0 = float(self.k0)
            k1 = float(self.k1)
            robot_radius = float(self.robot_radius)
            
            def _sample(p2):
                """Sample SDF texture at positions p2."""
                # p2: (..., 2)
                p2 = jnp.asarray(p2, dtype=jnp.float32)
                shp = p2.shape[:-1]
                pts = p2.reshape((-1, 2))
                ix = (pts[:, 0] - x_min) / res
                iy = (pts[:, 1] - y_min) / res
                ix = jnp.clip(ix, 0.0, W - 1.0)
                iy = jnp.clip(iy, 0.0, H - 1.0)
                x0_idx = jnp.floor(ix).astype(jnp.int32)
                y0_idx = jnp.floor(iy).astype(jnp.int32)
                x1_idx = jnp.minimum(x0_idx + 1, W - 1)
                y1_idx = jnp.minimum(y0_idx + 1, H - 1)
                wx = (ix - x0_idx).astype(jnp.float32)
                wy = (iy - y0_idx).astype(jnp.float32)
                v00 = tex_j[:, y0_idx, x0_idx]
                v10 = tex_j[:, y0_idx, x1_idx]
                v01 = tex_j[:, y1_idx, x0_idx]
                v11 = tex_j[:, y1_idx, x1_idx]
                v0 = v00 * (1.0 - wx) + v10 * wx
                v1 = v01 * (1.0 - wx) + v11 * wx
                v = v0 * (1.0 - wy) + v1 * wy
                sdf = v[0].reshape(shp)
                grad = jnp.stack([v[1], v[2]], axis=-1).reshape(shp + (2,))
                return sdf, grad
            
            def filter_fn(state, action, hard_clearance, hard_enabled):
                """JAX filter function."""
                # state: (..., 4), action: (..., 2)
                state = jnp.asarray(state, dtype=jnp.float32)
                action = jnp.asarray(action, dtype=jnp.float32)
                p = state[..., 0:2]
                v = state[..., 2:4]
                sdf, grad = _sample(p)
                gnorm = jnp.linalg.norm(grad, axis=-1)
                n = grad / (gnorm[..., None] + 1e-12)
                margin = robot_radius + jnp.asarray(hard_clearance, dtype=jnp.float32)
                h = sdf - margin
                hdot = jnp.sum(n * v, axis=-1)
                b = -(k1 * hdot + k0 * h)
                lhs = jnp.sum(n * action, axis=-1)
                delta = jnp.clip(b - lhs, a_min=0.0)
                do = (gnorm > 1e-6) & (h < tau) & jnp.asarray(hard_enabled, dtype=bool)
                action_new = action + delta[..., None] * n
                return jnp.where(do[..., None], action_new, action)
            
            self._jax_filter_fn = filter_fn
        
        def filter_actions(
            self,
            x0: State,
            actions: np.ndarray,
            step: Optional[int],
            total_steps: Optional[int],
            **kwargs
        ) -> np.ndarray:
            """
            Filter actions to ensure safety using JAX implementation.
            
            This method converts inputs to JAX, applies the filter, and converts back.
            For better performance, use make_jax_filter() to get a JIT-compiled function.
            
            Args:
                x0: Initial state
                actions: Actions to filter, shape (T, action_dim)
                step: Current step (for scheduling)
                total_steps: Total steps (for scheduling)
                **kwargs: Additional parameters
                
            Returns:
                Filtered actions, shape (T, action_dim)
            """
            if self._jax_filter_fn is None:
                # Fallback: return actions unchanged if JAX filter not available
                # (This should not happen if obstacles support JAX)
                return np.asarray(actions, dtype=np.float32)
            
            # Convert to JAX arrays
            x0_jax = jnp.asarray(x0, dtype=jnp.float32)
            actions_jax = jnp.asarray(actions, dtype=jnp.float32)
            clearance = self._get_clearance(self.schedule_manager, step, total_steps)
            hard_enabled = True
            
            # Apply filter step by step (JAX filter is designed for single-step filtering)
            x = x0_jax
            filtered_actions = []
            for t in range(actions.shape[0]):
                action = actions_jax[t:t+1]  # (1, 2)
                state = x[None]  # (1, 4)
                
                # Apply filter
                filtered_action = self._jax_filter_fn(
                    state, action, clearance, hard_enabled
                )
                filtered_action = filtered_action[0]  # Remove batch dimension
                filtered_actions.append(filtered_action)
                
                # Integrate one step (double integrator)
                p = x[:2]
                v = x[2:4]
                a = filtered_action
                v_next = v + self.dt * a
                p_next = p + self.dt * v_next
                x = jnp.concatenate([p_next, v_next])
            
            # Stack and convert back to NumPy
            filtered_jax = jnp.stack(filtered_actions, axis=0)
            return np.asarray(filtered_jax)
        
        def make_jax_filter(self) -> Optional[Callable]:
            """
            Return a JAX-callable filter function for JIT compilation.
            
            Returns:
                JAX function (state, action, hard_clearance, hard_enabled) -> filtered_action,
                or None if not available
            """
            return self._jax_filter_fn

else:
    # JAX not available - don't register the class
    pass

