from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from enerdynamics.core.constraints.legacy.base import ActionFilterOperator
from enerdynamics.envs.obstacles.base import ObstacleManager

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# Import registry system
try:
    from enerdynamics.core.registry.action_filters import get_action_filter_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    get_action_filter_registry = None

# Import RuntimeBackendManager to detect backend
try:
    from enerdynamics.core.backends.runtime import RuntimeBackendManager
    BACKEND_MANAGER_AVAILABLE = True
except ImportError:
    BACKEND_MANAGER_AVAILABLE = False
    RuntimeBackendManager = None

if TYPE_CHECKING:
    from enerdynamics.core.constraints.legacy.schedule import ConstraintScheduleManager


@dataclass
class CBFDoubleIntegrator2DActionFilter(ActionFilterOperator):
    """
    Legacy CBF Action Filter (DEPRECATED).
    
    ⚠️ This class is deprecated. Use the new architecture instead:
    - Replace with: CBFConvexifier + PerStepQPFilter
    - See legacy/MIGRATION_GUIDE.md for migration instructions
    
    This class is kept for backward compatibility only.
    """
    """
    CBF-style action filter for a 2D double-integrator state:
        x = [px, py, vx, vy]
        u = [ax, ay]

    Uses an SDF barrier h(p) = sdf(p) - (robot_radius + clearance) with a
    relative-degree-2 condition (like the MDOC implementation):
        n^T a >= -(k1 * hdot + k0 * h)
    where n is the normalized SDF gradient and hdot = n^T v.

    This is intended to be used as a hard "safety layer" during rollouts.
    """

    obstacles: ObstacleManager
    robot_radius: float = 0.05
    dt: float = 0.1
    tau: float = 0.05  # only activate filter when h < tau
    k0: float = 1.0
    k1: float = 4.0
    schedule_manager: Optional["ConstraintScheduleManager"] = None
    
    def __post_init__(self):
        """Initialize backend implementation after dataclass initialization."""
        # Determine backend name from RuntimeBackendManager
        backend_name = None
        if BACKEND_MANAGER_AVAILABLE and RuntimeBackendManager is not None:
            try:
                backend = RuntimeBackendManager.get_backend()
                backend_name = backend.name
            except Exception:
                # If backend detection fails, default to "numpy"
                backend_name = "numpy"
        else:
            backend_name = "numpy"  # Default to numpy if manager not available
        
        # Get backend implementation from registry
        self._backend_impl = None
        if REGISTRY_AVAILABLE and get_action_filter_registry is not None:
            registry = get_action_filter_registry()
            impl_class = registry.get("cbf_double_integrator_2d", backend_name)
            if impl_class is not None:
                # Create backend implementation instance
                config = {
                    'robot_radius': self.robot_radius,
                    'dt': self.dt,
                    'tau': self.tau,
                    'k0': self.k0,
                    'k1': self.k1,
                    'schedule_manager': self.schedule_manager,
                }
                try:
                    self._backend_impl = impl_class(self.obstacles, **config)
                    print(f"[CBF ActionFilter] Using {backend_name} backend implementation from registry")
                except Exception as e:
                    print(f"[CBF ActionFilter] WARNING: Failed to create {backend_name} backend implementation: {e}")
                    self._backend_impl = None
            else:
                available_backends = registry.list_backends("cbf_double_integrator_2d")
                print(f"[CBF ActionFilter] WARNING: {backend_name} backend not found in registry. "
                      f"Available backends: {available_backends}. Falling back to legacy implementation.")

    def _get_clearance(self, step: Optional[int], total_steps: Optional[int]) -> float:
        if self.schedule_manager is None:
            return 0.0
        return float(
            self.schedule_manager.get_hard_clearance(default=0.0, step=step, total_steps=total_steps)
        )

    def filter_actions_numpy(
        self,
        x0,
        actions: np.ndarray,
        *,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Filter actions using backend implementation or legacy NumPy implementation."""
        # Use registry-based backend implementation if available
        if self._backend_impl is not None:
            return self._backend_impl.filter_actions(x0, actions, step, total_steps)
        
        # Fallback to legacy NumPy implementation
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != 2:
            return actions
        if self.obstacles.get_sdf_texture_2d() is None:
            # No texture: fallback to do nothing (caller can decide to build texture)
            return actions

        tex = self.obstacles.get_sdf_texture_2d()
        clearance = self._get_clearance(step, total_steps)
        margin = float(self.robot_radius + clearance)
        dt = float(self.dt)

        x = np.asarray(x0, dtype=np.float32).copy()
        for t in range(actions.shape[0]):
            p = x[:2]
            v = x[2:4]
            a = actions[t]
            sdf, grad = tex.sample(p, backend="numpy")
            sdf = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
            grad = np.asarray(grad, dtype=np.float32).reshape(2)
            gnorm = float(np.linalg.norm(grad))
            if gnorm > 1e-6:
                n = grad / gnorm
                h = sdf - margin
                if h < self.tau:
                    hdot = float(np.dot(n, v))
                    b = -(self.k1 * hdot + self.k0 * h)
                    lhs = float(np.dot(n, a))
                    delta = max(0.0, b - lhs)  # ||n||^2 ~ 1
                    a = a + delta * n
                    actions[t] = a.astype(np.float32)

            # integrate one step (double integrator)
            # NOTE: We intentionally do not clamp here; caller/environment handles bounds.
            v_next = v + dt * a
            p_next = p + dt * v_next
            x = np.concatenate([p_next, v_next]).astype(np.float32)

        return actions

    def make_jax_filter(self):
        """Create JAX filter function using backend implementation or legacy code."""
        # Try to use backend implementation first
        if self._backend_impl is not None:
            if hasattr(self._backend_impl, 'make_jax_filter'):
                jax_filter = self._backend_impl.make_jax_filter()
                if jax_filter is not None:
                    return jax_filter
        
        # Fallback to legacy JAX implementation
        if jnp is None or jax is None:
            return None
        tex = self.obstacles.get_sdf_texture_2d()
        if tex is None:
            return None
        tex_j = tex.to_jax()  # (3,H,W)
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
            # p2: (...,2)
            p2 = jnp.asarray(p2, dtype=jnp.float32)
            shp = p2.shape[:-1]
            pts = p2.reshape((-1, 2))
            ix = (pts[:, 0] - x_min) / res
            iy = (pts[:, 1] - y_min) / res
            ix = jnp.clip(ix, 0.0, W - 1.0)
            iy = jnp.clip(iy, 0.0, H - 1.0)
            x0 = jnp.floor(ix).astype(jnp.int32)
            y0 = jnp.floor(iy).astype(jnp.int32)
            x1 = jnp.minimum(x0 + 1, W - 1)
            y1 = jnp.minimum(y0 + 1, H - 1)
            wx = (ix - x0).astype(jnp.float32)
            wy = (iy - y0).astype(jnp.float32)
            v00 = tex_j[:, y0, x0]
            v10 = tex_j[:, y0, x1]
            v01 = tex_j[:, y1, x0]
            v11 = tex_j[:, y1, x1]
            v0 = v00 * (1.0 - wx) + v10 * wx
            v1 = v01 * (1.0 - wx) + v11 * wx
            v = v0 * (1.0 - wy) + v1 * wy
            sdf = v[0].reshape(shp)
            grad = jnp.stack([v[1], v[2]], axis=-1).reshape(shp + (2,))
            return sdf, grad

        def filter_fn(state, action, hard_clearance, hard_enabled):
            # state: (...,4), action: (...,2)
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

        return filter_fn

    def make_torch_filter(self):
        """Create PyTorch filter function using backend implementation or legacy code."""
        # Try to use backend implementation first
        if self._backend_impl is not None:
            if hasattr(self._backend_impl, 'make_torch_filter'):
                torch_filter = self._backend_impl.make_torch_filter()
                if torch_filter is not None:
                    return torch_filter
        
        # Fallback to legacy PyTorch implementation
        if torch is None:
            return None
        tex = self.obstacles.get_sdf_texture_2d()
        if tex is None:
            return None
        # Use the texture sampler for torch via grid_sample
        # We return a small closure that expects tensors on the same device.
        tau = float(self.tau)
        k0 = float(self.k0)
        k1 = float(self.k1)
        robot_radius = float(self.robot_radius)

        def filter_fn(state, action, hard_clearance, hard_enabled):
            # state: (...,4), action: (...,2)
            device = action.device
            p = state[..., 0:2]
            v = state[..., 2:4]
            sdf, grad = tex.sample(p, backend="torch", device=device)
            gnorm = torch.linalg.norm(grad, dim=-1).clamp_min(1e-12)
            n = grad / gnorm.unsqueeze(-1)
            margin = robot_radius + hard_clearance
            h = sdf - margin
            hdot = (n * v).sum(dim=-1)
            b = -(k1 * hdot + k0 * h)
            lhs = (n * action).sum(dim=-1)
            delta = (b - lhs).clamp_min(0.0)
            do = (gnorm > 1e-6) & (h < tau) & hard_enabled
            action_new = action + delta.unsqueeze(-1) * n
            return torch.where(do.unsqueeze(-1), action_new, action)

        return filter_fn

