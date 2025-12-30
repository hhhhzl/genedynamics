"""
CBF (Control Barrier Function) convexifier.

Converts nonlinear safety constraints into linear inequalities in control space
using CBF theory: h_dot >= -alpha * h => linear constraint in u.
"""

from typing import Optional, Callable, Any
import numpy as np

from enerdynamics.core.constraints.convexify.base import Convexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import get_registry, register
from enerdynamics.core.types import State
from enerdynamics.envs.obstacles.base import ObstacleManager


@register("convexifier", "cbf", "numpy")
class CBFConvexifier(Convexifier):
    """
    CBF convexifier: Nonlinear safety -> linear inequalities in u.
    
    For a double integrator with CBF h(p) = sdf(p) - margin, this generates
    linear constraints in control space: n^T a >= -(k1 * hdot + k0 * h)
    where n is the normalized SDF gradient.
    
    Supports per-step constraints (one constraint per time step).
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        dynamics: Any,  # Dynamics model
        robot_radius: float = 0.05,
        dt: float = 0.1,
        k0: float = 1.0,
        k1: float = 4.0,
        tau: float = 0.05,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        backend: str = "numpy",
        **kwargs
    ):
        """
        Initialize CBF convexifier.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            dynamics: Dynamics model (for computing hdot)
            robot_radius: Robot radius
            dt: Time step
            k0: CBF gain k0
            k1: CBF gain k1
            tau: Activation threshold (only activate when h < tau)
            position_extractor: Function to extract position from state
            backend: Backend to use ("numpy", "jax", "torch")
            **kwargs: Additional arguments
        """
        self.obstacles = obstacles
        self.dynamics = dynamics
        self.robot_radius = robot_radius
        self.dt = dt
        self.k0 = k0
        self.k1 = k1
        self.tau = tau
        self.position_extractor = position_extractor or self._default_extract_position
        self.backend = backend
        
        # Get backend implementation from registry
        registry = get_registry()
        impl_class = registry.get("convexifier", "cbf", backend)
        if impl_class is not None and impl_class != CBFConvexifier:
            self._backend_impl = impl_class(obstacles, dynamics, **kwargs)
        else:
            self._backend_impl = None
    
    def build_constraints(
        self,
        ref: Any,  # Can be trajectory or (state, action) tuple
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CBF constraints.
        
        Args:
            ref: Reference trajectory or (state, action) tuple
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            ConvexConstraint with per-step constraints
        """
        if self._backend_impl is not None:
            return self._backend_impl.build_constraints(ref, params, state)
        
        # Fallback to basic implementation
        return self._build_constraints_basic(ref, params, state)
    
    def _build_constraints_basic(
        self,
        ref: Any,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """Basic implementation (fallback)."""
        # Handle trajectory or single state
        if hasattr(ref, 'states'):
            # Trajectory
            states = ref.states
            actions = ref.actions if hasattr(ref, 'actions') else [None] * len(states)
        else:
            # Single state-action pair
            states = [ref[0]] if isinstance(ref, tuple) else [ref]
            actions = [ref[1]] if isinstance(ref, tuple) else [None]
        
        # Build constraints for each time step
        As = []
        bs = []
        
        for i, (x, u) in enumerate(zip(states, actions)):
            # Extract position and velocity
            x_np = np.asarray(x, dtype=np.float32)
            p = self.position_extractor(x_np)
            v = x_np[2:4] if len(x_np) >= 4 else np.zeros(2, dtype=np.float32)
            
            # Get SDF and gradient
            sdf = self.obstacles.sdf(p)
            if hasattr(self.obstacles, 'gradient'):
                grad = self.obstacles.gradient(p)
            else:
                grad = self._finite_difference_gradient(p)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                # No valid gradient, skip constraint
                continue
            
            n = grad / grad_norm
            
            # CBF value
            margin = params.margin + self.robot_radius
            h = sdf - margin
            
            # Only activate if h < tau
            if h >= self.tau:
                continue
            
            # hdot = n^T v
            hdot = np.dot(n, v)
            
            # CBF constraint: n^T a >= -(k1 * hdot + k0 * h)
            A_row = n[None, :]  # (1, action_dim)
            b_val = -(self.k1 * hdot + self.k0 * h)
            
            As.append(A_row)
            bs.append(b_val)
        
        # Stack constraints
        if As:
            A = np.vstack(As)  # (m, action_dim)
            b = np.array(bs, dtype=np.float32)  # (m,)
        else:
            # No active constraints
            A = np.zeros((0, 2), dtype=np.float32)  # action_dim = 2 for 2D
            b = np.zeros((0,), dtype=np.float32)
        
        return ConvexConstraint(
            A=A,
            b=b,
            meta={"per_step": True, "type": "cbf"}
        )
    
    def _finite_difference_gradient(self, point: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Compute gradient using finite differences."""
        point = np.asarray(point, dtype=np.float32)
        dim = point.shape[0]
        grad = np.zeros(dim, dtype=np.float32)
        
        for i in range(dim):
            xp = point.copy()
            xm = point.copy()
            xp[i] += eps
            xm[i] -= eps
            
            sdf_p = self.obstacles.sdf(xp)
            sdf_m = self.obstacles.sdf(xm)
            
            grad[i] = (sdf_p - sdf_m) / (2 * eps)
        
        return grad
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """Default position extractor."""
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:  # 2D double integrator
            return state_np[:2]
        elif len(state_np) == 2:  # 1D double integrator
            return state_np[:1]
        return state_np[:min(2, len(state_np))]

