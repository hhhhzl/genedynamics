"""
CBF (Control Barrier Function) convexifier.

Converts nonlinear safety constraints into linear inequalities in control space
using CBF theory: h_dot >= -alpha * h => linear constraint in u.
"""

from typing import Optional, Callable, Any
import numpy as np

from genedynamics.core.constraints.convexify.base import Convexifier
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from genedynamics.core.constraints.core.registry import get_registry, register
from genedynamics.core.types import State
from genedynamics.core.task_spec import legacy_extract_position
from genedynamics.envs.obstacles.base import ObstacleManager


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
        build_traj_qp: bool = False,
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
        self.position_extractor = position_extractor or legacy_extract_position
        self.backend = backend
        self.build_traj_qp = bool(build_traj_qp)
        
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
        per_step_constraints = []  # list of (t, A_row, b_val)
        
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
            
            per_step_constraints.append((i, A_row, b_val))
        
        # No constraints at all
        if len(per_step_constraints) == 0:
            A_empty = np.zeros((0, len(actions[0]) if actions and actions[0] is not None else 2), dtype=np.float32)
            b_empty = np.zeros((0,), dtype=np.float32)
            return ConvexConstraint(
                A=A_empty,
                b=b_empty,
                meta={"per_step": True, "type": "cbf"}
            )

        # If not building trajectory-level QP: per-step stacked (existing behavior)
        if not self.build_traj_qp:
            As = [row for _, row, _ in per_step_constraints]
            bs = [val for _, _, val in per_step_constraints]
            A = np.vstack(As)
            b = np.asarray(bs, dtype=np.float32)
            return ConvexConstraint(
                A=A,
                b=b,
                meta={"per_step": True, "type": "cbf"}
            )

        # Trajectory-level: block-insert each timestep's control constraint into a single matrix
        H = len(actions)
        action_dim = per_step_constraints[0][1].shape[1]
        total_m = len(per_step_constraints)
        A_traj = np.zeros((total_m, H * action_dim), dtype=np.float32)
        b_traj = np.zeros((total_m,), dtype=np.float32)

        for idx, (t, A_row, b_val) in enumerate(per_step_constraints):
            start = t * action_dim
            A_traj[idx, start:start + action_dim] = A_row.flatten()
            b_traj[idx] = b_val

        return ConvexConstraint(
            A=A_traj,
            b=b_traj,
            meta={"per_step": False, "type": "cbf", "constrains": "actions"}
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
    
    # Uses legacy_extract_position when no custom position_extractor.

