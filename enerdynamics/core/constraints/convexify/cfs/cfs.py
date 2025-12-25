"""
CFS (Convex Feasible Set) convexifier.

Converts non-convex obstacle constraints into local convex corridors
by linearizing SDF constraints around reference points.
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
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager


@register("convexifier", "cfs", "numpy")
class CFSConvexifier(Convexifier):
    """
    CFS convexifier: Non-convex obstacles -> local convex corridors.
    
    This convexifier linearizes SDF constraints around reference points
    to create local convex feasible sets (half-spaces).
    
    The output is a ConvexConstraint with A and b matrices representing
    linear inequalities A x >= b.
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        backend: str = "numpy",
        **kwargs
    ):
        """
        Initialize CFS convexifier.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            position_extractor: Function to extract position from state
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint tightening
            backend: Backend to use ("numpy", "jax", "torch")
            **kwargs: Additional arguments
        """
        self.obstacles = obstacles
        self.position_extractor = position_extractor or self._default_extract_position
        self.max_constraints_per_point = max_constraints_per_point
        self.constraint_margin = constraint_margin
        self.backend = backend
        
        # Get backend implementation from registry
        registry = get_registry()
        impl_class = registry.get("convexifier", "cfs", backend)
        if impl_class is not None and impl_class != CFSConvexifier:
            self._backend_impl = impl_class(obstacles, position_extractor, **kwargs)
        else:
            self._backend_impl = None
    
    def build_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CFS constraints from reference trajectory.
        
        Args:
            ref: Reference trajectory
            params: Schedule parameters (margin, etc.)
            state: Schedule state
            
        Returns:
            ConvexConstraint with A and b matrices
        """
        if self._backend_impl is not None:
            return self._backend_impl.build_constraints(ref, params, state)
        
        # Fallback to basic implementation
        return self._build_constraints_basic(ref, params, state)
    
    def _build_constraints_basic(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """Basic implementation (fallback)."""
        # Extract positions
        positions = [self.position_extractor(s) for s in ref.states]
        positions = np.stack(positions, axis=0)  # (H, dim)
        
        # Compute SDF and gradients for all positions
        margin = params.margin + self.constraint_margin
        constraints_list = []
        
        for pos in positions:
            # Get SDF and gradient
            sdf = self.obstacles.sdf(pos)
            if hasattr(self.obstacles, 'gradient'):
                grad = self.obstacles.gradient(pos)
            else:
                # Finite difference gradient
                grad = self._finite_difference_gradient(pos)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                n = grad / grad_norm
                # Constraint: n^T (x - pos) >= margin - sdf
                # => n^T x >= margin - sdf + n^T pos
                A_row = n[None, :]  # (1, dim)
                b_val = margin - sdf + np.dot(n, pos)
                constraints_list.append((A_row, b_val))
        
        # Stack constraints
        if constraints_list:
            A = np.vstack([c[0] for c in constraints_list])  # (m, dim)
            b = np.array([c[1] for c in constraints_list])  # (m,)
        else:
            A = np.zeros((0, positions.shape[1]), dtype=np.float32)
            b = np.zeros((0,), dtype=np.float32)
        
        return ConvexConstraint(
            A=A,
            b=b,
            meta={"per_step": False, "type": "cfs"}
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
