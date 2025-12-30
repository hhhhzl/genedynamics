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
        _skip_backend_lookup: bool = False,
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
            _skip_backend_lookup: Internal flag to skip backend lookup (prevents recursion)
            **kwargs: Additional arguments
        """
        self.obstacles = obstacles
        self.position_extractor = position_extractor or self._default_extract_position
        self.max_constraints_per_point = max_constraints_per_point
        self.constraint_margin = constraint_margin
        self.backend = backend
        
        # Get backend implementation from registry
        # Skip if this is already a backend implementation (prevents recursion)
        # Backend implementations (CFSNumpyConvexifier, CFSJAXConvexifier) should not
        # try to look up another backend implementation
        if not _skip_backend_lookup:
            # Check if this class is registered as a backend implementation
            # If so, we are already a backend implementation and should not look up another one
            registry = get_registry()
            current_class = type(self)
            is_backend_impl = False
            
            # Check if current class is registered as a backend
            for reg_backend in ["numpy", "jax", "torch"]:
                reg_class = registry.get("convexifier", "cfs", reg_backend)
                if reg_class is not None and reg_class == current_class:
                    is_backend_impl = True
                    break
            
            # Only look up backend if we're not already a backend implementation
            if not is_backend_impl:
                impl_class = registry.get("convexifier", "cfs", backend)
                # Only use backend impl if it's different from this class (prevents recursion)
                if impl_class is not None and impl_class != CFSConvexifier and impl_class != current_class:
                    # Pass _skip_backend_lookup=True to prevent recursion
                    self._backend_impl = impl_class(
                        obstacles, position_extractor,
                        max_constraints_per_point=max_constraints_per_point,
                        constraint_margin=constraint_margin,
                        backend=backend,
                        _skip_backend_lookup=True,
                        **kwargs
                    )
                else:
                    self._backend_impl = None
            else:
                # We are already a backend implementation, use ourselves
                self._backend_impl = None
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
        
        # If no backend implementation is found, raise an error
        # Backend implementations should be registered and available
        raise RuntimeError(
            f"No backend implementation found for CFS convexifier with backend '{self.backend}'. "
            f"Please ensure the backend implementation is properly registered."
        )
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """
        Default position extractor.
        
        Extracts position from state vector. Assumes:
        - 2D double integrator: state = [x, y, vx, vy] (len=4) -> position = [x, y]
        - 2D single integrator: state = [x, y] (len=2) -> position = [x, y]
        - 1D double integrator: state = [x, vx] (len=2) -> position = [x]
        - Otherwise: take first 2 elements (or fewer if state is shorter)
        
        Note: For len=2, we assume it's 2D position (single integrator) by default.
        If you need 1D double integrator, provide a custom position_extractor.
        """
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:  # 2D double integrator: [x, y, vx, vy]
            return state_np[:2]
        elif len(state_np) == 2:  # Assume 2D single integrator: [x, y]
            # For single integrator, the entire state is position
            return state_np[:2]
        # For other lengths, take first 2 elements (or all if shorter)
        return state_np[:min(2, len(state_np))]
