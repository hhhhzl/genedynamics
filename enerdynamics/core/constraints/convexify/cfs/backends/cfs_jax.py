"""
JAX backend for CFS convexifier.

This implementation uses JAX for JIT compilation and GPU acceleration.
"""

from typing import Optional, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import register
from enerdynamics.core.types import Trajectory


@register("convexifier", "cfs", "jax")
class CFSJAXConvexifier(CFSConvexifier):
    """
    JAX backend for CFS convexifier.
    
    This implementation uses JAX for:
    - JIT compilation
    - GPU acceleration
    - Automatic differentiation
    """
    
    def __init__(
        self,
        obstacles,
        position_extractor: Optional[Callable] = None,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        backend: str = "jax",
        use_jit: bool = True,
        **kwargs
    ):
        """
        Initialize JAX backend CFS convexifier.
        
        Args:
            obstacles: ObstacleManager
            position_extractor: Function to extract position from state
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint tightening
            backend: Backend ("jax")
            use_jit: Whether to JIT compile
            **kwargs: Additional arguments
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install JAX to use CFSJAXConvexifier.")
        
        super().__init__(
            obstacles, position_extractor, max_constraints_per_point,
            constraint_margin, backend, **kwargs
        )
        
        self.use_jit = use_jit
        if use_jit:
            self._jit_build = jax.jit(self._build_constraints_jax, static_argnums=(2,))
    
    def build_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CFS constraints using JAX.
        
        Args:
            ref: Reference trajectory
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            ConvexConstraint
        """
        if self.use_jit:
            # Convert to JAX arrays
            ref_states_jax = jnp.stack([jnp.asarray(s) for s in ref.states])
            return self._jit_build(ref_states_jax, params, state)
        else:
            return self._build_constraints_jax(
                jnp.stack([jnp.asarray(s) for s in ref.states]),
                params,
                state
            )
    
    def _build_constraints_jax(
        self,
        ref_states: jnp.ndarray,  # (H+1, state_dim)
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        JAX implementation of constraint building using JAX autodiff.
        
        This is the function that gets JIT compiled.
        Uses JAX-compatible obstacle methods or JAX autodiff for gradients.
        """
        # Extract positions
        positions = ref_states[:, :2]  # Assume first 2 dims are position
        
        # Try to use JAX-compatible obstacle methods
        # Check if obstacle manager or obstacles support JAX
        use_jax_obstacles = False
        if hasattr(self.obstacles, 'jax_sdf'):
            use_jax_obstacles = True
        elif hasattr(self.obstacles, 'obstacles') and len(self.obstacles.obstacles) > 0:
            # Check if individual obstacles support JAX
            use_jax_obstacles = hasattr(self.obstacles.obstacles[0], 'jax_sdf')
        
        if use_jax_obstacles:
            # Use JAX-compatible obstacle methods
            if hasattr(self.obstacles, 'jax_sdf'):
                # ObstacleManager has jax_sdf method
                sdfs_jax = self.obstacles.jax_sdf(positions)  # (H+1,)
                
                # Compute gradients using JAX autodiff
                # Define SDF function for autodiff
                def sdf_fn(pos):
                    return self.obstacles.jax_sdf(pos)
                
                # Compute gradients using jax.grad
                grad_fn = jax.vmap(jax.grad(sdf_fn))
                grads_jax = grad_fn(positions)  # (H+1, 2)
            else:
                # Individual obstacles have jax_sdf
                # For now, use autodiff on a wrapper function
                def sdf_fn(pos):
                    # Find closest obstacle and use its jax_sdf
                    min_sdf = jnp.inf
                    for obs in self.obstacles.obstacles:
                        if hasattr(obs, 'jax_sdf'):
                            sdf_val = obs.jax_sdf(pos)
                            min_sdf = jnp.minimum(min_sdf, sdf_val)
                    return min_sdf
                
                # Compute SDF values
                sdfs_jax = jax.vmap(sdf_fn)(positions)  # (H+1,)
                
                # Compute gradients using JAX autodiff
                grad_fn = jax.vmap(jax.grad(sdf_fn))
                grads_jax = grad_fn(positions)  # (H+1, 2)
        else:
            # Fallback: Use JAX autodiff on NumPy SDF function
            # This requires converting to/from numpy, but still uses JAX for gradients
            def sdf_fn_np(pos_jax):
                """Wrapper that converts to numpy, calls sdf, converts back."""
                pos_np = np.asarray(pos_jax)
                sdf_val = self.obstacles.sdf(pos_np)
                return jnp.asarray(sdf_val)
            
            # Compute SDF values (with numpy conversion)
            sdfs_jax = jax.vmap(sdf_fn_np)(positions)  # (H+1,)
            
            # Compute gradients using JAX autodiff (this is the key improvement!)
            grad_fn = jax.vmap(jax.grad(sdf_fn_np))
            grads_jax = grad_fn(positions)  # (H+1, 2)
        
        # Normalize gradients
        grad_norms = jnp.linalg.norm(grads_jax, axis=-1, keepdims=True)
        grad_norms = jnp.maximum(grad_norms, 1e-8)
        n = grads_jax / grad_norms  # (H+1, dim)
        
        # Build constraints: n^T (x - pos) >= margin - sdf
        # => n^T x >= margin - sdf + n^T pos
        margin = params.margin + self.constraint_margin
        A = n  # (H+1, dim)
        b = margin - sdfs_jax + jnp.sum(n * positions, axis=-1)
        
        # Keep as JAX arrays (ConvexConstraint should handle BackendArray)
        from enerdynamics.core.constraints.core.array_interface import BackendArray
        return ConvexConstraint(
            A=BackendArray(A, "jax"),
            b=BackendArray(b, "jax"),
            meta={"per_step": False, "type": "cfs", "backend": "jax"}
        )

