"""
JAX backend for CBF convexifier (double integrator 2D).

This implementation uses JAX for JIT compilation and GPU acceleration.
"""

from typing import Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.convexify.cbf.cbf import CBFConvexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import register


@register("convexifier", "cbf", "jax")
class CBFJAXConvexifier(CBFConvexifier):
    """
    JAX backend for CBF convexifier (double integrator 2D).
    
    This implementation uses JAX for:
    - JIT compilation
    - GPU acceleration
    - Automatic differentiation
    """
    
    def __init__(
        self,
        obstacles,
        dynamics,
        robot_radius: float = 0.05,
        alpha: float = 1.0,
        dt: float = 0.1,
        backend: str = "jax",
        use_jit: bool = True,
        build_traj_qp: bool = False,
        **kwargs
    ):
        """
        Initialize JAX backend CBF convexifier.
        
        Args:
            obstacles: ObstacleManager
            dynamics: Dynamics model
            robot_radius: Robot radius
            alpha: CBF parameter
            dt: Time step
            backend: Backend ("jax")
            use_jit: Whether to JIT compile
            **kwargs: Additional arguments
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install JAX to use CBFJAXConvexifier.")
        
        super().__init__(
            obstacles, dynamics, robot_radius, alpha, dt, backend,
            build_traj_qp=build_traj_qp, **kwargs
        )
        
        self.use_jit = use_jit
        if use_jit:
            self._jit_build = jax.jit(self._build_constraints_jax, static_argnums=(2,))
    
    def build_constraints(
        self,
        ref,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CBF constraints using JAX.
        
        Args:
            ref: Reference (state, action) pair
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            ConvexConstraint
        """
        if getattr(self, "build_traj_qp", False):
            raise NotImplementedError("CBF JAX trajectory-level QP not implemented; use numpy backend or per-step mode.")
        if self.use_jit:
            state_jax = jnp.asarray(ref[0])
            action_jax = jnp.asarray(ref[1])
            return self._jit_build(state_jax, action_jax, params, state)
        else:
            return self._build_constraints_jax(
                jnp.asarray(ref[0]),
                jnp.asarray(ref[1]),
                params,
                state
            )
    
    def _build_constraints_jax(
        self,
        state: jnp.ndarray,  # (4,) for 2D double integrator
        action: jnp.ndarray,  # (2,) for 2D
        params: ScheduleParams,
        state_schedule: ScheduleState
    ) -> ConvexConstraint:
        """
        JAX implementation of CBF constraint building using JAX autodiff.
        
        This is the function that gets JIT compiled.
        Uses JAX autodiff for gradient computation instead of finite differences.
        """
        # Extract position and velocity
        pos = state[:2]  # (2,)
        vel = state[2:4]  # (2,)
        
        # Try to use JAX-compatible obstacle methods
        use_jax_obstacles = False
        if hasattr(self.obstacles, 'jax_sdf'):
            use_jax_obstacles = True
        elif hasattr(self.obstacles, 'obstacles') and len(self.obstacles.obstacles) > 0:
            use_jax_obstacles = hasattr(self.obstacles.obstacles[0], 'jax_sdf')
        
        if use_jax_obstacles:
            # Use JAX-compatible obstacle methods
            if hasattr(self.obstacles, 'jax_sdf'):
                sdf = self.obstacles.jax_sdf(pos)  # scalar
                
                # Compute gradient using JAX autodiff
                def sdf_fn(p):
                    return self.obstacles.jax_sdf(p)
                
                grad_jax = jax.grad(sdf_fn)(pos)  # (2,)
            else:
                # Individual obstacles have jax_sdf
                def sdf_fn(p):
                    min_sdf = jnp.inf
                    for obs in self.obstacles.obstacles:
                        if hasattr(obs, 'jax_sdf'):
                            sdf_val = obs.jax_sdf(p)
                            min_sdf = jnp.minimum(min_sdf, sdf_val)
                    return min_sdf
                
                sdf = sdf_fn(pos)
                grad_jax = jax.grad(sdf_fn)(pos)  # (2,)
        else:
            # Fallback: Use JAX autodiff on NumPy SDF function
            def sdf_fn_np(pos_jax):
                """Wrapper that converts to numpy, calls sdf, converts back."""
                pos_np = np.asarray(pos_jax)
                sdf_val = self.obstacles.sdf(pos_np)
                return jnp.asarray(sdf_val)
            
            sdf = sdf_fn_np(pos)  # scalar
            grad_jax = jax.grad(sdf_fn_np)(pos)  # (2,) - This is the key improvement!
        
        # Normalize gradient
        grad_norm = jnp.linalg.norm(grad_jax)
        grad_norm = jnp.maximum(grad_norm, 1e-8)
        n = grad_jax / grad_norm
        
        # CBF: h = sdf - robot_radius
        h = sdf - self.robot_radius
        
        # CBF derivative: h_dot = grad^T * vel
        h_dot = jnp.dot(grad_jax, vel)
        
        # CBF constraint: h_dot >= -alpha * h
        # => grad^T * vel >= -alpha * h
        # => grad^T * (vel + dt * action) >= -alpha * h (for next state)
        # => grad^T * dt * action >= -alpha * h - grad^T * vel
        # => (dt * grad)^T * action >= -alpha * h - grad^T * vel
        
        margin = params.margin
        alpha = self.alpha
        
        # Constraint: (dt * grad)^T * u >= -alpha * h - grad^T * vel - margin
        A = (self.dt * grad_jax)[None, :]  # (1, 2)
        b = jnp.array([-alpha * h - h_dot - margin])
        
        # Keep as JAX arrays (ConvexConstraint should handle BackendArray)
        from enerdynamics.core.constraints.core.array_interface import BackendArray
        return ConvexConstraint(
            A=BackendArray(A, "jax"),
            b=BackendArray(b, "jax"),
            meta={"per_step": True, "type": "cbf", "backend": "jax"}
        )

