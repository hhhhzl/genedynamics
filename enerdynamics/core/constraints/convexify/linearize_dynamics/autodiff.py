"""
Automatic differentiation for dynamics linearization.

This module provides utilities for computing Jacobians of dynamics
with respect to states and controls, enabling sensitivity analysis.
"""

from typing import Optional, Callable, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


class LinearizeDynamics:
    """
    Dynamics linearization using automatic differentiation.
    
    Computes Jacobians of dynamics:
    - d f(x, u) / d x: State Jacobian
    - d f(x, u) / d u: Control Jacobian
    
    These Jacobians are useful for:
    - Sensitivity analysis
    - Linearization-based constraint convexification
    - Trajectory optimization
    """
    
    def __init__(
        self,
        dynamics_fn: Callable,
        use_jax: bool = True
    ):
        """
        Initialize dynamics linearizer.
        
        Args:
            dynamics_fn: Dynamics function f(x, u) -> x_next
            use_jax: Whether to use JAX for automatic differentiation
        """
        self.dynamics_fn = dynamics_fn
        self.use_jax = use_jax and JAX_AVAILABLE
        
        if self.use_jax:
            # JIT compile Jacobian functions
            self._state_jacobian = jax.jit(jax.jacfwd(self.dynamics_fn, argnums=0))
            self._control_jacobian = jax.jit(jax.jacfwd(self.dynamics_fn, argnums=1))
    
    def state_jacobian(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Compute state Jacobian: d f(x, u) / d x.
        
        Args:
            state: Current state
            action: Current action
            
        Returns:
            State Jacobian matrix (state_dim, state_dim)
        """
        if self.use_jax:
            state_jax = jnp.asarray(state, dtype=jnp.float32)
            action_jax = jnp.asarray(action, dtype=jnp.float32)
            jac = self._state_jacobian(state_jax, action_jax)
            return np.asarray(jac)
        else:
            # Finite difference fallback
            return self._finite_difference_state_jacobian(state, action)
    
    def control_jacobian(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Compute control Jacobian: d f(x, u) / d u.
        
        Args:
            state: Current state
            action: Current action
            
        Returns:
            Control Jacobian matrix (state_dim, action_dim)
        """
        if self.use_jax:
            state_jax = jnp.asarray(state, dtype=jnp.float32)
            action_jax = jnp.asarray(action, dtype=jnp.float32)
            jac = self._control_jacobian(state_jax, action_jax)
            return np.asarray(jac)
        else:
            # Finite difference fallback
            return self._finite_difference_control_jacobian(state, action)
    
    def linearize(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize dynamics around (state, action).
        
        Returns: f(x, u) ≈ f(x0, u0) + A (x - x0) + B (u - u0)
        
        Args:
            state: Reference state
            action: Reference action
            
        Returns:
            Tuple of (f0, A, B)
            - f0: f(x0, u0)
            - A: State Jacobian (state_dim, state_dim)
            - B: Control Jacobian (state_dim, action_dim)
        """
        f0 = np.asarray(self.dynamics_fn(state, action), dtype=np.float32)
        A = self.state_jacobian(state, action)
        B = self.control_jacobian(state, action)
        
        return f0, A, B
    
    def _finite_difference_state_jacobian(
        self,
        state: np.ndarray,
        action: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """Compute state Jacobian using finite differences."""
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        
        state_dim = len(state)
        f0 = np.asarray(self.dynamics_fn(state, action), dtype=np.float32)
        output_dim = len(f0)
        
        jac = np.zeros((output_dim, state_dim), dtype=np.float32)
        
        for i in range(state_dim):
            state_perturbed = state.copy()
            state_perturbed[i] += eps
            f_perturbed = np.asarray(self.dynamics_fn(state_perturbed, action), dtype=np.float32)
            jac[:, i] = (f_perturbed - f0) / eps
        
        return jac
    
    def _finite_difference_control_jacobian(
        self,
        state: np.ndarray,
        action: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """Compute control Jacobian using finite differences."""
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        
        action_dim = len(action)
        f0 = np.asarray(self.dynamics_fn(state, action), dtype=np.float32)
        output_dim = len(f0)
        
        jac = np.zeros((output_dim, action_dim), dtype=np.float32)
        
        for i in range(action_dim):
            action_perturbed = action.copy()
            action_perturbed[i] += eps
            f_perturbed = np.asarray(self.dynamics_fn(state, action_perturbed), dtype=np.float32)
            jac[:, i] = (f_perturbed - f0) / eps
        
        return jac


