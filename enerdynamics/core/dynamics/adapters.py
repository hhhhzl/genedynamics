"""
Environment adapters for dynamics models.

This module provides adapters that bridge between environments and dynamics models:
- EnvDynamicsAdapter: Wraps an environment to implement DynamicsModel
- DynamicsToEnvAdapter: Converts a DynamicsModel to an environment-like object
"""

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

from enerdynamics.core.dynamics.base import DynamicsModel
from enerdynamics.core.types import State, Action


class EnvDynamicsAdapter(DynamicsModel):
    """
    Adapter that wraps an environment-like object to implement DynamicsModel.
    
    This allows existing environments to work with the new architecture
    without requiring full refactoring.
    """
    
    def __init__(self, env):
        """
        Initialize adapter with an environment.
        
        Args:
            env: Environment object with transition() and jax_transition() methods
        """
        self.env = env
        if hasattr(env, "act_dim"):
            self.act_dim = env.act_dim
        elif hasattr(env, "action_size"):
            self.act_dim = env.action_size
        else:
            self.act_dim = 1

    def __getattr__(self, name: str):
        """Forward attribute lookups (e.g. target) to the wrapped env for MBD/EDOC terminal cost."""
        return getattr(self.env, name)

    def step(self, x: State, u: Action) -> State:
        """
        Single-step dynamics using environment's transition.
        
        Args:
            x: Current state (can be numpy array or JAX array)
            u: Control action (can be numpy array or JAX array)
            
        Returns:
            Next state
        """
        x_data = np.asarray(x, dtype=np.float32) if not (jnp and isinstance(x, jnp.ndarray)) else x
        u_data = np.asarray(u, dtype=np.float32) if not (jnp and isinstance(u, jnp.ndarray)) else u
        
        if (jnp and (isinstance(x_data, jnp.ndarray) or isinstance(u_data, jnp.ndarray))):
            # Use JAX transition if available
            if hasattr(self.env, 'jax_model_transition'):
                return self.env.jax_model_transition(x_data, u_data)
            elif hasattr(self.env, 'jax_transition'):
                return self.env.jax_transition(x_data, u_data)
        
        # Fall back to numpy transition
        return self.env.transition(x_data, u_data)

    def jax_sdf(self, pos):
        """
        Pass-through JAX SDF if the wrapped environment provides it.
        If the wrapped env does not provide jax_sdf (e.g. D3IL plan env), return a fallback
        that is large positive everywhere (no obstacle) so EB-MBD and other barrier solvers
        can run without crashing; obstacle avoidance then relies on cost only.
        """
        target = getattr(self.env, "jax_sdf", None)
        if target is not None:
            return target(pos)
        if jnp is None:
            raise AttributeError("Underlying environment does not provide jax_sdf")
        # Fallback for envs without jax_sdf (e.g. d3il_unified plan env): free space everywhere
        return jnp.ones(pos.shape[:-1], dtype=jnp.float32) * 1e5


class DynamicsToEnvAdapter:
    """
    Adapter that converts a DynamicsModel or environment object to an
    environment-like object that EDOCPlanner expects.
    
    This adapter can wrap either:
    1. An environment object (e.g., DoubleIntegratorBoxEnv) - directly delegates
    2. A DynamicsModel - wraps step() method
    """
    
    def __init__(self, dynamics_or_env, dt: float = None, act_dim: int = None):
        """
        Initialize adapter.
        
        Args:
            dynamics_or_env: Either a DynamicsModel or an environment object
            dt: Time step (optional, inferred from env if available)
            act_dim: Action dimension (inferred if not provided)
        """
        self.dynamics = dynamics_or_env
        self.dt = dt
        
        # Try to infer dt and act_dim from environment if available
        if hasattr(dynamics_or_env, 'dt') and self.dt is None:
            self.dt = dynamics_or_env.dt
        if hasattr(dynamics_or_env, 'act_dim'):
            self.act_dim = dynamics_or_env.act_dim
        elif hasattr(dynamics_or_env, 'action_dim'):
            self.act_dim = dynamics_or_env.action_dim
        elif act_dim is not None:
            self.act_dim = act_dim
        else:
            # Default fallback
            self.act_dim = 1

    def __getattr__(self, name: str):
        """Forward attribute lookups (e.g. robot_radius, target) to wrapped dynamics/env."""
        return getattr(self.dynamics, name)
    
    def transition(self, state, action):
        """Numpy transition function."""
        state_np = np.asarray(state, dtype=np.float32)
        action_np = np.asarray(action, dtype=np.float32)
        
        # If it's an environment, use its transition method
        if hasattr(self.dynamics, 'transition'):
            return self.dynamics.transition(state_np, action_np)
        elif hasattr(self.dynamics, 'step'):
            next_state = self.dynamics.step(state_np, action_np)
            return np.asarray(next_state, dtype=np.float32)
        else:
            raise ValueError("Dynamics object must have 'transition' or 'step' method")
    
    def jax_transition(self, state, action):
        """
        JAX transition function.
        
        This method must be JAX-compatible and cannot use np.asarray()
        when called from JIT-compiled functions.
        
        The wrapped object (self.dynamics) can be either:
        1. An environment object with jax_transition/jax_model_transition methods
        2. A DynamicsModel that needs to be called via step()
        """
        # Case 1: If dynamics is an environment object (most common case in run_diffusion)
        # Directly use its JAX transition methods - these are already JAX-compatible
        if hasattr(self.dynamics, 'jax_model_transition'):
            # Use jax_model_transition if available (preferred for model-based planning)
            return self.dynamics.jax_model_transition(state, action)
        elif hasattr(self.dynamics, 'jax_transition'):
            # Fall back to jax_transition
            return self.dynamics.jax_transition(state, action)
        
        # Case 2: If dynamics is a DynamicsModel, we need to call step()
        # This assumes step() can handle JAX arrays and return JAX arrays
        # If not, the DynamicsModel needs to be JAX-compatible
        if hasattr(self.dynamics, 'step'):
            next_state = self.dynamics.step(state, action)
            # Ensure result is JAX array
            if jnp and isinstance(next_state, jnp.ndarray):
                return next_state
            else:
                # This should not happen in JIT context, but handle it for non-JIT calls
                # Use jnp.asarray instead of np.asarray to avoid tracer conversion error
                if jnp:
                    return jnp.asarray(next_state, dtype=jnp.float32)
                else:
                    return np.asarray(next_state, dtype=np.float32)
        
        raise ValueError(
            "Dynamics object must have 'jax_transition', 'jax_model_transition', or 'step' method"
        )
    
    def jax_model_transition(self, state, action):
        """JAX model transition (same as jax_transition for now)."""
        return self.jax_transition(state, action)
    
    def jax_env_transition(self, state, action):
        """JAX environment transition (same as jax_transition for now)."""
        return self.jax_transition(state, action)

    def jax_sdf(self, pos):
        """
        Pass-through JAX SDF if underlying env/dynamics provides it.
        """
        target = getattr(self.dynamics, "jax_sdf", None)
        if target is None:
            raise AttributeError("Underlying environment does not provide jax_sdf")
        return target(pos)
    
    def reset(self, rng=None):
        """Reset environment (returns dummy state)."""
        # EDOCPlanner will use the provided initial state, so this is not critical
        dummy_state = np.zeros(self.act_dim * 2, dtype=np.float32)  # Assume state_dim = 2 * act_dim
        return dummy_state, {}
    
    def step(self, state, action, t, info):
        """Environment step (for compatibility)."""
        next_state = self.transition(state, action)
        cost = 0.0  # Cost is computed by energy functional
        done = False
        return next_state, cost, done, info
    
    def cost(self, state):
        """Cost function (placeholder, energy handles this)."""
        return 0.0

    @property
    def target(self):
        """Pass-through target if underlying env/dynamics provides it."""
        return getattr(self.dynamics, "target", None)
