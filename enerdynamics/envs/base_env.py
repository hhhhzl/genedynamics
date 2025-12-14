"""
Base environment interface for the energy-driven control framework.

This module defines the BaseEnv protocol that all environments should implement.
It provides a unified interface for:
- State transitions
- Cost/reward computation
- Obstacle checking
- Rendering
- Multi-backend support (JAX, PyTorch, NumPy, Rust)
"""

from typing import Protocol, Optional, Tuple, Any, List, runtime_checkable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from enerdynamics.core.types import State, Action
from enerdynamics.core.backends import Backend
from enerdynamics.core.backends.render import RenderBackend
from enerdynamics.core.backends.physics import PhysicsBackend


@runtime_checkable
class BaseEnv(Protocol):
    """
    Protocol for environment interface.
    
    All environments should implement this interface to be compatible with
    the energy-driven control framework. This allows environments to work
    with different backends (JAX, PyTorch, NumPy, Rust) and solvers.
    
    Attributes:
        dt: Time step (float)
        horizon: Planning horizon (int)
        state_dim: State dimension (int, optional)
        act_dim: Action dimension (int, optional)
        backend: Computational backend (Backend, optional)
        obstacles: List of obstacles (List[Obstacle], optional)
    """
    
    dt: float
    horizon: int
    state_dim: Optional[int] = None
    act_dim: Optional[int] = None
    backend: Optional[Backend] = None
    obstacles: Optional[List[Any]] = None  # List[Obstacle], but avoid circular import
    render_backend: Optional[RenderBackend] = None
    physics_backend: Optional[PhysicsBackend] = None
    
    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.
        
        Args:
            rng: Random key/seed (JAX Array, int, or None)
            **kwargs: Additional reset parameters
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        ...
    
    def step(
        self,
        state: State,
        action: Action,
        t: Optional[int] = None,
        info: Optional[dict] = None
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.
        
        Args:
            state: Current state
            action: Action to take
            t: Current time step (optional)
            info: Additional info dict (optional)
            
        Returns:
            Tuple of (next_state, cost, done, info)
            Note: cost is used (not reward), where cost = -reward
        """
        ...
    
    def transition(self, state: State, action: Action) -> np.ndarray:
        """
        Compute next state from current state and action (deterministic).
        
        This is used by planners that need to simulate trajectories.
        Unlike step(), this does not modify internal environment state.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (numpy array)
        """
        ...
    
    def jax_transition(self, state: Any, action: Any) -> Any:
        """
        JAX-compatible transition function.
        
        This should be JIT-compilable and work with JAX arrays.
        If not implemented, will fall back to transition() with conversion.
        
        Args:
            state: Current state (JAX array)
            action: Action to take (JAX array)
            
        Returns:
            Next state (JAX array)
        """
        ...
    
    def cost(self, state: State) -> float:
        """
        Compute cost for a given state.
        
        Args:
            state: State to evaluate
            
        Returns:
            Cost (float, lower is better)
        """
        ...
    
    def jax_cost(self, state: Any) -> Any:
        """
        JAX-compatible cost function.
        
        Should work with JAX arrays and support vectorization.
        
        Args:
            state: State to evaluate (JAX array, can be batched)
            
        Returns:
            Cost (JAX array, can be batched)
        """
        ...
    
    def render(self, state: Optional[State] = None, mode: str = "human", **kwargs) -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            state: Optional state to render (if None, uses current state)
            mode: Rendering mode ("human", "rgb_array", "depth", etc.)
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        ...
    
    def check_collision(self, state: State) -> bool:
        """
        Check if state collides with obstacles.
        
        Args:
            state: State to check
            
        Returns:
            True if collision detected, False otherwise
        """
        ...


class BaseEnvMixin:
    """
    Mixin class providing default implementations for BaseEnv.
    
    Environments can inherit from this to get default behavior,
    then override specific methods as needed.
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 100,
        backend: Optional[Backend] = None,
        obstacles: Optional[List[Any]] = None,
        render_backend: Optional[RenderBackend] = None,
        physics_backend: Optional[PhysicsBackend] = None,
    ):
        """
        Initialize base environment.
        
        Args:
            dt: Time step
            horizon: Planning horizon
            backend: Computational backend
            obstacles: List of obstacles
            render_backend: Rendering backend (optional)
            physics_backend: Physics simulation backend (optional)
        """
        self.dt = dt
        self.horizon = horizon
        self.backend = backend
        self.obstacles = obstacles or []
        self.render_backend = render_backend
        self.physics_backend = physics_backend
        
        # If physics backend is provided, sync obstacles
        if self.physics_backend is not None and self.obstacles:
            for obstacle in self.obstacles:
                self.physics_backend.add_obstacle(obstacle)
    
    def check_collision(self, state: State) -> bool:
        """
        Default collision checking using obstacles.
        
        If obstacles are provided, checks if state collides with any obstacle.
        Subclasses can override for custom collision logic.
        """
        if not self.obstacles:
            return False
        
        # Extract position from state (assume first 3 dimensions are position)
        # This is a default assumption; subclasses should override if different
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) >= 3:
            position = state_np[:3]
        else:
            position = state_np
        
        # Check each obstacle
        for obstacle in self.obstacles:
            if hasattr(obstacle, 'contains'):
                if obstacle.contains(position):
                    return True
        
        return False
    
    def render(self, state: Optional[State] = None, mode: str = "human", **kwargs) -> Optional[Any]:
        """
        Default render using render_backend if available.
        
        Subclasses should override to provide actual rendering.
        """
        if self.render_backend is not None:
            return self.render_backend.render(state, mode=mode, **kwargs)
        
        # Fallback: no-op
        if mode == "human":
            print(f"Environment state: {self}")
        return None
    
    def jax_transition(self, state: Any, action: Any) -> Any:
        """
        Default JAX transition: convert to numpy, call transition(), convert back.
        
        Subclasses should override for better performance.
        """
        if jnp is None:
            raise RuntimeError("JAX is required for jax_transition")
        
        # Convert to numpy
        state_np = np.asarray(state)
        action_np = np.asarray(action)
        
        # Call numpy transition
        next_state_np = self.transition(state_np, action_np)
        
        # Convert back to JAX
        return jnp.asarray(next_state_np)
    
    def jax_cost(self, state: Any) -> Any:
        """
        Default JAX cost: convert to numpy, call cost(), convert back.
        
        Subclasses should override for better performance and vectorization.
        """
        if jnp is None:
            raise RuntimeError("JAX is required for jax_cost")
        
        # Convert to numpy
        state_np = np.asarray(state)
        
        # Call numpy cost (handle batching)
        if state_np.ndim > 1:
            # Batched
            costs = [self.cost(s) for s in state_np]
            return jnp.array(costs)
        else:
            # Single state
            return jnp.array(self.cost(state_np))
    
    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Default reset implementation.
        
        Subclasses should override this method.
        
        Args:
            rng: Random key/seed
            **kwargs: Additional reset parameters
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        # Default: return empty state
        return np.array([], dtype=np.float32), {}
    
    def step(
        self,
        state: State,
        action: Action,
        t: Optional[int] = None,
        info: Optional[dict] = None
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Default step implementation.
        
        Subclasses should override this method.
        
        Args:
            state: Current state
            action: Action to take
            t: Current time step (optional)
            info: Additional info dict (optional)
            
        Returns:
            Tuple of (next_state, cost, done, info)
        """
        # Default: use transition and cost
        next_state = self.transition(state, action)
        cost = self.cost(next_state)
        done = False
        return next_state, cost, done, {}
