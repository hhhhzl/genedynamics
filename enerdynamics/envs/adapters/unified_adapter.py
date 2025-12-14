"""
Unified environment adapter.

This adapter automatically detects environment type and wraps it with the
appropriate adapter (Gymnasium, Brax, or native enerdynamics).
"""

from typing import Optional, Any, Tuple
import numpy as np

from enerdynamics.core.types import State, Action
from enerdynamics.core.backends import Backend


class UnifiedEnvAdapter:
    """
    Unified adapter that automatically detects and wraps environments.
    
    Supports:
    - Gymnasium environments (via GymnasiumEnvAdapter)
    - Brax environments (via BraxEnvAdapter)
    - Native enerdynamics environments (direct passthrough)
    
    This allows users to use any environment type without knowing which
    adapter to use.
    """
    
    def __init__(
        self,
        env: Any,
        backend: str = "numpy",
        obstacles: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize unified adapter.
        
        Args:
            env: Environment to wrap (Gymnasium, Brax, or native)
            backend: Computational backend
            obstacles: List of obstacles (optional)
            **kwargs: Additional adapter-specific parameters
        """
        self.original_env = env
        self.backend = backend
        self.obstacles = obstacles or []
        
        # Auto-detect environment type and create appropriate adapter
        if self._is_gymnasium(env):
            from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
            self.adapter = GymnasiumEnvAdapter(
                env,
                backend=backend,
                obstacles=obstacles,
                **kwargs
            )
            self.env_type = "gymnasium"
        elif self._is_brax(env):
            from enerdynamics.envs.adapters.brax_adapter import BraxEnvAdapter
            self.adapter = BraxEnvAdapter(
                env,
                backend=backend,
                obstacles=obstacles,
                **kwargs
            )
            self.env_type = "brax"
        else:
            # Assume native enerdynamics environment
            self.adapter = env
            self.env_type = "native"
            # Add obstacles if environment supports it
            if obstacles:
                if hasattr(env, 'obstacles'):
                    if env.obstacles is None:
                        env.obstacles = obstacles.copy()
                    else:
                        env.obstacles.extend(obstacles)
                # Also store in UnifiedAdapter for fallback
                self.obstacles = obstacles.copy()
        
        # Extract common properties
        self.dt = getattr(self.adapter, 'dt', 0.1)
        self.horizon = getattr(self.adapter, 'horizon', 1000)
        self.state_dim = getattr(self.adapter, 'state_dim', None)
        self.act_dim = getattr(self.adapter, 'act_dim', None)
        
        # Store obstacles for fallback collision checking
        self.obstacles = obstacles or []
    
    def _is_gymnasium(self, env: Any) -> bool:
        """
        Check if environment is a Gymnasium environment.
        
        Args:
            env: Environment to check
            
        Returns:
            True if Gymnasium environment
        """
        try:
            import gymnasium as gym
            return isinstance(env, gym.Env)
        except ImportError:
            # Check by attributes
            return (
                hasattr(env, 'reset') and
                hasattr(env, 'step') and
                hasattr(env, 'observation_space') and
                hasattr(env, 'action_space')
            )
    
    def _is_brax(self, env: Any) -> bool:
        """
        Check if environment is a Brax environment.
        
        Args:
            env: Environment to check
            
        Returns:
            True if Brax environment
        """
        try:
            import brax
            return (
                hasattr(env, 'step') and
                hasattr(env, 'reset') and
                hasattr(env, 'observation_size') and
                hasattr(env, 'action_size') and
                hasattr(env, 'get_obs')
            )
        except ImportError:
            return False
    
    # Delegate all methods to adapter
    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        return self.adapter.reset(rng, **kwargs)
    
    def step(
        self,
        state: Optional[State] = None,
        action: Optional[Action] = None,
        t: Optional[int] = None,
        info: Optional[dict] = None
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one environment step."""
        # Handle different step() signatures
        if self.env_type == "brax":
            # Brax step() requires state PyTree
            if state is None:
                raise ValueError("Brax step() requires state PyTree")
            return self.adapter.step(state, action, t, info)
        elif self.env_type == "gymnasium":
            # Gymnasium step() only needs action
            return self.adapter.step(state, action, t, info)
        else:
            # Native enerdynamics
            if action is None:
                raise ValueError("action is required")
            return self.adapter.step(state, action, t, info)
    
    def transition(self, state: State, action: Action) -> np.ndarray:
        """Compute next state from current state and action."""
        return self.adapter.transition(state, action)
    
    def cost(self, state: State) -> float:
        """Compute cost for a given state."""
        return self.adapter.cost(state)
    
    def check_collision(self, state: State) -> bool:
        """Check if state collides with obstacles."""
        if hasattr(self.adapter, 'check_collision'):
            return self.adapter.check_collision(state)
        elif hasattr(self.adapter, 'obstacle_manager') and self.adapter.obstacle_manager:
            # If adapter has obstacle_manager, use it directly
            return self.adapter.obstacle_manager.collision_check(state)
        elif self.obstacles:
            # Fallback: use obstacles directly
            from enerdynamics.envs.obstacles.base import ObstacleManager
            manager = ObstacleManager(self.obstacles)
            return manager.collision_check(state)
        else:
            return False
    
    def render(self, mode: str = "human", **kwargs) -> Optional[Any]:
        """Render the environment."""
        return self.adapter.render(mode, **kwargs)
    
    def close(self) -> None:
        """Close environment."""
        if hasattr(self.adapter, 'close'):
            self.adapter.close()
    
    def add_obstacle(self, obstacle: Any) -> None:
        """Add obstacle to the environment."""
        if hasattr(self.adapter, 'add_obstacle'):
            self.adapter.add_obstacle(obstacle)
        else:
            # For native environments, add to our obstacles list
            if obstacle not in self.obstacles:
                self.obstacles.append(obstacle)
            # Also try to add to adapter if it has obstacles attribute
            if hasattr(self.adapter, 'obstacles'):
                if self.adapter.obstacles is None:
                    self.adapter.obstacles = [obstacle]
                elif obstacle not in self.adapter.obstacles:
                    self.adapter.obstacles.append(obstacle)
    
    def remove_obstacle(self, obstacle: Any) -> None:
        """Remove obstacle from the environment."""
        if hasattr(self.adapter, 'remove_obstacle'):
            self.adapter.remove_obstacle(obstacle)
        else:
            # For native environments, remove from our obstacles list
            if obstacle in self.obstacles:
                self.obstacles.remove(obstacle)
            # Also try to remove from adapter if it has obstacles attribute
            if hasattr(self.adapter, 'obstacles') and self.adapter.obstacles:
                if obstacle in self.adapter.obstacles:
                    self.adapter.obstacles.remove(obstacle)
