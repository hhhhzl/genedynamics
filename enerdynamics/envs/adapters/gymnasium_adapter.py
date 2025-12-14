"""
Gymnasium environment adapter.

This adapter wraps Gymnasium environments (including Gymnasium-Robotics,
gym-pybullet-drones, etc.) to work with the enerdynamics framework.
"""

from typing import Optional, Tuple, Any, Callable
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    Env = None

from enerdynamics.core.types import State, Action
from enerdynamics.core.backends import Backend, get_backend
from enerdynamics.envs.obstacles.base import ObstacleManager


class GymnasiumEnvAdapter:
    """
    Adapter that wraps a Gymnasium environment for use with enerdynamics.
    
    Supports:
    - Standard Gymnasium environments
    - Gymnasium-Robotics environments
    - gym-pybullet-drones environments
    - Any environment following Gymnasium API
    
    Features:
    - Automatic state extraction from observations
    - Obstacle integration
    - Multi-backend support
    - Collision checking
    """
    
    def __init__(
        self,
        gym_env: Any,  # gym.Env
        state_extractor: Optional[Callable] = None,
        backend: str = "numpy",
        obstacles: Optional[list] = None,
        collision_penalty: float = 1000.0,
    ):
        """
        Initialize Gymnasium environment adapter.
        
        Args:
            gym_env: Gymnasium environment instance
            state_extractor: Function to extract state from observation.
                            If None, uses observation directly as state.
                            Signature: state = state_extractor(obs) -> np.ndarray
            backend: Computational backend ("numpy", "jax", "torch")
            obstacles: List of obstacles (optional)
            collision_penalty: Penalty for collision (added to cost)
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "Gymnasium is required. Install with: pip install gymnasium"
            )
        
        self.gym_env = gym_env
        self.backend = get_backend(backend)
        self.collision_penalty = collision_penalty
        
        # State extractor: default uses observation directly
        if state_extractor is None:
            self.state_extractor = self._default_state_extractor
        else:
            self.state_extractor = state_extractor
        
        # Obstacle manager
        if obstacles:
            self.obstacle_manager = ObstacleManager(obstacles)
        else:
            self.obstacle_manager = ObstacleManager()
        
        # Get environment properties
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        
        # Infer dimensions
        try:
            if hasattr(self.observation_space, 'shape'):
                obs_shape = self.observation_space.shape
                if isinstance(obs_shape, tuple) and len(obs_shape) > 0:
                    self.state_dim = int(obs_shape[0])
                elif isinstance(obs_shape, (int, np.integer)):
                    self.state_dim = int(obs_shape)
                else:
                    self.state_dim = None
            else:
                self.state_dim = None
        except Exception:
            self.state_dim = None
        
        try:
            if hasattr(self.action_space, 'shape'):
                act_shape = self.action_space.shape
                if isinstance(act_shape, tuple) and len(act_shape) > 0:
                    self.act_dim = int(act_shape[0])
                elif isinstance(act_shape, (int, np.integer)):
                    self.act_dim = int(act_shape)
                else:
                    self.act_dim = None
            elif hasattr(self.action_space, 'n'):
                # Discrete action space
                self.act_dim = 1  # Discrete actions are scalar
            else:
                self.act_dim = None
        except Exception:
            self.act_dim = None
        
        # Environment metadata
        # Use unwrapped or get_wrapper_attr to avoid deprecation warnings
        try:
            # Try unwrapped first
            self.dt = getattr(gym_env.unwrapped, 'dt', 0.1)
        except AttributeError:
            # Fallback to get_wrapper_attr if available
            try:
                self.dt = gym_env.get_wrapper_attr('dt') if hasattr(gym_env, 'get_wrapper_attr') else 0.1
            except (AttributeError, ValueError):
                self.dt = 0.1  # Default time step
        
        try:
            # Try unwrapped first
            self.horizon = getattr(gym_env.unwrapped, 'max_episode_steps', 1000)
        except AttributeError:
            # Fallback to get_wrapper_attr if available
            try:
                self.horizon = gym_env.get_wrapper_attr('max_episode_steps') if hasattr(gym_env, 'get_wrapper_attr') else 1000
            except (AttributeError, ValueError):
                # Try spec if available
                if hasattr(gym_env, 'spec') and gym_env.spec is not None:
                    self.horizon = getattr(gym_env.spec, 'max_episode_steps', 1000)
                else:
                    self.horizon = 1000  # Default horizon
    
    def _default_state_extractor(self, obs: Any) -> np.ndarray:
        """
        Default state extractor: use observation directly.
        
        Args:
            obs: Observation from Gymnasium environment
            
        Returns:
            State as numpy array
        """
        if isinstance(obs, dict):
            # If observation is dict, try to extract 'state' or 'observation'
            if 'state' in obs:
                return np.asarray(obs['state'], dtype=np.float32)
            elif 'observation' in obs:
                return np.asarray(obs['observation'], dtype=np.float32)
            else:
                # Use first value
                return np.asarray(list(obs.values())[0], dtype=np.float32)
        else:
            return np.asarray(obs, dtype=np.float32)
    
    def reset(self, rng: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.
        
        Args:
            rng: Random key/seed (JAX Array, int, or None)
            **kwargs: Additional reset parameters
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        if rng is not None:
            # Convert rng to seed for Gymnasium
            if isinstance(rng, int):
                seed = rng
            elif hasattr(rng, '__getitem__'):
                # JAX rng key
                seed = int(rng[0]) if len(rng) > 0 else 0
            else:
                seed = hash(rng) % (2**32)
            obs, info = self.gym_env.reset(seed=seed, **kwargs)
        else:
            obs, info = self.gym_env.reset(**kwargs)
        
        state = self.state_extractor(obs)
        return np.asarray(state, dtype=np.float32), info
    
    def step(
        self,
        state: Optional[State] = None,
        action: Action = None,
        t: Optional[int] = None,
        info: Optional[dict] = None
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.
        
        Note: Gymnasium's step() doesn't take state as input (it uses internal state).
        If state is provided, it's ignored. Use transition() for state-based simulation.
        
        Args:
            state: Current state (ignored, uses internal Gymnasium state)
            action: Action to take
            t: Current time step (optional)
            info: Additional info dict (optional)
            
        Returns:
            Tuple of (next_state, cost, done, info)
            Note: cost = -reward (enerdynamics uses cost, Gymnasium uses reward)
        """
        if action is None:
            raise ValueError("action is required for step()")
        
        # Gymnasium step
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        state = self.state_extractor(obs)
        cost = -float(reward)  # Convert reward to cost
        done = terminated or truncated
        
        # Check obstacle collision
        if self.obstacle_manager and len(self.obstacle_manager) > 0:
            collision = self.obstacle_manager.collision_check(state)
            if collision:
                cost += self.collision_penalty
                info['collision'] = True
                info['obstacle_collision'] = True
        
        return np.asarray(state, dtype=np.float32), cost, done, info
    
    def transition(self, state: State, action: Action) -> np.ndarray:
        """
        Compute next state from current state and action (deterministic).
        
        This is used by planners that need to simulate trajectories.
        Note: Many Gymnasium environments don't expose transition() directly.
        This method attempts to:
        1. Use environment's transition() if available
        2. Use set_state() + step() if available
        3. Fall back to approximate dynamics (if implemented)
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Next state (numpy array)
        """
        state_np = np.asarray(state, dtype=np.float32)
        action_np = np.asarray(action, dtype=np.float32)
        
        # Try to use environment's transition if available
        if hasattr(self.gym_env, 'transition'):
            next_state = self.gym_env.transition(state_np, action_np)
            return np.asarray(next_state, dtype=np.float32)
        
        # Try to use set_state() + step()
        if hasattr(self.gym_env, 'set_state'):
            # Save current state
            old_state = None
            if hasattr(self.gym_env, 'get_state'):
                old_state = self.gym_env.get_state()
            
            # Set state and step
            self.gym_env.set_state(state_np)
            obs, _, _, _, _ = self.gym_env.step(action_np)
            next_state = self.state_extractor(obs)
            
            # Restore old state
            if old_state is not None:
                self.gym_env.set_state(old_state)
            
            return np.asarray(next_state, dtype=np.float32)
        
        # Fallback: warn and return state (no transition)
        import warnings
        warnings.warn(
            f"Environment {type(self.gym_env)} does not support transition(). "
            "Consider using a dynamics model or implementing transition() in the environment."
        )
        return state_np
    
    def cost(self, state: State) -> float:
        """
        Compute cost for a given state.
        
        Note: Gymnasium environments typically don't expose cost() directly.
        This is a placeholder that can be overridden.
        
        Args:
            state: State to evaluate
            
        Returns:
            Cost (float, lower is better)
        """
        # Default: distance from origin (placeholder)
        state_np = np.asarray(state, dtype=np.float32)
        base_cost = float(np.sum(state_np ** 2))
        
        # Add obstacle penalty
        if self.obstacle_manager and len(self.obstacle_manager) > 0:
            if self.obstacle_manager.collision_check(state_np):
                base_cost += self.collision_penalty
        
        return base_cost
    
    def check_collision(self, state: State) -> bool:
        """
        Check if state collides with obstacles.
        
        Args:
            state: State to check
            
        Returns:
            True if collision detected, False otherwise
        """
        if not self.obstacle_manager or len(self.obstacle_manager) == 0:
            return False
        
        state_np = np.asarray(state, dtype=np.float32)
        return self.obstacle_manager.collision_check(state_np)
    
    def render(self, mode: str = "human", **kwargs) -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ("human", "rgb_array", etc.)
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output (depends on mode)
        """
        return self.gym_env.render(mode=mode, **kwargs)
    
    def close(self) -> None:
        """Close environment and free resources."""
        self.gym_env.close()
    
    def add_obstacle(self, obstacle: Any) -> None:
        """
        Add obstacle to the environment.
        
        Args:
            obstacle: Obstacle to add (must implement Obstacle protocol)
        """
        self.obstacle_manager.add(obstacle)
    
    def remove_obstacle(self, obstacle: Any) -> None:
        """
        Remove obstacle from the environment.
        
        Args:
            obstacle: Obstacle to remove
        """
        self.obstacle_manager.remove(obstacle)
