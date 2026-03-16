"""
Brax environment adapter.

This adapter wraps Brax environments to work with the genedynamics framework.
Brax is JAX-native, so this adapter provides efficient vectorization and JIT compilation.
"""

from typing import Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # For type checking only
    import jax
    import jax.numpy as jnp

try:
    import brax
    from brax import envs
    import jax
    import jax.numpy as jnp
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False
    jax = None
    jnp = None

from genedynamics.core.types import State, Action
from genedynamics.core.backends import Backend, get_backend
from genedynamics.envs.obstacles.base import ObstacleManager


class BraxEnvAdapter:
    """
    Adapter that wraps a Brax environment for use with genedynamics.
    
    Brax is JAX-native, so this adapter:
    - Supports efficient vectorization (vmap)
    - Supports JIT compilation
    - Works with JAX arrays natively
    - Provides batch operations
    
    Features:
    - Obstacle integration
    - Vectorized rollouts
    - JAX-compatible transitions
    """
    
    def __init__(
        self,
        brax_env: Any,  # brax.envs.Env
        backend: str = "jax",
        obstacles: Optional[list] = None,
        collision_penalty: float = 1000.0,
    ):
        """
        Initialize Brax environment adapter.
        
        Args:
            brax_env: Brax environment instance
            backend: Computational backend (should be "jax" for Brax)
            obstacles: List of obstacles (optional)
            collision_penalty: Penalty for collision (added to cost)
        """
        if not BRAX_AVAILABLE:
            raise ImportError(
                "Brax is required. Install with: pip install brax"
            )
        
        self.brax_env = brax_env
        self.backend = get_backend(backend)
        self.collision_penalty = collision_penalty
        
        # Obstacle manager
        if obstacles:
            self.obstacle_manager = ObstacleManager(obstacles)
        else:
            self.obstacle_manager = ObstacleManager()
        
        # Get environment properties
        self.observation_size = brax_env.observation_size
        self.action_size = brax_env.action_size
        self.state_dim = self.observation_size
        self.act_dim = self.action_size
        
        # Environment metadata
        self.dt = getattr(brax_env, 'dt', 0.02)  # Brax default
        self.horizon = getattr(brax_env, 'max_episode_steps', 1000)
    
    def reset(self, rng: Optional[Any] = None) -> Tuple[Any, dict]:
        """
        Reset Brax environment to initial state.
        
        Args:
            rng: JAX random key (optional)
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        if jax is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        # Brax reset returns state (PyTree), not observation
        state = self.brax_env.reset(rng)
        obs = self.brax_env.get_obs(state)
        
        return obs, {"state": state}
    
    def step(
        self,
        state: Any,  # Brax state (PyTree)
        action: jnp.ndarray,
        t: Optional[int] = None,
        info: Optional[dict] = None
    ) -> Tuple[jnp.ndarray, float, bool, dict]:
        """
        Execute one Brax environment step.
        
        Note: Brax's step() requires the state PyTree as input (functional style).
        This is different from Gymnasium which uses internal state.
        
        Args:
            state: Brax state (PyTree), not observation
            action: Action to take (JAX array)
            t: Current time step (optional)
            info: Additional info dict (optional)
            
        Returns:
            Tuple of (next_observation, cost, done, info)
        """
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        # Brax step
        next_state = self.brax_env.step(state, action)
        obs = self.brax_env.get_obs(next_state)
        reward = self.brax_env.reward(next_state, action)
        done = self.brax_env.done(next_state)
        
        cost = -float(reward)  # Convert reward to cost
        collision = False
        if self.obstacle_manager and len(self.obstacle_manager) > 0:
            obs_np = np.asarray(obs)
            collision = self.obstacle_manager.collision_check(obs_np)
            if collision:
                cost += self.collision_penalty
                done = True  # Terminate on collision
        
        info = info or {}
        info['state'] = next_state
        if collision:
            info['collision'] = True
            info['obstacle_collision'] = True
        
        return obs, cost, bool(done), info
    
    def transition(self, state: Any, action: jnp.ndarray) -> jnp.ndarray:
        """
        Compute next observation from current state and action.
        
        Args:
            state: Brax state (PyTree)
            action: Action to take (JAX array)
            
        Returns:
            Next observation (JAX array)
        """
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        next_state = self.brax_env.step(state, action)
        return self.brax_env.get_obs(next_state)
    
    def jax_transition(self, state: Any, action: Any) -> Any:
        """
        JAX-compatible transition: flat [qpos; qvel] -> step -> flat next state.
        Converts flat obs to Brax State, steps, returns next obs.
        """
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        from brax.envs.base import State
        nq = self.brax_env.sys.q_size()
        qpos = state[:nq]
        qvel = state[nq:]
        pipeline_state = self.brax_env.pipeline_init(qpos, qvel)
        brax_state = State(pipeline_state, state, 0.0, False, {})
        next_brax_state = self.brax_env.step(brax_state, action)
        return next_brax_state.obs
    
    def rollout_batch(
        self,
        rng: Any,
        initial_states: Any,
        actions: Any,
    ) -> Any:
        """
        Roll out multiple trajectories in parallel (vectorized).
        
        This is a key advantage of Brax: efficient batch rollouts.
        
        Args:
            rng: JAX random key
            initial_states: Initial states (PyTree or array), shape (batch, ...)
            actions: Action sequences, shape (batch, horizon, action_dim)
            
        Returns:
            Observation trajectories, shape (batch, horizon+1, obs_dim)
        """
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        
        def rollout_single(rng, init_state, actions_seq):
            # Reset
            state = self.brax_env.reset(rng)
            obs = self.brax_env.get_obs(state)
            obs_list = [obs]
            
            # Step through actions
            for action in actions_seq:
                state = self.brax_env.step(state, action)
                obs = self.brax_env.get_obs(state)
                obs_list.append(obs)
            
            return jnp.stack(obs_list)
        
        # Vectorize over batch
        return jax.vmap(rollout_single)(rng, initial_states, actions)
    
    def cost(self, state: Any) -> float:
        """
        Compute cost for a given observation.
        
        Args:
            state: Observation to evaluate (JAX array)
            
        Returns:
            Cost (float, lower is better)
        """
        if jax is None or jnp is None:
            raise RuntimeError("JAX is required for BraxEnvAdapter")
        # Default: distance from origin
        base_cost = float(jnp.sum(state ** 2))
        
        # Add obstacle penalty (convert to numpy for checking)
        if self.obstacle_manager and len(self.obstacle_manager) > 0:
            state_np = np.asarray(state)
            if self.obstacle_manager.collision_check(state_np):
                base_cost += self.collision_penalty
        
        return base_cost
    
    def check_collision(self, state: Any) -> bool:
        """
        Check if observation collides with obstacles.
        
        Args:
            state: Observation to check (JAX array)
            
        Returns:
            True if collision detected, False otherwise
        """
        if not self.obstacle_manager or len(self.obstacle_manager) == 0:
            return False
        
        state_np = np.asarray(state)
        return self.obstacle_manager.collision_check(state_np)
    
    def render(self, mode: str = "human", **kwargs) -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output
        """
        # Brax doesn't have built-in rendering
        # Would need to use external renderer or convert to other format
        if mode == "human":
            print("Brax environments don't have built-in rendering.")
            print("Use brax.io.html or convert to other format for visualization.")
        return None
    
    def close(self) -> None:
        """Close environment (Brax environments are stateless, so this is a no-op)."""
        pass
    
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
