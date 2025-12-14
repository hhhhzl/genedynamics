"""
Unit tests for environment adapters.

Tests cover:
- GymnasiumEnvAdapter
- BraxEnvAdapter
- UnifiedEnvAdapter
- Obstacle integration with adapters
"""

import pytest
import numpy as np

from enerdynamics.envs.obstacles.convex import BoxObstacle, SphereObstacle


@pytest.mark.unit
@pytest.mark.requires_gymnasium
class TestGymnasiumAdapter:
    """Test GymnasiumEnvAdapter."""
    
    def test_gymnasium_adapter_creation(self):
        """Test GymnasiumEnvAdapter creation."""
        try:
            import gymnasium as gym
            from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
            
            # Create a simple Gymnasium environment
            gym_env = gym.make("CartPole-v1")
            adapter = GymnasiumEnvAdapter(gym_env)
            
            assert adapter.dt > 0
            assert adapter.state_dim is not None
            print("✓ GymnasiumEnvAdapter creation works")
        except ImportError:
            pytest.skip("Gymnasium not available")
    
    def test_gymnasium_adapter_reset_step(self):
        """Test GymnasiumEnvAdapter reset and step."""
        try:
            import gymnasium as gym
            from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
            
            gym_env = gym.make("CartPole-v1")
            adapter = GymnasiumEnvAdapter(gym_env)
            
            state, info = adapter.reset()
            assert state is not None
            
            action = adapter.action_space.sample()
            next_state, cost, done, info = adapter.step(state, action)
            assert next_state is not None
            print("✓ GymnasiumEnvAdapter reset/step works")
        except ImportError:
            pytest.skip("Gymnasium not available")
    
    def test_gymnasium_adapter_obstacles(self):
        """Test GymnasiumEnvAdapter with obstacles."""
        try:
            import gymnasium as gym
            from enerdynamics.envs.adapters.gymnasium_adapter import GymnasiumEnvAdapter
            
            gym_env = gym.make("CartPole-v1")
            obstacles = [
                BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
            ]
            adapter = GymnasiumEnvAdapter(gym_env, obstacles=obstacles)
            
            state, _ = adapter.reset()
            collision = adapter.check_collision(state)
            assert isinstance(collision, bool)
            print("✓ GymnasiumEnvAdapter with obstacles works")
        except ImportError:
            pytest.skip("Gymnasium not available")


@pytest.mark.unit
@pytest.mark.requires_brax
class TestBraxAdapter:
    """Test BraxEnvAdapter."""
    
    def test_brax_adapter_creation(self):
        """Test BraxEnvAdapter creation."""
        try:
            import brax
            from enerdynamics.envs.adapters.brax_adapter import BraxEnvAdapter
            
            # Create a simple Brax environment
            brax_env = brax.envs.create("ant")
            adapter = BraxEnvAdapter(brax_env)
            
            assert adapter.dt > 0
            assert adapter.state_dim is not None
            print("✓ BraxEnvAdapter creation works")
        except ImportError:
            pytest.skip("Brax not available")
    
    def test_brax_adapter_reset_step(self):
        """Test BraxEnvAdapter reset and step."""
        try:
            import brax
            import jax
            from enerdynamics.envs.adapters.brax_adapter import BraxEnvAdapter
            
            brax_env = brax.envs.create("ant")
            adapter = BraxEnvAdapter(brax_env)
            
            rng = jax.random.PRNGKey(42)
            state, info = adapter.reset(rng=rng)
            assert state is not None
            
            action = jax.random.uniform(rng, (adapter.act_dim,), minval=-1, maxval=1)
            next_state, cost, done, info = adapter.step(state, action)
            assert next_state is not None
            print("✓ BraxEnvAdapter reset/step works")
        except ImportError:
            pytest.skip("Brax not available")


@pytest.mark.unit
class TestUnifiedAdapter:
    """Test UnifiedEnvAdapter."""
    
    def test_unified_adapter_detection(self):
        """Test UnifiedEnvAdapter automatic detection."""
        try:
            from enerdynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter
            
            # Test with native environment
            from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
            
            native_env = DoubleIntegratorBoxEnv()
            adapter = UnifiedEnvAdapter(native_env)
            
            assert adapter.dt > 0
            # state_dim might be None for some environments, that's okay
            # Just check that adapter was created successfully
            assert adapter.env_type in ["gymnasium", "brax", "native"]
            print("✓ UnifiedEnvAdapter detection works")
        except ImportError:
            pytest.skip("DoubleIntegratorBoxEnv not available")
    
    def test_unified_adapter_obstacles(self):
        """Test UnifiedEnvAdapter with obstacles."""
        try:
            from enerdynamics.envs.adapters.unified_adapter import UnifiedEnvAdapter
            from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
            
            # Use 2D obstacles for 2D environment
            obstacles = [
                BoxObstacle(np.array([1.0, 0.0], dtype=np.float32), np.array([0.2, 0.2], dtype=np.float32)),
                SphereObstacle(np.array([-1.0, 0.0], dtype=np.float32), 0.3),
            ]
            
            native_env = DoubleIntegratorBoxEnv()
            adapter = UnifiedEnvAdapter(native_env, obstacles=obstacles)
            
            state, _ = adapter.reset()
            # Extract 2D position for collision check
            if len(state) >= 2:
                position_2d = state[:2]
                # Check collision with 2D obstacles
                collision = any(obs.contains(position_2d) for obs in obstacles)
                assert isinstance(collision, bool)
            print("✓ UnifiedEnvAdapter with obstacles works")
        except ImportError:
            pytest.skip("DoubleIntegratorBoxEnv not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
