"""
Unit tests for BaseEnv interface.

Tests cover:
- BaseEnv Protocol
- BaseEnvMixin
- Integration with backends
- Integration with obstacles
"""

import pytest
import numpy as np

from genedynamics.envs.base_env import BaseEnvMixin
from genedynamics.envs.obstacles.convex import BoxObstacle
from genedynamics.core.backends.physics import DummyPhysicsBackend


@pytest.mark.unit
class TestBaseEnvMixin:
    """Test BaseEnvMixin."""
    
    def test_base_env_creation(self):
        """Test BaseEnvMixin creation."""
        env = BaseEnvMixin(dt=0.1, horizon=100)
        
        assert env.dt == 0.1
        assert env.horizon == 100
        assert env.obstacles == []
        print("✓ BaseEnvMixin creation works")
    
    def test_base_env_with_obstacles(self):
        """Test BaseEnvMixin with obstacles."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ]
        env = BaseEnvMixin(obstacles=obstacles)
        
        assert len(env.obstacles) == 1
        print("✓ BaseEnvMixin with obstacles works")
    
    def test_base_env_collision_checking(self):
        """Test BaseEnvMixin collision checking."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ]
        env = BaseEnvMixin(obstacles=obstacles)
        
        # Point in obstacle
        state_in_obstacle = np.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert env.check_collision(state_in_obstacle)
        
        # Point outside obstacle
        state_outside = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert not env.check_collision(state_outside)
        print("✓ BaseEnvMixin collision checking works")
    
    def test_base_env_with_physics_backend(self):
        """Test BaseEnvMixin with physics backend."""
        physics_backend = DummyPhysicsBackend()
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ]
        
        env = BaseEnvMixin(physics_backend=physics_backend, obstacles=obstacles)
        
        assert env.physics_backend is not None
        assert env.physics_backend.name == "dummy"
        # Obstacles should be automatically added to physics backend
        print("✓ BaseEnvMixin with physics backend works")


@pytest.mark.unit
class TestBaseEnvIntegration:
    """Test BaseEnv integration with other components."""
    
    def test_env_with_render_backend(self):
        """Test environment with render backend."""
        from genedynamics.core.backends.render_impl import NullRenderer
        
        render_backend = NullRenderer()
        env = BaseEnvMixin(render_backend=render_backend)
        
        assert env.render_backend is not None
        result = env.render()
        assert result is None
        print("✓ Environment with render backend works")
    
    def test_env_full_integration(self):
        """Test full environment integration."""
        from genedynamics.core.backends.render_impl import NullRenderer
        
        physics_backend = DummyPhysicsBackend()
        render_backend = NullRenderer()
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ]
        
        env = BaseEnvMixin(
            dt=0.1,
            horizon=100,
            physics_backend=physics_backend,
            render_backend=render_backend,
            obstacles=obstacles
        )
        
        assert env.dt == 0.1
        assert env.horizon == 100
        assert env.physics_backend is not None
        assert env.render_backend is not None
        assert len(env.obstacles) == 1
        print("✓ Full environment integration works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
