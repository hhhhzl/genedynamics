"""
End-to-end integration tests.

Tests complete workflows from environment creation to solver execution.
"""

import pytest
import numpy as np

from enerdynamics.envs.base_env import BaseEnvMixin
from enerdynamics.envs.obstacles.convex import BoxObstacle, SphereObstacle
from enerdynamics.core.backends.physics import DummyPhysicsBackend
from enerdynamics.core.backends.render_impl import NullRenderer


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_simple_environment_workflow(self):
        """Test simple environment workflow."""
        # Create a simple environment that implements required methods
        class SimpleEnv(BaseEnvMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            def reset(self, rng=None, **kwargs):
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                return self.state, {}
            
            def transition(self, state, action):
                return state + action * self.dt
            
            def cost(self, state):
                return float(np.sum(state ** 2))
        
        # Create environment
        env = SimpleEnv(dt=0.1, horizon=100)
        
        # Reset
        state, info = env.reset()
        assert state is not None
        
        # Step
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        next_state, cost, done, info = env.step(state, action)
        assert next_state is not None
        assert isinstance(cost, (int, float))
        assert isinstance(done, bool)
        
        print("✓ Simple environment workflow works")
    
    def test_environment_with_obstacles_workflow(self):
        """Test environment with obstacles workflow."""
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)),
            SphereObstacle(np.array([-1.0, 0.0, 0.0], dtype=np.float32), 0.3),
        ]
        
        class TestEnv(BaseEnvMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            def reset(self, rng=None, **kwargs):
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                return self.state, {}
            
            def step(self, state, action, t=None, info=None):
                next_state = state + action * self.dt
                cost = float(np.sum(next_state ** 2))
                done = False
                return next_state, cost, done, {}
            
            def transition(self, state, action):
                return state + action * self.dt
            
            def cost(self, state):
                return float(np.sum(state ** 2))
        
        env = TestEnv(obstacles=obstacles, dt=0.1)
        
        # Reset
        state, info = env.reset()
        
        # Check collision
        collision = env.check_collision(state)
        assert isinstance(collision, bool)
        
        # Step
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        next_state, cost, done, info = env.step(state, action)
        
        print("✓ Environment with obstacles workflow works")
    
    def test_environment_with_physics_backend_workflow(self):
        """Test environment with physics backend workflow."""
        physics_backend = DummyPhysicsBackend()
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32))
        ]
        
        class TestEnv(BaseEnvMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            def reset(self, rng=None, **kwargs):
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if self.physics_backend is not None:
                    self.physics_backend.reset()
                return self.state, {}
            
            def step(self, state, action, t=None, info=None):
                if self.physics_backend is not None:
                    next_state_dict = self.physics_backend.step(action)
                    if isinstance(next_state_dict, dict):
                        qpos = next_state_dict.get('qpos', np.array([]))
                        qvel = next_state_dict.get('qvel', np.array([]))
                        next_state = np.concatenate([qpos, qvel]) if len(qpos) > 0 or len(qvel) > 0 else state
                    else:
                        next_state = np.asarray(next_state_dict) if len(next_state_dict) > 0 else state
                else:
                    next_state = state + action * self.dt
                
                cost = float(np.sum(next_state ** 2))
                done = False
                return next_state, cost, done, {}
            
            def transition(self, state, action):
                if self.physics_backend is not None:
                    self.physics_backend.set_state(state)
                    next_state_dict = self.physics_backend.step(action)
                    if isinstance(next_state_dict, dict):
                        qpos = next_state_dict.get('qpos', np.array([]))
                        qvel = next_state_dict.get('qvel', np.array([]))
                        return np.concatenate([qpos, qvel]) if len(qpos) > 0 or len(qvel) > 0 else state
                    else:
                        return np.asarray(next_state_dict) if len(next_state_dict) > 0 else state
                else:
                    return state + action * self.dt
            
            def cost(self, state):
                return float(np.sum(state ** 2))
        
        env = TestEnv(physics_backend=physics_backend, obstacles=obstacles, dt=0.1)
        
        # Reset
        state, info = env.reset()
        
        # Step
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        next_state, cost, done, info = env.step(state, action)
        
        print("✓ Environment with physics backend workflow works")
    
    def test_full_integration_workflow(self):
        """Test full integration with all components."""
        physics_backend = DummyPhysicsBackend()
        render_backend = NullRenderer()
        obstacles = [
            BoxObstacle(np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)),
            SphereObstacle(np.array([-1.0, 0.0, 0.0], dtype=np.float32), 0.3),
        ]
        
        class TestEnv(BaseEnvMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            def reset(self, rng=None, **kwargs):
                self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if self.physics_backend is not None:
                    self.physics_backend.reset()
                return self.state, {}
            
            def step(self, state, action, t=None, info=None):
                next_state = state + action * self.dt
                cost = float(np.sum(next_state ** 2))
                done = False
                return next_state, cost, done, {}
            
            def transition(self, state, action):
                return state + action * self.dt
            
            def cost(self, state):
                return float(np.sum(state ** 2))
        
        env = TestEnv(
            physics_backend=physics_backend,
            render_backend=render_backend,
            obstacles=obstacles,
            dt=0.1,
            horizon=100
        )
        
        # Reset
        state, info = env.reset()
        
        # Step
        action = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        next_state, cost, done, info = env.step(state, action)
        
        # Check collision
        collision = env.check_collision(next_state)
        
        # Render
        render_result = env.render(next_state)
        
        print("✓ Full integration workflow works")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_obstacle_query_performance(self):
        """Benchmark obstacle query performance."""
        import time
        
        obstacles = [
            BoxObstacle(
                center=np.array([i, 0.0, 0.0], dtype=np.float32),
                half_extents=np.array([0.1, 0.1, 0.1], dtype=np.float32)
            )
            for i in range(100)
        ]
        
        from enerdynamics.envs.obstacles.base import ObstacleManager
        manager = ObstacleManager(obstacles)
        
        # Benchmark contains queries
        points = np.random.randn(1000, 3).astype(np.float32)
        
        start_time = time.time()
        for point in points:
            manager.contains(point)
        elapsed = time.time() - start_time
        
        print(f"✓ Obstacle query performance: {elapsed:.4f}s for 1000 queries ({elapsed/1000*1000:.2f}ms per query)")
        assert elapsed < 1.0, "Obstacle queries should be fast"
    
    def test_sdf_computation_performance(self):
        """Benchmark SDF computation performance."""
        import time
        
        obstacles = [
            BoxObstacle(
                center=np.array([i, 0.0, 0.0], dtype=np.float32),
                half_extents=np.array([0.1, 0.1, 0.1], dtype=np.float32)
            )
            for i in range(10)
        ]
        
        from enerdynamics.envs.obstacles.base import ObstacleManager
        manager = ObstacleManager(obstacles)
        
        # Benchmark SDF queries
        points = np.random.randn(1000, 3).astype(np.float32)
        
        start_time = time.time()
        sdf_vals = manager.sdf(points)
        elapsed = time.time() - start_time
        
        print(f"✓ SDF computation performance: {elapsed:.4f}s for 1000 points ({elapsed/1000*1000:.2f}ms per point)")
        assert elapsed < 1.0, "SDF computation should be fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
