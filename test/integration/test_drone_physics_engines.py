"""
Integration tests for drone physics engine adapters.

This module tests the MuJoCo and Isaac Sim adapters for the full 3D quadrotor
environment, including obstacle integration and state conversion.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from genedynamics.envs.factories import make_env
from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle


@pytest.mark.requires_mujoco
class TestDroneFull3DMujoco:
    """Test MuJoCo adapter for full 3D quadrotor."""
    
    def test_mujoco_env_creation(self):
        """Test creating MuJoCo environment."""
        try:
            env = make_env(
                "drone_full_3d_mujoco",
                dt=0.01,
                horizon=80,
                use_mujoco_physics=True,
            )
            assert env is not None
            assert env.physics_backend == 'mujoco'
            env.close()
        except ImportError:
            pytest.skip("MuJoCo not available")
    
    def test_mujoco_reset(self):
        """Test MuJoCo environment reset."""
        try:
            env = make_env("drone_full_3d_mujoco", dt=0.01)
            state, info = env.reset()
            assert state.shape == (12,)
            assert state.dtype == np.float32
            env.close()
        except ImportError:
            pytest.skip("MuJoCo not available")
    
    def test_mujoco_transition(self):
        """Test MuJoCo transition."""
        try:
            env = make_env("drone_full_3d_mujoco", dt=0.01)
            state, _ = env.reset()
            action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            next_state = env.transition(state, action)
            assert next_state.shape == (12,)
            assert next_state.dtype == np.float32
            env.close()
        except ImportError:
            pytest.skip("MuJoCo not available")
    
    def test_mujoco_with_obstacles(self):
        """Test MuJoCo environment with obstacles."""
        try:
            obstacles = ObstacleManager()
            obstacles.add(SphereObstacle(center=[0.5, 0.5, 0.5], radius=0.15))
            obstacles.add(BoxObstacle(center=[-0.5, -0.5, 0.8], half_extents=[0.1, 0.1, 0.15]))
            
            env = make_env(
                "drone_full_3d_mujoco",
                dt=0.01,
                obstacles=obstacles,
                use_mujoco_physics=True,
            )
            assert env is not None
            assert env.obstacles is not None
            assert len(env.obstacles) == 2
            env.close()
        except ImportError:
            pytest.skip("MuJoCo not available")


@pytest.mark.requires_isaac
class TestDroneFull3DIsaac:
    """Test Isaac Sim adapter for full 3D quadrotor."""
    
    def test_isaac_env_creation(self):
        """Test creating Isaac Sim environment."""
        try:
            env = make_env(
                "drone_full_3d_isaac",
                dt=0.01,
                horizon=80,
                use_isaac_physics=True,
                use_gpu=False,  # Use CPU for testing
            )
            assert env is not None
            assert env.physics_backend == 'isaac'
            env.close()
        except ImportError:
            pytest.skip("Isaac Sim not available")
    
    def test_isaac_reset(self):
        """Test Isaac Sim environment reset."""
        try:
            env = make_env("drone_full_3d_isaac", dt=0.01, use_gpu=False)
            state, info = env.reset()
            assert state.shape == (12,)
            assert state.dtype == np.float32
            env.close()
        except ImportError:
            pytest.skip("Isaac Sim not available")
    
    def test_isaac_transition(self):
        """Test Isaac Sim transition."""
        try:
            env = make_env("drone_full_3d_isaac", dt=0.01, use_gpu=False)
            state, _ = env.reset()
            action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            next_state = env.transition(state, action)
            assert next_state.shape == (12,)
            assert next_state.dtype == np.float32
            env.close()
        except ImportError:
            pytest.skip("Isaac Sim not available")
    
    def test_isaac_with_obstacles(self):
        """Test Isaac Sim environment with obstacles."""
        try:
            obstacles = ObstacleManager()
            obstacles.add(SphereObstacle(center=[0.5, 0.5, 0.5], radius=0.15))
            obstacles.add(BoxObstacle(center=[-0.5, -0.5, 0.8], half_extents=[0.1, 0.1, 0.15]))
            
            env = make_env(
                "drone_full_3d_isaac",
                dt=0.01,
                obstacles=obstacles,
                use_isaac_physics=True,
                use_gpu=False,
            )
            assert env is not None
            assert env.obstacles is not None
            assert len(env.obstacles) == 2
            env.close()
        except ImportError:
            pytest.skip("Isaac Sim not available")


class TestModelGenerators:
    """Test model file generators."""
    
    def test_mujoco_model_generator(self):
        """Test MuJoCo model generation."""
        try:
            from genedynamics.envs.utils.mujoco_model_generator import (
                create_base_quadrotor_xml,
                generate_mujoco_xml_with_obstacles,
            )
            
            # Create base model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                base_path = f.name
            
            create_base_quadrotor_xml(base_path, mass=0.5, arm_length=0.17)
            assert os.path.exists(base_path)
            
            # Generate with obstacles
            obstacles = ObstacleManager()
            obstacles.add(SphereObstacle(center=[0.5, 0.5, 0.5], radius=0.15))
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                output_path = f.name
            
            xml_content = generate_mujoco_xml_with_obstacles(
                base_path,
                obstacles,
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            assert 'obstacle_0' in xml_content
            assert 'sphere' in xml_content.lower()
            
            # Cleanup
            os.unlink(base_path)
            os.unlink(output_path)
        except ImportError:
            pytest.skip("MuJoCo utilities not available")
    
    def test_isaac_usd_generator(self):
        """Test Isaac Sim USD generation."""
        try:
            from genedynamics.envs.utils.isaac_usd_generator import (
                create_quadrotor_usd_with_obstacles,
            )
            
            obstacles = ObstacleManager()
            obstacles.add(SphereObstacle(center=[0.5, 0.5, 0.5], radius=0.15))
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.usd', delete=False) as f:
                output_path = f.name
            
            create_quadrotor_usd_with_obstacles(
                output_path,
                obstacles=obstacles,
                mass=0.5,
                arm_length=0.17
            )
            
            assert os.path.exists(output_path)
            
            # Cleanup
            os.unlink(output_path)
        except ImportError:
            pytest.skip("Isaac Sim utilities not available (PXR not installed)")


class TestStateConversion:
    """Test state conversion utilities."""
    
    def test_state_12d_to_mujoco(self):
        """Test 12D to MuJoCo state conversion."""
        from genedynamics.envs.utils.state_converter import (
            state_12d_to_mujoco,
            mujoco_to_state_12d,
        )
        
        state_12d = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7], dtype=np.float32)
        mujoco_state = state_12d_to_mujoco(state_12d)
        
        assert 'qpos' in mujoco_state
        assert 'qvel' in mujoco_state
        assert len(mujoco_state['qpos']) == 7  # 3 pos + 4 quat
        assert len(mujoco_state['qvel']) == 6  # 3 vel + 3 ang_vel
        
        # Test round-trip conversion
        recovered_state = mujoco_to_state_12d(mujoco_state)
        np.testing.assert_allclose(state_12d[:9], recovered_state[:9], atol=1e-5)
    
    def test_state_12d_to_isaac(self):
        """Test 12D to Isaac Sim state conversion."""
        from genedynamics.envs.utils.state_converter import (
            state_12d_to_isaac,
            isaac_to_state_12d,
        )
        
        state_12d = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7], dtype=np.float32)
        isaac_state = state_12d_to_isaac(state_12d)
        
        assert 'qpos' in isaac_state
        assert 'qvel' in isaac_state
        assert len(isaac_state['qpos']) == 7  # 3 pos + 4 quat
        assert len(isaac_state['qvel']) == 6  # 3 vel + 3 ang_vel
        
        # Test round-trip conversion
        recovered_state = isaac_to_state_12d(isaac_state)
        np.testing.assert_allclose(state_12d[:9], recovered_state[:9], atol=1e-5)


class TestJAXDynamics:
    """Test JAX dynamics implementation."""
    
    def test_jax_quadrotor_step(self):
        """Test JAX quadrotor dynamics step."""
        try:
            from genedynamics.envs.utils.jax_dynamics import jax_quadrotor_step
            import jax.numpy as jnp
            
            state = jnp.zeros(12, dtype=jnp.float32)
            action = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32)
            I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
            
            next_state = jax_quadrotor_step(
                state, action, 0.01,
                0.5, I, 0.17,
                3.16e-10, 7.94e-12, 9.81
            )
            
            assert next_state.shape == (12,)
            assert next_state.dtype == jnp.float32
        except ImportError:
            pytest.skip("JAX not available")
    
    def test_jax_batch_computation(self):
        """Test batched JAX computation."""
        try:
            from genedynamics.envs.utils.jax_dynamics import jax_quadrotor_step_batch
            import jax.numpy as jnp
            
            batch_size = 10
            states = jnp.zeros((batch_size, 12), dtype=jnp.float32)
            actions = jnp.ones((batch_size, 4), dtype=jnp.float32) * 0.5
            I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
            
            next_states = jax_quadrotor_step_batch(
                states, actions, 0.01,
                0.5, I, 0.17,
                3.16e-10, 7.94e-12, 9.81
            )
            
            assert next_states.shape == (batch_size, 12)
            assert next_states.dtype == jnp.float32
        except ImportError:
            pytest.skip("JAX not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])










