"""
Integration tests for JAX integration and optimization.

This module tests JAX functionality including JIT compilation, batch processing,
hybrid mode, and performance optimizations.
"""

import pytest
import numpy as np
import time

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


@pytest.mark.requires_jax
class TestJAXDynamics:
    """Test JAX dynamics functions."""
    
    def test_jax_quadrotor_step(self):
        """Test JAX quadrotor step function."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_dynamics import jax_quadrotor_step
        
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
    
    def test_jax_batch_computation(self):
        """Test batched JAX computation."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_dynamics import jax_quadrotor_step_batch
        
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
    
    def test_jax_jit_compilation(self):
        """Test JIT compilation performance."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_dynamics import jax_quadrotor_step
        
        state = jnp.zeros(12, dtype=jnp.float32)
        action = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32)
        I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
        
        # First call (compilation + execution)
        start = time.time()
        _ = jax_quadrotor_step(
            state, action, 0.01,
            0.5, I, 0.17,
            3.16e-10, 7.94e-12, 9.81
        )
        first_time = time.time() - start
        
        # Second call (execution only, should be faster)
        start = time.time()
        for _ in range(100):
            _ = jax_quadrotor_step(
                state, action, 0.01,
                0.5, I, 0.17,
                3.16e-10, 7.94e-12, 9.81
            )
        avg_time = (time.time() - start) / 100
        
        # JIT-compiled function should be fast after compilation
        assert avg_time < 0.01  # Should be very fast (< 10ms per call)


@pytest.mark.requires_jax
class TestJAXRollout:
    """Test JAX rollout functions."""
    
    def test_jax_rollout_single(self):
        """Test single trajectory rollout."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_rollout import jax_rollout_single
        
        initial_state = jnp.zeros(12, dtype=jnp.float32)
        horizon = 10
        actions = jnp.ones((horizon, 4), dtype=jnp.float32) * 0.5
        I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
        
        trajectory = jax_rollout_single(
            initial_state, actions, 0.01,
            0.5, I, 0.17,
            3.16e-10, 7.94e-12, 9.81,
            2.0, 2.0, 1.0
        )
        
        assert trajectory.shape == (horizon + 1, 12)
        assert trajectory.dtype == jnp.float32
    
    def test_jax_rollout_batch(self):
        """Test batched trajectory rollout."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_rollout import jax_rollout_batch
        
        batch_size = 5
        horizon = 10
        initial_states = jnp.zeros((batch_size, 12), dtype=jnp.float32)
        actions = jnp.ones((batch_size, horizon, 4), dtype=jnp.float32) * 0.5
        I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
        
        trajectories = jax_rollout_batch(
            initial_states, actions, 0.01,
            0.5, I, 0.17,
            3.16e-10, 7.94e-12, 9.81,
            2.0, 2.0, 1.0
        )
        
        assert trajectories.shape == (batch_size, horizon + 1, 12)
        assert trajectories.dtype == jnp.float32
    
    def test_jax_rollout_hybrid(self):
        """Test hybrid rollout (NumPy -> JAX -> NumPy)."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_rollout import jax_rollout_hybrid
        
        initial_state = np.zeros(12, dtype=np.float32)
        horizon = 10
        actions = np.ones((horizon, 4), dtype=np.float32) * 0.5
        I = np.array([0.0023, 0.0023, 0.0046], dtype=np.float32)
        
        trajectory = jax_rollout_hybrid(
            initial_state, actions, 0.01,
            0.5, I, 0.17,
            3.16e-10, 7.94e-12, 9.81,
            2.0, 2.0, 1.0,
            use_jit=True
        )
        
        assert trajectory.shape == (horizon + 1, 12)
        assert isinstance(trajectory, np.ndarray)
        assert trajectory.dtype == np.float32


@pytest.mark.requires_jax
class TestJAXEnvironment:
    """Test JAX integration in environment classes."""
    
    def test_jax_transition(self):
        """Test JAX transition in environment."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.factories import make_env
        
        env = make_env(
            "drone_full_3d_physics",
            use_jax_dynamics=True,
            jax_jit=True,
            physics_backend="drone_model",
        )
        
        state, _ = env.reset()
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # Convert to JAX
        state_jax = jnp.asarray(state)
        action_jax = jnp.asarray(action)
        
        next_state_jax = env.jax_transition(state_jax, action_jax)
        
        assert next_state_jax.shape == (12,)
        assert isinstance(next_state_jax, jnp.ndarray)
        
        env.close()
    
    def test_jax_cost(self):
        """Test JAX cost function."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.factories import make_env
        
        env = make_env(
            "drone_full_3d_physics",
            use_jax_dynamics=True,
            jax_jit=True,
        )
        
        state = np.zeros(12, dtype=np.float32)
        state_jax = jnp.asarray(state)
        
        cost = env.jax_cost(state_jax)
        
        assert cost.shape == ()  # Scalar
        assert isinstance(cost, jnp.ndarray)
        
        env.close()
    
    def test_jax_batch_cost(self):
        """Test batched JAX cost computation."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.factories import make_env
        
        env = make_env(
            "drone_full_3d_physics",
            use_jax_dynamics=True,
            jax_jit=True,
        )
        
        batch_size = 10
        states = np.zeros((batch_size, 12), dtype=np.float32)
        states_jax = jnp.asarray(states)
        
        costs = env.jax_cost(states_jax)
        
        assert costs.shape == (batch_size,)
        assert isinstance(costs, jnp.ndarray)
        
        env.close()
    
    def test_hybrid_rollout(self):
        """Test hybrid rollout in environment."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.factories import make_env
        
        env = make_env(
            "drone_full_3d_physics",
            use_jax_dynamics=True,
            jax_jit=True,
            physics_backend="drone_model",
        )
        
        state, _ = env.reset()
        horizon = 10
        actions = np.ones((horizon, 4), dtype=np.float32) * 0.5
        
        # Use rollout_actions which should use JAX if available
        trajectory = env.rollout_actions(state, actions)
        
        assert trajectory.shape == (horizon + 1, 12)
        assert isinstance(trajectory, np.ndarray)
        
        env.close()


@pytest.mark.requires_jax
class TestJAXPerformance:
    """Test JAX performance optimizations."""
    
    def test_batch_vs_sequential(self):
        """Compare batch vs sequential computation performance."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        
        from enerdynamics.envs.utils.jax_dynamics import (
            jax_quadrotor_step,
            jax_quadrotor_step_batch,
        )
        
        batch_size = 100
        states = jnp.zeros((batch_size, 12), dtype=jnp.float32)
        actions = jnp.ones((batch_size, 4), dtype=jnp.float32) * 0.5
        I = jnp.array([0.0023, 0.0023, 0.0046], dtype=jnp.float32)
        
        # Sequential computation
        start = time.time()
        sequential_results = []
        for i in range(batch_size):
            result = jax_quadrotor_step(
                states[i], actions[i], 0.01,
                0.5, I, 0.17,
                3.16e-10, 7.94e-12, 9.81
            )
            sequential_results.append(result)
        sequential_time = time.time() - start
        
        # Batched computation
        start = time.time()
        batched_results = jax_quadrotor_step_batch(
            states, actions, 0.01,
            0.5, I, 0.17,
            3.16e-10, 7.94e-12, 9.81
        )
        batched_time = time.time() - start
        
        # Batched should be faster (or at least comparable)
        # Note: First call includes compilation time
        print(f"Sequential: {sequential_time:.4f}s, Batched: {batched_time:.4f}s")
        
        # Verify results are similar
        sequential_stack = jnp.stack(sequential_results)
        np.testing.assert_allclose(
            sequential_stack, batched_results, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





