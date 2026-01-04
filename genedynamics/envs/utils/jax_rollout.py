"""
High-performance JAX rollout utilities for trajectory generation.

This module provides optimized JAX-based rollout functions for generating
trajectories with support for JIT compilation and batch processing.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.envs.utils.jax_dynamics import (
    jax_quadrotor_step,
    jax_project_state,
    jax_quadrotor_step_batch,
    jax_project_state_batch,
)


if JAX_AVAILABLE:
    @jax.jit
    def jax_rollout_single(
        initial_state: jnp.ndarray,
        actions: jnp.ndarray,
        dt: float,
        mass: float,
        I: jnp.ndarray,
        arm_length: float,
        kf: float,
        km: float,
        gravity: float,
        p_max: float,
        v_max: float,
        control_limit: float,
    ) -> jnp.ndarray:
        """
        Rollout a single trajectory using JAX (JIT-compiled).
        
        High-performance single trajectory rollout with automatic differentiation support.
        
        Args:
            initial_state: Initial state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
            actions: Sequence of actions, shape (horizon, 4)
            dt: Time step
            mass: Quadrotor mass
            I: Moments of inertia [Ixx, Iyy, Izz], shape (3,)
            arm_length: Distance from center to motor
            kf: Thrust coefficient
            km: Moment coefficient
            gravity: Gravity acceleration
            p_max: Maximum position bound
            v_max: Maximum velocity bound
            control_limit: Maximum control input
            
        Returns:
            Trajectory of states, shape (horizon+1, 12)
        """
        horizon = actions.shape[0]
        
        def step_fn(state, action):
            # Clip action
            motor_thrusts = jnp.clip(action, 0.0, control_limit)
            
            # Compute next state
            next_state = jax_quadrotor_step(
                state, motor_thrusts, dt,
                mass, I, arm_length, kf, km, gravity
            )
            
            # Project state
            next_state = jax_project_state(next_state, p_max, v_max)
            
            return next_state, next_state
        
        # Use scan for efficient sequential computation
        _, trajectory = jax.lax.scan(step_fn, initial_state, actions)
        
        # Prepend initial state
        trajectory = jnp.concatenate([initial_state[None, :], trajectory], axis=0)
        
        return trajectory
    
    @jax.jit
    def jax_rollout_batch(
        initial_states: jnp.ndarray,
        actions: jnp.ndarray,
        dt: float,
        mass: float,
        I: jnp.ndarray,
        arm_length: float,
        kf: float,
        km: float,
        gravity: float,
        p_max: float,
        v_max: float,
        control_limit: float,
    ) -> jnp.ndarray:
        """
        Rollout multiple trajectories in parallel using JAX (JIT-compiled).
        
        High-performance batched trajectory rollout with parallel processing.
        
        Args:
            initial_states: Initial states, shape (batch, 12)
            actions: Sequence of actions, shape (batch, horizon, 4)
            dt: Time step
            mass: Quadrotor mass
            I: Moments of inertia [Ixx, Iyy, Izz], shape (3,)
            arm_length: Distance from center to motor
            kf: Thrust coefficient
            km: Moment coefficient
            gravity: Gravity acceleration
            p_max: Maximum position bound
            v_max: Maximum velocity bound
            control_limit: Maximum control input
            
        Returns:
            Trajectories of states, shape (batch, horizon+1, 12)
        """
        batch_size = initial_states.shape[0]
        horizon = actions.shape[1]
        
        def step_fn_batch(states, actions_t):
            # Clip actions
            motor_thrusts = jnp.clip(actions_t, 0.0, control_limit)
            
            # Compute next states (batched)
            next_states = jax_quadrotor_step_batch(
                states, motor_thrusts, dt,
                mass, I, arm_length, kf, km, gravity
            )
            
            # Project states (batched)
            next_states = jax_project_state_batch(next_states, p_max, v_max)
            
            return next_states, next_states
        
        # Use scan for efficient sequential computation over time
        # actions shape: (horizon, batch, 4), need to transpose
        actions_transposed = jnp.transpose(actions, (1, 0, 2))  # (horizon, batch, 4)
        
        _, trajectories = jax.lax.scan(step_fn_batch, initial_states, actions_transposed)
        
        # Transpose back: (horizon, batch, 12) -> (batch, horizon, 12)
        trajectories = jnp.transpose(trajectories, (1, 0, 2))
        
        # Prepend initial states
        trajectories = jnp.concatenate([initial_states[:, None, :], trajectories], axis=1)
        
        return trajectories
    
    def jax_rollout_hybrid(
        initial_state: np.ndarray,
        actions: np.ndarray,
        dt: float,
        mass: float,
        I: np.ndarray,
        arm_length: float,
        kf: float,
        km: float,
        gravity: float,
        p_max: float,
        v_max: float,
        control_limit: float,
        use_jit: bool = True,
    ) -> np.ndarray:
        """
        Hybrid rollout: NumPy input, JAX computation, NumPy output.
        
        Convenience function for using JAX dynamics from NumPy code.
        Automatically handles type conversion and JIT compilation.
        
        Args:
            initial_state: Initial state (NumPy array), shape (12,)
            actions: Sequence of actions (NumPy array), shape (horizon, 4)
            dt: Time step
            mass: Quadrotor mass
            I: Moments of inertia [Ixx, Iyy, Izz] (NumPy array), shape (3,)
            arm_length: Distance from center to motor
            kf: Thrust coefficient
            km: Moment coefficient
            gravity: Gravity acceleration
            p_max: Maximum position bound
            v_max: Maximum velocity bound
            control_limit: Maximum control input
            use_jit: Whether to use JIT-compiled function (default: True)
            
        Returns:
            Trajectory of states (NumPy array), shape (horizon+1, 12)
        """
        # Convert to JAX arrays
        state_jax = jnp.asarray(initial_state, dtype=jnp.float32)
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)
        I_jax = jnp.asarray(I, dtype=jnp.float32)
        
        # Use JIT-compiled function
        if use_jit:
            trajectory_jax = jax_rollout_single(
                state_jax, actions_jax, dt,
                mass, I_jax, arm_length, kf, km, gravity,
                p_max, v_max, control_limit
            )
        else:
            # Non-JIT version (for debugging)
            horizon = actions.shape[0]
            trajectory = [state_jax]
            state = state_jax
            for action in actions_jax:
                motor_thrusts = jnp.clip(action, 0.0, control_limit)
                state = jax_quadrotor_step(
                    state, motor_thrusts, dt,
                    mass, I_jax, arm_length, kf, km, gravity
                )
                state = jax_project_state(state, p_max, v_max)
                trajectory.append(state)
            trajectory_jax = jnp.stack(trajectory)
        
        # Convert back to NumPy
        return np.asarray(trajectory_jax)
    
    def jax_rollout_batch_hybrid(
        initial_states: np.ndarray,
        actions: np.ndarray,
        dt: float,
        mass: float,
        I: np.ndarray,
        arm_length: float,
        kf: float,
        km: float,
        gravity: float,
        p_max: float,
        v_max: float,
        control_limit: float,
        use_jit: bool = True,
    ) -> np.ndarray:
        """
        Hybrid batched rollout: NumPy input, JAX computation, NumPy output.
        
        Convenience function for using JAX batched dynamics from NumPy code.
        
        Args:
            initial_states: Initial states (NumPy array), shape (batch, 12)
            actions: Sequence of actions (NumPy array), shape (batch, horizon, 4)
            dt: Time step
            mass: Quadrotor mass
            I: Moments of inertia [Ixx, Iyy, Izz] (NumPy array), shape (3,)
            arm_length: Distance from center to motor
            kf: Thrust coefficient
            km: Moment coefficient
            gravity: Gravity acceleration
            p_max: Maximum position bound
            v_max: Maximum velocity bound
            control_limit: Maximum control input
            use_jit: Whether to use JIT-compiled function (default: True)
            
        Returns:
            Trajectories of states (NumPy array), shape (batch, horizon+1, 12)
        """
        # Convert to JAX arrays
        states_jax = jnp.asarray(initial_states, dtype=jnp.float32)
        actions_jax = jnp.asarray(actions, dtype=jnp.float32)
        I_jax = jnp.asarray(I, dtype=jnp.float32)
        
        # Use JIT-compiled function
        if use_jit:
            trajectories_jax = jax_rollout_batch(
                states_jax, actions_jax, dt,
                mass, I_jax, arm_length, kf, km, gravity,
                p_max, v_max, control_limit
            )
        else:
            # Non-JIT version (for debugging)
            batch_size = initial_states.shape[0]
            horizon = actions.shape[1]
            trajectories = []
            for i in range(batch_size):
                traj = jax_rollout_hybrid(
                    initial_states[i], actions[i], dt,
                    mass, I, arm_length, kf, km, gravity,
                    p_max, v_max, control_limit, use_jit=False
                )
                trajectories.append(traj)
            trajectories_jax = jnp.stack(trajectories)
        
        # Convert back to NumPy
        return np.asarray(trajectories_jax)

else:
    # Placeholder functions when JAX is not available
    jax_rollout_single = None
    jax_rollout_batch = None
    jax_rollout_hybrid = None
    jax_rollout_batch_hybrid = None


