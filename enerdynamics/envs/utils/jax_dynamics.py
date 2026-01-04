"""
Pure JAX implementation of quadrotor dynamics.

This module provides high-performance JAX-native quadrotor dynamics
with support for JIT compilation, vectorization, and automatic differentiation.
"""

from typing import Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


def jax_euler_to_quaternion(euler: jnp.ndarray) -> jnp.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    High-performance JAX implementation with support for batched inputs.
    
    Args:
        euler: Euler angles [roll, pitch, yaw], shape (..., 3)
        
    Returns:
        Quaternion [w, x, y, z], shape (..., 4)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for jax_euler_to_quaternion")
    
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    
    # Half angles for quaternion conversion
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return jnp.stack([w, x, y, z], axis=-1)


def jax_quaternion_to_euler(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    
    High-performance JAX implementation with support for batched inputs.
    
    Args:
        quat: Quaternion [w, x, y, z], shape (..., 4)
        
    Returns:
        Euler angles [roll, pitch, yaw], shape (..., 3)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for jax_quaternion_to_euler")
    
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp sinp to [-1, 1] to avoid numerical issues
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    
    return jnp.stack([roll, pitch, yaw], axis=-1)


def jax_rotation_matrix(roll: jnp.ndarray, pitch: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """
    Compute rotation matrix from Euler angles (ZYX convention).
    
    High-performance JAX implementation. For single values, returns (3, 3) matrix.
    For batched inputs, use jax.vmap or compute element-wise.
    
    Args:
        roll: Roll angle (x-axis rotation), scalar or array
        pitch: Pitch angle (y-axis rotation), scalar or array
        yaw: Yaw angle (z-axis rotation), scalar or array
        
    Returns:
        Rotation matrix, shape (3, 3) for scalars
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for jax_rotation_matrix")
    
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    
    # Rotation matrix (ZYX convention)
    R = jnp.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    
    return R


if JAX_AVAILABLE:
    # JIT-compile the core dynamics function for maximum performance
    @jax.jit
    def jax_quadrotor_step(
        state: jnp.ndarray,
        motor_thrusts: jnp.ndarray,
        dt: float,
        mass: float,
        I: jnp.ndarray,
        arm_length: float,
        kf: float,
        km: float,
        gravity: float
    ) -> jnp.ndarray:
    """
    Single step of quadrotor dynamics using pure JAX.
    
    JIT-compiled for high performance. Supports automatic differentiation.
    
    Args:
        state: Current state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
        motor_thrusts: Motor thrusts [T1, T2, T3, T4] (normalized 0-1), shape (4,)
        dt: Time step
        mass: Quadrotor mass
        I: Moments of inertia [Ixx, Iyy, Izz], shape (3,)
        arm_length: Distance from center to motor
        kf: Thrust coefficient
        km: Moment coefficient
        gravity: Gravity acceleration
        
    Returns:
        Next state, shape (12,)
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("JAX is required for jax_quadrotor_step")
    
    # Extract state components
    pos = state[0:3]
    vel = state[3:6]
    euler = state[6:9]
    ang_vel = state[9:12]
    
    # Compute forces and moments from motor thrusts
    T_total = jnp.sum(motor_thrusts) * kf
    
    # Motor configuration: front-left, front-right, back-right, back-left
    T1, T2, T3, T4 = motor_thrusts[0], motor_thrusts[1], motor_thrusts[2], motor_thrusts[3]
    
    # Moments
    Mx = arm_length * kf * (T2 - T4)  # Roll moment
    My = arm_length * kf * (T1 - T3)  # Pitch moment
    Mz = km * (T1 - T2 + T3 - T4)     # Yaw moment
    
    # Compute rotation matrix from Euler angles
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    
    # Rotation matrix (ZYX convention) - computed element-wise for performance
    R = jnp.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    
    # Thrust in body frame (upward)
    thrust_body = jnp.array([0.0, 0.0, T_total])
    thrust_world = R @ thrust_body
    
    # Gravity
    gravity_vec = jnp.array([0.0, 0.0, -gravity])
    
    # Linear acceleration
    accel = (thrust_world / mass) + gravity_vec
    
    # Angular acceleration
    moments = jnp.array([Mx, My, Mz])
    I_inv = 1.0 / I
    ang_accel = I_inv * moments
    
    # Integrate using Euler method
    new_vel = vel + accel * dt
    new_pos = pos + vel * dt + 0.5 * accel * dt ** 2
    
    new_ang_vel = ang_vel + ang_accel * dt
    
    # Integrate angular velocity to get new orientation
    # Simplified: use small angle approximation
    new_euler = euler + ang_vel * dt
    
    # Combine new state
    new_state = jnp.concatenate([
        new_pos,
        new_vel,
        new_euler,
        new_ang_vel
    ])
    
        return new_state
    
    # JIT-compile state projection
    @jax.jit
    def jax_project_state(
        state: jnp.ndarray,
        p_max: float,
        v_max: float
    ) -> jnp.ndarray:
        """
        Project state to valid bounds using JAX operations.
        
        High-performance JAX implementation with support for batched inputs.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (..., 12)
            p_max: Maximum position magnitude
            v_max: Maximum velocity magnitude
            
        Returns:
            Projected state, shape (..., 12)
        """
        # Position bounds
        state = state.at[..., 0:3].set(jnp.clip(state[..., 0:3], -p_max, p_max))
        
        # Velocity bounds
        state = state.at[..., 3:6].set(jnp.clip(state[..., 3:6], -v_max, v_max))
        
        # Orientation bounds (Euler angles)
        state = state.at[..., 6:9].set(jnp.clip(state[..., 6:9], -jnp.pi, jnp.pi))
        
        # Pitch bounds (special case: -pi/2 to pi/2)
        state = state.at[..., 8].set(jnp.clip(state[..., 8], -jnp.pi/2, jnp.pi/2))
        
        # Angular velocity bounds
        state = state.at[..., 9:12].set(jnp.clip(state[..., 9:12], -5.0, 5.0))
        
        return state


# Create JIT-compiled and batched versions for high-performance batch processing
if JAX_AVAILABLE:
    # Create batched version using vmap (already JIT-compiled via decorator)
    jax_quadrotor_step_batch = jax.vmap(
        jax_quadrotor_step,
        in_axes=(0, 0, None, None, None, None, None, None, None),
        out_axes=0
    )
    # JIT-compile the batched version for maximum performance
    jax_quadrotor_step_batch = jax.jit(jax_quadrotor_step_batch)
    """Batched version of jax_quadrotor_step for processing multiple states/actions in parallel."""
    
    # Create batched version of state projection (already JIT-compiled via decorator)
    jax_project_state_batch = jax.vmap(
        jax_project_state,
        in_axes=(0, None, None),
        out_axes=0
    )
    # JIT-compile the batched version
    jax_project_state_batch = jax.jit(jax_project_state_batch)
    """Batched version of jax_project_state for processing multiple states in parallel."""
else:
    # Placeholder functions when JAX is not available
    jax_quadrotor_step = None
    jax_project_state = None
    jax_quadrotor_step_batch = None
    jax_project_state_batch = None

