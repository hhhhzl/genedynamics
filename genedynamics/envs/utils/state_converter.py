"""
State conversion utilities for quadrotor environments.

This module provides conversion functions between different state representations:
- 12D state: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
- MuJoCo format: {'qpos': [x, y, z, qw, qx, qy, qz], 'qvel': [vx, vy, vz, wx, wy, wz]}
- Isaac Sim format: Similar to MuJoCo but with PyTorch tensors

Supports both NumPy and JAX implementations for high performance.
"""

from typing import Dict, Tuple, Union
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# ============================================================================
# NumPy implementations (for NumPy backend and MuJoCo)
# ============================================================================

def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    NumPy implementation for use with MuJoCo and NumPy backends.
    
    Args:
        euler: Euler angles [roll, pitch, yaw], shape (3,) or (..., 3)
        
    Returns:
        Quaternion [w, x, y, z], shape (4,) or (..., 4)
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    if euler.ndim == 1:
        return np.array([w, x, y, z], dtype=np.float32)
    else:
        return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    
    NumPy implementation for use with MuJoCo and NumPy backends.
    
    Args:
        quat: Quaternion [w, x, y, z], shape (4,) or (..., 4)
        
    Returns:
        Euler angles [roll, pitch, yaw], shape (3,) or (..., 3)
    """
    if quat.ndim == 1:
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    else:
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if quat.ndim == 1:
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
    else:
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    if quat.ndim == 1:
        return np.array([roll, pitch, yaw], dtype=np.float32)
    else:
        return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def state_12d_to_mujoco(state: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert 12D state to MuJoCo state format.
    
    Args:
        state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
        
    Returns:
        MuJoCo state dict with 'qpos' and 'qvel' keys
    """
    pos = state[0:3]
    vel = state[3:6]
    euler = state[6:9]
    ang_vel = state[9:12]
    
    # Convert Euler to quaternion
    quat = euler_to_quaternion(euler)
    
    return {
        'qpos': np.concatenate([pos, quat]).astype(np.float64),
        'qvel': np.concatenate([vel, ang_vel]).astype(np.float64),
    }


def mujoco_to_state_12d(mujoco_state: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert MuJoCo state to 12D state.
    
    Args:
        mujoco_state: MuJoCo state dict with 'qpos' and 'qvel' keys
        
    Returns:
        12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
    """
    qpos = mujoco_state['qpos']
    qvel = mujoco_state['qvel']
    
    pos = qpos[0:3]
    quat = qpos[3:7]  # qw, qx, qy, qz
    vel = qvel[0:3]
    ang_vel = qvel[3:6]
    
    # Convert quaternion to Euler
    euler = quaternion_to_euler(quat)
    
    return np.concatenate([pos, vel, euler, ang_vel]).astype(np.float32)


def state_12d_to_isaac(state: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert 12D state to Isaac Sim state format.
    
    Args:
        state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
        
    Returns:
        Isaac Sim state dict with 'qpos' and 'qvel' keys (NumPy arrays)
    """
    pos = state[0:3]
    vel = state[3:6]
    euler = state[6:9]
    ang_vel = state[9:12]
    
    # Convert Euler to quaternion
    quat = euler_to_quaternion(euler)
    
    return {
        'qpos': np.concatenate([pos, quat]).astype(np.float32),
        'qvel': np.concatenate([vel, ang_vel]).astype(np.float32),
    }


def isaac_to_state_12d(isaac_state: Union[Dict[str, np.ndarray], Dict[str, 'torch.Tensor']]) -> np.ndarray:
    """
    Convert Isaac Sim state to 12D state.
    
    Handles both NumPy arrays and PyTorch tensors.
    
    Args:
        isaac_state: Isaac Sim state dict with 'qpos' and 'qvel' keys
        
    Returns:
        12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
    """
    # Convert to NumPy if needed
    if TORCH_AVAILABLE and torch is not None:
        if isinstance(isaac_state['qpos'], torch.Tensor):
            qpos = isaac_state['qpos'].cpu().numpy()
            qvel = isaac_state['qvel'].cpu().numpy()
        else:
            qpos = np.asarray(isaac_state['qpos'])
            qvel = np.asarray(isaac_state['qvel'])
    else:
        qpos = np.asarray(isaac_state['qpos'])
        qvel = np.asarray(isaac_state['qvel'])
    
    pos = qpos[0:3]
    quat = qpos[3:7]  # qw, qx, qy, qz
    vel = qvel[0:3]
    ang_vel = qvel[3:6]
    
    # Convert quaternion to Euler
    euler = quaternion_to_euler(quat)
    
    return np.concatenate([pos, vel, euler, ang_vel]).astype(np.float32)


# ============================================================================
# JAX implementations (for JAX backend and high-performance planning)
# ============================================================================

if JAX_AVAILABLE:
    try:
        from .jax_dynamics import jax_euler_to_quaternion, jax_quaternion_to_euler
    except ImportError:
        # JAX available but jax_dynamics not importable
        JAX_AVAILABLE = False
        jax_euler_to_quaternion = None
        jax_quaternion_to_euler = None
    
    def jax_state_12d_to_mujoco(state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Convert 12D state to MuJoCo state format using JAX.
        
        Args:
            state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
            
        Returns:
            MuJoCo state dict with 'qpos' and 'qvel' keys (JAX arrays)
        """
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        ang_vel = state[9:12]
        
        # Convert Euler to quaternion
        quat = jax_euler_to_quaternion(euler)
        
        return {
            'qpos': jnp.concatenate([pos, quat]),
            'qvel': jnp.concatenate([vel, ang_vel]),
        }
    
    
    def jax_mujoco_to_state_12d(mujoco_state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Convert MuJoCo state to 12D state using JAX.
        
        Args:
            mujoco_state: MuJoCo state dict with 'qpos' and 'qvel' keys (JAX arrays)
            
        Returns:
            12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
        """
        qpos = mujoco_state['qpos']
        qvel = mujoco_state['qvel']
        
        pos = qpos[0:3]
        quat = qpos[3:7]  # qw, qx, qy, qz
        vel = qvel[0:3]
        ang_vel = qvel[3:6]
        
        # Convert quaternion to Euler
        euler = jax_quaternion_to_euler(quat)
        
        return jnp.concatenate([pos, vel, euler, ang_vel])
    
    
    def jax_state_12d_to_isaac(state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Convert 12D state to Isaac Sim state format using JAX.
        
        Args:
            state: 12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
            
        Returns:
            Isaac Sim state dict with 'qpos' and 'qvel' keys (JAX arrays)
        """
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        ang_vel = state[9:12]
        
        # Convert Euler to quaternion
        quat = jax_euler_to_quaternion(euler)
        
        return {
            'qpos': jnp.concatenate([pos, quat]),
            'qvel': jnp.concatenate([vel, ang_vel]),
        }
    
    
    def jax_isaac_to_state_12d(isaac_state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Convert Isaac Sim state to 12D state using JAX.
        
        Args:
            isaac_state: Isaac Sim state dict with 'qpos' and 'qvel' keys (JAX arrays)
            
        Returns:
            12D state [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz], shape (12,)
        """
        qpos = isaac_state['qpos']
        qvel = isaac_state['qvel']
        
        pos = qpos[0:3]
        quat = qpos[3:7]  # qw, qx, qy, qz
        vel = qvel[0:3]
        ang_vel = qvel[3:6]
        
        # Convert quaternion to Euler
        euler = jax_quaternion_to_euler(quat)
        
        return jnp.concatenate([pos, vel, euler, ang_vel])
else:
    # Placeholder functions when JAX is not available
    def jax_state_12d_to_mujoco(state):
        raise RuntimeError("JAX is required for jax_state_12d_to_mujoco")
    
    def jax_mujoco_to_state_12d(mujoco_state):
        raise RuntimeError("JAX is required for jax_mujoco_to_state_12d")
    
    def jax_state_12d_to_isaac(state):
        raise RuntimeError("JAX is required for jax_state_12d_to_isaac")
    
    def jax_isaac_to_state_12d(isaac_state):
        raise RuntimeError("JAX is required for jax_isaac_to_state_12d")

