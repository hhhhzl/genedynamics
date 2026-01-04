"""
Utility modules for environment implementations.

This package provides utility functions for:
- JAX dynamics implementations
- State conversion between different formats
- Model file generation
"""

from .jax_dynamics import (
    jax_euler_to_quaternion,
    jax_quaternion_to_euler,
    jax_rotation_matrix,
    jax_quadrotor_step,
    jax_project_state,
    jax_quadrotor_step_batch,
    jax_project_state_batch,
)

from .state_converter import (
    euler_to_quaternion,
    quaternion_to_euler,
    state_12d_to_mujoco,
    mujoco_to_state_12d,
    state_12d_to_isaac,
    isaac_to_state_12d,
)

from .mujoco_model_generator import (
    generate_obstacle_xml,
    generate_mujoco_xml_with_obstacles,
    create_base_quadrotor_xml,
)

from .isaac_usd_generator import (
    add_obstacle_to_usd,
    create_quadrotor_usd_with_obstacles,
    create_base_quadrotor_usd,
)

from .jax_rollout import (
    jax_rollout_single,
    jax_rollout_batch,
    jax_rollout_hybrid,
    jax_rollout_batch_hybrid,
)

__all__ = [
    # JAX dynamics
    'jax_euler_to_quaternion',
    'jax_quaternion_to_euler',
    'jax_rotation_matrix',
    'jax_quadrotor_step',
    'jax_project_state',
    'jax_quadrotor_step_batch',
    'jax_project_state_batch',
    # State conversion
    'euler_to_quaternion',
    'quaternion_to_euler',
    'state_12d_to_mujoco',
    'mujoco_to_state_12d',
    'state_12d_to_isaac',
    'isaac_to_state_12d',
    # MuJoCo model generation
    'generate_obstacle_xml',
    'generate_mujoco_xml_with_obstacles',
    'create_base_quadrotor_xml',
    # Isaac Sim USD generation
    'add_obstacle_to_usd',
    'create_quadrotor_usd_with_obstacles',
    'create_base_quadrotor_usd',
    # JAX rollout utilities
    'jax_rollout_single',
    'jax_rollout_batch',
    'jax_rollout_hybrid',
    'jax_rollout_batch_hybrid',
]

