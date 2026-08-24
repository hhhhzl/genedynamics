"""Robot-facing Cartesian impedance operations for MJX manipulators."""

from __future__ import annotations

import jax.numpy as jnp


def end_effector_kinematics(system, pipeline_state, site_id: int, body_id: int):
    """Return pose, Jacobians, and Cartesian velocities at a semantic TCP."""
    from mujoco.mjx._src import support as mjx_support

    pos = pipeline_state.site_xpos[site_id]
    rot = pipeline_state.site_xmat[site_id].reshape(3, 3)
    jacp, jacr = mjx_support.jac(system, pipeline_state, pos, body_id)
    return (
        pos,
        rot,
        jacp,
        jacr,
        jacp.T @ pipeline_state.qvel,
        jacr.T @ pipeline_state.qvel,
    )


def orientation_error(current_rotation, desired_rotation):
    """World-frame SO(3) error used by the legacy impedance law."""
    return 0.5 * (
        jnp.cross(current_rotation[:, 0], desired_rotation[:, 0])
        + jnp.cross(current_rotation[:, 1], desired_rotation[:, 1])
        + jnp.cross(current_rotation[:, 2], desired_rotation[:, 2])
    )


def map_cartesian_wrench(
    jacp,
    jacr,
    force,
    moment,
    bias_force,
    *,
    gravity_compensation: float,
    torque_limits,
):
    """Map a Cartesian wrench to the robot actuator order and clip limits."""
    tau = (
        jacp @ force
        + jacr @ moment
        + gravity_compensation * bias_force
    )
    limits = jnp.asarray(torque_limits, tau.dtype)
    return jnp.clip(tau, -limits, limits)
