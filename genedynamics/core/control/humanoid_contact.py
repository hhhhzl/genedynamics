"""Robot-facing contact and whole-body torque control for humanoid tasks."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from brax import math as brax_math
from mujoco.mjx._src import support as mjx_support


def hand_contact_wrench(
    system,
    pipeline_state,
    *,
    body_id: int,
    geom_id: int,
    target_position,
    contact_normal,
    stiffness,
    desired_force,
    damping: float,
    local_push_axis,
):
    """Cartesian impedance at the physical hand contact geom."""
    hand_position = pipeline_state.geom_xpos[geom_id]
    push_axis = brax_math.rotate(
        jnp.asarray(local_push_axis), pipeline_state.x.rot[body_id - 1]
    )
    jacp, _ = mjx_support.jac(
        system, pipeline_state, hand_position, body_id
    )
    hand_velocity = jacp.T @ pipeline_state.qvel
    wrench = (
        stiffness @ (target_position - hand_position)
        - damping * hand_velocity
        - desired_force * contact_normal
    )
    normal_force = jnp.maximum(-jnp.dot(wrench, contact_normal), 0.0)
    tangent_force = wrench + normal_force * contact_normal
    tangent_velocity = hand_velocity - jnp.dot(
        hand_velocity, contact_normal
    ) * contact_normal
    return {
        "p_c": target_position,
        "p_hand": hand_position,
        "push_axis": push_axis,
        "jacp": jacp,
        "wrench": wrench,
        "f_n": normal_force,
        "f_t": tangent_force,
        "slip": jnp.linalg.norm(tangent_velocity),
    }


@dataclass(frozen=True)
class HumanoidWholeBodyController:
    """Embodiment controller translating semantic hand contact into torques."""

    binding: object
    config: object
    default_pose: object
    stance_pose: object
    kp: object
    kd: object
    torque_limits: object
    arm_push_pose: object
    right_arm: tuple
    left_arm: tuple
    sagittal_legs: tuple
    n_planner: int
    pelvis_body_id: int
    feet_site_ids: object
    stance_com_x0: object
    is_walk: bool = False
    gait_controller: object = None

    def torque(self, pipeline_state, contact, action, info, primitive_width: int):
        cfg = self.config
        qidx = jnp.asarray(self.binding.qpos_indices)
        didx = jnp.asarray(self.binding.dof_indices)
        q = pipeline_state.qpos[qidx]
        qd = pipeline_state.qvel[didx]
        load = jnp.minimum(contact["F_n"], cfg.f_target)
        hip_load = jnp.maximum(load - cfg.stance_force_hip_deadband, 0.0)

        if self.is_walk:
            leg_residual = action[primitive_width:]
            com_x = pipeline_state.x.pos[self.pelvis_body_id - 1, 0]
            feet_x = pipeline_state.site_xpos[self.feet_site_ids, 0].mean()
            com_vx = pipeline_state.xd.vel[self.pelvis_body_id - 1, 0]
            com_y = pipeline_state.x.pos[self.pelvis_body_id - 1, 1]
            feet_y = pipeline_state.site_xpos[self.feet_site_ids, 1].mean()
            com_vy = pipeline_state.xd.vel[self.pelvis_body_id - 1, 1]
            gait_ref = self.gait_controller(
                jnp.asarray(info["step"], jnp.float32) * cfg.dt,
                com_x - feet_x,
                com_vx,
                cfg.target_vx,
                com_y - feet_y,
                com_vy,
            )
            gait_ref = gait_ref * jnp.clip(
                jnp.asarray(info["step"], jnp.float32) * cfg.dt
                / max(float(cfg.gait_ramp_time), 1e-6),
                0.0,
                1.0,
            )
            q_target = self.default_pose
            for hip, _, ankle in self.sagittal_legs:
                q_target = (
                    q_target.at[hip].add(
                        cfg.stance_hip_bias + cfg.stance_force_hip_gain * hip_load
                    ).at[ankle].add(cfg.stance_ankle_bias)
                )
            q_target = q_target + gait_ref
            q_target = q_target.at[:self.n_planner].add(
                cfg.leg_scale * jnp.clip(leg_residual, -1.0, 1.0)
            )
        else:
            q_target = self.stance_pose
            for hip, _, ankle in self.sagittal_legs:
                q_target = (
                    q_target.at[hip].add(
                        cfg.stance_hip_bias + cfg.stance_force_hip_gain * hip_load
                    ).at[ankle].add(cfg.stance_ankle_bias)
                )

        tau_pd = self.kp * (q_target - q) - self.kd * qd
        tau_pd = tau_pd.at[:self.n_planner].add(
            cfg.leg_grav_comp * pipeline_state.qfrc_bias[didx[:self.n_planner]]
        )
        load_scale = load / max(float(cfg.stance_force_reference), 1e-6)
        brace_tau = cfg.stance_force_ankle_gain * load_scale * load
        com_x = pipeline_state.x.pos[self.pelvis_body_id - 1, 0]
        feet_x = pipeline_state.site_xpos[self.feet_site_ids, 0].mean()
        com_correction = cfg.stance_com_ankle_gain * (
            com_x - feet_x - self.stance_com_x0
        ) + cfg.stance_com_ankle_damping * pipeline_state.xd.vel[
            self.pelvis_body_id - 1, 0
        ]
        for _, _, ankle in self.sagittal_legs:
            tau_pd = tau_pd.at[ankle].add(brace_tau + com_correction)

        tau = tau_pd
        for hand, joint_ids in (
            (contact, self.right_arm),
            (contact["left"], self.left_arm),
        ):
            local_ids = jnp.asarray(joint_ids)
            arm_dofs = didx[local_ids]
            jacobian = hand["jacp"].T[:, arm_dofs]
            eye3 = jnp.eye(3, dtype=jacobian.dtype)
            eye_arm = jnp.eye(len(joint_ids), dtype=jacobian.dtype)
            nullspace = eye_arm - jacobian.T @ jnp.linalg.solve(
                jacobian @ jacobian.T + cfg.arm_null_damping * eye3,
                jacobian,
            )
            posture = (
                cfg.arm_posture_kp * (self.arm_push_pose[local_ids] - q[local_ids])
                - cfg.arm_posture_kd * qd[local_ids]
            )
            tau_arm = (
                (hand["jacp"] @ hand["wrench"])[arm_dofs]
                + nullspace @ posture
                + cfg.arm_grav_comp * pipeline_state.qfrc_bias[arm_dofs]
            )
            tau = tau.at[local_ids].set(tau_arm)
        return jnp.clip(tau, -self.torque_limits, self.torque_limits)
