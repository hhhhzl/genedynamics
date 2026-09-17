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
    joint_limits: object = None
    planner_joint_bounds: object = None
    planner_action_reference: object = None
    planner_reference_residual_scale: float | None = None

    def __post_init__(self):
        direct = self.is_walk and getattr(self.config, "walk_leg_control", "legacy") == "joint_target"
        if self.planner_joint_bounds is None:
            if direct:
                raise ValueError("joint_target requires robot-owned planner_joint_bounds")
            return
        import numpy as np

        bounds = np.asarray(self.planner_joint_bounds)
        if (bounds.shape != (self.n_planner, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])):
            raise ValueError("planner_joint_bounds must be finite ordered (n_planner, 2) bounds")
        if self.joint_limits is not None:
            limits = np.asarray(self.joint_limits)[:self.n_planner]
            if (limits.shape != bounds.shape or np.any(bounds[:, 0] < limits[:, 0])
                    or np.any(bounds[:, 1] > limits[:, 1])):
                raise ValueError("planner_joint_bounds must lie within physical joint limits")
        if self.planner_action_reference is not None:
            reference = np.asarray(self.planner_action_reference)
            if (reference.ndim != 2 or reference.shape[1] != self.n_planner
                    or reference.shape[0] < 2 or not np.isfinite(reference).all()
                    or np.any(np.abs(reference) > 1.0)):
                raise ValueError(
                    "planner_action_reference must be finite normalized (T, n_planner)"
                )
            scale = self.planner_reference_residual_scale
            if scale is None or not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError("planner reference residual scale must be in [0, 1]")

    def planner_action_from_joints(self, joints):
        """Inverse affine command chart; outside targets project to its bounds."""
        if self.planner_joint_bounds is None:
            raise ValueError("planner joint inverse requires planner_joint_bounds")
        bounds = jnp.asarray(self.planner_joint_bounds)
        return jnp.clip(2.0 * (joints - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0,
                        -1.0, 1.0)

    def joint_targets(self, pipeline_state, contact, action, info, primitive_width: int):
        """Nominal gait and bounded residuals, before torque feedback."""
        cfg = self.config
        if self.is_walk and getattr(cfg, "walk_leg_control", "legacy") == "joint_target":
            # Without a reference, the last coordinates are absolute H1 joint
            # commands.  With the shared DIAL motion reference, they are small
            # normalized residuals around its time-indexed joint targets.
            bounds = jnp.asarray(self.planner_joint_bounds)
            normalized_action = jnp.clip(action[primitive_width:], -1.0, 1.0)
            if self.planner_action_reference is not None:
                index = jnp.clip(
                    jnp.asarray(
                        info.get("walk_reference_step", info["step"]),
                        jnp.int32,
                    ),
                    0,
                    self.planner_action_reference.shape[0] - 1,
                )
                normalized_action = jnp.clip(
                    self.planner_action_reference[index]
                    + self.planner_reference_residual_scale * normalized_action,
                    -1.0,
                    1.0,
                )
            normalized = (normalized_action + 1.0) / 2.0
            targets = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
            return self.default_pose.at[:self.n_planner].set(targets)
        load = jnp.minimum(contact.get("support_load", contact["F_n"]), cfg.f_target)
        hip_load = jnp.maximum(load - cfg.stance_force_hip_deadband, 0.0)

        if self.is_walk:
            leg_residual = action[primitive_width:]
            com_x = pipeline_state.x.pos[self.pelvis_body_id - 1, 0]
            feet_x = pipeline_state.site_xpos[self.feet_site_ids, 0].mean()
            com_vx = pipeline_state.xd.vel[self.pelvis_body_id - 1, 0]
            com_y = pipeline_state.x.pos[self.pelvis_body_id - 1, 1]
            feet_y = pipeline_state.site_xpos[self.feet_site_ids, 1].mean()
            com_vy = pipeline_state.xd.vel[self.pelvis_body_id - 1, 1]
            if "support_reference_xy" in contact:
                feet_x, feet_y = contact["support_reference_xy"]
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
            if "stance_support_weights" in contact and "swing_foot_pitch_target" in contact:
                # Level the nominal swing foot before applying planner actions.
                # Applying this after the residual would erase the ankle action
                # and couple its authority to the hip/knee residuals instead.
                for side, (hip, knee, ankle) in enumerate(self.sagittal_legs):
                    foot_level = (
                        contact["swing_foot_pitch_target"]
                        - q_target[hip] - q_target[knee]
                    )
                    q_target = q_target.at[ankle].set(jnp.where(
                        contact["swing_feet"][side], foot_level, q_target[ankle]
                    ))
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

        if "stance_support_weights" in contact and self.is_walk:
            if self.joint_limits is not None:
                q_target = q_target.at[:self.n_planner].set(jnp.clip(
                    q_target[:self.n_planner],
                    self.joint_limits[:self.n_planner, 0] + 0.02,
                    self.joint_limits[:self.n_planner, 1] - 0.02,
                ))
        return q_target

    def torque(self, pipeline_state, contact, action, info, primitive_width: int):
        cfg = self.config
        qidx = jnp.asarray(self.binding.qpos_indices)
        didx = jnp.asarray(self.binding.dof_indices)
        q = pipeline_state.qpos[qidx]
        qd = pipeline_state.qvel[didx]
        q_target = self.joint_targets(pipeline_state, contact, action, info, primitive_width)
        load = jnp.minimum(contact.get("support_load", contact["F_n"]), cfg.f_target)
        tau_pd = self.kp * (q_target - q) - self.kd * qd
        direct = self.is_walk and getattr(cfg, "walk_leg_control", "legacy") == "joint_target"
        # Gravity compensation remains an explicit, independent setting.
        # joint_target + leg_grav_comp=0 is the original DIAL leg PD law.
        if not direct or cfg.leg_grav_comp != 0.0:
            tau_pd = tau_pd.at[:self.n_planner].add(
                cfg.leg_grav_comp * pipeline_state.qfrc_bias[didx[:self.n_planner]]
            )
        if direct:
            # DIAL supplies the nominal walking joint targets, while the
            # Cartesian hand controller supplies a measured external load.
            # Compensate only that load at the physically supporting ankle;
            # otherwise an unloaded DIAL gait is pushed backwards as soon as
            # contact is acquired.  The term is exactly zero without measured
            # hand contact and therefore preserves the original DIAL law.
            load_scale = load / max(float(cfg.stance_force_reference), 1e-6)
            brace_tau = cfg.stance_force_ankle_gain * load_scale * load
            support_weights = contact.get("stance_support_weights", jnp.ones(2))
            for side, (_, _, ankle) in enumerate(self.sagittal_legs):
                tau_pd = tau_pd.at[ankle].add(support_weights[side] * brace_tau)
        else:
            load_scale = load / max(float(cfg.stance_force_reference), 1e-6)
            brace_tau = cfg.stance_force_ankle_gain * load_scale * load
            com_x = pipeline_state.x.pos[self.pelvis_body_id - 1, 0]
            feet_x = pipeline_state.site_xpos[self.feet_site_ids, 0].mean()
            # ``load`` and the COM correction are whole-body loads/moments,
            # not per-foot quantities.  A fixed two-foot stance therefore
            # shares them across the ankles; applying the full value to both
            # feet double-counts the requested contact load.  Walking tasks
            # provide explicit support weights in the direct branch above.
            support_weights = contact.get(
                "stance_support_weights", 0.5 * jnp.ones(2)
            )
            if "support_reference_xy" in contact:
                # The task supplies the same physical-support reference used by
                # gait capture, including its continuous low-load/flight limit.
                feet_x = contact["support_reference_xy"][0]
            com_correction = cfg.stance_com_ankle_gain * (
                com_x - feet_x - self.stance_com_x0
            ) + cfg.stance_com_ankle_damping * pipeline_state.xd.vel[
                self.pelvis_body_id - 1, 0
            ]
            for side, (_, _, ankle) in enumerate(self.sagittal_legs):
                tau_pd = tau_pd.at[ankle].add(
                    support_weights[side] * (brace_tau + com_correction)
                )

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
                cfg.arm_posture_kp
                * contact.get("arm_posture_position_scale", 1.0)
                * (self.arm_push_pose[local_ids] - q[local_ids])
                - cfg.arm_posture_kd * qd[local_ids]
            )
            tau_arm = (
                contact.get("arm_task_scale", 1.0)
                * (hand["jacp"] @ hand["wrench"])[arm_dofs]
                + nullspace @ posture
                + cfg.arm_grav_comp * pipeline_state.qfrc_bias[arm_dofs]
            )
            tau_arm = tau_arm * contact.get("arm_control_scale", 1.0)
            tau = tau.at[local_ids].set(tau_arm)
        return jnp.clip(tau, -self.torque_limits, self.torque_limits)
