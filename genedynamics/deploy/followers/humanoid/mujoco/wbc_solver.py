"""
Whole-body inverse-dynamics controller for humanoid corridor following in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.followers.common.controller_output import JointTargets
from genedynamics.deploy.followers.humanoid.models.g1_model import G1ModelSpec
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec


@dataclass
class G1WBCTaskStackConfig:
    contact_position_kp: float = 8.0
    contact_position_kd: float = 6.0
    contact_position_error_clip: float = 0.01
    contact_accel_limit: float = 6.0
    com_kp: float = 45.0
    com_kd: float = 14.0
    pelvis_orientation_kp: float = 40.0
    pelvis_orientation_kd: float = 10.0
    torso_orientation_kp: float = 36.0
    torso_orientation_kd: float = 9.0
    swing_foot_position_kp: float = 20.0
    swing_foot_position_kd: float = 6.0
    swing_foot_orientation_kp: float = 8.0
    swing_foot_orientation_kd: float = 3.0
    posture_kp: float = 18.0
    posture_kd: float = 5.0
    waist_kp: float = 24.0
    waist_kd: float = 6.0
    arm_kp: float = 14.0
    arm_kd: float = 4.0
    contact_task_weight: float = 80.0
    com_task_weight: float = 8.0
    pelvis_task_weight: float = 7.0
    torso_task_weight: float = 6.0
    swing_foot_task_weight: float = 8.0
    swing_foot_orientation_weight: float = 2.5
    lower_body_posture_weight: float = 2.0
    waist_posture_weight: float = 1.6
    arm_posture_weight: float = 1.2
    posture_weight: float = 0.5
    lambda_weight: float = 1e-3
    ddq_weight: float = 1e-4
    lambda_normal_target: float = 35.0
    lambda_normal_weight: float = 0.05
    lambda_tangent_weight: float = 0.02
    friction_coeff: float = 0.60
    lambda_min_normal: float = 20.0
    lambda_max_normal: float = 550.0
    lambda_max_tangent: float = 220.0
    through_gap_com_weight_scale: float = 1.35
    through_gap_pelvis_weight_scale: float = 1.25
    through_gap_torso_weight_scale: float = 2.20
    through_gap_swing_weight_scale: float = 0.70
    through_gap_arm_weight_scale: float = 2.40
    through_gap_waist_weight_scale: float = 2.00
    through_gap_lower_body_weight_scale: float = 1.40
    single_support_com_weight_scale: float = 1.20
    single_support_pelvis_weight_scale: float = 0.95
    single_support_torso_weight_scale: float = 0.35
    single_support_swing_weight_scale: float = 0.12
    single_support_arm_weight_scale: float = 0.03
    single_support_waist_weight_scale: float = 0.05
    single_support_lower_body_weight_scale: float = 0.18
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    crouch_hip_pitch_gain: float = 0.35
    crouch_knee_gain: float = 0.95
    crouch_ankle_pitch_gain: float = -0.45
    pelvis_forward_hip_pitch_gain: float = 0.20
    pelvis_lateral_hip_roll_gain: float = 0.50
    max_joint_accel: float = 40.0
    max_base_accel: float = 20.0
    max_qd_ref: float = 8.0
    max_q_step: float = 0.12
    torque_limit_scale: float = 0.85
    qp_regularization: float = 1e-8
    use_osqp: bool = True
    osqp_maxiter: int = 4000
    osqp_polish: bool = True
    osqp_verbose: bool = False
    osqp_accept_constraint_tol: float = 5e-4
    use_slsqp: bool = True
    slsqp_maxiter: int = 120
    use_trust_constr: bool = False
    trust_constr_maxiter: int = 80
    use_trust_constr_repair: bool = True
    trust_constr_repair_maxiter: int = 40
    trust_constr_repair_violation_threshold: float = 1e-3
    single_support_trust_constr_maxiter: int = 160
    active_set_refine_iters: int = 8
    projection_repair_iters: int = 40
    constraint_tol: float = 1e-6


class G1WholeBodyDynamicWBCSolver:
    """
    Contact-aware inverse-dynamics whole-body controller.

    The solver optimizes generalized accelerations and support forces and
    recovers actuator torques from the floating-base dynamics:

      M(q) ddq + h(q, dq) = S^T tau + J_c(q)^T lambda

    Support-foot translational contacts are enforced as hard equalities, while
    CoM, pelvis, torso, swing-foot, and posture objectives enter a weighted
    least-squares solve in acceleration space.
    """

    solver_name = "wbc_inverse_dynamics_qp_hard_contact"

    def __init__(
        self,
        model: object,
        data: object,
        model_spec: G1ModelSpec,
        cfg: Optional[G1WBCTaskStackConfig] = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.model_spec = model_spec
        self.cfg = cfg or G1WBCTaskStackConfig()
        self._last_osqp_solution: Optional[np.ndarray] = None

    def build_task_stack(self, tasks: HumanoidTaskSpec) -> Dict[str, object]:
        return {
            "pelvis": tasks.pelvis,
            "torso_yaw": tasks.torso_yaw,
            "left_foot": tasks.left_foot,
            "right_foot": tasks.right_foot,
            "left_arm": tasks.left_arm,
            "right_arm": tasks.right_arm,
            "joint_hints": dict(tasks.joint_hints),
        }

    def solve(
        self,
        tasks: HumanoidTaskSpec,
        *,
        qpos: Sequence[float],
        qvel: Optional[Sequence[float]] = None,
        dt: float = 0.0,
    ) -> JointTargets:
        dt = float(max(dt, 1e-6))
        q_full = np.asarray(qpos, dtype=np.float64).reshape(-1)
        qvel_full = np.zeros((self.model.nv,), dtype=np.float64) if qvel is None else np.asarray(qvel, dtype=np.float64).reshape(-1)

        self.data.qpos[:] = q_full
        self.data.qvel[:] = qvel_full
        self.mujoco.mj_forward(self.model, self.data)

        q_act = self.model_spec.actuated_qpos_from_full(q_full)
        qd_act = self._actuated_qvel_from_full(qvel_full)
        q_hint = self.model_spec.joint_dict_to_vector(tasks.joint_hints, base=self.model_spec.stand_ctrl)
        lower_body_target, lower_body_meta = self._build_lower_body_posture_target(tasks, q_act)

        M = self._mass_matrix()
        bias = np.asarray(self.data.qfrc_bias, dtype=np.float64).copy()
        ddq_full, lambda_ref, task_meta = self._solve_inverse_dynamics(
            tasks,
            q_full,
            qvel_full,
            q_act,
            qd_act,
            q_hint,
            lower_body_target,
            M,
            bias,
        )

        tau_ff = self._recover_actuated_torques(M, bias, ddq_full, task_meta["support_jacobian"], lambda_ref)
        torque_limit = self.cfg.torque_limit_scale * self.model_spec.torque_limit_vector()
        torque_bound_violation = np.maximum(np.abs(tau_ff) - torque_limit, 0.0)
        torque_bound_violation_max = float(np.max(torque_bound_violation)) if tau_ff.size else 0.0
        if torque_bound_violation_max > 1e-6:
            tau_clipped = np.clip(tau_ff, -torque_limit, torque_limit)
            torque_saturation = float(np.max(np.abs(tau_ff - tau_clipped))) if tau_ff.size else 0.0
            tau_ff = tau_clipped
        else:
            torque_saturation = 0.0

        ddq_act = ddq_full[self.model_spec.actuated_dof_indices]
        qd_ref = np.clip(qd_act + ddq_act * dt, -float(self.cfg.max_qd_ref), float(self.cfg.max_qd_ref))
        q_ref = q_act + qd_ref * dt
        q_ref = np.clip(q_ref - q_act, -float(self.cfg.max_q_step), float(self.cfg.max_q_step)) + q_act
        q_ref = self.model_spec.clip_to_joint_limits(q_ref)

        metadata = {
            "solver": self.solver_name,
            "task_errors": task_meta["task_errors"],
            "contact_feet": task_meta["contact_feet"],
            "contact_loads": task_meta["contact_loads"],
            "torque_saturation_max": torque_saturation,
            "torque_bound_violation_max": torque_bound_violation_max,
            "lower_body_target": lower_body_meta,
            "support_jacobian_rows": int(task_meta["support_jacobian"].shape[0]),
            "support_acc_cmd": task_meta["support_acc_cmd"],
            "num_ineq_constraints": int(task_meta["num_ineq_constraints"]),
            "eq_residual_norm": float(task_meta.get("eq_residual_norm", 0.0)),
            "ineq_violation_max": float(task_meta.get("ineq_violation_max", 0.0)),
            "qp_method": str(task_meta.get("qp_method", "unknown")),
            "note": "Inverse-dynamics QP over ddq/lambda with hard floating-base dynamics, support-contact equalities, friction pyramids, and actuator torque bounds.",
        }
        return JointTargets(
            q_ref=q_ref,
            qd_ref=qd_ref,
            ddq_ref=ddq_full,
            tau_ff=tau_ff,
            lambda_ref=lambda_ref,
            metadata=metadata,
        )

    def _solve_inverse_dynamics(
        self,
        tasks: HumanoidTaskSpec,
        q_full: np.ndarray,
        qvel_full: np.ndarray,
        q_act: np.ndarray,
        qd_act: np.ndarray,
        q_hint: np.ndarray,
        lower_body_target: np.ndarray,
        M: np.ndarray,
        bias: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        support_blocks = self._support_contact_blocks(tasks, qvel_full)
        num_contacts = len(support_blocks)
        n_lambda = 3 * num_contacts
        nv = self.model.nv

        dyn_C = np.zeros((6, nv + n_lambda), dtype=np.float64)
        dyn_C[:, :nv] = M[:6, :]
        for i, block in enumerate(support_blocks):
            lam_slice = slice(nv + 3 * i, nv + 3 * (i + 1))
            dyn_C[:, lam_slice] = -block["J"][:3, :6].T
        dyn_d = -bias[:6]

        if support_blocks:
            contact_C = np.zeros((3 * num_contacts, nv + n_lambda), dtype=np.float64)
            contact_d = np.zeros((3 * num_contacts,), dtype=np.float64)
            for i, block in enumerate(support_blocks):
                row = slice(3 * i, 3 * (i + 1))
                contact_C[row, :nv] = block["J"]
                contact_d[row] = block["a_des"]
            C_eq = np.vstack([dyn_C, contact_C])
            d_eq = np.concatenate([dyn_d, contact_d], axis=0)
        else:
            C_eq = dyn_C
            d_eq = dyn_d

        A_obj, b_obj, task_errors = self._build_objective_system(
            tasks,
            q_full,
            qvel_full,
            q_act,
            qd_act,
            q_hint,
            lower_body_target,
            support_blocks,
        )
        G_ineq, h_ineq = self._build_inequality_constraints(M, bias, support_blocks, nv, n_lambda)
        x, qp_meta = self._solve_constrained_qp(
            C_eq,
            d_eq,
            A_obj,
            b_obj,
            G_ineq,
            h_ineq,
            prefer_trust_constr=(len(support_blocks) == 1),
        )

        ddq_full = np.asarray(x[:nv], dtype=np.float64).copy()
        lambda_ref = np.asarray(x[nv:], dtype=np.float64).copy() if n_lambda > 0 else np.zeros((0,), dtype=np.float64)

        return ddq_full, lambda_ref, {
            "task_errors": task_errors,
            "contact_feet": [block["name"] for block in support_blocks],
            "contact_loads": self._lambda_dict([block["name"] for block in support_blocks], lambda_ref),
            "support_jacobian": np.vstack([block["J"] for block in support_blocks]) if support_blocks else np.zeros((0, nv), dtype=np.float64),
            "support_acc_cmd": np.concatenate([block["a_des"] for block in support_blocks], axis=0) if support_blocks else np.zeros((0,), dtype=np.float64),
            "num_ineq_constraints": int(G_ineq.shape[0]),
            "eq_residual_norm": float(np.linalg.norm(C_eq @ x - d_eq)) if C_eq.size else 0.0,
            "ineq_violation_max": float(np.max(np.maximum(G_ineq @ x - h_ineq, 0.0))) if G_ineq.size else 0.0,
            "qp_method": str(qp_meta.get("method", "unknown")),
        }

    def _build_objective_system(
        self,
        tasks: HumanoidTaskSpec,
        q_full: np.ndarray,
        qvel_full: np.ndarray,
        q_act: np.ndarray,
        qd_act: np.ndarray,
        q_hint: np.ndarray,
        lower_body_target: np.ndarray,
        support_blocks: Sequence[Dict[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        rows: List[np.ndarray] = []
        rhs: List[np.ndarray] = []
        errors: Dict[str, float] = {}
        nv = self.model.nv
        n_lambda = 3 * len(support_blocks)
        narrowness = float(np.clip(tasks.extras.get("narrowness", 0.0), 0.0, 1.0))
        weight_scales = self._through_gap_weight_scales(tasks)

        def add_task(weight: float, J: np.ndarray, target: np.ndarray, name: str) -> None:
            if J.size == 0:
                return
            row = np.zeros((J.shape[0], nv + n_lambda), dtype=np.float64)
            row[:, :nv] = np.sqrt(max(weight, 1e-8)) * J
            rows.append(row)
            rhs.append(np.sqrt(max(weight, 1e-8)) * target)
            errors[name] = float(np.linalg.norm(target))

        J_com, a_com = self._com_task(tasks, qvel_full)
        support_weight_scales = self._support_weight_scales(len(support_blocks))

        add_task(
            self.cfg.com_task_weight * weight_scales["com"] * support_weight_scales["com"],
            J_com,
            a_com,
            "com",
        )

        J_pelvis, a_pelvis = self._orientation_task(
            site_name="imu_in_pelvis",
            target_rpy=(tasks.pelvis.roll_world, tasks.pelvis.pitch_world, tasks.pelvis.yaw_world),
            target_angular_velocity=np.asarray(tasks.pelvis.angular_velocity_world, dtype=np.float64),
            kp=float(self.cfg.pelvis_orientation_kp),
            kd=float(self.cfg.pelvis_orientation_kd),
            qvel_full=qvel_full,
        )
        add_task(
            self.cfg.pelvis_task_weight * weight_scales["pelvis"] * support_weight_scales["pelvis"],
            J_pelvis,
            a_pelvis,
            "pelvis_orientation",
        )

        torso_yaw_world = float(tasks.pelvis.yaw_world + tasks.torso_yaw)
        J_torso, a_torso = self._orientation_task(
            site_name="imu_in_torso",
            target_rpy=(tasks.pelvis.roll_world, tasks.pelvis.pitch_world, torso_yaw_world),
            target_angular_velocity=np.asarray([0.0, 0.0, tasks.plan_frame.omega + tasks.plan_frame.psi_dot_torso], dtype=np.float64),
            kp=float(self.cfg.torso_orientation_kp),
            kd=float(self.cfg.torso_orientation_kd),
            qvel_full=qvel_full,
        )
        add_task(
            self.cfg.torso_task_weight * weight_scales["torso"] * support_weight_scales["torso"],
            J_torso,
            a_torso,
            "torso_orientation",
        )

        for foot_name, foot_task in (("left_foot", tasks.left_foot), ("right_foot", tasks.right_foot)):
            if foot_task.in_contact:
                continue
            J_lin, a_lin = self._site_linear_task(
                site_name=foot_name,
                target_position=np.asarray(foot_task.position_world, dtype=np.float64),
                target_velocity=np.asarray(foot_task.velocity_world, dtype=np.float64),
                kp=float(self.cfg.swing_foot_position_kp),
                kd=float(self.cfg.swing_foot_position_kd),
                qvel_full=qvel_full,
            )
            add_task(
                self.cfg.swing_foot_task_weight * foot_task.weight * weight_scales["swing"] * support_weight_scales["swing"],
                J_lin,
                a_lin,
                f"{foot_name}_swing",
            )

            J_rot, a_rot = self._orientation_task(
                site_name=foot_name,
                target_rpy=(foot_task.roll_world, foot_task.pitch_world, foot_task.yaw_world),
                target_angular_velocity=np.asarray(foot_task.angular_velocity_world, dtype=np.float64),
                kp=float(self.cfg.swing_foot_orientation_kp),
                kd=float(self.cfg.swing_foot_orientation_kd),
                qvel_full=qvel_full,
            )
            add_task(
                self.cfg.swing_foot_orientation_weight * foot_task.weight * weight_scales["swing"] * support_weight_scales["swing"],
                J_rot,
                a_rot,
                f"{foot_name}_orientation",
            )

        add_task(
            self.cfg.lower_body_posture_weight * weight_scales["lower_body"] * support_weight_scales["lower_body"],
            self._joint_selector(tuple(self.model_spec.left_leg_joints) + tuple(self.model_spec.right_leg_joints), nv),
            self._joint_accel_command(
                lower_body_target,
                q_act,
                qd_act,
                self.cfg.posture_kp,
                self.cfg.posture_kd,
                joints=self.model_spec.left_leg_joints + self.model_spec.right_leg_joints,
            ),
            "lower_body_posture",
        )
        add_task(
            self.cfg.waist_posture_weight * weight_scales["waist"] * support_weight_scales["waist"],
            self._joint_selector(tuple(self.model_spec.waist_joints), nv),
            self._joint_accel_command(q_hint, q_act, qd_act, self.cfg.waist_kp, self.cfg.waist_kd, joints=self.model_spec.waist_joints),
            "waist_posture",
        )
        add_task(
            self.cfg.arm_posture_weight * weight_scales["arm"] * support_weight_scales["arm"],
            self._joint_selector(tuple(self.model_spec.left_arm_joints) + tuple(self.model_spec.right_arm_joints), nv),
            self._joint_accel_command(
                q_hint,
                q_act,
                qd_act,
                self.cfg.arm_kp,
                self.cfg.arm_kd,
                joints=self.model_spec.left_arm_joints + self.model_spec.right_arm_joints,
            ),
            "arm_posture",
        )
        add_task(
            self.cfg.posture_weight,
            self._joint_selector(self.model_spec.actuated_joints, nv),
            self._joint_accel_command(self.model_spec.stand_ctrl, q_act, qd_act, self.cfg.posture_kp, self.cfg.posture_kd),
            "posture",
        )

        row_ddq = np.zeros((nv, nv + n_lambda), dtype=np.float64)
        row_ddq[:, :nv] = np.sqrt(float(self.cfg.ddq_weight)) * np.eye(nv, dtype=np.float64)
        rows.append(row_ddq)
        rhs.append(np.zeros((nv,), dtype=np.float64))

        if n_lambda > 0:
            row_lambda = np.zeros((n_lambda, nv + n_lambda), dtype=np.float64)
            row_lambda[:, nv:] = np.sqrt(float(self.cfg.lambda_weight)) * np.eye(n_lambda, dtype=np.float64)
            rows.append(row_lambda)
            rhs.append(np.zeros((n_lambda,), dtype=np.float64))

            lambda_target = np.zeros((n_lambda,), dtype=np.float64)
            for i in range(len(support_blocks)):
                lambda_target[3 * i + 2] = float(self.cfg.lambda_normal_target)
            row_lambda_target = np.zeros((n_lambda, nv + n_lambda), dtype=np.float64)
            row_lambda_target[:, nv:] = np.diag(
                [self.cfg.lambda_tangent_weight, self.cfg.lambda_tangent_weight, self.cfg.lambda_normal_weight] * len(support_blocks)
            )
            rows.append(row_lambda_target)
            rhs.append(row_lambda_target[:, nv:] @ lambda_target)

        A_obj = np.vstack(rows) if rows else np.zeros((0, nv + n_lambda), dtype=np.float64)
        b_obj = np.concatenate(rhs, axis=0) if rhs else np.zeros((0,), dtype=np.float64)
        errors["narrowness"] = narrowness
        errors["gap_severity"] = float(np.clip(tasks.extras.get("gap_severity", narrowness), 0.0, 1.0))
        return A_obj, b_obj, errors

    def _support_contact_blocks(
        self,
        tasks: HumanoidTaskSpec,
        qvel_full: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        blocks: List[Dict[str, np.ndarray]] = []
        for site_name, foot_task in (("left_foot", tasks.left_foot), ("right_foot", tasks.right_foot)):
            if not foot_task.in_contact:
                continue
            J, pos_err, vel = self._site_linear_kinematics(site_name, foot_task.position_world, qvel_full)
            if J is None or pos_err is None or vel is None:
                continue
            pos_err = np.clip(
                np.asarray(pos_err, dtype=np.float64),
                -float(self.cfg.contact_position_error_clip),
                float(self.cfg.contact_position_error_clip),
            )
            a_des = float(self.cfg.contact_position_kp) * pos_err + float(self.cfg.contact_position_kd) * (-vel)
            a_des = np.clip(a_des, -float(self.cfg.contact_accel_limit), float(self.cfg.contact_accel_limit))
            blocks.append({"name": site_name, "J": J, "a_des": a_des})
        return blocks

    def _mass_matrix(self) -> np.ndarray:
        M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        self.mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def _recover_actuated_torques(
        self,
        M: np.ndarray,
        bias: np.ndarray,
        ddq_full: np.ndarray,
        support_jacobian: np.ndarray,
        lambda_ref: np.ndarray,
    ) -> np.ndarray:
        tau_full = M @ ddq_full + bias
        if lambda_ref.size > 0 and support_jacobian.size > 0:
            tau_full -= support_jacobian.T @ lambda_ref
        return np.asarray(tau_full[self.model_spec.actuated_dof_indices], dtype=np.float64).copy()

    def _com_task(self, tasks: HumanoidTaskSpec, qvel_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pelvis_body_id = self.model_spec.body_id.get("pelvis")
        if pelvis_body_id is None:
            return np.zeros((0, self.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSubtreeCom(self.model, self.data, jacp, pelvis_body_id)
        current = np.asarray(self.data.subtree_com[pelvis_body_id], dtype=np.float64).copy()
        vel = jacp @ qvel_full
        desired = np.asarray(tasks.pelvis.position_world, dtype=np.float64).copy()
        target_vel = np.asarray(tasks.pelvis.linear_velocity_world, dtype=np.float64).copy()
        target_vel[2] = float(tasks.plan_frame.h_dot)
        a_des = float(self.cfg.com_kp) * (desired - current) + float(self.cfg.com_kd) * (target_vel - vel)
        return jacp, a_des

    def _site_linear_task(
        self,
        site_name: str,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        kp: float,
        kd: float,
        qvel_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        J, pos_err, vel = self._site_linear_kinematics(site_name, target_position, qvel_full)
        if J is None or pos_err is None or vel is None:
            return np.zeros((0, self.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        a_des = float(kp) * pos_err + float(kd) * (np.asarray(target_velocity, dtype=np.float64) - vel)
        return J, a_des

    def _site_linear_kinematics(
        self,
        site_name: str,
        target_position: Sequence[float],
        qvel_full: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        sid = self.model_spec.site_id.get(site_name)
        if sid is None:
            return None, None, None
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
        current = np.asarray(self.data.site_xpos[sid], dtype=np.float64).copy()
        vel = jacp @ qvel_full
        err = np.asarray(target_position, dtype=np.float64).reshape(3) - current
        return jacp, err, vel

    def _orientation_task(
        self,
        site_name: str,
        target_rpy: Tuple[float, float, float],
        target_angular_velocity: np.ndarray,
        kp: float,
        kd: float,
        qvel_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sid = self.model_spec.site_id.get(site_name)
        if sid is None:
            return np.zeros((0, self.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, None, jacr, sid)
        current = np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        target = self._rotation_matrix_from_rpy(*target_rpy)
        ang_vel = jacr @ qvel_full
        err = 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )
        a_des = float(kp) * err + float(kd) * (np.asarray(target_angular_velocity, dtype=np.float64) - ang_vel)
        return jacr, a_des

    def _joint_accel_command(
        self,
        q_target: np.ndarray,
        q_act: np.ndarray,
        qd_act: np.ndarray,
        kp: float,
        kd: float,
        joints: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        out = np.zeros((len(joints) if joints is not None else self.model_spec.num_actuated,), dtype=np.float64)
        names = self.model_spec.actuated_joints if joints is None else tuple(joints)
        for i, name in enumerate(names):
            j = self.model_spec.actuated_joints.index(name)
            out[i] = float(kp) * (float(q_target[j]) - float(q_act[j])) + float(kd) * (0.0 - float(qd_act[j]))
        return out

    def _joint_selector(self, joint_names: Sequence[str], nv: int) -> np.ndarray:
        valid = [name for name in joint_names if name in self.model_spec.actuated_joints]
        J = np.zeros((len(valid), nv), dtype=np.float64)
        for row, name in enumerate(valid):
            dof = self.model_spec.joint_dof_index[name]
            J[row, dof] = 1.0
        return J

    def _build_inequality_constraints(
        self,
        M: np.ndarray,
        bias: np.ndarray,
        support_blocks: Sequence[Dict[str, np.ndarray]],
        nv: int,
        n_lambda: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        total_dim = nv + n_lambda

        if n_lambda > 0:
            mu = float(self.cfg.friction_coeff)
            for i in range(len(support_blocks)):
                lam = slice(nv + 3 * i, nv + 3 * (i + 1))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 0] = 1.0
                row[lam.start + 2] = -mu
                rows.append(row)
                rhs.append(0.0)

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 0] = -1.0
                row[lam.start + 2] = -mu
                rows.append(row)
                rhs.append(0.0)

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 1] = 1.0
                row[lam.start + 2] = -mu
                rows.append(row)
                rhs.append(0.0)

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 1] = -1.0
                row[lam.start + 2] = -mu
                rows.append(row)
                rhs.append(0.0)

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 2] = -1.0
                rows.append(row)
                rhs.append(-float(self.cfg.lambda_min_normal))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 2] = 1.0
                rows.append(row)
                rhs.append(float(self.cfg.lambda_max_normal))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 0] = 1.0
                rows.append(row)
                rhs.append(float(self.cfg.lambda_max_tangent))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 0] = -1.0
                rows.append(row)
                rhs.append(float(self.cfg.lambda_max_tangent))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 1] = 1.0
                rows.append(row)
                rhs.append(float(self.cfg.lambda_max_tangent))

                row = np.zeros((total_dim,), dtype=np.float64)
                row[lam.start + 1] = -1.0
                rows.append(row)
                rhs.append(float(self.cfg.lambda_max_tangent))

        act_idx = self.model_spec.actuated_dof_indices
        tau_map = np.zeros((self.model_spec.num_actuated, total_dim), dtype=np.float64)
        tau_map[:, :nv] = M[act_idx, :]
        if n_lambda > 0 and support_blocks:
            support_jacobian = np.vstack([block["J"] for block in support_blocks])
            tau_map[:, nv:] = -support_jacobian[:, act_idx].T
        tau_bias = np.asarray(bias[act_idx], dtype=np.float64)
        torque_limit = self.cfg.torque_limit_scale * self.model_spec.torque_limit_vector()
        finite_mask = np.isfinite(torque_limit)
        for i, finite in enumerate(finite_mask):
            if not finite:
                continue
            rows.append(tau_map[i])
            rhs.append(float(torque_limit[i] - tau_bias[i]))
            rows.append(-tau_map[i])
            rhs.append(float(torque_limit[i] + tau_bias[i]))

        for i in range(nv):
            bound = float(self.cfg.max_base_accel if i < 6 else self.cfg.max_joint_accel)
            row = np.zeros((total_dim,), dtype=np.float64)
            row[i] = 1.0
            rows.append(row)
            rhs.append(bound)
            row = np.zeros((total_dim,), dtype=np.float64)
            row[i] = -1.0
            rows.append(row)
            rhs.append(bound)

        if not rows:
            return np.zeros((0, total_dim), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        return np.vstack(rows), np.asarray(rhs, dtype=np.float64)

    def _solve_constrained_qp(
        self,
        C: np.ndarray,
        d: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
        *,
        prefer_trust_constr: bool = False,
    ) -> tuple[np.ndarray, Dict[str, object]]:
        total_dim = A.shape[1] if A.size > 0 else (C.shape[1] if C.size > 0 else G.shape[1])
        H = A.T @ A + float(self.cfg.qp_regularization) * np.eye(total_dim, dtype=np.float64)
        f = -(A.T @ b)
        accept_tol = float(self.cfg.constraint_tol)
        best_candidate: Optional[tuple[np.ndarray, Dict[str, object]]] = None
        best_score: Optional[tuple[float, float]] = None

        def register_candidate(
            x_candidate: np.ndarray,
            method: str,
            *,
            eq_residual: Optional[float] = None,
            ineq_violation: Optional[float] = None,
        ) -> Optional[tuple[np.ndarray, Dict[str, object]]]:
            nonlocal best_candidate, best_score
            x_candidate = np.asarray(x_candidate, dtype=np.float64)
            if not np.all(np.isfinite(x_candidate)):
                return None
            eq_residual = float(np.linalg.norm(C @ x_candidate - d)) if eq_residual is None and C.size else float(eq_residual or 0.0)
            ineq_violation = float(np.max(np.maximum(G @ x_candidate - h, 0.0))) if ineq_violation is None and G.size else float(ineq_violation or 0.0)
            score = (ineq_violation, eq_residual)
            if best_candidate is None or best_score is None or score < best_score:
                best_candidate = (
                    x_candidate.copy(),
                    {
                        "method": method,
                        "eq_residual_norm": eq_residual,
                        "ineq_violation_max": ineq_violation,
                    },
                )
                best_score = score
            if eq_residual <= accept_tol and ineq_violation <= accept_tol:
                return (
                    x_candidate,
                    {
                        "method": method,
                        "eq_residual_norm": eq_residual,
                        "ineq_violation_max": ineq_violation,
                    },
                )
            return None

        full_osqp = self._solve_full_space_osqp(H, f, C, d, G, h)
        if full_osqp is not None:
            x_osqp, meta = full_osqp
            candidate = register_candidate(
                x_osqp,
                str(meta.get("method", "osqp")),
                eq_residual=float(meta.get("eq_residual_norm", 0.0)),
                ineq_violation=float(meta.get("ineq_violation_max", 0.0)),
            )
            if candidate is not None:
                return candidate

        if C.size == 0:
            x0 = np.zeros((total_dim,), dtype=np.float64)
            N = np.eye(total_dim, dtype=np.float64)
        else:
            CCt = C @ C.T
            x0 = C.T @ self._safe_solve(CCt + float(self.cfg.qp_regularization) * np.eye(CCt.shape[0], dtype=np.float64), d)
            _, S, Vt = np.linalg.svd(C, full_matrices=True)
            rank = int(np.sum(S > 1e-8))
            N = Vt[rank:].T
            if N.size == 0:
                candidate = register_candidate(x0, "equality_only")
                if candidate is not None:
                    return candidate
                if best_candidate is not None:
                    return best_candidate
                return x0, {"method": "equality_only"}
        Hr = N.T @ H @ N
        fr = N.T @ (H @ x0 + f)

        if G.size > 0:
            Gr = G @ N
            hr = h - G @ x0
        else:
            Gr = np.zeros((0, N.shape[1]), dtype=np.float64)
            hr = np.zeros((0,), dtype=np.float64)

        if Hr.size == 0:
            candidate = register_candidate(x0, "degenerate")
            if candidate is not None:
                return candidate
            if best_candidate is not None:
                return best_candidate
            return x0, {"method": "degenerate"}

        if Gr.size == 0:
            y = -self._safe_solve(Hr, fr)
            x_candidate = x0 + N @ y
            candidate = register_candidate(x_candidate, "reduced_ls")
            if candidate is not None:
                return candidate
            if best_candidate is not None:
                return best_candidate
            return x_candidate, {"method": "reduced_ls"}

        if bool(self.cfg.use_osqp):
            try:
                import osqp
                from scipy import sparse

                P = 0.5 * (Hr + Hr.T)
                P = sparse.csc_matrix(P)
                q = np.asarray(fr, dtype=np.float64)
                Aineq = sparse.csc_matrix(Gr)
                l = np.full(hr.shape, -np.inf, dtype=np.float64)
                u = np.asarray(hr, dtype=np.float64)

                solver = osqp.OSQP()
                solver.setup(
                    P=P,
                    q=q,
                    A=Aineq,
                    l=l,
                    u=u,
                    verbose=bool(self.cfg.osqp_verbose),
                    polish=bool(self.cfg.osqp_polish),
                    max_iter=int(self.cfg.osqp_maxiter),
                    eps_abs=float(self.cfg.constraint_tol),
                    eps_rel=float(self.cfg.constraint_tol),
                )
                result = solver.solve()
                status = str(getattr(result.info, "status", "")).lower()
                if result.x is not None and ("solved" in status):
                    y = np.asarray(result.x, dtype=np.float64)
                    candidate = register_candidate(
                        x0 + N @ y,
                        "osqp_reduced",
                    )
                    if candidate is not None:
                        return candidate
            except Exception:
                pass

        def _trust_constr_attempt(maxiter: int) -> Optional[np.ndarray]:
            try:
                from scipy.optimize import LinearConstraint, minimize

                objective = lambda y: 0.5 * float(y @ (Hr @ y)) + float(fr @ y)
                gradient = lambda y: Hr @ y + fr
                constraints = [LinearConstraint(Gr, -np.inf * np.ones_like(hr), hr)]
                y0 = np.zeros((N.shape[1],), dtype=np.float64)
                result = minimize(
                    objective,
                    y0,
                    jac=gradient,
                    method="trust-constr",
                    constraints=constraints,
                    options={"maxiter": int(maxiter), "verbose": 0},
                )
                if result.success and np.all(np.isfinite(result.x)):
                    y = np.asarray(result.x, dtype=np.float64)
                    x_candidate = x0 + N @ y
                    candidate = register_candidate(x_candidate, "trust_constr_candidate")
                    if candidate is not None:
                        return y
                    return y
            except Exception:
                return None
            return None

        if bool(self.cfg.use_slsqp):
            try:
                from scipy.optimize import minimize

                objective = lambda y: 0.5 * float(y @ (Hr @ y)) + float(fr @ y)
                gradient = lambda y: Hr @ y + fr
                constraints = [
                    {
                        "type": "ineq",
                        "fun": lambda y, G=Gr, h=hr: h - G @ y,
                        "jac": lambda y, G=Gr, h=hr: -G,
                    }
                ]
                y0 = np.zeros((N.shape[1],), dtype=np.float64)
                result = minimize(
                    objective,
                    y0,
                    jac=gradient,
                    method="SLSQP",
                    constraints=constraints,
                    options={"maxiter": int(self.cfg.slsqp_maxiter), "ftol": 1e-8, "disp": False},
                )
                if result.success and np.all(np.isfinite(result.x)):
                    y = np.asarray(result.x, dtype=np.float64)
                    candidate = register_candidate(x0 + N @ y, "slsqp")
                    if candidate is not None:
                        return candidate
            except Exception:
                pass

        if prefer_trust_constr:
            trust_y = _trust_constr_attempt(int(self.cfg.single_support_trust_constr_maxiter))
            if trust_y is not None:
                candidate = register_candidate(x0 + N @ trust_y, "trust_constr_single_support")
                if candidate is not None:
                    return candidate

        if bool(self.cfg.use_trust_constr):
            trust_y = _trust_constr_attempt(int(self.cfg.trust_constr_maxiter))
            if trust_y is not None:
                candidate = register_candidate(x0 + N @ trust_y, "trust_constr")
                if candidate is not None:
                    return candidate

        y = -self._safe_solve(Hr, fr)
        if Gr.size > 0:
            violation = Gr @ y - hr
            if np.any(violation > 1e-6):
                for _ in range(int(self.cfg.active_set_refine_iters)):
                    active = violation > 1e-6
                    if not np.any(active):
                        break
                    G_active = Gr[active]
                    h_active = hr[active]
                    KKT = np.block(
                        [
                            [Hr, G_active.T],
                            [G_active, np.zeros((G_active.shape[0], G_active.shape[0]), dtype=np.float64)],
                        ]
                    )
                    rhs = np.concatenate([-fr, h_active], axis=0)
                    sol = self._safe_solve(KKT, rhs)
                    y = sol[: Hr.shape[0]]
                    violation = Gr @ y - hr
        if Gr.size > 0 and np.max(Gr @ y - hr) > float(self.cfg.constraint_tol):
            y_proj = y.copy()
            for _ in range(int(self.cfg.projection_repair_iters)):
                max_violation = 0.0
                for i in range(Gr.shape[0]):
                    gi = Gr[i]
                    viol = float(gi @ y_proj - hr[i])
                    if viol <= float(self.cfg.constraint_tol):
                        continue
                    denom = float(gi @ gi) + float(self.cfg.qp_regularization)
                    if denom <= 0.0:
                        continue
                    y_proj = y_proj - (viol / denom) * gi
                    max_violation = max(max_violation, viol)
                if max_violation <= float(self.cfg.constraint_tol):
                    y = y_proj
                    break
            if np.max(Gr @ y_proj - hr) <= float(self.cfg.constraint_tol):
                candidate = register_candidate(x0 + N @ y_proj, "projection_repair")
                if candidate is not None:
                    return candidate
            register_candidate(x0 + N @ y_proj, "projection_repair_candidate")
        if (
            bool(self.cfg.use_trust_constr_repair)
            and np.max(Gr @ y - hr) > float(self.cfg.trust_constr_repair_violation_threshold)
        ):
            trust_y = _trust_constr_attempt(int(self.cfg.trust_constr_repair_maxiter))
            if trust_y is not None:
                candidate = register_candidate(x0 + N @ trust_y, "trust_constr_repair")
                if candidate is not None:
                    return candidate
        register_candidate(x0 + N @ y, "active_set")
        if best_candidate is not None:
            return best_candidate
        return x0 + N @ y, {"method": "active_set"}

    def _solve_full_space_osqp(
        self,
        H: np.ndarray,
        f: np.ndarray,
        C: np.ndarray,
        d: np.ndarray,
        G: np.ndarray,
        h: np.ndarray,
    ) -> Optional[tuple[np.ndarray, Dict[str, object]]]:
        if not bool(self.cfg.use_osqp):
            return None
        try:
            import osqp
            from scipy import sparse

            P = sparse.csc_matrix(0.5 * (H + H.T))
            q = np.asarray(f, dtype=np.float64)
            blocks = []
            lower = []
            upper = []
            if C.size > 0:
                blocks.append(sparse.csc_matrix(C))
                lower.append(np.asarray(d, dtype=np.float64))
                upper.append(np.asarray(d, dtype=np.float64))
            if G.size > 0:
                blocks.append(sparse.csc_matrix(G))
                lower.append(np.full(h.shape, -np.inf, dtype=np.float64))
                upper.append(np.asarray(h, dtype=np.float64))
            A = sparse.vstack(blocks, format="csc") if blocks else sparse.csc_matrix((0, H.shape[0]))
            l = np.concatenate(lower, axis=0) if lower else np.zeros((0,), dtype=np.float64)
            u = np.concatenate(upper, axis=0) if upper else np.zeros((0,), dtype=np.float64)

            solver = osqp.OSQP()
            solver.setup(
                P=P,
                q=q,
                A=A,
                l=l,
                u=u,
                verbose=bool(self.cfg.osqp_verbose),
                polish=bool(self.cfg.osqp_polish),
                max_iter=int(self.cfg.osqp_maxiter),
                eps_abs=float(self.cfg.constraint_tol),
                eps_rel=float(self.cfg.constraint_tol),
            )
            if self._last_osqp_solution is not None and self._last_osqp_solution.shape == q.shape:
                solver.warm_start(x=self._last_osqp_solution)
            result = solver.solve()
            status = str(getattr(result.info, "status", "")).lower()
            if result.x is None or "solved" not in status:
                return None

            x = np.asarray(result.x, dtype=np.float64)
            eq_residual = float(np.linalg.norm(C @ x - d)) if C.size else 0.0
            ineq_violation = float(np.max(np.maximum(G @ x - h, 0.0))) if G.size else 0.0
            self._last_osqp_solution = x.copy()
            return x, {
                "method": "osqp",
                "eq_residual_norm": eq_residual,
                "ineq_violation_max": ineq_violation,
            }
        except Exception:
            return None
        return None

    def _through_gap_weight_scales(self, tasks: HumanoidTaskSpec) -> Dict[str, float]:
        narrowness = float(
            np.clip(
                tasks.extras.get("gap_severity", tasks.extras.get("narrowness", 0.0)),
                0.0,
                1.0,
            )
        )
        return {
            "com": 1.0 + (float(self.cfg.through_gap_com_weight_scale) - 1.0) * narrowness,
            "pelvis": 1.0 + (float(self.cfg.through_gap_pelvis_weight_scale) - 1.0) * narrowness,
            "torso": 1.0 + (float(self.cfg.through_gap_torso_weight_scale) - 1.0) * narrowness,
            "swing": 1.0 + (float(self.cfg.through_gap_swing_weight_scale) - 1.0) * narrowness,
            "arm": 1.0 + (float(self.cfg.through_gap_arm_weight_scale) - 1.0) * narrowness,
            "waist": 1.0 + (float(self.cfg.through_gap_waist_weight_scale) - 1.0) * narrowness,
            "lower_body": 1.0 + (float(self.cfg.through_gap_lower_body_weight_scale) - 1.0) * narrowness,
        }

    def _support_weight_scales(self, num_contacts: int) -> Dict[str, float]:
        if int(num_contacts) >= 2:
            return {
                "com": 1.0,
                "pelvis": 1.0,
                "torso": 1.0,
                "swing": 1.0,
                "arm": 1.0,
                "waist": 1.0,
                "lower_body": 1.0,
            }
        return {
            "com": float(self.cfg.single_support_com_weight_scale),
            "pelvis": float(self.cfg.single_support_pelvis_weight_scale),
            "torso": float(self.cfg.single_support_torso_weight_scale),
            "swing": float(self.cfg.single_support_swing_weight_scale),
            "arm": float(self.cfg.single_support_arm_weight_scale),
            "waist": float(self.cfg.single_support_waist_weight_scale),
            "lower_body": float(self.cfg.single_support_lower_body_weight_scale),
        }

    @staticmethod
    def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x

    def _build_lower_body_posture_target(
        self,
        tasks: HumanoidTaskSpec,
        q_current: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        q_target = np.asarray(q_current, dtype=np.float64).copy()
        body_height = float(tasks.pelvis.position_world[2])
        crouch_ratio = self._crouch_ratio(body_height)
        support_points = []
        if bool(tasks.left_foot.in_contact):
            support_points.append(np.asarray(tasks.left_foot.position_world[:2], dtype=np.float64))
        if bool(tasks.right_foot.in_contact):
            support_points.append(np.asarray(tasks.right_foot.position_world[:2], dtype=np.float64))
        if not support_points:
            support_points = [
                np.asarray(tasks.left_foot.position_world[:2], dtype=np.float64),
                np.asarray(tasks.right_foot.position_world[:2], dtype=np.float64),
            ]
        support_mid = np.mean(np.stack(support_points, axis=0), axis=0)
        yaw = float(tasks.pelvis.yaw_world)
        rot = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        pelvis_xy = np.asarray(tasks.pelvis.position_world[:2], dtype=np.float64)
        pelvis_offset_local = rot.T @ (pelvis_xy - support_mid)
        forward = float(pelvis_offset_local[0])
        lateral = float(pelvis_offset_local[1])

        leg_targets: Dict[str, float] = {}
        for prefix, roll_sign in (("left", 1.0), ("right", -1.0)):
            leg_targets[f"{prefix}_hip_pitch_joint"] = (
                self._get_joint_value(self.model_spec.stand_ctrl, f"{prefix}_hip_pitch_joint")
                + float(self.cfg.crouch_hip_pitch_gain) * crouch_ratio
                + float(self.cfg.pelvis_forward_hip_pitch_gain) * forward
            )
            leg_targets[f"{prefix}_knee_joint"] = (
                self._get_joint_value(self.model_spec.stand_ctrl, f"{prefix}_knee_joint")
                + float(self.cfg.crouch_knee_gain) * crouch_ratio
            )
            leg_targets[f"{prefix}_ankle_pitch_joint"] = (
                self._get_joint_value(self.model_spec.stand_ctrl, f"{prefix}_ankle_pitch_joint")
                + float(self.cfg.crouch_ankle_pitch_gain) * crouch_ratio
            )
            leg_targets[f"{prefix}_hip_roll_joint"] = (
                self._get_joint_value(self.model_spec.stand_ctrl, f"{prefix}_hip_roll_joint")
                - roll_sign * float(self.cfg.pelvis_lateral_hip_roll_gain) * lateral
            )

        q_target = self.model_spec.joint_dict_to_vector(leg_targets, base=q_target)
        q_target = self.model_spec.clip_to_joint_limits(q_target)
        return q_target, {
            "body_height": body_height,
            "crouch_ratio": crouch_ratio,
            "pelvis_offset_forward": forward,
            "pelvis_offset_lateral": lateral,
        }

    def _crouch_ratio(self, body_height: float) -> float:
        denom = max(float(self.cfg.body_height_nominal) - float(self.cfg.body_height_min), 1e-6)
        return float(np.clip((float(self.cfg.body_height_nominal) - body_height) / denom, 0.0, 1.0))

    def _actuated_qvel_from_full(self, qvel_full: np.ndarray) -> np.ndarray:
        idx = self.model_spec.actuated_dof_indices
        if idx.size == 0 or int(np.max(idx)) >= qvel_full.size:
            return np.zeros((self.model_spec.num_actuated,), dtype=np.float64)
        return np.asarray(qvel_full[idx], dtype=np.float64).copy()

    @staticmethod
    def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(float(roll))
        sr = np.sin(float(roll))
        cp = np.cos(float(pitch))
        sp = np.sin(float(pitch))
        cy = np.cos(float(yaw))
        sy = np.sin(float(yaw))
        return np.asarray(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _get_joint_value(self, q_ref: np.ndarray, joint_name: str) -> float:
        idx = self.model_spec.actuated_joints.index(joint_name)
        return float(q_ref[idx])

    @staticmethod
    def _lambda_dict(contact_names: Sequence[str], lambda_ref: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(contact_names):
            lam = np.asarray(lambda_ref[3 * i: 3 * (i + 1)], dtype=np.float64)
            out[name] = {"fx": float(lam[0]), "fy": float(lam[1]), "fz": float(lam[2])}
        return out


# Backward-compatible import name while the pipeline still uses the old symbol.
G1WholeBodySolverSkeleton = G1WholeBodyDynamicWBCSolver
