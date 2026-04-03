"""
Contact-constrained QP/WBC prototype for humanoid corridor following in MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from genedynamics.deploy.followers.common.controller_output import JointTargets
from genedynamics.deploy.followers.humanoid.models.g1_model import G1ModelSpec
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec


@dataclass
class G1WBCTaskStackConfig:
    max_iterations: int = 8
    damping: float = 1e-4
    step_size: float = 0.7
    max_delta_per_step: float = 0.08
    position_tolerance: float = 5e-3
    posture_weight: float = 0.03
    joint_hint_weight: float = 0.30
    lower_body_hint_weight: float = 0.18
    waist_hint_weight: float = 0.40
    arm_hint_weight: float = 0.22
    velocity_damping_weight: float = 0.08
    swing_foot_weight: float = 2.0
    com_task_weight: float = 1.10
    torso_orientation_weight: float = 0.85
    contact_slack_weight: float = 250.0
    contact_slack_regularization: float = 1e-6
    body_height_nominal: float = 0.75
    body_height_min: float = 0.55
    com_height_gain: float = 0.45
    max_com_xy_error: float = 0.08
    max_com_z_error: float = 0.06
    crouch_hip_pitch_gain: float = 0.35
    crouch_knee_gain: float = 0.95
    crouch_ankle_pitch_gain: float = -0.45
    pelvis_forward_hip_pitch_gain: float = 0.20
    pelvis_lateral_hip_roll_gain: float = 0.50
    use_contact_constraints: bool = True
    use_friction_cones: bool = False
    use_torque_limits: bool = False
    pelvis_task_weight: float = 1.0
    swing_foot_task_weight: float = 1.0
    torso_task_weight: float = 0.5
    arm_task_weight: float = 0.2


class G1WholeBodySolverSkeleton:
    """
    Contact-constrained QP/WBC prototype.

    This is still a MuJoCo-native configuration-space controller, not a full
    torque-level inverse-dynamics stack. The important architectural change is
    that stance-foot contact enters the solve as an explicit equality
    constraint with slack, while swing-foot tracking and posture/joint-hint
    tracking remain weighted objectives.
    """

    solver_name = "wbc_qp_contact_com_torso"

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

    def build_task_stack(self, tasks: HumanoidTaskSpec) -> Dict[str, object]:
        return {
            "pelvis": tasks.pelvis,
            "torso_yaw": tasks.torso_yaw,
            "left_foot": tasks.left_foot,
            "right_foot": tasks.right_foot,
            "left_arm": tasks.left_arm,
            "right_arm": tasks.right_arm,
            "joint_hints": dict(tasks.joint_hints),
            "weights": {
                "pelvis": self.cfg.pelvis_task_weight,
                "com": self.cfg.com_task_weight,
                "swing_foot": self.cfg.swing_foot_task_weight,
                "torso": self.cfg.torso_task_weight,
                "torso_orientation": self.cfg.torso_orientation_weight,
                "arms": self.cfg.arm_task_weight,
            },
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
        qd_full = None if qvel is None else np.asarray(qvel, dtype=np.float64).reshape(-1)
        try:
            q_current = self.model_spec.actuated_qpos_from_full(q_full)
        except ValueError:
            q_current = self.model_spec.stand_ctrl.copy()
        qd_current = self._actuated_qvel_from_full(qd_full)

        q_hint = self.model_spec.joint_dict_to_vector(
            tasks.joint_hints,
            base=self.model_spec.stand_ctrl,
        )
        lower_body_target, lower_body_meta = self._build_lower_body_posture_target(tasks, q_current)
        q_ref, meta = self._solve_qp(q_current, q_full, tasks, q_hint, lower_body_target, qd_current, dt)
        meta["lower_body_target"] = lower_body_meta
        qd_ref = (q_ref - q_current) / dt
        return JointTargets(
            q_ref=q_ref,
            qd_ref=qd_ref,
            metadata=meta,
        )

    def _solve_qp(
        self,
        q_current: np.ndarray,
        q_full: np.ndarray,
        tasks: HumanoidTaskSpec,
        q_hint: np.ndarray,
        lower_body_target: np.ndarray,
        qd_current: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, Dict[str, object]]:
        q_work = np.asarray(q_current, dtype=np.float64).copy()
        iterations_used = 0
        final_foot_errors: Dict[str, float] = {}
        final_contact_slack: Dict[str, float] = {}
        final_contact_feet: list[str] = []
        final_com_error = float("nan")
        final_torso_orientation_error = float("nan")

        for it in range(int(self.cfg.max_iterations)):
            full_work = self.model_spec.apply_actuated_qpos(q_full, q_work)
            self.data.qpos[:] = full_work
            self.mujoco.mj_forward(self.model, self.data)

            H, g = self._build_objective(q_work, tasks, q_hint, lower_body_target, qd_current, dt)
            A_eq, b_eq, contact_meta = self._build_contact_constraints(tasks)
            dq, slack = self._solve_constrained_step(H, g, A_eq, b_eq)
            dq = np.clip(dq, -float(self.cfg.max_delta_per_step), float(self.cfg.max_delta_per_step))

            q_next = self.model_spec.clip_to_joint_limits(q_work + float(self.cfg.step_size) * dq)
            iterations_used = it + 1
            q_work = q_next

            final_foot_errors = self._measure_foot_errors(q_full, q_work, tasks)
            final_com_error = self._measure_com_error(q_full, q_work, tasks)
            final_torso_orientation_error = self._measure_torso_orientation_error(q_full, q_work, tasks)
            final_contact_feet = list(contact_meta["contact_feet"])
            final_contact_slack = self._slack_dict(contact_meta["row_slices"], slack)

            if self._max_task_error(
                final_foot_errors,
                final_contact_slack,
                final_com_error,
                final_torso_orientation_error,
            ) <= float(self.cfg.position_tolerance):
                break

        q_work = self.model_spec.clip_to_joint_limits(q_work)
        return q_work, {
            "solver": self.solver_name,
            "weights": {
                "posture": self.cfg.posture_weight,
                "joint_hint": self.cfg.joint_hint_weight,
                "lower_body_hint": self.cfg.lower_body_hint_weight,
                "waist_hint": self.cfg.waist_hint_weight,
                "arm_hint": self.cfg.arm_hint_weight,
                "velocity_damping": self.cfg.velocity_damping_weight,
                "com": self.cfg.com_task_weight,
                "swing_foot": self.cfg.swing_foot_weight,
                "contact_slack": self.cfg.contact_slack_weight,
                "pelvis": self.cfg.pelvis_task_weight,
                "torso": self.cfg.torso_task_weight,
                "torso_orientation": self.cfg.torso_orientation_weight,
                "arms": self.cfg.arm_task_weight,
            },
            "joint_hint_names": sorted(tasks.joint_hints.keys()),
            "iterations": iterations_used,
            "contact_feet": final_contact_feet,
            "foot_position_error": final_foot_errors,
            "com_error": final_com_error,
            "contact_slack_norm": final_contact_slack,
            "torso_orientation_error": final_torso_orientation_error,
            "note": "Support feet enforced as equality constraints with slack; CoM and torso orientation enter the objective alongside swing-foot and posture tasks.",
        }

    def _build_objective(
        self,
        q_work: np.ndarray,
        tasks: HumanoidTaskSpec,
        q_hint: np.ndarray,
        lower_body_target: np.ndarray,
        qd_current: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.model_spec.num_actuated
        H = float(self.cfg.damping) * np.eye(n, dtype=np.float64)
        g = np.zeros(n, dtype=np.float64)

        J_com, err_com = self._subtree_com_jacobian_and_error(tasks)
        if J_com is not None and err_com is not None:
            weight = float(self.cfg.com_task_weight) * float(max(self.cfg.pelvis_task_weight, 1e-6))
            H += weight * (J_com.T @ J_com)
            g += -weight * (J_com.T @ err_com)

        J_torso_rot, err_torso_rot = self._torso_orientation_jacobian_and_error(tasks)
        if J_torso_rot is not None and err_torso_rot is not None:
            weight = float(self.cfg.torso_orientation_weight) * float(max(self.cfg.torso_task_weight, 1e-6))
            H += weight * (J_torso_rot.T @ J_torso_rot)
            g += -weight * (J_torso_rot.T @ err_torso_rot)

        for site_name, foot_task in (
            ("left_foot", tasks.left_foot),
            ("right_foot", tasks.right_foot),
        ):
            if foot_task.in_contact:
                continue
            J, err = self._site_jacobian_and_error(site_name, foot_task.position_world)
            if J is None or err is None:
                continue
            weight = float(self.cfg.swing_foot_weight) * float(max(foot_task.weight, 1e-6))
            H += weight * (J.T @ J)
            g += -weight * (J.T @ err)

        delta_hint = q_hint - q_work
        arm_selector = self._joint_selector(tuple(self.model_spec.left_arm_joints) + tuple(self.model_spec.right_arm_joints))
        waist_selector = self._joint_selector(tuple(self.model_spec.waist_joints))
        hinted_joint_names = tuple(name for name in tasks.joint_hints.keys() if name in self.model_spec.actuated_joints)
        generic_selector = self._joint_selector(
            tuple(name for name in hinted_joint_names if name not in self.model_spec.waist_joints and name not in self.model_spec.left_arm_joints and name not in self.model_spec.right_arm_joints)
        )

        H += float(self.cfg.waist_hint_weight) * waist_selector
        g += -float(self.cfg.waist_hint_weight) * (waist_selector @ delta_hint)

        H += float(self.cfg.arm_hint_weight) * arm_selector
        g += -float(self.cfg.arm_hint_weight) * (arm_selector @ delta_hint)

        H += float(self.cfg.joint_hint_weight) * generic_selector
        g += -float(self.cfg.joint_hint_weight) * (generic_selector @ delta_hint)

        lower_body_delta = lower_body_target - q_work
        lower_body_selector = self._lower_body_selector()
        H += float(self.cfg.lower_body_hint_weight) * lower_body_selector
        g += -float(self.cfg.lower_body_hint_weight) * (lower_body_selector @ lower_body_delta)

        delta_posture = self.model_spec.stand_ctrl - q_work
        H += float(self.cfg.posture_weight) * np.eye(n, dtype=np.float64)
        g += -float(self.cfg.posture_weight) * delta_posture

        if qd_current.size == n and float(self.cfg.velocity_damping_weight) > 0.0:
            desired_dq = -qd_current * dt
            H += float(self.cfg.velocity_damping_weight) * np.eye(n, dtype=np.float64)
            g += -float(self.cfg.velocity_damping_weight) * desired_dq
        return H, g

    def _build_contact_constraints(
        self,
        tasks: HumanoidTaskSpec,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        rows = []
        rhs = []
        row_slices: Dict[str, tuple[int, int]] = {}
        contact_feet: list[str] = []
        start = 0

        for site_name, foot_task in (
            ("left_foot", tasks.left_foot),
            ("right_foot", tasks.right_foot),
        ):
            if not foot_task.in_contact or not bool(self.cfg.use_contact_constraints):
                continue
            J, err = self._site_jacobian_and_error(site_name, foot_task.position_world)
            if J is None or err is None:
                continue
            rows.append(J)
            rhs.append(err)
            row_slices[site_name] = (start, start + J.shape[0])
            start += J.shape[0]
            contact_feet.append(site_name)

        if not rows:
            A_eq = np.zeros((0, self.model_spec.num_actuated), dtype=np.float64)
            b_eq = np.zeros((0,), dtype=np.float64)
        else:
            A_eq = np.vstack(rows)
            b_eq = np.concatenate(rhs)
        return A_eq, b_eq, {
            "contact_feet": contact_feet,
            "row_slices": row_slices,
        }

    def _solve_constrained_step(
        self,
        H: np.ndarray,
        g: np.ndarray,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = H.shape[0]
        m = A_eq.shape[0]
        if m == 0:
            dq = -self._safe_solve(H, g)
            return dq, np.zeros((0,), dtype=np.float64)

        P = np.zeros((n + m, n + m), dtype=np.float64)
        P[:n, :n] = H
        P[n:, n:] = (
            float(self.cfg.contact_slack_weight) + float(self.cfg.contact_slack_regularization)
        ) * np.eye(m, dtype=np.float64)
        c = np.zeros(n + m, dtype=np.float64)
        c[:n] = g

        A = np.zeros((m, n + m), dtype=np.float64)
        A[:, :n] = A_eq
        A[:, n:] = -np.eye(m, dtype=np.float64)

        KKT = np.block(
            [
                [P, A.T],
                [A, np.zeros((m, m), dtype=np.float64)],
            ]
        )
        rhs = np.concatenate([-c, b_eq], axis=0)
        sol = self._safe_solve(KKT, rhs)
        z = sol[: n + m]
        dq = z[:n]
        slack = z[n:]
        return dq, slack

    def _site_jacobian_and_error(
        self,
        site_name: str,
        target_world: Sequence[float],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        sid = self.model_spec.site_id.get(site_name)
        if sid is None:
            return None, None
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
        J = np.asarray(jacp[:, self.model_spec.actuated_dof_indices], dtype=np.float64)
        current = np.asarray(self.data.site_xpos[sid], dtype=np.float64).copy()
        err = np.asarray(target_world, dtype=np.float64).reshape(3) - current
        return J, err

    def _subtree_com_jacobian_and_error(
        self,
        tasks: HumanoidTaskSpec,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pelvis_body_id = self.model_spec.body_id.get("pelvis")
        if pelvis_body_id is None:
            return None, None
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSubtreeCom(self.model, self.data, jacp, pelvis_body_id)
        J = np.asarray(jacp[:, self.model_spec.actuated_dof_indices], dtype=np.float64)
        current = np.asarray(self.data.subtree_com[pelvis_body_id], dtype=np.float64).copy()
        desired = current.copy()
        desired[:2] = np.asarray(tasks.pelvis.position_world[:2], dtype=np.float64)
        desired[2] = current[2] + float(self.cfg.com_height_gain) * (
            float(tasks.pelvis.position_world[2]) - float(self.cfg.body_height_nominal)
        )
        err = desired - current
        err[:2] = np.clip(err[:2], -float(self.cfg.max_com_xy_error), float(self.cfg.max_com_xy_error))
        err[2] = float(np.clip(err[2], -float(self.cfg.max_com_z_error), float(self.cfg.max_com_z_error)))
        return J, err

    def _torso_orientation_jacobian_and_error(
        self,
        tasks: HumanoidTaskSpec,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        sid = self.model_spec.site_id.get("imu_in_torso")
        if sid is None:
            return None, None
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacSite(self.model, self.data, None, jacr, sid)
        J = np.asarray(jacr[:, self.model_spec.actuated_dof_indices], dtype=np.float64)
        current = np.asarray(self.data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        target = self._rotation_matrix_from_rpy(
            float(tasks.pelvis.roll_world),
            float(tasks.pelvis.pitch_world),
            float(tasks.pelvis.yaw_world + tasks.torso_yaw),
        )
        err = 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )
        return J, err

    def _measure_foot_errors(
        self,
        q_full: np.ndarray,
        q_actuated: np.ndarray,
        tasks: HumanoidTaskSpec,
    ) -> Dict[str, float]:
        full_work = self.model_spec.apply_actuated_qpos(q_full, q_actuated)
        self.data.qpos[:] = full_work
        self.mujoco.mj_forward(self.model, self.data)
        out: Dict[str, float] = {}
        for site_name, foot_task in (
            ("left_foot", tasks.left_foot),
            ("right_foot", tasks.right_foot),
        ):
            _, err = self._site_jacobian_and_error(site_name, foot_task.position_world)
            out[site_name] = float(np.linalg.norm(err)) if err is not None else float("nan")
        return out

    def _measure_com_error(
        self,
        q_full: np.ndarray,
        q_actuated: np.ndarray,
        tasks: HumanoidTaskSpec,
    ) -> float:
        full_work = self.model_spec.apply_actuated_qpos(q_full, q_actuated)
        self.data.qpos[:] = full_work
        self.mujoco.mj_forward(self.model, self.data)
        _, err = self._subtree_com_jacobian_and_error(tasks)
        return float(np.linalg.norm(err)) if err is not None else float("nan")

    def _measure_torso_orientation_error(
        self,
        q_full: np.ndarray,
        q_actuated: np.ndarray,
        tasks: HumanoidTaskSpec,
    ) -> float:
        full_work = self.model_spec.apply_actuated_qpos(q_full, q_actuated)
        self.data.qpos[:] = full_work
        self.mujoco.mj_forward(self.model, self.data)
        _, err = self._torso_orientation_jacobian_and_error(tasks)
        return float(np.linalg.norm(err)) if err is not None else float("nan")

    @staticmethod
    def _slack_dict(
        row_slices: Dict[str, tuple[int, int]],
        slack: np.ndarray,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, (start, end) in row_slices.items():
            out[name] = float(np.linalg.norm(slack[start:end])) if end > start else 0.0
        return out

    @staticmethod
    def _max_task_error(
        foot_errors: Dict[str, float],
        contact_slack: Dict[str, float],
        com_error: float,
        torso_orientation_error: float,
    ) -> float:
        values = [v for v in foot_errors.values() if np.isfinite(v)]
        values.extend(v for v in contact_slack.values() if np.isfinite(v))
        if np.isfinite(com_error):
            values.append(float(com_error))
        if np.isfinite(torso_orientation_error):
            values.append(float(torso_orientation_error))
        return float(max(values)) if values else 0.0

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
        support_mid = 0.5 * (
            np.asarray(tasks.left_foot.position_world[:2], dtype=np.float64)
            + np.asarray(tasks.right_foot.position_world[:2], dtype=np.float64)
        )
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

    def _lower_body_selector(self) -> np.ndarray:
        return self._joint_selector(tuple(self.model_spec.left_leg_joints) + tuple(self.model_spec.right_leg_joints))

    def _joint_selector(self, joint_names: Sequence[str]) -> np.ndarray:
        selector = np.zeros((self.model_spec.num_actuated, self.model_spec.num_actuated), dtype=np.float64)
        valid = [name for name in joint_names if name in self.model_spec.actuated_joints]
        for name in valid:
            idx = self.model_spec.actuated_joints.index(name)
            selector[idx, idx] = 1.0
        return selector

    def _actuated_qvel_from_full(self, qvel_full: Optional[np.ndarray]) -> np.ndarray:
        if qvel_full is None:
            return np.zeros((self.model_spec.num_actuated,), dtype=np.float64)
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
