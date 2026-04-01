"""
Offline stepping follower for Go2 (walk / trot).

Pipeline:
  midline / pair plan -> gait scheduler (walk or trot) -> swing-foot targets
  -> Jacobian IK -> joint PD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from genedynamics.tasks.stepping_stones import (
    decode_plan_states,
    estimate_yaw_from_mid,
    pair_to_virtual_feet,
    states_to_lr_pairs,
)
from genedynamics.tasks.stepping_stones.kinematics import MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR


LEG_ORDER = ("FL", "RR", "FR", "RL")
WALK_SWING_GROUPS = (("FL",), ("RR",), ("FR",), ("RL",))
TROT_SWING_GROUPS = (("FL", "RR"), ("FR", "RL"))
LEG_JOINTS = {
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
}
LEG_TIP_GEOM = {
    "FL": "FL",
    "FR": "FR",
    "RL": "RL",
    "RR": "RR",
}
LEG_TIP_BODY = {
    "FL": "FL_calf",
    "FR": "FR_calf",
    "RL": "RL_calf",
    "RR": "RR_calf",
}
LEG_BASE_OFFSETS = {
    "FL": np.array([1.0, 1.0], dtype=np.float64),
    "FR": np.array([1.0, -1.0], dtype=np.float64),
    "RL": np.array([-1.0, 1.0], dtype=np.float64),
    "RR": np.array([-1.0, -1.0], dtype=np.float64),
}


def _quat_wxyz_yaw(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _wrap(yaw: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(yaw), np.cos(yaw))


def _interp_angle(a0: float, a1: float, alpha: float) -> float:
    d = np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))
    return float(a0 + alpha * d)


def _smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _quat_to_rpy_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    w, x, y, z = q[:4]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _swing_group_from_mode(mode: int) -> Tuple[str, ...]:
    m = int(mode) % 4
    if m in (MODE_DS_FL_RR, MODE_QS_AFTER_FL_RR):
        return ("FR", "RL")
    return ("FL", "RR")


def _stance_group_from_mode(mode: int) -> Tuple[str, ...]:
    m = int(mode) % 4
    if m == MODE_DS_FL_RR:
        return ("FL", "RR")
    if m == MODE_QS_AFTER_FL_RR:
        return LEG_ORDER
    if m == 2:
        return ("FR", "RL")
    return LEG_ORDER


@dataclass
class WalkFollowerConfig:
    gait: str = "walk"  # walk | trot
    sim_dt: float = 0.01
    phase_steps: int = 24
    swing_height: float = 0.05
    leg_half_length: float = 0.18
    step_width: float = 0.30
    kp: float = 35.0
    kd: float = 1.8
    ik_iters: int = 8
    ik_damping: float = 1e-3
    ik_tol: float = 2e-3
    control_clip: float = 3.0
    settle_steps: int = 0
    phase_extend_factor: float = 2.5
    stance_contact_min: int = 2
    release_contact_steps: int = 2
    swing_release_height: float = 0.08
    touchdown_contact_steps: int = 5
    touchdown_xy_tol: float = 0.05
    support_base_blend: float = 0.85
    lift_phase_ratio: float = 0.25
    advance_phase_ratio: float = 0.5
    swing_task_kp_xy: float = 220.0
    swing_task_kp_z: float = 260.0
    swing_task_kd_xy: float = 18.0
    swing_task_kd_z: float = 22.0
    stance_task_kp_xy: float = 140.0
    stance_task_kp_z: float = 180.0
    stance_task_kd_xy: float = 20.0
    stance_task_kd_z: float = 24.0
    task_torque_clip: float = 8.0
    base_height_offset: float = 0.02
    min_foot_z: float = 0.015
    lock_base_pose: bool = False
    base_z_min: float = 0.22
    use_base_pd: bool = True
    touchdown_alpha_min: float = 0.55
    base_kp_xy: float = 40.0
    base_kd_xy: float = 10.0
    base_kp_z: float = 220.0
    base_kd_z: float = 30.0
    base_kp_rp: float = 80.0
    base_kd_rp: float = 10.0
    base_kp_yaw: float = 25.0
    base_kd_yaw: float = 4.0
    base_weight_comp: float = 1.0
    base_support_contact_min: int = 2
    base_force_xy_clip: float = 80.0
    base_force_z_clip: float = 200.0
    base_torque_clip: float = 40.0
    contact_stance_kp_scale_xy: float = 2.0
    contact_stance_kp_scale_z: float = 1.8
    contact_stance_kd_scale_xy: float = 1.6
    contact_stance_kd_scale_z: float = 1.6
    contact_base_kp_scale_xy: float = 2.2
    contact_base_kp_scale_z: float = 1.4
    contact_base_kp_scale_rp: float = 1.55
    contact_base_kp_scale_yaw: float = 2.0
    contact_base_kd_scale_xy: float = 1.4
    contact_base_kd_scale_z: float = 1.25
    contact_base_kd_scale_rp: float = 1.40
    contact_base_kd_scale_yaw: float = 1.4
    contact_base_kp_scale_x: float = 2.2
    contact_base_kd_scale_x: float = 1.4
    contact_base_kp_scale_y: float = 1.50
    contact_base_kd_scale_y: float = 1.20
    contact_phase_steps_scale: float = 1.35
    contact_swing_height_scale: float = 0.70
    contact_touchdown_immediate_relock: bool = True
    contact_base_y_support_blend: float = 0.75
    contact_stance_slip_kp_xy: float = 120.0
    contact_stance_slip_kd_xy: float = 35.0
    contact_stance_slip_kp_z: float = 40.0
    contact_stance_slip_kd_z: float = 12.0
    contact_stance_anchor_err_clip: float = 0.03
    contact_progress_min_support_ratio: float = 0.90
    contact_early_switch_on_touchdown: bool = False
    contact_early_switch_min_phase_ratio: float = 0.35
    contact_post_touchdown_stabilize_steps: int = 10
    contact_base_vel_kd_scale_x: float = 1.0
    contact_base_vel_kd_scale_y: float = 0.4
    contact_base_yaw_rate_kd_scale: float = 1.0
    contact_base_x_support_blend: float = 0.20
    contact_qp_base_pd_blend: float = 0.0
    contact_qp_enable: bool = True
    contact_qp_mu: float = 0.8
    contact_qp_w_base: float = 120.0
    contact_qp_w_stance: float = 80.0
    contact_qp_w_swing: float = 120.0
    contact_qp_w_tau: float = 1e-4
    contact_qp_w_lambda: float = 1e-4
    contact_qp_w_lambda_balance: float = 2e-2
    contact_qp_w_joint_accel: float = 0.35
    contact_qp_torque_limit_scale: float = 3.0
    contact_liftoff_lambda_z_max: float = 8.0
    contact_qs_hold_max_steps: int = 12
    landing_xy_correction_gain: float = 0.18
    landing_xy_correction_clip: float = 0.04
    landing_descent_vel_scale: float = 0.6
    touchdown_force_z_min: float = 20.0
    touchdown_slip_speed_max: float = 0.12
    touchdown_roll_pitch_max: float = 0.30
    phase_hold_roll_pitch_max: float = 0.22


@dataclass
class ContactExecutionReference:
    mode: int
    stance_legs: Tuple[str, ...]
    swing_legs: Tuple[str, ...]
    base_xy: np.ndarray
    base_vel_xy: np.ndarray
    base_yaw: float
    base_yaw_rate: float
    stance_anchors: Dict[str, np.ndarray]
    swing_targets: Dict[str, np.ndarray]
    swing_target_vels: Dict[str, np.ndarray]
    phase_tau: float


@dataclass
class ContactQPSolution:
    tau: np.ndarray
    lambda_by_leg: Dict[str, np.ndarray]


@dataclass
class WrenchReference:
    desired_wrench: np.ndarray
    desired_lambda_by_leg: Dict[str, np.ndarray]
    unload_legs: Tuple[str, ...]


class ContactPhaseManager:
    def __init__(self, cfg: WalkFollowerConfig) -> None:
        self.cfg = cfg

    def progress_tau(
        self,
        *,
        tau0: float,
        tau1: float,
        alpha: float,
        stance_legs: Tuple[str, ...],
        stance_contacts: int,
        base_stable: bool,
    ) -> float:
        tau_nom = (1.0 - alpha) * float(tau0) + alpha * float(tau1)
        support_ratio = float(stance_contacts) / float(max(1, len(stance_legs)))
        progress_ratio = float(
            np.clip(
                support_ratio / max(float(self.cfg.contact_progress_min_support_ratio), 1e-6),
                0.0,
                1.0,
            )
        )
        if not bool(base_stable):
            progress_ratio *= 0.35
        return float(tau0 + progress_ratio * (tau_nom - tau0))

    def should_early_switch(
        self,
        *,
        swing_legs: Tuple[str, ...],
        touchdown_locked: Dict[str, bool],
        base_stable: bool,
        sub: int,
        substeps: int,
    ) -> bool:
        return bool(
            swing_legs
            and bool(self.cfg.contact_early_switch_on_touchdown)
            and all(touchdown_locked.values())
            and bool(base_stable)
            and float(sub + 1) / float(max(1, substeps)) >= float(self.cfg.contact_early_switch_min_phase_ratio)
        )

    def liftoff_legs(self, mode: int, next_mode: Optional[int]) -> Tuple[str, ...]:
        if next_mode is None:
            return tuple()
        stance_legs = _stance_group_from_mode(mode)
        next_stance = _stance_group_from_mode(int(next_mode))
        return tuple(leg for leg in stance_legs if leg not in next_stance)

    def should_hold_qs(
        self,
        *,
        mode: int,
        next_mode: Optional[int],
        lambda_by_leg: Dict[str, np.ndarray],
        base_stable: bool,
    ) -> bool:
        stance_legs = _stance_group_from_mode(mode)
        if len(stance_legs) != len(LEG_ORDER) or next_mode is None:
            return False
        if not bool(base_stable):
            return True
        unload_legs = self.liftoff_legs(mode, next_mode)
        if not unload_legs:
            return False
        for leg in unload_legs:
            lam = np.asarray(lambda_by_leg.get(leg, np.zeros((3,), dtype=np.float64)), dtype=np.float64)
            if lam.shape[0] < 3 or float(lam[2]) > float(self.cfg.contact_liftoff_lambda_z_max):
                return True
        return False


class CentroidalWrenchMapper:
    def __init__(self, cfg: WalkFollowerConfig, total_mass: float, base_inertia_diag: np.ndarray) -> None:
        self.cfg = cfg
        self.total_mass = float(total_mass)
        self.base_inertia_diag = np.asarray(base_inertia_diag, dtype=np.float64)

    def _solve_vertical_distribution(
        self,
        rel_points: np.ndarray,
        desired_force_z: float,
        desired_tau_xy: np.ndarray,
        unload_mask: np.ndarray,
    ) -> np.ndarray:
        n = int(rel_points.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.float64)
        A = np.vstack(
            [
                np.ones((n,), dtype=np.float64),
                rel_points[:, 1],
                -rel_points[:, 0],
            ]
        )
        b = np.asarray(
            [float(max(desired_force_z, 1e-3)), float(desired_tau_xy[0]), float(desired_tau_xy[1])],
            dtype=np.float64,
        )
        nominal = np.full((n,), float(max(desired_force_z, 1e-3)) / float(n), dtype=np.float64)
        if np.any(~unload_mask):
            nominal[unload_mask] *= 0.15
            rest = max(float(np.sum(nominal[~unload_mask])), 1e-6)
            scale = float(max(desired_force_z, 1e-3) - np.sum(nominal[unload_mask])) / rest
            nominal[~unload_mask] *= max(scale, 0.2)
        reg = np.where(unload_mask, 12.0, 0.15).astype(np.float64)
        lhs = A.T @ A + np.diag(reg)
        rhs = A.T @ b + reg * nominal
        fz = np.linalg.solve(lhs + 1e-8 * np.eye(n, dtype=np.float64), rhs)
        fz = np.maximum(fz, 0.0)
        s = float(np.sum(fz))
        if s <= 1e-6:
            return nominal
        return fz * (float(max(desired_force_z, 1e-3)) / s)

    def _solve_tangential_distribution(
        self,
        rel_points: np.ndarray,
        desired_force_xy: np.ndarray,
        desired_tau_z: float,
        fz: np.ndarray,
        unload_mask: np.ndarray,
        mu: float,
    ) -> np.ndarray:
        n = int(rel_points.shape[0])
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        A = np.zeros((3, 2 * n), dtype=np.float64)
        for i in range(n):
            A[0, 2 * i + 0] = 1.0
            A[1, 2 * i + 1] = 1.0
            A[2, 2 * i + 0] = -rel_points[i, 1]
            A[2, 2 * i + 1] = rel_points[i, 0]
        b = np.asarray([float(desired_force_xy[0]), float(desired_force_xy[1]), float(desired_tau_z)], dtype=np.float64)
        reg = np.repeat(np.where(unload_mask, 8.0, 0.2).astype(np.float64), 2)
        xy = np.linalg.solve(A.T @ A + np.diag(reg) + 1e-8 * np.eye(2 * n, dtype=np.float64), A.T @ b)
        out = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            fx = float(xy[2 * i + 0])
            fy = float(xy[2 * i + 1])
            budget = max(0.0, 0.85 * float(mu) * float(fz[i]))
            norm = float(np.hypot(fx, fy))
            if norm > budget and norm > 1e-9:
                scale = budget / norm
                fx *= scale
                fy *= scale
            out[i, 0] = fx
            out[i, 1] = fy
            out[i, 2] = float(fz[i])
        return out

    def build(
        self,
        *,
        mode: int,
        next_mode: Optional[int],
        stance_legs: Tuple[str, ...],
        base_accel: np.ndarray,
        contact_points: np.ndarray,
        com_pos: np.ndarray,
        mu: float,
    ) -> WrenchReference:
        base_accel = np.asarray(base_accel, dtype=np.float64)
        desired_force = np.array(
            [
                self.total_mass * float(base_accel[0]),
                self.total_mass * float(base_accel[1]),
                self.total_mass * (float(base_accel[2]) + 9.81),
            ],
            dtype=np.float64,
        )
        desired_torque = np.asarray(self.base_inertia_diag, dtype=np.float64) * np.asarray(base_accel[3:6], dtype=np.float64)
        desired_wrench = np.concatenate([desired_force, desired_torque], axis=0)
        unload_legs = tuple()
        if next_mode is not None:
            next_stance = _stance_group_from_mode(int(next_mode))
            unload_legs = tuple(leg for leg in stance_legs if leg not in next_stance)
        unload_mask = np.asarray([leg in unload_legs for leg in stance_legs], dtype=bool)
        rel_points = np.asarray(contact_points, dtype=np.float64) - np.asarray(com_pos, dtype=np.float64)[None, :]
        fz = self._solve_vertical_distribution(
            rel_points,
            desired_force_z=float(desired_force[2]),
            desired_tau_xy=np.asarray(desired_torque[:2], dtype=np.float64),
            unload_mask=unload_mask,
        )
        forces = self._solve_tangential_distribution(
            rel_points,
            desired_force_xy=np.asarray(desired_force[:2], dtype=np.float64),
            desired_tau_z=float(desired_torque[2]),
            fz=fz,
            unload_mask=unload_mask,
            mu=float(mu),
        )
        desired_lambda_by_leg: Dict[str, np.ndarray] = {}
        for i, leg in enumerate(stance_legs):
            desired_lambda_by_leg[leg] = np.asarray(forces[i], dtype=np.float64).copy()
        return WrenchReference(
            desired_wrench=desired_wrench,
            desired_lambda_by_leg=desired_lambda_by_leg,
            unload_legs=unload_legs,
        )


class HierarchicalContactQP:
    def __init__(self, cfg: WalkFollowerConfig) -> None:
        self.cfg = cfg

    def _solve_qp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        G: Optional[np.ndarray],
        h: Optional[np.ndarray],
        Aeq: np.ndarray,
        beq: np.ndarray,
    ) -> Optional[np.ndarray]:
        from qpsolvers import solve_qp

        for solver_name in ("clarabel", "osqp", "cvxopt"):
            try:
                sol = solve_qp(P, q, G, h, Aeq, beq, solver=solver_name)
            except Exception:
                sol = None
            if sol is not None and np.all(np.isfinite(sol)):
                return np.asarray(sol, dtype=np.float64)
        return None

    def solve(
        self,
        follower: "SteppingWalkFollower",
        *,
        ref: ContactExecutionReference,
        foot_target: Dict[str, np.ndarray],
        prev_foot_target: Dict[str, np.ndarray],
        q_joint_des: np.ndarray,
        next_mode: Optional[int],
    ) -> Optional[ContactQPSolution]:
        cfg = self.cfg
        stance_legs = tuple(ref.stance_legs)
        swing_legs = tuple(ref.swing_legs)
        n_c = 3 * len(stance_legs)
        nv = int(follower.model.nv)
        nu = int(len(follower.actuator_ids))
        n = nv + nu + n_c
        qdd_slice = slice(0, nv)
        tau_slice = slice(nv, nv + nu)
        lam_slice = slice(nv + nu, n)

        M = follower._mass_matrix()
        bias = np.asarray(follower.data.qfrc_bias, dtype=np.float64).copy()
        S = np.asarray(follower.actuator_selection, dtype=np.float64)
        Jc_blocks = []
        jdotq_blocks = []
        contact_points = []
        for leg in stance_legs:
            pos, _vel, J, jdot_qdot = follower._foot_contact_kinematics(leg)
            Jc_blocks.append(J)
            jdotq_blocks.append(jdot_qdot)
            contact_points.append(np.asarray(pos, dtype=np.float64))
        Jc = np.vstack(Jc_blocks) if Jc_blocks else np.zeros((0, nv), dtype=np.float64)
        jdotq = np.concatenate(jdotq_blocks, axis=0) if jdotq_blocks else np.zeros((0,), dtype=np.float64)
        base_accel = follower._base_task_accel(
            target_xy=ref.base_xy,
            target_vel_xy=ref.base_vel_xy,
            target_yaw=ref.base_yaw,
            target_yaw_rate=ref.base_yaw_rate,
        )
        wrench_ref = follower._wrench_mapper.build(
            mode=int(ref.mode),
            next_mode=next_mode,
            stance_legs=stance_legs,
            base_accel=base_accel,
            contact_points=np.asarray(contact_points, dtype=np.float64),
            com_pos=np.asarray(follower.data.qpos[:3], dtype=np.float64),
            mu=float(cfg.contact_qp_mu),
        )

        q_joint = np.asarray(follower.data.qpos[follower.joint_qidx_all], dtype=np.float64)
        qd_joint = np.asarray(follower.data.qvel[follower.joint_didx_all], dtype=np.float64)
        qdd_joint_des = float(cfg.kp) * (np.asarray(q_joint_des, dtype=np.float64) - q_joint) - float(cfg.kd) * qd_joint
        qdd_joint_des = np.clip(qdd_joint_des, -60.0, 60.0)

        Aeq_dyn = np.zeros((nv, n), dtype=np.float64)
        Aeq_dyn[:, qdd_slice] = M
        Aeq_dyn[:, tau_slice] = -S
        if n_c > 0:
            Aeq_dyn[:, lam_slice] = -Jc.T
        beq_dyn = -bias
        if n_c > 0:
            Aeq_contact = np.zeros((n_c, n), dtype=np.float64)
            Aeq_contact[:, qdd_slice] = Jc
            beq_contact = -jdotq
            Aeq = np.vstack([Aeq_dyn, Aeq_contact])
            beq = np.concatenate([beq_dyn, beq_contact], axis=0)
        else:
            Aeq = Aeq_dyn
            beq = beq_dyn

        G_blocks = []
        h_blocks = []
        tau_lim = float(cfg.control_clip) * float(cfg.contact_qp_torque_limit_scale)
        G_tau = np.zeros((2 * nu, n), dtype=np.float64)
        G_tau[:nu, tau_slice] = np.eye(nu, dtype=np.float64)
        G_tau[nu:, tau_slice] = -np.eye(nu, dtype=np.float64)
        h_tau = np.full((2 * nu,), tau_lim, dtype=np.float64)
        G_blocks.append(G_tau)
        h_blocks.append(h_tau)
        if n_c > 0:
            mu = float(cfg.contact_qp_mu)
            G_l = np.zeros((5 * len(stance_legs), n), dtype=np.float64)
            h_l = np.zeros((5 * len(stance_legs),), dtype=np.float64)
            for i in range(len(stance_legs)):
                col = lam_slice.start + 3 * i
                row = 5 * i
                G_l[row + 0, col + 0] = 1.0
                G_l[row + 0, col + 2] = -mu
                G_l[row + 1, col + 0] = -1.0
                G_l[row + 1, col + 2] = -mu
                G_l[row + 2, col + 1] = 1.0
                G_l[row + 2, col + 2] = -mu
                G_l[row + 3, col + 1] = -1.0
                G_l[row + 3, col + 2] = -mu
                G_l[row + 4, col + 2] = -1.0
            G_blocks.append(G_l)
            h_blocks.append(h_l)
        G = np.vstack(G_blocks) if G_blocks else None
        h = np.concatenate(h_blocks, axis=0) if h_blocks else None

        def add_quad(P: np.ndarray, qv: np.ndarray, A: np.ndarray, y: np.ndarray, w: float) -> Tuple[np.ndarray, np.ndarray]:
            if A.size == 0 or w <= 0.0:
                return P, qv
            P = P + 2.0 * float(w) * (A.T @ A)
            qv = qv - 2.0 * float(w) * (A.T @ y)
            return P, qv

        def add_var_reg(P: np.ndarray, qv: np.ndarray, target: np.ndarray, sl: slice, w: float) -> Tuple[np.ndarray, np.ndarray]:
            if w <= 0.0:
                return P, qv
            A = np.zeros((sl.stop - sl.start, n), dtype=np.float64)
            A[:, sl] = np.eye(sl.stop - sl.start, dtype=np.float64)
            return add_quad(P, qv, A, np.asarray(target, dtype=np.float64), w)

        P1 = np.eye(n, dtype=np.float64) * 1e-8
        q1 = np.zeros((n,), dtype=np.float64)
        A_base = np.zeros((6, n), dtype=np.float64)
        A_base[:, qdd_slice] = np.eye(6, nv, dtype=np.float64)
        A_base_core = A_base[:6, :]
        P1, q1 = add_quad(P1, q1, A_base_core, base_accel, 0.35 * float(cfg.contact_qp_w_base))
        P1, q1 = add_var_reg(P1, q1, np.zeros((nu,), dtype=np.float64), tau_slice, float(cfg.contact_qp_w_tau))
        if n_c > 0:
            W = np.zeros((6, n_c), dtype=np.float64)
            for i, p in enumerate(contact_points):
                r = np.asarray(p, dtype=np.float64) - np.asarray(follower.data.qpos[:3], dtype=np.float64)
                W[:3, 3 * i : 3 * i + 3] = np.eye(3, dtype=np.float64)
                W[3:, 3 * i : 3 * i + 3] = np.array(
                    [
                        [0.0, -r[2], r[1]],
                        [r[2], 0.0, -r[0]],
                        [-r[1], r[0], 0.0],
                    ],
                    dtype=np.float64,
                )
            A_wr = np.zeros((6, n), dtype=np.float64)
            A_wr[:, lam_slice] = W
            P1, q1 = add_quad(P1, q1, A_wr, np.asarray(wrench_ref.desired_wrench, dtype=np.float64), 2.2 * float(cfg.contact_qp_w_base))
            lam_target = np.concatenate(
                [np.asarray(wrench_ref.desired_lambda_by_leg.get(leg, np.zeros((3,), dtype=np.float64)), dtype=np.float64) for leg in stance_legs],
                axis=0,
            ) if stance_legs else np.zeros((0,), dtype=np.float64)
            P1, q1 = add_var_reg(P1, q1, lam_target, lam_slice, 45.0 * float(cfg.contact_qp_w_lambda))
            unload_rows = []
            unload_y = []
            for leg in wrench_ref.unload_legs:
                if leg not in stance_legs:
                    continue
                i = stance_legs.index(leg)
                row = np.zeros((1, n), dtype=np.float64)
                row[0, lam_slice.start + 3 * i + 2] = 1.0
                unload_rows.append(row)
                unload_y.append(np.zeros((1,), dtype=np.float64))
            if unload_rows:
                A_unload = np.vstack(unload_rows)
                y_unload = np.concatenate(unload_y, axis=0)
                P1, q1 = add_quad(P1, q1, A_unload, y_unload, 3.0 * float(cfg.contact_qp_w_stance))

        sol1 = self._solve_qp(P1, q1, G, h, Aeq, beq)
        if sol1 is None:
            return None

        P2 = np.eye(n, dtype=np.float64) * 1e-8
        q2 = np.zeros((n,), dtype=np.float64)
        base_qdd_slice = slice(qdd_slice.start, qdd_slice.start + 6)
        P2, q2 = add_var_reg(P2, q2, np.asarray(sol1[base_qdd_slice], dtype=np.float64), base_qdd_slice, 220.0)
        P2, q2 = add_var_reg(P2, q2, np.asarray(sol1[tau_slice], dtype=np.float64), tau_slice, 0.1)
        if n_c > 0:
            P2, q2 = add_var_reg(P2, q2, np.asarray(sol1[lam_slice], dtype=np.float64), lam_slice, 140.0)
        P2, q2 = add_quad(P2, q2, A_base_core, base_accel, 0.8 * float(cfg.contact_qp_w_base))

        swing_rows = []
        swing_acc = []
        for leg in swing_legs:
            _pos, vel, J = follower._foot_full_jacobian(leg)
            tgt = np.asarray(foot_target[leg], dtype=np.float64)
            tgt_vel = np.asarray(ref.swing_target_vels.get(leg, (tgt - prev_foot_target[leg]) / max(float(cfg.sim_dt), 1e-6)), dtype=np.float64)
            pos = np.asarray(follower.data.geom_xpos[follower.leg_geom_id[leg]], dtype=np.float64).copy()
            err = tgt - pos
            vel_err = tgt_vel - vel
            a_des = np.array(
                [
                    float(cfg.swing_task_kp_xy) * err[0] + float(cfg.swing_task_kd_xy) * vel_err[0],
                    float(cfg.swing_task_kp_xy) * err[1] + float(cfg.swing_task_kd_xy) * vel_err[1],
                    float(cfg.swing_task_kp_z) * err[2] + float(cfg.swing_task_kd_z) * vel_err[2],
                ],
                dtype=np.float64,
            )
            swing_rows.append(J)
            swing_acc.append(a_des)
        if swing_rows:
            Js = np.vstack(swing_rows)
            ys = np.concatenate(swing_acc, axis=0)
            A_sw = np.zeros((Js.shape[0], n), dtype=np.float64)
            A_sw[:, qdd_slice] = Js
            P2, q2 = add_quad(P2, q2, A_sw, ys, 0.55 * float(cfg.contact_qp_w_swing))

        A_ja = np.zeros((nu, n), dtype=np.float64)
        for i, did in enumerate(follower.actuator_dof_ids.tolist()):
            A_ja[i, qdd_slice.start + int(did)] = 1.0
        P2, q2 = add_quad(P2, q2, A_ja, qdd_joint_des, 0.6 * float(cfg.contact_qp_w_joint_accel))

        sol2 = self._solve_qp(P2, q2, G, h, Aeq, beq)
        sol = sol2 if sol2 is not None else sol1
        tau = np.asarray(sol[tau_slice], dtype=np.float64)
        lambda_by_leg: Dict[str, np.ndarray] = {}
        if n_c > 0:
            lam = np.asarray(sol[lam_slice], dtype=np.float64)
            for i, leg in enumerate(stance_legs):
                lambda_by_leg[leg] = lam[3 * i : 3 * i + 3].copy()
        return ContactQPSolution(
            tau=np.clip(tau, -tau_lim, tau_lim),
            lambda_by_leg=lambda_by_leg,
        )


class SteppingWalkFollower:
    def __init__(
        self,
        model_xml_path: Optional[str] = None,
        cfg: Optional[WalkFollowerConfig] = None,
        stepping_scene: Optional[Dict[str, Any]] = None,
    ):
        import mujoco
        from genedynamics.robots.registry import _get_go2_path
        from genedynamics.envs.utils.mujoco_model_generator import create_go2_sim_xml_with_stepping_scene

        self.cfg = cfg or WalkFollowerConfig()
        self.stepping_scene = stepping_scene or None
        if model_xml_path is None:
            p = Path(_get_go2_path() or "")
            if not p.exists():
                raise FileNotFoundError("Go2 model not found. Set MUJOCO_MENAGERIE_PATH.")
            scene = p.parent / "scene.xml"
            model_xml_path = str(scene if scene.exists() else p)

        temp_xml_path: Optional[Path] = None
        if self.stepping_scene:
            src = Path(str(model_xml_path))
            temp_xml_path = src.parent / "_walk_follow_scene_temp.xml"
            create_go2_sim_xml_with_stepping_scene(
                str(temp_xml_path),
                stepping_scene=self.stepping_scene,
                go2_xml_path=str(src),
            )
            model_xml_path = str(temp_xml_path)

        self.mujoco = mujoco
        try:
            self.model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        finally:
            if temp_xml_path is not None:
                try:
                    os.unlink(temp_xml_path)
                except OSError:
                    pass
        self.model.opt.timestep = float(self.cfg.sim_dt)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._build_indices()
        self._home_qpos = np.array(self.data.qpos, copy=True)
        self._base_height = float(self._home_qpos[2])
        self._total_mass = float(np.sum(self.model.body_mass[1:]))
        self._scene_bank_top_z = 0.0
        self._scene_river_depth = 0.10
        self._scene_stone_top_z = 0.035
        M0 = self._mass_matrix()
        self._base_inertia_diag = np.maximum(np.diag(M0)[3:6], 1e-3)
        self._last_contact_lambda_by_leg: Dict[str, np.ndarray] = {}
        self._phase_manager = ContactPhaseManager(self.cfg)
        self._wrench_mapper = CentroidalWrenchMapper(self.cfg, self._total_mass, self._base_inertia_diag)
        self._hierarchical_qp = HierarchicalContactQP(self.cfg)

    def _build_indices(self) -> None:
        m = self.model
        mujoco = self.mujoco

        self.leg_qidx: Dict[str, np.ndarray] = {}
        self.leg_didx: Dict[str, np.ndarray] = {}
        self.leg_jmin: Dict[str, np.ndarray] = {}
        self.leg_jmax: Dict[str, np.ndarray] = {}
        self.leg_body_id: Dict[str, int] = {}
        self.leg_geom_id: Dict[str, int] = {}
        self.actuator_ids: List[int] = []
        self.actuator_dof_ids: List[int] = []
        self.joint_qidx_all: List[int] = []
        self.joint_didx_all: List[int] = []
        self.support_geom_ids: Set[int] = set()
        self.world_geom_ids: Set[int] = set()

        for leg, joints in LEG_JOINTS.items():
            qids, dids, jmin, jmax = [], [], [], []
            for jn in joints:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise KeyError(f"Missing joint: {jn}")
                qid = int(m.jnt_qposadr[jid])
                did = int(m.jnt_dofadr[jid])
                qids.append(qid)
                dids.append(did)
                rng = m.jnt_range[jid]
                jmin.append(float(rng[0]))
                jmax.append(float(rng[1]))

            self.leg_qidx[leg] = np.asarray(qids, dtype=np.int32)
            self.leg_didx[leg] = np.asarray(dids, dtype=np.int32)
            self.leg_jmin[leg] = np.asarray(jmin, dtype=np.float64)
            self.leg_jmax[leg] = np.asarray(jmax, dtype=np.float64)

            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, LEG_TIP_BODY[leg])
            if bid < 0:
                raise KeyError(f"Missing body: {LEG_TIP_BODY[leg]}")
            self.leg_body_id[leg] = int(bid)
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, LEG_TIP_GEOM[leg])
            if gid < 0:
                raise KeyError(f"Missing geom: {LEG_TIP_GEOM[leg]}")
            self.leg_geom_id[leg] = int(gid)

            for an in (f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"):
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
                if aid < 0:
                    raise KeyError(f"Missing actuator: {an}")
                self.actuator_ids.append(int(aid))
                trnid = int(m.actuator_trnid[aid, 0])
                if trnid < 0:
                    raise KeyError(f"Actuator not tied to joint: {an}")
                self.actuator_dof_ids.append(int(m.jnt_dofadr[trnid]))

            self.joint_qidx_all.extend(qids)
            self.joint_didx_all.extend(dids)

        self.joint_qidx_all = np.asarray(self.joint_qidx_all, dtype=np.int32)
        self.joint_didx_all = np.asarray(self.joint_didx_all, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        self.actuator_dof_ids = np.asarray(self.actuator_dof_ids, dtype=np.int32)
        self.actuator_selection = np.zeros((self.model.nv, len(self.actuator_ids)), dtype=np.float64)
        for i, did in enumerate(self.actuator_dof_ids.tolist()):
            self.actuator_selection[int(did), i] = 1.0
        for gid in range(int(m.ngeom)):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name == "floor":
                self.support_geom_ids.add(int(gid))
                self.world_geom_ids.add(int(gid))
            elif name.startswith("stepping_stone_") or name.startswith("bank_"):
                self.support_geom_ids.add(int(gid))
                self.world_geom_ids.add(int(gid))
            elif name == "river_bottom":
                self.world_geom_ids.add(int(gid))

    def _set_base_pose(self, xy: np.ndarray, yaw: float) -> None:
        q = self.data.qpos
        q[0] = float(xy[0])
        q[1] = float(xy[1])
        q[2] = self._base_height + float(self.cfg.base_height_offset)
        q[3:7] = _quat_wxyz_yaw(float(yaw))

    def _stabilize_base(self, xy: np.ndarray, yaw: float) -> None:
        q = self.data.qpos
        v = self.data.qvel
        q[0] = float(xy[0])
        q[1] = float(xy[1])
        q[2] = max(float(self.cfg.base_z_min), float(q[2]))
        q[3:7] = _quat_wxyz_yaw(float(yaw))
        v[:6] = 0.0

    def _apply_base_pd(
        self,
        xy: np.ndarray,
        yaw: float,
        support_contacts: int,
        *,
        kp_x_scale: float = 1.0,
        kd_x_scale: float = 1.0,
        kp_y_scale: float = 1.0,
        kd_y_scale: float = 1.0,
        kp_z_scale: float = 1.0,
        kd_z_scale: float = 1.0,
        kp_rp_scale: float = 1.0,
        kd_rp_scale: float = 1.0,
        kp_yaw_scale: float = 1.0,
        kd_yaw_scale: float = 1.0,
        vel_xy_ref: Optional[np.ndarray] = None,
        yaw_rate_ref: float = 0.0,
    ) -> None:
        cfg = self.cfg
        q = self.data.qpos
        v = self.data.qvel
        vel_xy_ref = np.zeros((2,), dtype=np.float64) if vel_xy_ref is None else np.asarray(vel_xy_ref, dtype=np.float64)
        rpy = _quat_to_rpy_wxyz(q[3:7])
        yaw_err = np.arctan2(np.sin(float(yaw) - rpy[2]), np.cos(float(yaw) - rpy[2]))
        denom = max(1, int(cfg.base_support_contact_min))
        support_scale = float(np.clip(float(support_contacts) / float(denom), 0.0, 1.0))
        fx = (
            float(cfg.base_kp_xy) * float(kp_x_scale) * (float(xy[0]) - float(q[0]))
            - float(cfg.base_kd_xy) * float(kd_x_scale) * (float(v[0]) - float(vel_xy_ref[0]))
        )
        fy = (
            float(cfg.base_kp_xy) * float(kp_y_scale) * (float(xy[1]) - float(q[1]))
            - float(cfg.base_kd_xy) * float(kd_y_scale) * (float(v[1]) - float(vel_xy_ref[1]))
        )
        z_ref = self._base_height + float(cfg.base_height_offset)
        gravity_mag = float(abs(self.model.opt.gravity[2]))
        fz = (
            float(cfg.base_weight_comp) * self._total_mass * gravity_mag
            + float(cfg.base_kp_z) * float(kp_z_scale) * (z_ref - float(q[2]))
            - float(cfg.base_kd_z) * float(kd_z_scale) * float(v[2])
        )
        tx = -float(cfg.base_kp_rp) * float(kp_rp_scale) * float(rpy[0]) - float(cfg.base_kd_rp) * float(kd_rp_scale) * float(v[3])
        ty = -float(cfg.base_kp_rp) * float(kp_rp_scale) * float(rpy[1]) - float(cfg.base_kd_rp) * float(kd_rp_scale) * float(v[4])
        tz = (
            float(cfg.base_kp_yaw) * float(kp_yaw_scale) * yaw_err
            - float(cfg.base_kd_yaw) * float(kd_yaw_scale) * (float(v[5]) - float(yaw_rate_ref))
        )
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[0] = support_scale * np.clip(
            fx, -float(cfg.base_force_xy_clip), float(cfg.base_force_xy_clip)
        )
        self.data.qfrc_applied[1] = support_scale * np.clip(
            fy, -float(cfg.base_force_xy_clip), float(cfg.base_force_xy_clip)
        )
        self.data.qfrc_applied[2] = support_scale * np.clip(
            fz, -float(cfg.base_force_z_clip), float(cfg.base_force_z_clip)
        )
        self.data.qfrc_applied[3] = support_scale * np.clip(
            tx, -float(cfg.base_torque_clip), float(cfg.base_torque_clip)
        )
        self.data.qfrc_applied[4] = support_scale * np.clip(
            ty, -float(cfg.base_torque_clip), float(cfg.base_torque_clip)
        )
        self.data.qfrc_applied[5] = support_scale * np.clip(
            tz, -float(cfg.base_torque_clip), float(cfg.base_torque_clip)
        )

    def _has_contact_with_geom_set(self, leg: str, geom_ids: Set[int]) -> bool:
        if not geom_ids:
            return False
        gid = self.leg_geom_id[leg]
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 == gid and g2 in geom_ids) or (g2 == gid and g1 in geom_ids):
                return True
        return False

    def _has_support_contact(self, leg: str) -> bool:
        return self._has_contact_with_geom_set(leg, self.support_geom_ids)

    def _has_world_contact(self, leg: str) -> bool:
        return self._has_contact_with_geom_set(leg, self.world_geom_ids)

    def _ground_contact_legs(self) -> List[str]:
        return [leg for leg in LEG_ORDER if self._has_support_contact(leg)]

    def _base_roll_pitch(self) -> np.ndarray:
        return _quat_to_rpy_wxyz(np.asarray(self.data.qpos[3:7], dtype=np.float64))[:2]

    def _base_is_stable(self, *, max_abs_rp: Optional[float] = None) -> bool:
        limit = float(self.cfg.phase_hold_roll_pitch_max if max_abs_rp is None else max_abs_rp)
        rp = np.abs(self._base_roll_pitch())
        return bool(float(np.max(rp)) <= limit)

    def _measured_contact_force_z(self, leg: str) -> float:
        mujoco = self.mujoco
        gid = self.leg_geom_id[leg]
        total_fz = 0.0
        wrench = np.zeros((6,), dtype=np.float64)
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if not ((g1 == gid and g2 in self.support_geom_ids) or (g2 == gid and g1 in self.support_geom_ids)):
                continue
            wrench[:] = 0.0
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            total_fz += abs(float(wrench[0]))
        return float(total_fz)

    def _touchdown_quality(
        self,
        leg: str,
        goal_xy: np.ndarray,
    ) -> bool:
        if not self._has_support_contact(leg):
            return False
        curr = np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
        near_goal = float(np.linalg.norm(curr[:2] - np.asarray(goal_xy, dtype=np.float64)[:2])) <= float(self.cfg.touchdown_xy_tol)
        if not near_goal:
            return False
        fz = self._measured_contact_force_z(leg)
        if fz < float(self.cfg.touchdown_force_z_min):
            return False
        _pos, vel, _J = self._foot_full_jacobian(leg)
        if float(np.linalg.norm(vel[:2])) > float(self.cfg.touchdown_slip_speed_max):
            return False
        if not self._base_is_stable(max_abs_rp=float(self.cfg.touchdown_roll_pitch_max)):
            return False
        return True

    def _terrain_height_at(self, xy: np.ndarray) -> float:
        x = float(xy[0])
        y = float(xy[1])
        if not self.stepping_scene:
            return 0.0
        centers = np.asarray(self.stepping_scene.get("stones_centers", []), dtype=np.float64)
        radii = np.asarray(self.stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        n = min(len(centers), len(radii))
        for i in range(n):
            c = np.asarray(centers[i], dtype=np.float64).reshape(-1)
            if c.size < 2 or not np.isfinite(c[:2]).all():
                continue
            if (x - c[0]) ** 2 + (y - c[1]) ** 2 <= float(radii[i]) ** 2:
                return float(self._scene_stone_top_z)
        if not bool(self.stepping_scene.get("has_river", True)):
            return 0.0
        river_x = self.stepping_scene.get("river_x", [-0.2, 0.2])
        try:
            rx0, rx1 = float(river_x[0]), float(river_x[1])
        except Exception:
            rx0, rx1 = -0.2, 0.2
        if rx0 <= x <= rx1:
            return -float(self._scene_river_depth)
        return float(self._scene_bank_top_z)

    def _foot_pos_vel(self, leg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mujoco = self.mujoco
        gid = self.leg_geom_id[leg]
        didx = self.leg_didx[leg]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, gid)
        pos = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()
        J = jacp[:, didx]
        vel = J @ np.asarray(self.data.qvel[didx], dtype=np.float64)
        return pos, vel, J

    def _foot_full_jacobian(self, leg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mujoco = self.mujoco
        gid = self.leg_geom_id[leg]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, gid)
        pos = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()
        vel = jacp @ np.asarray(self.data.qvel, dtype=np.float64)
        return pos, vel, jacp

    def _foot_contact_kinematics(self, leg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mujoco = self.mujoco
        gid = self.leg_geom_id[leg]
        bid = self.leg_body_id[leg]
        point = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        jacpdot = np.zeros((3, self.model.nv), dtype=np.float64)
        jacrdot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, gid)
        mujoco.mj_jacDot(self.model, self.data, jacpdot, jacrdot, point, bid)
        vel = jacp @ np.asarray(self.data.qvel, dtype=np.float64)
        jdot_qdot = jacpdot @ np.asarray(self.data.qvel, dtype=np.float64)
        return point, vel, jacp, jdot_qdot

    def _mass_matrix(self) -> np.ndarray:
        M = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        self.mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def _body_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(self.data.qpos[:3], dtype=np.float64).copy()
        vel = np.asarray(self.data.qvel[:3], dtype=np.float64).copy()
        rpy = _quat_to_rpy_wxyz(np.asarray(self.data.qpos[3:7], dtype=np.float64))
        return pos, vel, rpy

    def _base_task_accel(
        self,
        *,
        target_xy: np.ndarray,
        target_vel_xy: np.ndarray,
        target_yaw: float,
        target_yaw_rate: float,
    ) -> np.ndarray:
        cfg = self.cfg
        q = self.data.qpos
        v = self.data.qvel
        rpy = _quat_to_rpy_wxyz(q[3:7])
        yaw_err = np.arctan2(np.sin(float(target_yaw) - rpy[2]), np.cos(float(target_yaw) - rpy[2]))
        acc = np.zeros((6,), dtype=np.float64)
        acc[0] = (
            float(cfg.base_kp_xy) * float(cfg.contact_base_kp_scale_x) * (float(target_xy[0]) - float(q[0]))
            + float(cfg.base_kd_xy) * float(cfg.contact_base_kd_scale_x) * (float(target_vel_xy[0]) - float(v[0]))
        )
        acc[1] = (
            float(cfg.base_kp_xy) * float(cfg.contact_base_kp_scale_y) * (float(target_xy[1]) - float(q[1]))
            + float(cfg.base_kd_xy) * float(cfg.contact_base_kd_scale_y) * (float(target_vel_xy[1]) - float(v[1]))
        )
        z_ref = self._base_height + float(cfg.base_height_offset)
        acc[2] = (
            float(cfg.base_kp_z) * float(cfg.contact_base_kp_scale_z) * (z_ref - float(q[2]))
            - float(cfg.base_kd_z) * float(cfg.contact_base_kd_scale_z) * float(v[2])
        )
        acc[3] = -float(cfg.base_kp_rp) * float(cfg.contact_base_kp_scale_rp) * float(rpy[0]) - float(cfg.base_kd_rp) * float(cfg.contact_base_kd_scale_rp) * float(v[3])
        acc[4] = -float(cfg.base_kp_rp) * float(cfg.contact_base_kp_scale_rp) * float(rpy[1]) - float(cfg.base_kd_rp) * float(cfg.contact_base_kd_scale_rp) * float(v[4])
        acc[5] = (
            float(cfg.base_kp_yaw) * float(cfg.contact_base_kp_scale_yaw) * yaw_err
            + float(cfg.base_kd_yaw) * float(cfg.contact_base_kd_scale_yaw) * (float(target_yaw_rate) - float(v[5]))
        )
        return acc

    def _project_swing_goal(
        self,
        goal: np.ndarray,
        base_xy: np.ndarray,
        plan_xy: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        corrected = np.asarray(goal, dtype=np.float64).copy()
        delta = np.asarray(plan_xy, dtype=np.float64) - np.asarray(base_xy, dtype=np.float64)
        corr = float(cfg.landing_xy_correction_gain) * delta[:2]
        n = float(np.linalg.norm(corr))
        clip = float(cfg.landing_xy_correction_clip)
        if n > clip and n > 1e-9:
            corr = corr * (clip / n)
        corrected[:2] += corr
        corrected = self._project_to_safe_support(corrected)
        return corrected

    def _project_to_safe_support(self, point: np.ndarray, *, margin: float = 0.006) -> np.ndarray:
        out = np.asarray(point, dtype=np.float64).copy()
        if not self.stepping_scene:
            return out
        centers = np.asarray(self.stepping_scene.get("stones_centers", []), dtype=np.float64)
        radii = np.asarray(self.stepping_scene.get("stones_radii", []), dtype=np.float64).reshape(-1)
        n = min(len(centers), len(radii))
        if n <= 0:
            return out
        centers = centers[:n, :2]
        radii = radii[:n]
        delta = out[:2][None, :] - centers
        dist = np.linalg.norm(delta, axis=1)
        support = radii - dist
        best_idx = int(np.argmax(support))
        if float(support[best_idx]) >= float(margin):
            return out
        direction = delta[best_idx]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([1.0, 0.0], dtype=np.float64)
            norm = 1.0
        target_r = max(0.0, float(radii[best_idx]) - float(margin))
        out[:2] = centers[best_idx] + direction / norm * target_r
        return out

    def _normal_contact_force(self, leg: str) -> float:
        lam = self._last_contact_lambda_by_leg.get(leg)
        if lam is None or len(lam) < 3:
            return 0.0
        return float(lam[2])

    def _task_space_leg_torque(
        self,
        leg: str,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        *,
        kp_xy: float,
        kp_z: float,
        kd_xy: float,
        kd_z: float,
    ) -> np.ndarray:
        pos, vel, J = self._foot_pos_vel(leg)
        err = np.asarray(target_pos, dtype=np.float64) - pos
        vel_err = np.asarray(target_vel, dtype=np.float64) - vel
        wrench = np.array(
            [
                float(kp_xy) * err[0] + float(kd_xy) * vel_err[0],
                float(kp_xy) * err[1] + float(kd_xy) * vel_err[1],
                float(kp_z) * err[2] + float(kd_z) * vel_err[2],
            ],
            dtype=np.float64,
        )
        tau = J.T @ wrench
        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    def _stance_antislip_torque(self, leg: str, anchor_pos: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pos, vel, J = self._foot_pos_vel(leg)
        anchor = np.asarray(anchor_pos, dtype=np.float64)
        err = anchor - pos
        err[0] = float(np.clip(err[0], -float(cfg.contact_stance_anchor_err_clip), float(cfg.contact_stance_anchor_err_clip)))
        err[1] = float(np.clip(err[1], -float(cfg.contact_stance_anchor_err_clip), float(cfg.contact_stance_anchor_err_clip)))
        wrench = np.array(
            [
                float(cfg.contact_stance_slip_kp_xy) * err[0] - float(cfg.contact_stance_slip_kd_xy) * vel[0],
                float(cfg.contact_stance_slip_kp_xy) * err[1] - float(cfg.contact_stance_slip_kd_xy) * vel[1],
                float(cfg.contact_stance_slip_kp_z) * err[2] - float(cfg.contact_stance_slip_kd_z) * vel[2],
            ],
            dtype=np.float64,
        )
        tau = J.T @ wrench
        return np.clip(tau, -float(self.cfg.task_torque_clip), float(self.cfg.task_torque_clip))

    def _build_execution_reference(
        self,
        *,
        mode: int,
        alpha: float,
        tau0: float,
        tau1: float,
        mid_k: np.ndarray,
        mid_k1: np.ndarray,
        yaw_k: float,
        yaw_k1: float,
        anchors: Dict[str, np.ndarray],
        swing_start: Dict[str, np.ndarray],
        swing_goal: Dict[str, np.ndarray],
        swing_locked: Dict[str, bool],
    ) -> ContactExecutionReference:
        cfg = self.cfg
        stance_legs = _stance_group_from_mode(mode)
        swing_legs = tuple(leg for leg in LEG_ORDER if leg not in stance_legs)
        base_xy_plan = (1.0 - alpha) * np.asarray(mid_k, dtype=np.float64) + alpha * np.asarray(mid_k1, dtype=np.float64)
        base_vel_xy_plan = (np.asarray(mid_k1, dtype=np.float64) - np.asarray(mid_k, dtype=np.float64)) / max(
            float(cfg.contact_phase_steps_scale) * float(cfg.phase_steps) * max(float(cfg.sim_dt), 1e-6),
            1e-6,
        )
        base_yaw = _interp_angle(float(yaw_k), float(yaw_k1), alpha)
        base_yaw_rate = np.arctan2(np.sin(float(yaw_k1) - float(yaw_k)), np.cos(float(yaw_k1) - float(yaw_k))) / max(
            float(cfg.contact_phase_steps_scale) * float(cfg.phase_steps) * max(float(cfg.sim_dt), 1e-6),
            1e-6,
        )
        support_xy = self._support_base_target(
            {leg: anchors[leg] for leg in stance_legs},
            base_yaw,
            step_width=float(cfg.step_width),
            leg_half_length=float(cfg.leg_half_length),
        )
        x_blend = float(np.clip(float(cfg.contact_base_x_support_blend), 0.0, 1.0))
        y_blend = float(np.clip(float(cfg.contact_base_y_support_blend), 0.0, 1.0))
        base_xy = np.asarray(
            [
                x_blend * float(support_xy[0]) + (1.0 - x_blend) * float(base_xy_plan[0]),
                y_blend * float(support_xy[1]) + (1.0 - y_blend) * float(base_xy_plan[1]),
            ],
            dtype=np.float64,
        )
        phase_tau = (1.0 - alpha) * float(tau0) + alpha * float(tau1)
        swing_targets: Dict[str, np.ndarray] = {}
        swing_target_vels: Dict[str, np.ndarray] = {}
        for leg in stance_legs:
            anchors[leg][2] = max(float(cfg.min_foot_z), float(anchors[leg][2]))
        for leg in swing_legs:
            if swing_locked.get(leg, False):
                swing_targets[leg] = np.asarray(anchors[leg], dtype=np.float64).copy()
                swing_target_vels[leg] = np.zeros((3,), dtype=np.float64)
                continue
            goal = self._project_swing_goal(swing_goal[leg], base_xy, base_xy_plan)
            goal[2] = max(float(cfg.min_foot_z), float(goal[2]))
            apex_z = max(
                float(cfg.min_foot_z),
                float(max(swing_start[leg][2], goal[2]) + float(cfg.swing_height) * float(cfg.contact_swing_height_scale)),
            )
            target = self._three_phase_swing_target(
                swing_start[leg],
                goal,
                phase_tau,
                apex_z=apex_z,
            )
            swing_targets[leg] = target
            vel = np.zeros((3,), dtype=np.float64)
            if phase_tau < float(cfg.lift_phase_ratio):
                vel[2] = float(cfg.landing_descent_vel_scale) * max(0.0, apex_z - float(swing_start[leg][2])) / max(
                    float(cfg.sim_dt) * max(float(cfg.phase_steps), 1.0),
                    1e-6,
                )
            elif phase_tau > float(cfg.lift_phase_ratio + cfg.advance_phase_ratio):
                vel[2] = -float(cfg.landing_descent_vel_scale) * max(0.0, apex_z - float(goal[2])) / max(
                    float(cfg.sim_dt) * max(float(cfg.phase_steps), 1.0),
                    1e-6,
                )
            else:
                vel[:2] = (goal[:2] - np.asarray(swing_start[leg][:2], dtype=np.float64)) / max(
                    float(cfg.sim_dt) * max(float(cfg.phase_steps), 1.0),
                    1e-6,
                )
            swing_target_vels[leg] = vel
        return ContactExecutionReference(
            mode=int(mode),
            stance_legs=stance_legs,
            swing_legs=swing_legs,
            base_xy=base_xy,
            base_vel_xy=np.asarray(base_vel_xy_plan, dtype=np.float64),
            base_yaw=float(base_yaw),
            base_yaw_rate=float(base_yaw_rate),
            stance_anchors={leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in stance_legs},
            swing_targets=swing_targets,
            swing_target_vels=swing_target_vels,
            phase_tau=float(phase_tau),
        )

    def _solve_contact_force_qp(
        self,
        ref: ContactExecutionReference,
        foot_target: Dict[str, np.ndarray],
        prev_foot_target: Dict[str, np.ndarray],
        q_joint_des: np.ndarray,
        next_mode: Optional[int],
    ) -> Optional[ContactQPSolution]:
        return self._hierarchical_qp.solve(
            self,
            ref=ref,
            foot_target=foot_target,
            prev_foot_target=prev_foot_target,
            q_joint_des=q_joint_des,
            next_mode=next_mode,
        )

    def _three_phase_swing_target(
        self,
        start_foot: np.ndarray,
        goal_foot: np.ndarray,
        alpha: float,
        *,
        apex_z: float,
    ) -> np.ndarray:
        cfg = self.cfg
        lift_ratio = float(np.clip(cfg.lift_phase_ratio, 0.05, 0.8))
        adv_ratio = float(np.clip(cfg.advance_phase_ratio, 0.05, 0.9))
        if lift_ratio + adv_ratio >= 0.98:
            adv_ratio = max(0.05, 0.98 - lift_ratio)
        descend_ratio = max(1e-4, 1.0 - lift_ratio - adv_ratio)
        a = float(np.clip(alpha, 0.0, 1.0))
        pos = np.asarray(start_foot, dtype=np.float64).copy()
        if a < lift_ratio:
            s = _smoothstep(a / max(lift_ratio, 1e-6))
            pos[:2] = start_foot[:2]
            pos[2] = (1.0 - s) * float(start_foot[2]) + s * float(apex_z)
            return pos
        if a < lift_ratio + adv_ratio:
            s = _smoothstep((a - lift_ratio) / max(adv_ratio, 1e-6))
            pos[:2] = (1.0 - s) * start_foot[:2] + s * goal_foot[:2]
            pos[2] = float(apex_z)
            return pos
        s = _smoothstep((a - lift_ratio - adv_ratio) / max(descend_ratio, 1e-6))
        pos[:2] = goal_foot[:2]
        pos[2] = (1.0 - s) * float(apex_z) + s * float(goal_foot[2])
        return pos

    def _support_base_target(
        self,
        anchors: Dict[str, np.ndarray],
        yaw: float,
        *,
        step_width: float,
        leg_half_length: float,
    ) -> np.ndarray:
        if not anchors:
            return np.zeros((2,), dtype=np.float64)
        cy = float(np.cos(yaw))
        sy = float(np.sin(yaw))
        est = []
        for leg, foot in anchors.items():
            sign = LEG_BASE_OFFSETS[leg]
            local = np.array(
                [float(sign[0]) * float(leg_half_length), float(sign[1]) * 0.5 * float(step_width)],
                dtype=np.float64,
            )
            world = np.array(
                [cy * local[0] - sy * local[1], sy * local[0] + cy * local[1]],
                dtype=np.float64,
            )
            est.append(np.asarray(foot[:2], dtype=np.float64) - world)
        return np.mean(np.asarray(est, dtype=np.float64), axis=0)

    def _plan_feet_to_3d(self, feet_ref: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feet3 = {}
        for leg in LEG_ORDER:
            xy = np.asarray(feet_ref[leg], dtype=np.float64)
            out = np.zeros((xy.shape[0], 3), dtype=np.float64)
            out[:, :2] = xy[:, :2]
            for i in range(xy.shape[0]):
                out[i, 2] = float(self._terrain_height_at(out[i, :2]))
            feet3[leg] = out
        return feet3

    def _contact_interval_segments(self, mode_ref: np.ndarray) -> List[Tuple[int, int, int]]:
        modes = np.asarray(mode_ref, dtype=np.int32).reshape(-1)
        if modes.size < 2:
            return []
        segs: List[Tuple[int, int, int]] = []
        start = 0
        curr = int(modes[0])
        for k in range(1, modes.size - 1):
            if int(modes[k]) != curr:
                segs.append((start, k - 1, curr))
                start = k
                curr = int(modes[k])
        segs.append((start, modes.size - 2, curr))
        return segs

    def _run_control_step(
        self,
        *,
        foot_target: Dict[str, np.ndarray],
        prev_foot_target: Dict[str, np.ndarray],
        swing_legs: Tuple[str, ...],
        stance_legs: Tuple[str, ...] = tuple(),
        stance_anchors: Optional[Dict[str, np.ndarray]] = None,
        base_xy: np.ndarray,
        base_yaw: float,
        base_vel_xy: Optional[np.ndarray] = None,
        base_yaw_rate: float = 0.0,
        contact_mode: bool = False,
        execution_ref: Optional[ContactExecutionReference] = None,
        next_mode: Optional[int] = None,
    ) -> np.ndarray:
        mujoco = self.mujoco
        cfg = self.cfg
        sim_qpos = np.array(self.data.qpos, dtype=np.float64).copy()
        sim_qvel = np.array(self.data.qvel, dtype=np.float64).copy()
        q_des = np.array(sim_qpos[self.joint_qidx_all], dtype=np.float64).copy()
        for i_leg, leg in enumerate(LEG_ORDER):
            q_leg = self._solve_leg_ik(leg, foot_target[leg])
            q_des[3 * i_leg : 3 * i_leg + 3] = q_leg
        self.data.qpos[:] = sim_qpos
        self.data.qvel[:] = sim_qvel
        mujoco.mj_forward(self.model, self.data)

        u: Optional[np.ndarray] = None
        if bool(contact_mode) and bool(cfg.contact_qp_enable) and execution_ref is not None:
            qp_sol = self._solve_contact_force_qp(execution_ref, foot_target, prev_foot_target, q_des, next_mode)
            if qp_sol is not None:
                u = np.asarray(qp_sol.tau, dtype=np.float64)
                self._last_contact_lambda_by_leg = {
                    leg: np.asarray(val, dtype=np.float64).copy()
                    for leg, val in qp_sol.lambda_by_leg.items()
                }
            else:
                self._last_contact_lambda_by_leg = {}
        else:
            self._last_contact_lambda_by_leg = {}

        if u is None:
            q = np.asarray(self.data.qpos[self.joint_qidx_all], dtype=np.float64)
            qd = np.asarray(self.data.qvel[self.joint_didx_all], dtype=np.float64)
            u = float(cfg.kp) * (q_des - q) - float(cfg.kd) * qd
        dt = max(float(cfg.sim_dt), 1e-6)
        contact_stance_legs = set(stance_legs if contact_mode else tuple())
        if execution_ref is None or u is None:
            for leg in LEG_ORDER:
                target_vel = (np.asarray(foot_target[leg], dtype=np.float64) - prev_foot_target[leg]) / dt
                in_hard_contact = leg in contact_stance_legs
                tau_task = self._task_space_leg_torque(
                    leg,
                    foot_target[leg],
                    target_vel if leg in swing_legs else np.zeros((3,), dtype=np.float64),
                    kp_xy=float(
                        cfg.swing_task_kp_xy
                        if leg in swing_legs
                        else cfg.stance_task_kp_xy * (cfg.contact_stance_kp_scale_xy if in_hard_contact else 1.0)
                    ),
                    kp_z=float(
                        cfg.swing_task_kp_z
                        if leg in swing_legs
                        else cfg.stance_task_kp_z * (cfg.contact_stance_kp_scale_z if in_hard_contact else 1.0)
                    ),
                    kd_xy=float(
                        cfg.swing_task_kd_xy
                        if leg in swing_legs
                        else cfg.stance_task_kd_xy * (cfg.contact_stance_kd_scale_xy if in_hard_contact else 1.0)
                    ),
                    kd_z=float(
                        cfg.swing_task_kd_z
                        if leg in swing_legs
                        else cfg.stance_task_kd_z * (cfg.contact_stance_kd_scale_z if in_hard_contact else 1.0)
                    ),
                )
                if in_hard_contact and stance_anchors is not None and leg in stance_anchors and self._has_support_contact(leg):
                    tau_task += self._stance_antislip_torque(leg, stance_anchors[leg])
                i_leg = LEG_ORDER.index(leg)
                u[3 * i_leg : 3 * i_leg + 3] += tau_task
        qp_active = bool(contact_mode) and bool(cfg.contact_qp_enable) and (execution_ref is not None)
        clip_limit = (
            float(cfg.control_clip) * float(cfg.contact_qp_torque_limit_scale)
            if qp_active
            else float(cfg.control_clip)
        )
        u = np.clip(u, -clip_limit, clip_limit)
        if qp_active:
            base_pd_blend = float(np.clip(float(cfg.contact_qp_base_pd_blend), 0.0, 1.0))
            if base_pd_blend > 0.0 and bool(cfg.use_base_pd):
                support_contacts = len(self._ground_contact_legs())
                self._apply_base_pd(
                    base_xy,
                    base_yaw,
                    support_contacts=support_contacts,
                    kp_x_scale=float(cfg.contact_base_kp_scale_x),
                    kd_x_scale=float(cfg.contact_base_kd_scale_x),
                    kp_y_scale=float(cfg.contact_base_kp_scale_y),
                    kd_y_scale=float(cfg.contact_base_kd_scale_y),
                    kp_z_scale=float(cfg.contact_base_kp_scale_z),
                    kd_z_scale=float(cfg.contact_base_kd_scale_z),
                    kp_rp_scale=float(cfg.contact_base_kp_scale_rp),
                    kd_rp_scale=float(cfg.contact_base_kd_scale_rp),
                    kp_yaw_scale=float(cfg.contact_base_kp_scale_yaw),
                    kd_yaw_scale=float(cfg.contact_base_kd_scale_yaw),
                    vel_xy_ref=np.asarray(
                        [
                            float((base_vel_xy[0] if base_vel_xy is not None else 0.0) * float(cfg.contact_base_vel_kd_scale_x)),
                            float((base_vel_xy[1] if base_vel_xy is not None else 0.0) * float(cfg.contact_base_vel_kd_scale_y)),
                        ],
                        dtype=np.float64,
                    ),
                    yaw_rate_ref=float(base_yaw_rate) * float(cfg.contact_base_yaw_rate_kd_scale),
                )
                self.data.qfrc_applied[:6] *= base_pd_blend
            else:
                self.data.qfrc_applied[:] = 0.0
        elif bool(cfg.use_base_pd) and bool(contact_mode):
            support_contacts = len(self._ground_contact_legs())
            self._apply_base_pd(
                base_xy,
                base_yaw,
                support_contacts=support_contacts,
                kp_x_scale=float(cfg.contact_base_kp_scale_x),
                kd_x_scale=float(cfg.contact_base_kd_scale_x),
                kp_y_scale=float(cfg.contact_base_kp_scale_y),
                kd_y_scale=float(cfg.contact_base_kd_scale_y),
                kp_z_scale=float(cfg.contact_base_kp_scale_z),
                kd_z_scale=float(cfg.contact_base_kd_scale_z),
                kp_rp_scale=float(cfg.contact_base_kp_scale_rp),
                kd_rp_scale=float(cfg.contact_base_kd_scale_rp),
                kp_yaw_scale=float(cfg.contact_base_kp_scale_yaw),
                kd_yaw_scale=float(cfg.contact_base_kd_scale_yaw),
                vel_xy_ref=np.asarray(
                    [
                        float((base_vel_xy[0] if base_vel_xy is not None else 0.0) * float(cfg.contact_base_vel_kd_scale_x)),
                        float((base_vel_xy[1] if base_vel_xy is not None else 0.0) * float(cfg.contact_base_vel_kd_scale_y)),
                    ],
                    dtype=np.float64,
                ),
                yaw_rate_ref=float(base_yaw_rate) * float(cfg.contact_base_yaw_rate_kd_scale),
            )
        elif bool(cfg.use_base_pd):
            support_contacts = len(self._ground_contact_legs())
            self._apply_base_pd(
                base_xy,
                base_yaw,
                support_contacts=support_contacts,
                vel_xy_ref=base_vel_xy,
                yaw_rate_ref=base_yaw_rate,
            )
        else:
            self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[self.actuator_ids] = u
        mujoco.mj_step(self.model, self.data)
        if bool(cfg.lock_base_pose):
            self._stabilize_base(base_xy, base_yaw)
            mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.ctrl, dtype=np.float64).copy()

    def _rollout_contact_plan(
        self,
        *,
        mid: np.ndarray,
        yaw: np.ndarray,
        feet_ref: Dict[str, np.ndarray],
        mode_ref: np.ndarray,
        tau_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mujoco = self.mujoco
        cfg = self.cfg
        feet3 = self._plan_feet_to_3d(feet_ref)
        tau_ref = (
            np.asarray(tau_ref, dtype=np.float64).reshape(-1)
            if tau_ref is not None
            else np.zeros((mid.shape[0],), dtype=np.float64)
        )

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._set_base_pose(mid[0], float(yaw[0]))
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Initialize joints to the first contact frame so the rollout starts on the 12D plan.
        for leg in LEG_ORDER:
            q_leg = self._solve_leg_ik(leg, feet3[leg][0])
            self.data.qpos[self.leg_qidx[leg]] = q_leg
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        anchors = {leg: feet3[leg][0].copy() for leg in LEG_ORDER}
        foot_target = {leg: anchors[leg].copy() for leg in LEG_ORDER}
        prev_foot_target = {leg: anchors[leg].copy() for leg in LEG_ORDER}

        qpos_hist: List[np.ndarray] = [np.array(self.data.qpos, dtype=np.float64).copy()]
        qvel_hist: List[np.ndarray] = [np.array(self.data.qvel, dtype=np.float64).copy()]
        ctrl_hist: List[np.ndarray] = []

        segs = self._contact_interval_segments(mode_ref)
        substeps = max(1, int(np.ceil(float(cfg.phase_steps) * float(cfg.contact_phase_steps_scale))))
        prev_stance_legs: Tuple[str, ...] = tuple()
        for seg_idx, (start_k, end_k, mode) in enumerate(segs):
            stance_legs = _stance_group_from_mode(mode)
            swing_legs = tuple(leg for leg in LEG_ORDER if leg not in stance_legs)
            execution_ref: Optional[ContactExecutionReference] = None
            next_mode = int(segs[seg_idx + 1][2]) if seg_idx + 1 < len(segs) else None
            new_stance_legs = tuple(leg for leg in stance_legs if leg not in prev_stance_legs)
            for leg in new_stance_legs:
                if self._has_support_contact(leg):
                    curr = np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
                    curr[2] = max(float(cfg.min_foot_z), float(curr[2]))
                    anchors[leg] = curr
            swing_start = {leg: anchors[leg].copy() for leg in swing_legs}
            swing_goal = {}
            touchdown_locked = {leg: False for leg in swing_legs}
            touchdown_counter = {leg: 0 for leg in swing_legs}
            touch_idx = end_k + 1
            for leg in swing_legs:
                goal = feet3[leg][touch_idx].copy()
                goal[2] = max(float(cfg.min_foot_z), float(goal[2]))
                swing_goal[leg] = self._project_to_safe_support(goal)

            early_switched = False
            for k in range(start_k, end_k + 1):
                tau0 = float(np.clip(tau_ref[k], 0.0, 1.0))
                tau1 = float(np.clip(tau_ref[k + 1], 0.0, 1.0))
                if int(mode_ref[k + 1]) != int(mode):
                    tau1 = 1.0
                if tau1 < tau0:
                    tau1 = tau0
                for sub in range(substeps):
                    alpha = float(sub + 1) / float(substeps)
                    execution_ref = self._build_execution_reference(
                        mode=mode,
                        alpha=alpha,
                        tau0=tau0,
                        tau1=tau1,
                        mid_k=np.asarray(mid[k], dtype=np.float64),
                        mid_k1=np.asarray(mid[k + 1], dtype=np.float64),
                        yaw_k=float(yaw[k]),
                        yaw_k1=float(yaw[k + 1]),
                        anchors=anchors,
                        swing_start=swing_start,
                        swing_goal=swing_goal,
                        swing_locked=touchdown_locked,
                    )
                    for leg in execution_ref.stance_legs:
                        foot_target[leg] = np.asarray(execution_ref.stance_anchors[leg], dtype=np.float64).copy()
                    if swing_legs:
                        stance_contacts = sum(1 for leg in stance_legs if self._has_support_contact(leg))
                        base_stable = self._base_is_stable()
                        tau = self._phase_manager.progress_tau(
                            tau0=tau0,
                            tau1=float(execution_ref.phase_tau),
                            alpha=1.0,
                            stance_legs=stance_legs,
                            stance_contacts=stance_contacts,
                            base_stable=base_stable,
                        )
                        for leg in swing_legs:
                            if touchdown_locked[leg]:
                                foot_target[leg] = anchors[leg].copy()
                                foot_target[leg][2] = max(float(cfg.min_foot_z), float(foot_target[leg][2]))
                                continue
                            foot_target[leg] = np.asarray(execution_ref.swing_targets[leg], dtype=np.float64).copy()
                    else:
                        for leg in LEG_ORDER:
                            foot_target[leg] = anchors[leg].copy()
                            foot_target[leg][2] = max(float(cfg.min_foot_z), float(foot_target[leg][2]))

                    ctrl = self._run_control_step(
                        foot_target=foot_target,
                        prev_foot_target=prev_foot_target,
                        swing_legs=swing_legs,
                        stance_legs=stance_legs,
                        stance_anchors={leg: anchors[leg] for leg in stance_legs},
                        base_xy=np.asarray(execution_ref.base_xy, dtype=np.float64),
                        base_yaw=float(execution_ref.base_yaw),
                        base_vel_xy=np.asarray(execution_ref.base_vel_xy, dtype=np.float64),
                        base_yaw_rate=float(execution_ref.base_yaw_rate),
                        contact_mode=True,
                        execution_ref=execution_ref,
                        next_mode=next_mode,
                    )
                    for leg in swing_legs:
                        if touchdown_locked[leg]:
                            continue
                        if tau < float(cfg.touchdown_alpha_min):
                            touchdown_counter[leg] = 0
                            continue
                        if self._touchdown_quality(leg, swing_goal[leg][:2]):
                            curr = np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
                            if bool(cfg.contact_touchdown_immediate_relock):
                                touchdown_counter[leg] = int(cfg.touchdown_contact_steps)
                            else:
                                touchdown_counter[leg] += 1
                            if touchdown_counter[leg] >= max(1, int(cfg.touchdown_contact_steps)):
                                curr[2] = max(float(cfg.min_foot_z), float(curr[2]))
                                anchors[leg] = curr
                                foot_target[leg] = anchors[leg].copy()
                                touchdown_locked[leg] = True
                        else:
                            touchdown_counter[leg] = 0
                    qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
                    qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
                    ctrl_hist.append(ctrl)
                    prev_foot_target = {leg: np.asarray(foot_target[leg], dtype=np.float64).copy() for leg in LEG_ORDER}
                    if self._phase_manager.should_early_switch(
                        swing_legs=swing_legs,
                        touchdown_locked=touchdown_locked,
                        base_stable=self._base_is_stable(),
                        sub=sub,
                        substeps=substeps,
                    ):
                        for _ in range(max(0, int(cfg.contact_post_touchdown_stabilize_steps))):
                            ctrl = self._run_control_step(
                                foot_target=anchors,
                                prev_foot_target=prev_foot_target,
                                swing_legs=tuple(),
                                stance_legs=tuple(sorted(set(stance_legs).union(set(swing_legs)))),
                                stance_anchors={leg: anchors[leg] for leg in set(stance_legs).union(set(swing_legs))},
                                base_xy=np.asarray(execution_ref.base_xy, dtype=np.float64),
                                base_yaw=float(execution_ref.base_yaw),
                                base_vel_xy=np.zeros((2,), dtype=np.float64),
                                base_yaw_rate=0.0,
                                contact_mode=True,
                                execution_ref=ContactExecutionReference(
                                    mode=int(mode),
                                    stance_legs=tuple(sorted(set(stance_legs).union(set(swing_legs)))),
                                    swing_legs=tuple(),
                                    base_xy=np.asarray(execution_ref.base_xy, dtype=np.float64),
                                    base_vel_xy=np.zeros((2,), dtype=np.float64),
                                    base_yaw=float(execution_ref.base_yaw),
                                    base_yaw_rate=0.0,
                                    stance_anchors={leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in set(stance_legs).union(set(swing_legs))},
                                    swing_targets={},
                                    swing_target_vels={},
                                    phase_tau=1.0,
                                ),
                                next_mode=next_mode,
                            )
                            qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
                            qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
                            ctrl_hist.append(ctrl)
                            prev_foot_target = {leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER}
                        early_switched = True
                        break
                if early_switched:
                    break

            if (
                execution_ref is not None
                and len(stance_legs) == len(LEG_ORDER)
                and next_mode is not None
            ):
                liftoff_legs = self._phase_manager.liftoff_legs(int(mode), next_mode)
                if liftoff_legs:
                    for _ in range(max(0, int(cfg.contact_qs_hold_max_steps))):
                        if not self._phase_manager.should_hold_qs(
                            mode=int(mode),
                            next_mode=next_mode,
                            lambda_by_leg=self._last_contact_lambda_by_leg,
                            base_stable=self._base_is_stable(),
                        ):
                            break
                        hold_ref = ContactExecutionReference(
                            mode=int(mode),
                            stance_legs=tuple(LEG_ORDER),
                            swing_legs=tuple(),
                            base_xy=np.asarray(execution_ref.base_xy, dtype=np.float64),
                            base_vel_xy=np.zeros((2,), dtype=np.float64),
                            base_yaw=float(execution_ref.base_yaw),
                            base_yaw_rate=0.0,
                            stance_anchors={leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER},
                            swing_targets={},
                            swing_target_vels={},
                            phase_tau=1.0,
                        )
                        ctrl = self._run_control_step(
                            foot_target={leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER},
                            prev_foot_target=prev_foot_target,
                            swing_legs=tuple(),
                            stance_legs=tuple(LEG_ORDER),
                            stance_anchors={leg: anchors[leg] for leg in LEG_ORDER},
                            base_xy=np.asarray(hold_ref.base_xy, dtype=np.float64),
                            base_yaw=float(hold_ref.base_yaw),
                            base_vel_xy=np.zeros((2,), dtype=np.float64),
                            base_yaw_rate=0.0,
                            contact_mode=True,
                            execution_ref=hold_ref,
                            next_mode=next_mode,
                        )
                        qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
                        qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
                        ctrl_hist.append(ctrl)
                        prev_foot_target = {leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER}

            for leg in swing_legs:
                if not touchdown_locked[leg]:
                    if self._has_support_contact(leg):
                        curr = np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
                        curr[2] = max(float(cfg.min_foot_z), float(curr[2]))
                        anchors[leg] = curr
                    else:
                        anchors[leg] = swing_goal[leg].copy()
            prev_stance_legs = stance_legs

        for _ in range(max(0, int(cfg.settle_steps))):
            settle_ref = ContactExecutionReference(
                mode=int(mode_ref[-1]) if len(mode_ref) > 0 else int(MODE_QS_AFTER_FL_RR),
                stance_legs=tuple(LEG_ORDER),
                swing_legs=tuple(),
                base_xy=np.asarray(mid[-1], dtype=np.float64),
                base_vel_xy=np.zeros((2,), dtype=np.float64),
                base_yaw=float(yaw[-1]),
                base_yaw_rate=0.0,
                stance_anchors={leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER},
                swing_targets={},
                swing_target_vels={},
                phase_tau=1.0,
            )
            ctrl = self._run_control_step(
                foot_target=anchors,
                prev_foot_target=prev_foot_target,
                swing_legs=tuple(),
                stance_legs=LEG_ORDER,
                stance_anchors=anchors,
                base_xy=np.asarray(mid[-1], dtype=np.float64),
                base_yaw=float(yaw[-1]),
                base_vel_xy=np.zeros((2,), dtype=np.float64),
                base_yaw_rate=0.0,
                contact_mode=True,
                execution_ref=settle_ref,
                next_mode=None,
            )
            qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
            qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
            ctrl_hist.append(ctrl)
            prev_foot_target = {leg: np.asarray(anchors[leg], dtype=np.float64).copy() for leg in LEG_ORDER}

        return (
            np.asarray(qpos_hist, dtype=np.float32),
            np.asarray(qvel_hist, dtype=np.float32),
            np.asarray(ctrl_hist, dtype=np.float32),
        )

    def _solve_leg_ik(self, leg: str, target_world: np.ndarray) -> np.ndarray:
        mujoco = self.mujoco
        qidx = self.leg_qidx[leg]
        didx = self.leg_didx[leg]
        q = np.array(self.data.qpos[qidx], copy=True, dtype=np.float64)
        jmin = self.leg_jmin[leg]
        jmax = self.leg_jmax[leg]
        gid = self.leg_geom_id[leg]
        target = np.asarray(target_world, dtype=np.float64).copy()
        target[2] = max(float(self.cfg.min_foot_z), float(target[2]))

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(int(self.cfg.ik_iters)):
            curr = np.asarray(self.data.geom_xpos[gid], dtype=np.float64).copy()
            err = target - curr
            if float(np.linalg.norm(err)) <= float(self.cfg.ik_tol):
                break

            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, gid)
            J = jacp[:, didx]
            JJt = J @ J.T
            step = np.linalg.solve(
                JJt + float(self.cfg.ik_damping) * np.eye(3, dtype=np.float64),
                err,
            )
            dq = J.T @ step
            q = np.clip(q + dq, jmin, jmax)
            self.data.qpos[qidx] = q
            mujoco.mj_forward(self.model, self.data)

        return q.astype(np.float64)

    def rollout(
        self,
        *,
        mid: np.ndarray,
        yaw: np.ndarray,
        step_width: Optional[float] = None,
        leg_half_length: Optional[float] = None,
        feet_ref: Optional[Dict[str, np.ndarray]] = None,
        mode_ref: Optional[np.ndarray] = None,
        tau_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rollout gait tracking foot placements derived from (mid, yaw).
        """
        mujoco = self.mujoco
        cfg = self.cfg
        step_width = float(step_width if step_width is not None else cfg.step_width)
        leg_half_length = float(leg_half_length if leg_half_length is not None else cfg.leg_half_length)

        mid = np.asarray(mid, dtype=np.float64)
        yaw = _wrap(np.asarray(yaw, dtype=np.float64).reshape(-1))
        if mid.ndim != 2 or mid.shape[0] < 2:
            raise ValueError("mid must be (T,2) with T>=2")
        if yaw.shape[0] != mid.shape[0]:
            raise ValueError("yaw length mismatch")

        if mode_ref is not None and feet_ref is not None:
            return self._rollout_contact_plan(
                mid=mid,
                yaw=yaw,
                feet_ref=feet_ref,
                mode_ref=mode_ref,
                tau_ref=tau_ref,
            )

        # Prefer exact per-foot references from the planner when available.
        if feet_ref is None:
            states3 = np.concatenate([mid, yaw[:, None]], axis=1)
            p_l, p_r, _ = states_to_lr_pairs(states3, step_width=step_width)
            lf_ref, rf_ref, lh_ref, rh_ref = pair_to_virtual_feet(
                p_l,
                p_r,
                yaw.astype(np.float32),
                half_pair_length=leg_half_length,
            )
            feet_ref = {"FL": lf_ref, "FR": rf_ref, "RL": lh_ref, "RR": rh_ref}
        else:
            feet_ref = {
                leg: np.asarray(feet_ref[leg], dtype=np.float64).reshape(mid.shape[0], -1)
                for leg in LEG_ORDER
            }
        if mode_ref is not None:
            mode_ref = np.asarray(mode_ref, dtype=np.int32).reshape(-1)
            if mode_ref.shape[0] != mid.shape[0]:
                raise ValueError("mode_ref length mismatch")

        # Reset model.
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._set_base_pose(mid[0], float(yaw[0]))
        self.data.qvel[:6] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Current world targets for each leg (start at current COM points).
        foot_target = {
            leg: np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
            for leg in LEG_ORDER
        }
        prev_foot_target = {leg: foot_target[leg].copy() for leg in LEG_ORDER}

        qpos_hist: List[np.ndarray] = [np.array(self.data.qpos, dtype=np.float64).copy()]
        qvel_hist: List[np.ndarray] = [np.array(self.data.qvel, dtype=np.float64).copy()]
        ctrl_hist: List[np.ndarray] = []

        gait = str(cfg.gait).strip().lower()
        swing_groups = TROT_SWING_GROUPS if gait == "trot" else WALK_SWING_GROUPS

        nseg = mid.shape[0] - 1
        for k in range(nseg):
            swing_legs = (
                _swing_group_from_mode(int(mode_ref[k]))
                if mode_ref is not None
                else swing_groups[k % len(swing_groups)]
            )
            stance_legs = tuple(leg for leg in LEG_ORDER if leg not in swing_legs)
            start_foots = {leg: foot_target[leg].copy() for leg in swing_legs}
            goal_foots = {}
            for leg in swing_legs:
                goal_xy = np.asarray(feet_ref[leg][k + 1][:2], dtype=np.float64)
                goal_z = float(self._terrain_height_at(goal_xy))
                goal_foots[leg] = np.array([goal_xy[0], goal_xy[1], goal_z], dtype=np.float64)

            stance_anchor = {}
            for leg in stance_legs:
                anchor = foot_target[leg].copy()
                if self._has_world_contact(leg):
                    anchor = np.asarray(self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64).copy()
                anchor[2] = float(self._terrain_height_at(anchor[:2]))
                foot_target[leg] = anchor.copy()
                stance_anchor[leg] = anchor

            touchdown = {leg: False for leg in swing_legs}
            touchdown_counter = {leg: 0 for leg in swing_legs}
            released = {leg: False for leg in swing_legs}
            release_counter = {leg: 0 for leg in swing_legs}
            progress_count = 0
            max_phase_iters = max(
                int(cfg.phase_steps),
                int(np.ceil(float(cfg.phase_steps) * float(cfg.phase_extend_factor))),
            )
            for _ in range(max_phase_iters):
                stance_contacts = sum(1 for leg in stance_legs if self._has_support_contact(leg))
                release_done = all(released.values()) if swing_legs else True
                if (
                    release_done
                    and progress_count < int(cfg.phase_steps)
                    and stance_contacts >= min(len(stance_legs), int(cfg.stance_contact_min))
                ):
                    progress_count += 1
                alpha = float(progress_count) / float(max(1, int(cfg.phase_steps)))
                base_yaw = _interp_angle(float(yaw[k]), float(yaw[k + 1]), alpha)

                # Stance legs hold support anchors; swing legs move only while support is stable.
                for leg in stance_legs:
                    if self._has_support_contact(leg):
                        stance_anchor[leg][2] = max(
                            float(cfg.min_foot_z),
                            float(self.data.geom_xpos[self.leg_geom_id[leg]][2]),
                        )
                    else:
                        stance_anchor[leg][2] = float(self._terrain_height_at(stance_anchor[leg][:2]))
                    foot_target[leg] = stance_anchor[leg].copy()

                base_xy_plan = (1.0 - alpha) * mid[k] + alpha * mid[k + 1]
                base_xy_support = self._support_base_target(
                    stance_anchor,
                    base_yaw,
                    step_width=step_width,
                    leg_half_length=leg_half_length,
                )
                blend = float(np.clip(float(cfg.support_base_blend), 0.0, 1.0))
                base_xy = blend * base_xy_support + (1.0 - blend) * base_xy_plan

                for leg in swing_legs:
                    if touchdown[leg]:
                        foot_target[leg] = goal_foots[leg].copy()
                        continue
                    if not released[leg]:
                        start_foot = start_foots[leg]
                        release_z = max(
                            float(cfg.min_foot_z),
                            float(self._terrain_height_at(start_foot[:2]) + float(cfg.swing_release_height)),
                        )
                        foot_target[leg][:2] = start_foot[:2]
                        foot_target[leg][2] = release_z
                        continue
                    start_foot = start_foots[leg]
                    goal_foot = goal_foots[leg]
                    apex_z = max(
                        float(cfg.min_foot_z),
                        float(max(start_foot[2], goal_foot[2]) + float(cfg.swing_height)),
                    )
                    foot_target[leg] = self._three_phase_swing_target(
                        start_foot,
                        goal_foot,
                        alpha,
                        apex_z=apex_z,
                    )

                # IK for all legs, then joint PD control.
                sim_qpos = np.array(self.data.qpos, dtype=np.float64).copy()
                sim_qvel = np.array(self.data.qvel, dtype=np.float64).copy()
                q_des = np.array(sim_qpos[self.joint_qidx_all], dtype=np.float64).copy()
                for i_leg, leg in enumerate(LEG_ORDER):
                    q_leg = self._solve_leg_ik(leg, foot_target[leg])
                    q_des[3 * i_leg : 3 * i_leg + 3] = q_leg
                self.data.qpos[:] = sim_qpos
                self.data.qvel[:] = sim_qvel
                mujoco.mj_forward(self.model, self.data)

                q = np.asarray(self.data.qpos[self.joint_qidx_all], dtype=np.float64)
                qd = np.asarray(self.data.qvel[self.joint_didx_all], dtype=np.float64)
                u = float(cfg.kp) * (q_des - q) - float(cfg.kd) * qd
                dt = max(float(cfg.sim_dt), 1e-6)
                for leg in LEG_ORDER:
                    target_vel = (np.asarray(foot_target[leg], dtype=np.float64) - prev_foot_target[leg]) / dt
                    tau_task = self._task_space_leg_torque(
                        leg,
                        foot_target[leg],
                        target_vel if leg in swing_legs else np.zeros((3,), dtype=np.float64),
                        kp_xy=float(cfg.swing_task_kp_xy if leg in swing_legs else cfg.stance_task_kp_xy),
                        kp_z=float(cfg.swing_task_kp_z if leg in swing_legs else cfg.stance_task_kp_z),
                        kd_xy=float(cfg.swing_task_kd_xy if leg in swing_legs else cfg.stance_task_kd_xy),
                        kd_z=float(cfg.swing_task_kd_z if leg in swing_legs else cfg.stance_task_kd_z),
                    )
                    i_leg = LEG_ORDER.index(leg)
                    u[3 * i_leg : 3 * i_leg + 3] += tau_task
                u = np.clip(u, -float(cfg.control_clip), float(cfg.control_clip))

                if bool(cfg.use_base_pd):
                    support_contacts = len(self._ground_contact_legs())
                    self._apply_base_pd(base_xy, base_yaw, support_contacts=support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0
                self.data.ctrl[self.actuator_ids] = u
                mujoco.mj_step(self.model, self.data)
                if bool(cfg.lock_base_pose):
                    self._stabilize_base(base_xy, base_yaw)
                    mujoco.mj_forward(self.model, self.data)
                for leg in swing_legs:
                    if released[leg]:
                        continue
                    if not self._has_support_contact(leg):
                        release_counter[leg] += 1
                    else:
                        release_counter[leg] = 0
                    if release_counter[leg] >= int(cfg.release_contact_steps):
                        released[leg] = True
                if alpha >= float(cfg.touchdown_alpha_min):
                    for leg in swing_legs:
                        if not released[leg]:
                            touchdown_counter[leg] = 0
                            continue
                        if self._has_support_contact(leg):
                            curr_xy = np.asarray(
                                self.data.geom_xpos[self.leg_geom_id[leg]][:2], dtype=np.float64
                            )
                            if np.linalg.norm(curr_xy - goal_foots[leg][:2]) <= float(cfg.touchdown_xy_tol):
                                touchdown_counter[leg] += 1
                            else:
                                touchdown_counter[leg] = 0
                        else:
                            touchdown_counter[leg] = 0
                        if touchdown_counter[leg] >= int(cfg.touchdown_contact_steps):
                            touchdown[leg] = True
                            foot_target[leg] = np.asarray(
                                self.data.geom_xpos[self.leg_geom_id[leg]], dtype=np.float64
                            ).copy()
                            foot_target[leg][2] = max(float(cfg.min_foot_z), float(foot_target[leg][2]))
                qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
                qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
                ctrl_hist.append(np.array(self.data.ctrl, dtype=np.float64).copy())
                prev_foot_target = {leg: np.asarray(foot_target[leg], dtype=np.float64).copy() for leg in LEG_ORDER}
                if progress_count >= int(cfg.phase_steps) and all(touchdown.values()):
                    break

            for leg in swing_legs:
                if not touchdown[leg]:
                    foot_target[leg] = goal_foots[leg].copy()

            for _ in range(max(0, int(cfg.settle_steps))):
                sim_qpos = np.array(self.data.qpos, dtype=np.float64).copy()
                sim_qvel = np.array(self.data.qvel, dtype=np.float64).copy()
                q_des = np.array(sim_qpos[self.joint_qidx_all], dtype=np.float64).copy()
                for i_leg, leg in enumerate(LEG_ORDER):
                    q_leg = self._solve_leg_ik(leg, foot_target[leg])
                    q_des[3 * i_leg : 3 * i_leg + 3] = q_leg
                self.data.qpos[:] = sim_qpos
                self.data.qvel[:] = sim_qvel
                mujoco.mj_forward(self.model, self.data)

                q = np.asarray(self.data.qpos[self.joint_qidx_all], dtype=np.float64)
                qd = np.asarray(self.data.qvel[self.joint_didx_all], dtype=np.float64)
                u = float(cfg.kp) * (q_des - q) - float(cfg.kd) * qd
                u = np.clip(u, -float(cfg.control_clip), float(cfg.control_clip))

                if bool(cfg.use_base_pd):
                    support_contacts = len(self._ground_contact_legs())
                    self._apply_base_pd(mid[k + 1], float(yaw[k + 1]), support_contacts=support_contacts)
                else:
                    self.data.qfrc_applied[:] = 0.0
                self.data.ctrl[self.actuator_ids] = u
                mujoco.mj_step(self.model, self.data)
                if bool(cfg.lock_base_pose):
                    self._stabilize_base(mid[k + 1], float(yaw[k + 1]))
                    mujoco.mj_forward(self.model, self.data)
                qpos_hist.append(np.array(self.data.qpos, dtype=np.float64).copy())
                qvel_hist.append(np.array(self.data.qvel, dtype=np.float64).copy())
                ctrl_hist.append(np.array(self.data.ctrl, dtype=np.float64).copy())

        return (
            np.asarray(qpos_hist, dtype=np.float32),
            np.asarray(qvel_hist, dtype=np.float32),
            np.asarray(ctrl_hist, dtype=np.float32),
        )


def load_midline_plan_from_seed_dir(
    seed_dir: str | Path,
    *,
    step_width: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    plan = load_stepping_plan_from_seed_dir(seed_dir, step_width=step_width)
    return plan["mid"].astype(np.float32), plan["yaw"].astype(np.float32)


def load_stepping_plan_from_seed_dir(
    seed_dir: str | Path,
    *,
    step_width: float = 0.30,
) -> Dict[str, Any]:
    import json

    seed_path = Path(seed_dir)
    traj_path = seed_path / "trajectory" / "trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    with open(traj_path, "r", encoding="utf-8") as f:
        blob = json.load(f) or {}
    cand = blob.get("candidate_states") or []
    if not cand:
        raise ValueError(f"No candidate_states in {traj_path}")
    idx = int(blob.get("best_idx", 0))
    idx = max(0, min(idx, len(cand) - 1))
    st = np.asarray(cand[idx], dtype=np.float32)
    if st.ndim != 2 or st.shape[1] < 3:
        raise ValueError(f"Unexpected state shape: {st.shape}")

    mid, yaw, feet, mode = decode_plan_states(
        st,
        step_width=step_width,
        half_pair_length=0.18,
    )
    tau = st[:, 11].astype(np.float32) if st.shape[1] >= 12 else np.zeros((st.shape[0],), dtype=np.float32)
    return {
        "states": st.astype(np.float32),
        "mid": mid.astype(np.float32),
        "yaw": yaw.astype(np.float32),
        "feet": {leg: np.asarray(feet[leg], dtype=np.float32) for leg in LEG_ORDER},
        "mode": np.asarray(mode, dtype=np.int32),
        "tau": tau,
    }
