"""Assemble the WBC QP and recover joint torques from its solution.

This module is the **only** place that knows the layout of the decision
vector ``x = [ddq; λ]``. It exposes:

* :func:`build_objective_rows` — build the stacked least-squares ``(A, b)``
  rows from the task list, the regularizers on ``ddq`` and ``λ``, and the
  desired contact-normal target. Internally it consults
  :mod:`task_stack` for per-task ``(J, a_des)`` pairs.
* :func:`assemble_qp` — assemble all rows from the contact, objective and
  inequality systems into the form expected by
  :class:`QPSolverAdapter`.
* :func:`recover_torques` — given a QP solution ``x``, project back to
  actuator torques via the floating-base dynamics.

Keeping this layout-knowledge in one file means swapping in a new task
(e.g. an angular momentum task) is a one-place change: add the new ``add``
call in :func:`build_objective_rows` and update the
:class:`TaskWeightsConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import (
    LimitsConfig,
    TaskGainsConfig,
    TaskWeightsConfig,
    WBCConfig,
)
from genedynamics.deploy.controllers.wbc.contact_blocks import (
    SupportContactBlock,
    build_contact_equality_rows,
    build_floating_base_dynamics_rows,
)
from genedynamics.deploy.controllers.wbc.friction_cone import build_inequality_rows
from genedynamics.deploy.controllers.wbc.task_stack import (
    com_task,
    joint_posture_task,
    joint_selector,
    orientation_task,
    site_linear_task,
    support_phase_scales,
    through_gap_scales,
)
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec
from genedynamics.deploy.io.mujoco_io import MujocoRobotIO
from genedynamics.robots.g1 import G1RobotSpec

__all__ = [
    "QPProblem",
    "build_objective_rows",
    "assemble_qp",
    "recover_torques",
]


@dataclass
class QPProblem:
    """Materialized QP problem ``min ½xᵀHx + fᵀx s.t. Cx=d, Gx≤h``.

    Carries the per-task error norms and contact bookkeeping needed by the
    controller to populate diagnostic fields in :class:`JointTargets`.
    """

    H: np.ndarray
    f: np.ndarray
    C_eq: np.ndarray
    d_eq: np.ndarray
    G_ineq: np.ndarray
    h_ineq: np.ndarray
    nv: int
    n_lambda: int
    blocks: List[SupportContactBlock]
    task_errors: Dict[str, float]
    support_jacobian: np.ndarray
    support_acc_cmd: np.ndarray


# ---------------------------------------------------------------------------
# Objective construction
# ---------------------------------------------------------------------------


def build_objective_rows(
    io: MujocoRobotIO,
    tasks: HumanoidTaskSpec,
    qvel_full: np.ndarray,
    q_act: np.ndarray,
    qd_act: np.ndarray,
    q_hint: np.ndarray,
    lower_body_target_q: np.ndarray,
    blocks: List[SupportContactBlock],
    *,
    weights: TaskWeightsConfig,
    gains: TaskGainsConfig,
    limits: LimitsConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build the stacked ``(A, b)`` system for the WBC objective.

    Each task contributes a block ``√w · J`` (rows in ``A``) and ``√w ·
    a_des`` (rows in ``b``); the QP then minimizes ``½‖A·x − b‖²``.
    """
    spec: G1RobotSpec = io.spec
    nv = io.model.nv
    n_lambda = 6 * len(blocks)
    rows: List[np.ndarray] = []
    rhs: List[np.ndarray] = []
    errors: Dict[str, float] = {}

    narrowness = float(
        np.clip(
            tasks.extras.get("gap_severity", tasks.extras.get("narrowness", 0.0)),
            0.0,
            1.0,
        )
    )
    gap = through_gap_scales(narrowness, weights)
    support = support_phase_scales(len(blocks), weights)

    def add(weight: float, J: np.ndarray, target: np.ndarray, name: str) -> None:
        if J.size == 0:
            return
        sw = float(np.sqrt(max(weight, 1e-8)))
        row = np.zeros((J.shape[0], nv + n_lambda), dtype=np.float64)
        row[:, :nv] = sw * J
        rows.append(row)
        rhs.append(sw * target)
        errors[name] = float(np.linalg.norm(target))

    # CoM task
    J_com, a_com = com_task(io, tasks, qvel_full, gains)
    add(weights.com * gap["com"] * support["com"], J_com, a_com, "com")

    # Pelvis orientation
    J_pelvis, a_pelvis = orientation_task(
        io,
        site_name="imu_in_pelvis",
        target_rpy=(tasks.pelvis.roll_world, tasks.pelvis.pitch_world, tasks.pelvis.yaw_world),
        target_angular_velocity=np.asarray(tasks.pelvis.angular_velocity_world, dtype=np.float64),
        qvel_full=qvel_full,
        kp=gains.pelvis_orientation_kp,
        kd=gains.pelvis_orientation_kd,
    )
    add(
        weights.pelvis * gap["pelvis"] * support["pelvis"],
        J_pelvis,
        a_pelvis,
        "pelvis_orientation",
    )

    # Torso orientation (decoupled yaw)
    torso_yaw_world = float(tasks.pelvis.yaw_world + tasks.torso_yaw)
    J_torso, a_torso = orientation_task(
        io,
        site_name="imu_in_torso",
        target_rpy=(tasks.pelvis.roll_world, tasks.pelvis.pitch_world, torso_yaw_world),
        target_angular_velocity=np.asarray(
            [0.0, 0.0, tasks.plan_frame.omega + tasks.plan_frame.psi_dot_torso],
            dtype=np.float64,
        ),
        qvel_full=qvel_full,
        kp=gains.torso_orientation_kp,
        kd=gains.torso_orientation_kd,
    )
    add(
        weights.torso * gap["torso"] * support["torso"],
        J_torso,
        a_torso,
        "torso_orientation",
    )

    # Swing-foot tasks (per non-contact foot)
    for foot_name, foot_task in (
        ("left_foot", tasks.left_foot),
        ("right_foot", tasks.right_foot),
    ):
        if foot_task.in_contact:
            continue
        J_lin, a_lin = site_linear_task(
            io,
            site_name=foot_name,
            target_position=np.asarray(foot_task.position_world, dtype=np.float64),
            target_velocity=np.asarray(foot_task.velocity_world, dtype=np.float64),
            qvel_full=qvel_full,
            kp=gains.swing_foot_position_kp,
            kd=gains.swing_foot_position_kd,
        )
        add(
            weights.swing_foot * foot_task.weight * gap["swing"] * support["swing"],
            J_lin,
            a_lin,
            f"{foot_name}_swing",
        )

        J_rot, a_rot = orientation_task(
            io,
            site_name=foot_name,
            target_rpy=(foot_task.roll_world, foot_task.pitch_world, foot_task.yaw_world),
            target_angular_velocity=np.asarray(
                foot_task.angular_velocity_world, dtype=np.float64
            ),
            qvel_full=qvel_full,
            kp=gains.swing_foot_orientation_kp,
            kd=gains.swing_foot_orientation_kd,
        )
        add(
            weights.swing_foot_orientation * foot_task.weight * gap["swing"] * support["swing"],
            J_rot,
            a_rot,
            f"{foot_name}_orientation",
        )

    # Lower-body posture
    leg_joints = tuple(spec.left_leg_joints) + tuple(spec.right_leg_joints)
    add(
        weights.lower_body_posture * gap["lower_body"] * support["lower_body"],
        joint_selector(spec, leg_joints, nv),
        joint_posture_task(
            spec,
            lower_body_target_q,
            q_act,
            qd_act,
            kp=gains.posture_kp,
            kd=gains.posture_kd,
            joints=leg_joints,
        ),
        "lower_body_posture",
    )

    # Waist posture
    add(
        weights.waist_posture * gap["waist"] * support["waist"],
        joint_selector(spec, spec.waist_joints, nv),
        joint_posture_task(
            spec,
            q_hint,
            q_act,
            qd_act,
            kp=gains.waist_kp,
            kd=gains.waist_kd,
            joints=spec.waist_joints,
        ),
        "waist_posture",
    )

    # Arm posture
    arm_joints = tuple(spec.left_arm_joints) + tuple(spec.right_arm_joints)
    add(
        weights.arm_posture * gap["arm"] * support["arm"],
        joint_selector(spec, arm_joints, nv),
        joint_posture_task(
            spec,
            q_hint,
            q_act,
            qd_act,
            kp=gains.arm_kp,
            kd=gains.arm_kd,
            joints=arm_joints,
        ),
        "arm_posture",
    )

    # Whole-body posture (light regularizer)
    add(
        weights.posture,
        joint_selector(spec, spec.actuated_joints, nv),
        joint_posture_task(
            spec,
            spec.stand_ctrl,
            q_act,
            qd_act,
            kp=gains.posture_kp,
            kd=gains.posture_kd,
        ),
        "posture",
    )

    # Tikhonov regularization on ddq
    row_ddq = np.zeros((nv, nv + n_lambda), dtype=np.float64)
    row_ddq[:, :nv] = float(np.sqrt(limits.ddq_weight)) * np.eye(nv, dtype=np.float64)
    rows.append(row_ddq)
    rhs.append(np.zeros((nv,), dtype=np.float64))

    # Lambda regularization + contact-normal target
    if n_lambda > 0:
        # Diagonal Tikhonov on lambda
        row_lambda = np.zeros((n_lambda, nv + n_lambda), dtype=np.float64)
        row_lambda[:, nv:] = float(np.sqrt(limits.lambda_weight)) * np.eye(
            n_lambda, dtype=np.float64
        )
        rows.append(row_lambda)
        rhs.append(np.zeros((n_lambda,), dtype=np.float64))

        # Per-axis weighted target on lambda (encourages a healthy normal load)
        target = np.zeros((n_lambda,), dtype=np.float64)
        for i in range(len(blocks)):
            target[6 * i + 2] = float(limits.lambda_normal_target)
        diag = np.empty(n_lambda, dtype=np.float64)
        for i in range(len(blocks)):
            base = 6 * i
            diag[base + 0] = limits.lambda_tangent_weight
            diag[base + 1] = limits.lambda_tangent_weight
            diag[base + 2] = limits.lambda_normal_weight
            diag[base + 3] = limits.lambda_moment_weight
            diag[base + 4] = limits.lambda_moment_weight
            diag[base + 5] = limits.lambda_moment_weight
        row_target = np.zeros((n_lambda, nv + n_lambda), dtype=np.float64)
        row_target[:, nv:] = np.diag(diag)
        rows.append(row_target)
        rhs.append(row_target[:, nv:] @ target)

    A = np.vstack(rows) if rows else np.zeros((0, nv + n_lambda), dtype=np.float64)
    b = np.concatenate(rhs, axis=0) if rhs else np.zeros((0,), dtype=np.float64)
    errors["narrowness"] = narrowness
    errors["gap_severity"] = float(
        np.clip(tasks.extras.get("gap_severity", narrowness), 0.0, 1.0)
    )
    return A, b, errors


# ---------------------------------------------------------------------------
# Full QP assembly
# ---------------------------------------------------------------------------


def assemble_qp(
    io: MujocoRobotIO,
    tasks: HumanoidTaskSpec,
    *,
    M: np.ndarray,
    bias: np.ndarray,
    qvel_full: np.ndarray,
    q_act: np.ndarray,
    qd_act: np.ndarray,
    q_hint: np.ndarray,
    lower_body_target_q: np.ndarray,
    cfg: WBCConfig,
) -> QPProblem:
    """One-call assembly of every QP matrix the WBC needs."""
    spec: G1RobotSpec = io.spec
    nv = io.model.nv

    blocks = _build_blocks_via_io(io, tasks, qvel_full, gains=cfg.gains, limits=cfg.limits)
    n_lambda = 6 * len(blocks)

    # Equality system
    dyn_C, dyn_d = build_floating_base_dynamics_rows(M, bias, blocks, nv=nv)
    if blocks:
        contact_C_local, contact_d = build_contact_equality_rows(blocks, nv=nv)
        contact_C = np.zeros((contact_C_local.shape[0], nv + n_lambda), dtype=np.float64)
        contact_C[:, :nv] = contact_C_local[:, :nv]
        C_eq = np.vstack([dyn_C, contact_C])
        d_eq = np.concatenate([dyn_d, contact_d], axis=0)
    else:
        C_eq = dyn_C
        d_eq = dyn_d

    # Objective
    A_obj, b_obj, errors = build_objective_rows(
        io,
        tasks,
        qvel_full,
        q_act,
        qd_act,
        q_hint,
        lower_body_target_q,
        blocks,
        weights=cfg.weights,
        gains=cfg.gains,
        limits=cfg.limits,
    )

    # Inequality
    G_ineq, h_ineq = build_inequality_rows(
        spec, M, bias, blocks, nv=nv, limits=cfg.limits
    )

    # H, f from least-squares system
    total_dim = nv + n_lambda
    H = A_obj.T @ A_obj + cfg.limits.qp_regularization * np.eye(total_dim, dtype=np.float64)
    f = -(A_obj.T @ b_obj)

    support_jacobian = (
        np.vstack([block.J for block in blocks])
        if blocks
        else np.zeros((0, nv), dtype=np.float64)
    )
    support_acc_cmd = (
        np.concatenate([block.a_des for block in blocks], axis=0)
        if blocks
        else np.zeros((0,), dtype=np.float64)
    )

    return QPProblem(
        H=H,
        f=f,
        C_eq=C_eq,
        d_eq=d_eq,
        G_ineq=G_ineq,
        h_ineq=h_ineq,
        nv=nv,
        n_lambda=n_lambda,
        blocks=blocks,
        task_errors=errors,
        support_jacobian=support_jacobian,
        support_acc_cmd=support_acc_cmd,
    )


def _build_blocks_via_io(
    io: MujocoRobotIO,
    tasks: HumanoidTaskSpec,
    qvel_full: np.ndarray,
    *,
    gains: TaskGainsConfig,
    limits: LimitsConfig,
) -> List[SupportContactBlock]:
    """Local re-export of :func:`build_support_blocks` to avoid a cyclic import.

    ``contact_blocks.py`` and ``qp_builder.py`` are intentionally split, but
    the builder needs to call back through the IO. We thread that here so
    each module imports only from "lower" layers.
    """
    from genedynamics.deploy.controllers.wbc.contact_blocks import build_support_blocks

    return build_support_blocks(io, tasks, qvel_full, gains=gains, limits=limits)


# ---------------------------------------------------------------------------
# Torque recovery
# ---------------------------------------------------------------------------


def recover_torques(
    spec: G1RobotSpec,
    M: np.ndarray,
    bias: np.ndarray,
    ddq_full: np.ndarray,
    support_jacobian: np.ndarray,
    lambda_ref: np.ndarray,
) -> np.ndarray:
    """Recover actuator torques from a solved ``ddq, λ`` pair.

    Floating-base inverse dynamics::

        τ_full = M · ddq + bias − Σᵢ Jcᵢᵀ · λᵢ

    Returns the actuated subset only.
    """
    tau_full = M @ ddq_full + bias
    if lambda_ref.size > 0 and support_jacobian.size > 0:
        tau_full = tau_full - support_jacobian.T @ lambda_ref
    return np.asarray(tau_full[spec.actuated_dof_indices], dtype=np.float64).copy()
