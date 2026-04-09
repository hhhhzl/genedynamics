"""Per-task Jacobians and desired-acceleration commands for the WBC.

This module exposes the small set of pure functions that turn a follower's
:class:`HumanoidTaskSpec` plus the current physics state into individual
``(J, a_des)`` rows of the WBC objective. Each row encodes one task::

    minimize ||J · ddq − a_des||²

Centralizing the per-task math here keeps :mod:`qp_builder` free of any
controller-specific bookkeeping; the QP builder just stacks rows weighted
by :class:`TaskWeightsConfig`.

Tasks provided:

* :func:`com_task`
* :func:`orientation_task`
* :func:`site_linear_task`
* :func:`joint_posture_task`

Helper utilities:

* :func:`joint_selector`        — selector matrix for a subset of actuated joints
* :func:`through_gap_scales`    — narrow-gap weight multipliers
* :func:`support_phase_scales`  — single-support weight multipliers
* :func:`lower_body_target`     — height-aware crouch + lateral bias for legs
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import (
    LimitsConfig,
    TaskGainsConfig,
    TaskWeightsConfig,
)
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec
from genedynamics.deploy.io.mujoco_io import MujocoRobotIO
from genedynamics.robots.g1 import G1RobotSpec

__all__ = [
    "com_task",
    "orientation_task",
    "site_linear_task",
    "joint_posture_task",
    "joint_selector",
    "through_gap_scales",
    "support_phase_scales",
    "lower_body_target",
    "rotation_matrix_from_rpy",
    "orientation_axis_error",
]


# ---------------------------------------------------------------------------
# CoM task
# ---------------------------------------------------------------------------


def com_task(
    io: MujocoRobotIO,
    tasks: HumanoidTaskSpec,
    qvel_full: np.ndarray,
    gains: TaskGainsConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """CoM tracking task in world frame.

    Returns ``(J, a_des)`` where ``J`` is the subtree-CoM Jacobian rooted
    at the pelvis body and ``a_des`` is the desired CoM acceleration::

        a_des = Kp · (com_target − com_current) + Kd · (vel_target − vel_current)

    The vertical velocity target is taken from the plan frame's ``h_dot``
    so the controller follows planned crouch/rise profiles smoothly.
    """
    if "pelvis" not in io.spec.body_id:
        return np.zeros((0, io.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    jacp, com = io.subtree_com_jacobian("pelvis")
    desired = np.asarray(tasks.pelvis.position_world, dtype=np.float64).copy()
    target_vel = np.asarray(tasks.pelvis.linear_velocity_world, dtype=np.float64).copy()
    target_vel[2] = float(tasks.plan_frame.h_dot)
    vel = jacp @ qvel_full
    a_des = float(gains.com_kp) * (desired - com) + float(gains.com_kd) * (target_vel - vel)
    return jacp, a_des


# ---------------------------------------------------------------------------
# Orientation tasks (pelvis, torso, swing foot)
# ---------------------------------------------------------------------------


def orientation_task(
    io: MujocoRobotIO,
    site_name: str,
    target_rpy: Tuple[float, float, float],
    target_angular_velocity: np.ndarray,
    qvel_full: np.ndarray,
    *,
    kp: float,
    kd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orientation tracking for a named MuJoCo site.

    Uses the column-error formulation::

        e = 0.5 · Σ (R_current[:, i] × R_target[:, i])

    which is well-conditioned for small/moderate angles and avoids the
    quaternion sign ambiguity.
    """
    if site_name not in io.spec.site_id:
        return np.zeros((0, io.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    _, jacr = io.site_jacobian(site_name)
    _, R_current = io.site_pose(site_name)
    R_target = rotation_matrix_from_rpy(*target_rpy)
    err = orientation_axis_error(R_target, R_current)
    ang_vel = jacr @ qvel_full
    a_des = float(kp) * err + float(kd) * (
        np.asarray(target_angular_velocity, dtype=np.float64) - ang_vel
    )
    return jacr, a_des


def site_linear_task(
    io: MujocoRobotIO,
    site_name: str,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    qvel_full: np.ndarray,
    *,
    kp: float,
    kd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Position tracking for a named MuJoCo site (used for swing foot)."""
    if site_name not in io.spec.site_id:
        return np.zeros((0, io.model.nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    jacp, _ = io.site_jacobian(site_name)
    pos, _ = io.site_pose(site_name)
    err = np.asarray(target_position, dtype=np.float64).reshape(3) - pos
    vel = jacp @ qvel_full
    a_des = float(kp) * err + float(kd) * (
        np.asarray(target_velocity, dtype=np.float64) - vel
    )
    return jacp, a_des


# ---------------------------------------------------------------------------
# Joint-space posture tasks
# ---------------------------------------------------------------------------


def joint_selector(
    spec: G1RobotSpec,
    joint_names: Sequence[str],
    nv: int,
) -> np.ndarray:
    """Selector matrix mapping ``ddq`` (length nv) to a subset of actuated DoFs.

    Rows are ordered like ``joint_names``; missing joints are silently
    skipped (matches the legacy behavior).
    """
    valid = [name for name in joint_names if name in spec.actuated_joints]
    J = np.zeros((len(valid), nv), dtype=np.float64)
    for row, name in enumerate(valid):
        J[row, spec.joint_dof_index[name]] = 1.0
    return J


def joint_posture_task(
    spec: G1RobotSpec,
    q_target: np.ndarray,
    q_current: np.ndarray,
    qd_current: np.ndarray,
    *,
    kp: float,
    kd: float,
    joints: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute the desired joint accelerations for a posture task.

    Returns a vector ``a_des`` of length ``len(joints)`` (or ``num_actuated``
    when ``joints`` is None) ordered to match ``joint_selector(spec, joints, nv)``.

    ``q_target`` and ``q_current`` are full actuated-joint vectors (length
    ``num_actuated``); the function indexes into them by joint name to
    produce per-DoF errors.
    """
    names = tuple(spec.actuated_joints) if joints is None else tuple(joints)
    out = np.zeros((len(names),), dtype=np.float64)
    for i, name in enumerate(names):
        j = spec.actuated_joints.index(name)
        out[i] = float(kp) * (float(q_target[j]) - float(q_current[j])) + float(kd) * (
            0.0 - float(qd_current[j])
        )
    return out


# ---------------------------------------------------------------------------
# Phase-aware weight scaling
# ---------------------------------------------------------------------------


def through_gap_scales(narrowness: float, weights: TaskWeightsConfig) -> Dict[str, float]:
    """Linear interpolation from 1.0 (open space) to the through-gap scales."""
    n = float(np.clip(narrowness, 0.0, 1.0))
    return {
        "com": 1.0 + (weights.through_gap_com_scale - 1.0) * n,
        "pelvis": 1.0 + (weights.through_gap_pelvis_scale - 1.0) * n,
        "torso": 1.0 + (weights.through_gap_torso_scale - 1.0) * n,
        "swing": 1.0 + (weights.through_gap_swing_scale - 1.0) * n,
        "arm": 1.0 + (weights.through_gap_arm_scale - 1.0) * n,
        "waist": 1.0 + (weights.through_gap_waist_scale - 1.0) * n,
        "lower_body": 1.0 + (weights.through_gap_lower_body_scale - 1.0) * n,
    }


def support_phase_scales(num_contacts: int, weights: TaskWeightsConfig) -> Dict[str, float]:
    """Single-support gets aggressive de-weighting for swing/arm/torso tasks."""
    if int(num_contacts) >= 2:
        return {k: 1.0 for k in ("com", "pelvis", "torso", "swing", "arm", "waist", "lower_body")}
    return {
        "com": weights.single_support_com_scale,
        "pelvis": weights.single_support_pelvis_scale,
        "torso": weights.single_support_torso_scale,
        "swing": weights.single_support_swing_scale,
        "arm": weights.single_support_arm_scale,
        "waist": weights.single_support_waist_scale,
        "lower_body": weights.single_support_lower_body_scale,
    }


# ---------------------------------------------------------------------------
# Lower-body posture target shaping
# ---------------------------------------------------------------------------


def lower_body_target(
    spec: G1RobotSpec,
    tasks: HumanoidTaskSpec,
    q_current: np.ndarray,
    *,
    gains: TaskGainsConfig,
    limits: LimitsConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build a height-aware leg-posture target.

    Mirrors the legacy ``_build_lower_body_posture_target``: starts from
    ``stand_ctrl``, biases hips/knees/ankles by a crouch ratio computed
    from the planned body height, and adds lateral hip-roll based on the
    pelvis offset relative to the support midpoint (in local pelvis-yaw
    frame).
    """
    body_height = float(tasks.pelvis.position_world[2])
    crouch_ratio = _crouch_ratio(body_height, gains.body_height_nominal, gains.body_height_min)

    support_points: list[np.ndarray] = []
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
    rot_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float64
    )
    pelvis_xy = np.asarray(tasks.pelvis.position_world[:2], dtype=np.float64)
    forward, lateral = rot_yaw.T @ (pelvis_xy - support_mid)

    leg_targets: Dict[str, float] = {}
    for prefix, roll_sign in (("left", 1.0), ("right", -1.0)):
        leg_targets[f"{prefix}_hip_pitch_joint"] = (
            _stand_value(spec, f"{prefix}_hip_pitch_joint")
            + gains.crouch_hip_pitch_gain * crouch_ratio
            + gains.pelvis_forward_hip_pitch_gain * float(forward)
        )
        leg_targets[f"{prefix}_knee_joint"] = (
            _stand_value(spec, f"{prefix}_knee_joint") + gains.crouch_knee_gain * crouch_ratio
        )
        leg_targets[f"{prefix}_ankle_pitch_joint"] = (
            _stand_value(spec, f"{prefix}_ankle_pitch_joint")
            + gains.crouch_ankle_pitch_gain * crouch_ratio
        )
        leg_targets[f"{prefix}_hip_roll_joint"] = (
            _stand_value(spec, f"{prefix}_hip_roll_joint")
            - roll_sign * gains.pelvis_lateral_hip_roll_gain * float(lateral)
        )

    q_target = spec.joint_dict_to_vector(leg_targets, base=q_current)
    q_target = spec.clip_to_joint_limits(q_target)
    return q_target, {
        "body_height": body_height,
        "crouch_ratio": crouch_ratio,
        "pelvis_offset_forward": float(forward),
        "pelvis_offset_lateral": float(lateral),
    }


def _crouch_ratio(body_height: float, nominal: float, minimum: float) -> float:
    denom = max(float(nominal) - float(minimum), 1e-6)
    return float(np.clip((float(nominal) - float(body_height)) / denom, 0.0, 1.0))


def _stand_value(spec: G1RobotSpec, joint_name: str) -> float:
    idx = spec.actuated_joints.index(joint_name)
    return float(spec.stand_ctrl[idx])


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) intrinsic rotation matrix."""
    cr, sr = np.cos(float(roll)), np.sin(float(roll))
    cp, sp = np.cos(float(pitch)), np.sin(float(pitch))
    cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def orientation_axis_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Column-cross orientation error (well-conditioned for small angles)."""
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )
