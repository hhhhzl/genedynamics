"""Support-contact equality blocks for the WBC.

The legacy WBC enforces ``J_c · ddq = a_des`` for every foot in contact,
where ``a_des`` is a small PD-corrected target around zero (so the foot
holds its current pose against numerical drift). This module builds the
``(J, a_des)`` blocks per foot and the floating-base dynamics rows that
relate ``ddq`` to support wrenches via::

    M[:6, :] · ddq − Σᵢ Jᵢᵀ · λᵢ = −bias[:6]

The number of contact blocks (``num_contacts ∈ {0, 1, 2}``) determines the
shape of the lambda variable in the QP and the rows added to the equality
system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from genedynamics.deploy.controllers.wbc.config import LimitsConfig, TaskGainsConfig
from genedynamics.deploy.controllers.wbc.task_stack import (
    orientation_axis_error,
    rotation_matrix_from_rpy,
)
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec
from genedynamics.deploy.io.mujoco_io import MujocoRobotIO

__all__ = [
    "SupportContactBlock",
    "build_support_blocks",
    "build_floating_base_dynamics_rows",
    "build_contact_equality_rows",
]


# ---------------------------------------------------------------------------
# Block dataclass
# ---------------------------------------------------------------------------


@dataclass
class SupportContactBlock:
    """One foot's contribution to the WBC contact constraints.

    Attributes:
        name: Site name (``"left_foot"`` or ``"right_foot"``).
        J: Stacked translational+rotational Jacobian, shape ``(6, nv)``.
        a_des: PD-corrected target spatial acceleration, shape ``(6,)``.
    """

    name: str
    J: np.ndarray
    a_des: np.ndarray


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


def build_support_blocks(
    io: MujocoRobotIO,
    tasks: HumanoidTaskSpec,
    qvel_full: np.ndarray,
    *,
    gains: TaskGainsConfig,
    limits: LimitsConfig,
) -> List[SupportContactBlock]:
    """Construct one :class:`SupportContactBlock` per in-contact foot.

    The PD correction reuses the contact gains from
    :class:`TaskGainsConfig` and clamps the position / orientation errors
    by :attr:`LimitsConfig.contact_position_error_clip` /
    :attr:`LimitsConfig.contact_orientation_error_clip` so a stale or wild
    target cannot dominate the QP.
    """
    blocks: List[SupportContactBlock] = []
    for site_name, foot_task in (
        ("left_foot", tasks.left_foot),
        ("right_foot", tasks.right_foot),
    ):
        if not foot_task.in_contact:
            continue
        if site_name not in io.spec.site_id:
            continue

        jacp, jacr = io.site_jacobian(site_name)
        pos, R_current = io.site_pose(site_name)
        vel = jacp @ qvel_full
        ang_vel = jacr @ qvel_full

        # Position error (clipped)
        pos_err = np.asarray(foot_task.position_world, dtype=np.float64) - pos
        pos_err = np.clip(
            pos_err,
            -limits.contact_position_error_clip,
            limits.contact_position_error_clip,
        )

        # Orientation error (clipped)
        R_target = rotation_matrix_from_rpy(
            foot_task.roll_world, foot_task.pitch_world, foot_task.yaw_world
        )
        ori_err = orientation_axis_error(R_target, R_current)
        ori_err = np.clip(
            ori_err,
            -limits.contact_orientation_error_clip,
            limits.contact_orientation_error_clip,
        )

        # PD-corrected accelerations (zero target velocity at contact)
        a_lin = gains.contact_position_kp * pos_err + gains.contact_position_kd * (-vel)
        a_ang = gains.contact_orientation_kp * ori_err + gains.contact_orientation_kd * (-ang_vel)
        a_lin = np.clip(a_lin, -limits.contact_accel_limit, limits.contact_accel_limit)
        a_ang = np.clip(
            a_ang, -limits.contact_angular_accel_limit, limits.contact_angular_accel_limit
        )

        J = np.vstack([jacp, jacr])
        a_des = np.concatenate([a_lin, a_ang], axis=0)
        blocks.append(SupportContactBlock(name=site_name, J=J, a_des=a_des))

    return blocks


# ---------------------------------------------------------------------------
# Equality rows for the QP
# ---------------------------------------------------------------------------


def build_floating_base_dynamics_rows(
    M: np.ndarray,
    bias: np.ndarray,
    blocks: List[SupportContactBlock],
    *,
    nv: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top six rows: floating-base Newton/Euler equations.

    ``M[:6, :] · ddq + Σᵢ (−Jᵢ[:, :6]ᵀ) · λᵢ = −bias[:6]``

    These hold the underactuated DoFs of the floating base (6 axes) to the
    dynamics; without them the QP would let the base "fly" freely.
    """
    n_lambda = 6 * len(blocks)
    C = np.zeros((6, nv + n_lambda), dtype=np.float64)
    C[:, :nv] = M[:6, :]
    for i, block in enumerate(blocks):
        col = slice(nv + 6 * i, nv + 6 * (i + 1))
        C[:, col] = -block.J[:, :6].T
    d = -bias[:6]
    return C, d


def build_contact_equality_rows(
    blocks: List[SupportContactBlock],
    *,
    nv: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-foot contact equality rows ``Jᵢ · ddq = a_des,i``.

    These constraints make the support feet hold their current spatial
    pose. Returns ``(C, d)`` of shape ``(6·n, nv+6·n)`` and ``(6·n,)``.
    """
    n = len(blocks)
    if n == 0:
        return np.zeros((0, nv), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    n_lambda = 6 * n
    C = np.zeros((6 * n, nv + n_lambda), dtype=np.float64)
    d = np.zeros((6 * n,), dtype=np.float64)
    for i, block in enumerate(blocks):
        row = slice(6 * i, 6 * (i + 1))
        C[row, :nv] = block.J
        d[row] = block.a_des
    return C, d
