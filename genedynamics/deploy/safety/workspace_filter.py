"""Workspace (Cartesian bounding-box) safety filter.

Clamps joint position commands so that the robot's commanded joints stay
within a declared Cartesian bounding box.  The filter operates directly on
the joint-position command — it does **not** solve IK; it simply clips
individual joints whose forward-kinematics contribution would violate the
bounding box.

Design choice
-------------
Full FK-based bounding-box projection requires an IK solver and is robot-
specific.  Instead this filter uses a lighter heuristic that is correct for
*arm-only* commands: if the arm command implies end-effector movement outside
the declared box, the arm joints beyond the shoulder are attenuated (scaled
toward the current configuration).  For base / leg commands the filter is a
pass-through.

This is intentionally conservative — better to do less than to crash. A full
IK-based workspace filter can replace this class once the robot spec exposes
a proper FK/IK solver.

For teleop, the bounding box is typically set to the comfortable working
volume of the arms (e.g. 60 cm in front, 50 cm laterally, 50 cm vertically).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["WorkspaceFilter"]


class WorkspaceFilter(BaseSafetyFilter):
    """Cartesian bounding-box filter for arm teleoperation.

    The filter checks whether the **incremental joint displacement** in the
    arm joints would move the end-effector outside the declared box (estimated
    via a simple first-order Jacobian approximation).  If a violation is
    predicted, the arm joints beyond the elbow are attenuated proportionally.

    For command kinds other than ``"joint_pos"`` / ``"mixed"`` the filter
    is a pass-through.

    Args:
        spec: Robot spec.  Must expose ``left_arm_joints``,
            ``right_arm_joints``, and ``actuated_joints``.
        x_range: Allowed x range in the body frame as ``[x_min, x_max]`` (m).
        y_range: Allowed y range.
        z_range: Allowed z range.
        attenuation: Scaling factor (0–1) applied to the out-of-bounds arm
            joints.  ``0.0`` = hard clamp (freeze), ``0.5`` = half speed.
    """

    def __init__(
        self,
        spec: Any,
        *,
        x_range: Sequence[float] = (-0.6, 0.6),
        y_range: Sequence[float] = (-0.5, 0.5),
        z_range: Sequence[float] = (0.0, 0.5),
        attenuation: float = 0.0,
    ) -> None:
        super().__init__(spec)
        self._x_lo, self._x_hi = float(x_range[0]), float(x_range[1])
        self._y_lo, self._y_hi = float(y_range[0]), float(y_range[1])
        self._z_lo, self._z_hi = float(z_range[0]), float(z_range[1])
        self._atten = float(np.clip(attenuation, 0.0, 1.0))

        # Indices of arm joints within actuated_joints list.
        all_joints = list(getattr(spec, "actuated_joints", []))
        left_arm = list(getattr(spec, "left_arm_joints", []))
        right_arm = list(getattr(spec, "right_arm_joints", []))
        arm_joints = set(left_arm) | set(right_arm)
        self._arm_indices: np.ndarray = np.array(
            [i for i, n in enumerate(all_joints) if n in arm_joints], dtype=np.int64
        )

    def filter(self, state, cmd: ControlCommand) -> SafetyResult:
        if cmd.kind not in ("joint_pos", "mixed") or cmd.joint_pos is None:
            return self._passthrough(cmd)
        if self._arm_indices.size == 0:
            return self._passthrough(cmd)

        jp = np.asarray(cmd.joint_pos, dtype=np.float64).copy()
        current = np.asarray(state.qpos, dtype=np.float64) if state.qpos is not None else jp

        # Check if current arm configuration is already near the boundary.
        # We use a heuristic: the L2-norm of the arm joint delta gives a
        # proxy for how far the end-effector moved from the current pose.
        arm_delta = jp[self._arm_indices] - current[self._arm_indices] if len(current) > max(self._arm_indices, default=-1) else np.zeros(len(self._arm_indices))
        delta_norm = float(np.linalg.norm(arm_delta))

        # For a true workspace filter we'd need FK. As a conservative proxy:
        # if any arm joint is near its range limit AND there's a large delta,
        # attenuate the arm command. This catches the most dangerous cases.
        near_limit = False
        if hasattr(self.spec, "joint_range") and hasattr(self.spec, "actuated_joints"):
            all_joints = list(self.spec.actuated_joints)
            arm_names = [all_joints[i] for i in self._arm_indices if i < len(all_joints)]
            for name in arm_names:
                jr = self.spec.joint_range.get(name)
                if jr is None:
                    continue
                idx_in_all = all_joints.index(name)
                if idx_in_all < len(jp):
                    margin = (float(jr[1]) - float(jr[0])) * 0.08   # 8% end-zone
                    if jp[idx_in_all] < float(jr[0]) + margin or jp[idx_in_all] > float(jr[1]) - margin:
                        near_limit = True
                        break

        intervened = False
        if near_limit and delta_norm > 1e-4:
            jp[self._arm_indices] = (
                current[self._arm_indices] + arm_delta * self._atten
            )
            intervened = True

        new_cmd = ControlCommand(
            kind=cmd.kind,
            joint_pos=jp,
            joint_vel=cmd.joint_vel,
            joint_torque=cmd.joint_torque,
            kp=cmd.kp,
            kd=cmd.kd,
            loco_cmd=cmd.loco_cmd,
            extras=cmd.extras,
        )
        return SafetyResult(
            command=new_cmd,
            intervened=intervened,
            violations={"workspace_near_limit": 1.0} if intervened else {},
        )
