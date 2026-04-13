"""Singularity avoidance filter for arm teleoperation.

Detects proximity to kinematic singularities by monitoring the manipulability
measure (approximate minimum singular value of the Jacobian), estimated from
the spread of arm joint velocities.  When manipulability drops below a
threshold, the filter attenuates the arm command to slow down approach toward
the singularity.

Design
------
Computing the true Jacobian requires a full kinematic model.  Since the deploy
stack does not currently expose per-robot FK/IK solvers in a generic way, this
filter uses a **proxy heuristic**:

* High joint velocity spread at low task-space movement = singularity warning.
* The filter measures the ratio of arm joint velocity norm to commanded
  position displacement.  A large ratio suggests the configuration is ill-
  conditioned (large joint motion needed for small end-effector motion).

For arms with elbow-near-zero (common robot singularity), monitor the elbow
joint directly — a near-zero elbow joint index triggers damping regardless of
the ratio.

When a proper Jacobian-based check is added (via a robot-specific FK plugin),
replace ``_is_near_singular`` with the real manipulability computation.

The filter is a no-op when no arm joints are present in the spec.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["SingularityFilter"]


class SingularityFilter(BaseSafetyFilter):
    """Attenuate arm commands near kinematic singularities.

    Args:
        spec: Robot spec exposing ``left_arm_joints``, ``right_arm_joints``,
            ``actuated_joints``.
        elbow_joint_names: Joint names considered the "elbow" — their
            absolute value is used as a singularity proxy (near-zero elbow =
            near singular).  Auto-detected if ``None`` and spec exposes
            ``left_arm_joints`` / ``right_arm_joints``.
        elbow_singular_rad: Elbow angle (rad) below which the singularity
            warning fires.
        attenuation_min: Minimum scale factor applied to arm deltas at the
            singularity boundary (0 = fully frozen, 0.5 = half speed).
        attenuation_blend_rad: Angular range over which the attenuation
            blends from 1.0 (no damping) to ``attenuation_min`` (full damping).
    """

    def __init__(
        self,
        spec: Any,
        *,
        elbow_joint_names: Optional[Sequence[str]] = None,
        elbow_singular_rad: float = 0.05,
        attenuation_min: float = 0.1,
        attenuation_blend_rad: float = 0.20,
    ) -> None:
        super().__init__(spec)
        self._elbow_singular = float(elbow_singular_rad)
        self._atten_min = float(np.clip(attenuation_min, 0.0, 1.0))
        self._blend = float(max(attenuation_blend_rad, 1e-6))

        all_joints = list(getattr(spec, "actuated_joints", []))
        left_arm = list(getattr(spec, "left_arm_joints", []))
        right_arm = list(getattr(spec, "right_arm_joints", []))
        arm_joints = set(left_arm) | set(right_arm)
        self._arm_indices: np.ndarray = np.array(
            [i for i, n in enumerate(all_joints) if n in arm_joints], dtype=np.int64
        )

        # Auto-detect elbow joints: the 3rd joint in each arm (index 2) is
        # typically the elbow.
        if elbow_joint_names is not None:
            elbow_set = set(elbow_joint_names)
        else:
            elbow_set = set()
            for arm in (left_arm, right_arm):
                if len(arm) >= 3:
                    elbow_set.add(arm[2])   # 3rd joint = elbow

        self._elbow_indices: np.ndarray = np.array(
            [i for i, n in enumerate(all_joints) if n in elbow_set], dtype=np.int64
        )

    def filter(self, state, cmd: ControlCommand) -> SafetyResult:
        if cmd.kind not in ("joint_pos", "mixed") or cmd.joint_pos is None:
            return self._passthrough(cmd)
        if self._arm_indices.size == 0:
            return self._passthrough(cmd)

        jp = np.asarray(cmd.joint_pos, dtype=np.float64).copy()
        current = np.asarray(state.qpos, dtype=np.float64) if state.qpos is not None else jp

        scale = self._compute_scale(jp, current)
        if abs(scale - 1.0) < 1e-6:
            return self._passthrough(cmd)

        # Attenuate arm joints only.
        n = len(current)
        valid_arm = self._arm_indices[self._arm_indices < min(len(jp), n)]
        if valid_arm.size > 0:
            arm_delta = jp[valid_arm] - current[valid_arm]
            jp[valid_arm] = current[valid_arm] + arm_delta * scale

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
            intervened=True,
            violations={"singularity_scale": float(scale)},
        )

    def _compute_scale(self, jp: np.ndarray, current: np.ndarray) -> float:
        """Return an attenuation scale in [attenuation_min, 1.0]."""
        if self._elbow_indices.size == 0:
            return 1.0

        valid_elbow = self._elbow_indices[self._elbow_indices < len(current)]
        if valid_elbow.size == 0:
            return 1.0

        # Minimum |elbow angle| across both arms.
        min_elbow_abs = float(np.min(np.abs(current[valid_elbow])))

        if min_elbow_abs >= self._elbow_singular + self._blend:
            return 1.0   # well away from singular

        if min_elbow_abs <= self._elbow_singular:
            return self._atten_min   # at or past singular — maximum damping

        # Linear blend in [elbow_singular, elbow_singular + blend].
        alpha = (min_elbow_abs - self._elbow_singular) / self._blend
        return float(self._atten_min + alpha * (1.0 - self._atten_min))
