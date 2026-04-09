"""Self-collision avoidance filter.

This filter performs a coarse self-collision check using the body-pair
distances reported by the robot's MuJoCo model. When a configured pair of
bodies comes within ``min_clearance`` of each other, the filter shrinks the
joint position command toward the *current* qpos by a configurable retreat
factor — i.e. it produces an interpolation between "where we are" and
"where the controller wanted to go".

This is intentionally **best-effort**: it neither solves a QP nor unrolls
the dynamics. Pairs and the body distance lookup are read from the spec the
filter is constructed against. The check is skipped (and the command passed
through) when the spec does not expose distance helpers — letting the
filter compose with non-MuJoCo IO without raising.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.interfaces.safety import SafetyResult
from genedynamics.deploy.safety.base import BaseSafetyFilter

__all__ = ["SelfCollisionFilter"]


# Default G1 body pairs that are most likely to bash into each other
# during arm-tucking and corridor traversal. Pairs use spec body names.
_DEFAULT_G1_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("left_ankle_roll_link", "right_ankle_roll_link"),
    ("left_wrist_yaw_link", "torso_link"),
    ("right_wrist_yaw_link", "torso_link"),
    ("left_elbow_link", "torso_link"),
    ("right_elbow_link", "torso_link"),
)


class SelfCollisionFilter(BaseSafetyFilter):
    """Coarse capsule-distance self-collision filter.

    Args:
        spec: Robot spec exposing ``body_id``. The filter only consults the
            ``body_id`` map; clearance distances are looked up via the
            optional ``robot`` argument.
        robot: Optional callable / object exposing
            ``body_pair_distance(body_a, body_b) -> float`` (e.g. a
            :class:`G1RobotModel`). When ``None``, the filter degrades to
            a no-op so it can be inserted into pipelines that don't have
            full kinematics on hand (mjx, real hardware without MuJoCo).
        pairs: Body name pairs to monitor. Defaults to a small G1 set.
        min_clearance: Distance threshold (meters). Below this the filter
            intervenes.
        retreat_factor: Interpolation weight ``α`` ∈ [0, 1] used to compute
            the safe target as ``q_safe = α * q_current + (1 − α) * q_cmd``.
            Higher → more conservative.
    """

    def __init__(
        self,
        spec: Any,
        *,
        robot: Optional[Any] = None,
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
        min_clearance: float = 0.03,
        retreat_factor: float = 0.5,
    ) -> None:
        super().__init__(spec)
        self.robot = robot
        self.pairs: Tuple[Tuple[str, str], ...] = tuple(pairs) if pairs is not None else _DEFAULT_G1_PAIRS
        self.min_clearance = float(min_clearance)
        if not (0.0 <= retreat_factor <= 1.0):
            raise ValueError(f"retreat_factor must be in [0, 1], got {retreat_factor}")
        self.retreat_factor = float(retreat_factor)

    def filter(self, state: RobotState, cmd: ControlCommand) -> SafetyResult:
        if cmd.kind not in ("joint_pos", "mixed") or cmd.joint_pos is None:
            return self._passthrough(cmd)
        if self.robot is None or not hasattr(self.robot, "body_pair_distance"):
            return self._passthrough(cmd)

        violators: list[Tuple[str, str, float]] = []
        for a, b in self.pairs:
            try:
                d = float(self.robot.body_pair_distance(a, b))
            except Exception:
                continue
            if d < self.min_clearance:
                violators.append((a, b, d))

        if not violators:
            return self._passthrough(cmd)

        # Pull the current actuated qpos vector and interpolate the command toward it.
        try:
            q_cur_full = np.asarray(state.qpos, dtype=np.float64)
            q_cur_actuated = self._extract_actuated(q_cur_full)
        except Exception:
            return self._passthrough(cmd)

        jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1).copy()
        n = min(jp.size, q_cur_actuated.size)
        alpha = self.retreat_factor
        jp[:n] = alpha * q_cur_actuated[:n] + (1.0 - alpha) * jp[:n]

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
        worst = min(d for _, _, d in violators)
        return SafetyResult(
            command=new_cmd,
            intervened=True,
            violations={"self_collision_min_dist_m": worst},
            extras={"violators": [{"a": a, "b": b, "dist": d} for a, b, d in violators]},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_actuated(self, qpos_full: np.ndarray) -> np.ndarray:
        """Extract actuated qpos from a full ``qpos`` vector."""
        idx = getattr(self.spec, "actuated_qpos_indices", None)
        if idx is None:
            # Spec doesn't know how to slice — assume legs+waist+arms start at offset 7.
            n = len(self.spec.actuated_joints)
            return qpos_full[7 : 7 + n].astype(np.float64)
        return qpos_full[np.asarray(idx)].astype(np.float64)
