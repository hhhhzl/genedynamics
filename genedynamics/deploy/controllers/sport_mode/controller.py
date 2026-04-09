"""Sport-mode controller (mixed: loco-driven legs + PD-driven upper body).

This is the cheapest controller that solves the corridor follower task in
its entirety:

* Lower body (12 leg joints) — driven by a :class:`LocoClient`. In sim
  that's :class:`SparkRLLocoClient` (spark's pretrained motion.pt PPO
  policy); on real hardware it's :class:`RealLocoClient` which wraps
  Unitree's stock sport-mode SDK.
* Upper body (3 waist + 14 arm joints) — driven by direct PD targets
  reused from :class:`HumanoidUpperBodyMapper`. The mapper already
  decodes the 14D plan's ``psi_torso`` / ``a_left/right`` /
  ``p_left/right`` fields into joint hints, so this controller gets the
  arm-tucking + torso-twist behavior for free.

The controller emits a single :class:`ControlCommand` of kind
``"joint_pos"`` with all 29 actuated joints populated and per-DoF Kp / Kd
gains. ``MujocoRobotIO`` (and ``MjxRobotIO``) accept this directly; on
real G1 the IO splits the command back into a sport-mode call + low-level
joint write.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from genedynamics.deploy.controllers.sport_mode.loco_client import LocoClient
from genedynamics.deploy.followers.humanoid.task_spec import HumanoidTaskSpec
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    LocoCommand,
    RobotState,
)
from genedynamics.deploy.io.mujoco_io import MujocoRobotIO
from genedynamics.robots.g1 import G1RobotSpec

__all__ = ["SportModeController"]


class SportModeController:
    """Mixed-mode controller: legs from LocoClient, upper body from PD targets.

    Args:
        io: A :class:`MujocoRobotIO` (or :class:`MjxRobotIO`). Used only for
            ``spec`` and to read the initial pelvis pose at reset.
        loco_client: Any :class:`LocoClient` implementation.
        leg_kp: Position gain applied to the 12 leg joints.
        leg_kd: Velocity gain applied to the 12 leg joints.
        upper_body_kp: Position gain applied to the 17 waist+arm joints.
            Held high enough that the upper body tracks the mapper's output
            against gravity and inertial coupling from the legs.
        upper_body_kd: Velocity gain on waist+arms.
    """

    runtime: str = "numpy"
    produces: Tuple[str, ...] = ("joint_pos",)

    def __init__(
        self,
        io: MujocoRobotIO,
        *,
        loco_client: LocoClient,
        leg_kp: float = 100.0,
        leg_kd: float = 4.0,
        upper_body_kp: float = 80.0,
        upper_body_kd: float = 3.0,
    ) -> None:
        self.io = io
        self.spec: G1RobotSpec = io.spec
        self.loco_client = loco_client
        self.leg_kp = float(leg_kp)
        self.leg_kd = float(leg_kd)
        self.upper_body_kp = float(upper_body_kp)
        self.upper_body_kd = float(upper_body_kd)

        # Pre-compute joint name → actuator index map (one-time hash lookup
        # in the hot loop is much cheaper than .index() calls).
        self._joint_index = {
            name: i for i, name in enumerate(self.spec.actuated_joints)
        }
        self._leg_joint_names = (
            tuple(self.spec.left_leg_joints) + tuple(self.spec.right_leg_joints)
        )
        self._upper_joint_names = (
            tuple(self.spec.waist_joints)
            + tuple(self.spec.left_arm_joints)
            + tuple(self.spec.right_arm_joints)
        )

        # Pre-built per-joint gain vectors. Updated only on construction;
        # the controller does not reach into the spec on every call.
        n = self.spec.num_actuated
        self._kp = np.full(n, self.leg_kp, dtype=np.float64)
        self._kd = np.full(n, self.leg_kd, dtype=np.float64)
        for name in self._upper_joint_names:
            idx = self._joint_index.get(name)
            if idx is not None:
                self._kp[idx] = self.upper_body_kp
                self._kd[idx] = self.upper_body_kd

    # ------------------------------------------------------------------
    # Controller protocol
    # ------------------------------------------------------------------

    def reset(self, io: Optional[MujocoRobotIO] = None) -> None:
        """Re-bind to ``io`` (if given) and re-anchor the loco client."""
        if io is not None:
            self.io = io
            self.spec = io.spec
        state = self.io.get_state()
        pelvis_world, pelvis_yaw = self._pelvis_pose_from_state(state)
        self.loco_client.reset(pelvis_world=pelvis_world, pelvis_yaw=pelvis_yaw)

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        """Build a single 29-joint position command for legs + upper body."""
        # 1. Read pelvis pose so the loco client can advance its world model.
        pelvis_world, pelvis_yaw = self._pelvis_pose_from_state(state)

        # 2. Build the LocoCommand from the intent.
        loco_cmd = LocoCommand(
            vx=float(intent.base_lin_vel[0]),
            vy=float(intent.base_lin_vel[1]),
            yaw_rate=float(intent.base_yaw_rate),
            body_height=float(intent.base_height),
        )

        # 3. Resolve dt from intent.extras (set by the pipeline) or fall back.
        dt = float((intent.extras or {}).get("dt", 0.02))

        # 4. Ask the loco client for leg targets.
        leg_targets = self.loco_client.step(
            cmd=loco_cmd,
            dt=dt,
            pelvis_world=pelvis_world,
            pelvis_yaw=pelvis_yaw,
            state=state,
        )

        # 5. Pull upper-body targets from the follower's HumanoidTaskSpec.
        upper_targets = self._upper_body_targets_from_intent(intent)

        # 6. Compose the full 29-joint vector starting from the stand pose.
        q_target = self.spec.stand_ctrl.copy()
        for name, val in leg_targets.items():
            idx = self._joint_index.get(name)
            if idx is not None:
                q_target[idx] = float(val)
        for name, val in upper_targets.items():
            idx = self._joint_index.get(name)
            if idx is not None:
                q_target[idx] = float(val)
        q_target = self.spec.clip_to_joint_limits(q_target)

        return ControlCommand(
            kind="joint_pos",
            joint_pos=q_target,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
            extras={
                "loco_cmd": loco_cmd,
                "swing_foot": self.loco_client.swing_foot,
                "phase": self.loco_client.phase,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _upper_body_targets_from_intent(self, intent: Intent) -> dict[str, float]:
        """Extract waist + arm joint targets from ``intent.extras``.

        Two upstream sources are supported:

        1. ``extras["upper_body_targets"]`` — any object exposing a
           ``joint_hints: dict`` attribute. This is the lightweight path:
           a follower can build a :class:`UpperBodyTargets` directly from
           the 14D plan frame via :class:`HumanoidUpperBodyMapper.map`,
           bypassing the full :class:`HumanoidTaskBuilder` pipeline. Used
           by the corridor diagnose scripts.

        2. ``extras["humanoid_tasks"]`` — a :class:`HumanoidTaskSpec`
           assembled by the WBC follower stack. Used by the legacy
           :class:`HumanoidMujocoPipeline` and any consumer that already
           has a full task spec on hand.

        When both are missing — e.g. during a unit test — return an empty
        dict so the controller falls back to the spec's stand pose.
        """
        extras = intent.extras or {}
        upper = extras.get("upper_body_targets")
        if upper is not None and hasattr(upper, "joint_hints"):
            joint_hints = getattr(upper, "joint_hints")
            if joint_hints:
                return {k: float(v) for k, v in joint_hints.items()}
        tasks = extras.get("humanoid_tasks")
        if isinstance(tasks, HumanoidTaskSpec) and tasks.joint_hints:
            return {k: float(v) for k, v in tasks.joint_hints.items()}
        return {}

    @staticmethod
    def _pelvis_pose_from_state(state: RobotState) -> Tuple[np.ndarray, float]:
        """Extract ``(pelvis_world, pelvis_yaw)`` from a :class:`RobotState`.

        Falls back to ``(zeros, 0)`` if the state has no floating base.
        """
        if state.qpos is None or state.qpos.shape[0] < 7:
            return np.zeros(3, dtype=np.float64), 0.0
        qpos = np.asarray(state.qpos, dtype=np.float64)
        pos = qpos[0:3].copy()
        quat = qpos[3:7]  # mujoco quat order: w, x, y, z
        yaw = _yaw_from_quat_wxyz(quat)
        return pos, yaw


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Extract yaw (z-axis rotation) from a unit quaternion in (w, x, y, z) order."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
