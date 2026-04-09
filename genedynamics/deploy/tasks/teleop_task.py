"""Teleoperation execution task.

Monitors a live teleoperation episode and signals termination when:

* the operator triggers E-stop via the follower (``intent.extras["estop"]``)
* the robot falls (base height drops below ``fall_height_m``)
* ``max_steps`` elapse without any of the above

Per-step :class:`StepInfo` metrics:

* ``"vx_cmd"`` / ``"vy_cmd"`` / ``"yaw_rate_cmd"`` — current intent commands
* ``"body_height"`` — current intent height target
* ``"base_z"`` — measured base height from state
* ``"roll_deg"`` / ``"pitch_deg"`` — IMU roll/pitch in degrees (if available)

The task also accumulates a ``"distance_travelled_m"`` odometry estimate in
:meth:`summary`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.tasks.base import BaseExecutionTask

__all__ = ["TeleopTask"]


def _rpy_from_quat_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    # yaw (not returned but kept for symmetry)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class TeleopTask(BaseExecutionTask):
    """Per-episode teleoperation monitor.

    Args:
        max_steps: Hard episode length cap. ``None`` = unlimited (use only
            for manual-stop workflows).
        fall_height_m: Base height (m) below which the robot is considered
            fallen and the episode is terminated.
        fall_angle_deg: Roll or pitch (degrees) above which the robot is
            considered fallen.
        name: Display name forwarded to observers.
    """

    def __init__(
        self,
        *,
        max_steps: Optional[int] = None,
        fall_height_m: float = 0.25,
        fall_angle_deg: float = 45.0,
        name: str = "teleop",
    ) -> None:
        super().__init__(name=name)
        self.max_steps = max_steps
        self.fall_height_m = float(fall_height_m)
        self.fall_angle_rad = math.radians(float(fall_angle_deg))

        self._prev_xy: Optional[np.ndarray] = None
        self._distance_m: float = 0.0
        self._termination_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # ExecutionTask
    # ------------------------------------------------------------------

    def reset(self, io: Any) -> None:
        super().reset(io)
        self._prev_xy = None
        self._distance_m = 0.0
        self._termination_reason = None

    def step(
        self,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
    ) -> StepInfo:
        self._step_count += 1
        metrics: dict[str, float] = {}

        # Intent metrics.
        metrics["vx_cmd"] = float(intent.base_lin_vel[0]) if intent.base_lin_vel is not None else 0.0
        metrics["vy_cmd"] = float(intent.base_lin_vel[1]) if intent.base_lin_vel is not None and len(intent.base_lin_vel) > 1 else 0.0
        metrics["yaw_rate_cmd"] = float(intent.base_yaw_rate)
        metrics["body_height"] = float(intent.base_height)

        # Base pose.
        base_z: Optional[float] = None
        roll: Optional[float] = None
        pitch: Optional[float] = None
        quat: Optional[np.ndarray] = None

        if state.base_pose is not None and len(state.base_pose) >= 3:
            base_z = float(state.base_pose[2])
            if len(state.base_pose) >= 7:
                quat = np.asarray(state.base_pose[3:7], dtype=np.float64)
        elif state.qpos is not None and state.qpos.shape[0] >= 7:
            base_z = float(state.qpos[2])
            quat = np.asarray(state.qpos[3:7], dtype=np.float64)

        if base_z is not None:
            metrics["base_z"] = base_z
        if quat is not None:
            roll, pitch, _ = _rpy_from_quat_wxyz(quat)
            metrics["roll_deg"] = math.degrees(roll)
            metrics["pitch_deg"] = math.degrees(pitch)

        # Odometry accumulation.
        xy = self._base_xy(state)
        if xy is not None:
            if self._prev_xy is not None:
                self._distance_m += float(np.linalg.norm(xy - self._prev_xy))
            self._prev_xy = xy.copy()
        metrics["distance_m"] = self._distance_m

        # Termination checks — order: E-stop > fall > max_steps
        estop = bool(intent.extras.get("estop", False)) or bool(intent.extras.get("stop", False))
        if estop:
            self._termination_reason = "estop"
            return StepInfo(done=True, success=False, metrics=metrics)

        if base_z is not None and base_z < self.fall_height_m:
            self._termination_reason = "fall_height"
            return StepInfo(done=True, success=False, metrics=metrics)

        if roll is not None and pitch is not None:
            if abs(roll) > self.fall_angle_rad or abs(pitch) > self.fall_angle_rad:
                self._termination_reason = "fall_angle"
                return StepInfo(done=True, success=False, metrics=metrics)

        if self.max_steps is not None and self._step_count >= self.max_steps:
            self._termination_reason = "max_steps"
            return StepInfo(done=True, success=False, metrics=metrics)

        return StepInfo(done=False, metrics=metrics)

    def summary(self) -> Mapping[str, Any]:
        return {
            "steps": self._step_count,
            "distance_travelled_m": self._distance_m,
            "termination_reason": self._termination_reason,
        }

    @staticmethod
    def _base_xy(state: RobotState) -> Optional[np.ndarray]:
        if state.base_pose is not None and len(state.base_pose) >= 2:
            return np.asarray(state.base_pose[:2], dtype=np.float64)
        if state.qpos is not None and state.qpos.shape[0] >= 2:
            return np.asarray(state.qpos[:2], dtype=np.float64)
        return None
