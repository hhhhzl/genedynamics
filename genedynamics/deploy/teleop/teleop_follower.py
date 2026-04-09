"""Teleoperation follower — bridges an :class:`InputSource` to the deploy loop.

:class:`TeleopFollower` implements the :class:`TrajectoryFollower` protocol
with ``horizon = None`` (open-ended), turning each control step into an
:class:`Intent` derived from the latest :class:`TeleopCommand`.

The follower maintains a small amount of state:
* ``_body_height`` — accumulated height target (integrated from delta commands)
* ``_base_yaw`` — smoothed yaw estimate (low-pass from ``state.qpos``)

``is_finished()`` returns ``True`` only when the source signals ``stop=True``
(E-stop pressed) or when ``max_steps`` is exceeded.

Usage (in a script)::

    from genedynamics.deploy.teleop import TeleopFollower, GamepadSource

    src = GamepadSource(vx_max=0.4)
    follower = TeleopFollower(source=src, base_height_nominal=0.75)

    # plug into runner.py or a hand-rolled loop
    follower.reset(plan=None)
    intent = follower.step(t, state)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from genedynamics.deploy.interfaces.messages import Intent, RobotState
from genedynamics.deploy.teleop.input_source import InputSource, TeleopCommand

__all__ = ["TeleopFollower"]


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Extract yaw from a (w, x, y, z) quaternion."""
    qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class TeleopFollower:
    """Open-ended follower driven by a live :class:`InputSource`.

    Args:
        source: An :class:`InputSource` implementation (gamepad, keyboard,
            ROS2 topic, …). The follower calls ``source.open()`` in
            :meth:`reset` and ``source.close()`` in :meth:`close`.
        base_height_nominal: Default body height target in metres.
        base_height_min: Lower clamp for height integration.
        base_height_max: Upper clamp for height integration.
        yaw_filter_alpha: Low-pass coefficient for the yaw estimate from state
            (0 = use raw state yaw, 1 = freeze). Tune to avoid jerky intent.
        max_steps: Hard episode length cap. ``is_finished()`` returns ``True``
            when this is exceeded. Set to ``None`` for unlimited.
    """

    # TrajectoryFollower protocol attributes
    plan_dt: float = 0.02   # 50 Hz nominal; runner uses its own sim_dt
    horizon: Optional[float] = None  # open-ended

    def __init__(
        self,
        source: InputSource,
        *,
        base_height_nominal: float = 0.75,
        base_height_min: float = 0.55,
        base_height_max: float = 0.90,
        yaw_filter_alpha: float = 0.05,
        max_steps: Optional[int] = None,
    ) -> None:
        self._source = source
        self._h_nom = float(base_height_nominal)
        self._h_min = float(base_height_min)
        self._h_max = float(base_height_max)
        self._yaw_alpha = float(yaw_filter_alpha)
        self._max_steps = max_steps

        self._body_height: float = self._h_nom
        self._base_yaw: float = 0.0
        self._step_count: int = 0
        self._stopped: bool = False
        self._source_open: bool = False

    # ------------------------------------------------------------------
    # TrajectoryFollower protocol
    # ------------------------------------------------------------------

    def reset(self, plan: Any = None) -> None:
        """Open the input source and reset internal state.

        ``plan`` is ignored (teleoperation has no pre-defined plan).
        """
        if self._source_open:
            try:
                self._source.close()
            except Exception:
                pass
        self._source.open()
        self._source_open = True
        self._body_height = self._h_nom
        self._base_yaw = 0.0
        self._step_count = 0
        self._stopped = False

    def step(self, t: float, state: RobotState) -> Intent:
        """Sample the input source and return an :class:`Intent`."""
        self._step_count += 1

        cmd: TeleopCommand = self._source.read()

        if cmd.stop:
            self._stopped = True

        # Integrate height delta.
        self._body_height = float(
            np.clip(
                self._body_height + cmd.body_height_delta,
                self._h_min,
                self._h_max,
            )
        )

        # Update yaw estimate from state (low-pass).
        meas_yaw = self._extract_yaw(state)
        if meas_yaw is not None:
            self._base_yaw = (1.0 - self._yaw_alpha) * meas_yaw + self._yaw_alpha * self._base_yaw

        # Build Intent — rotate body-frame velocity to world frame using yaw.
        cos_y = math.cos(self._base_yaw)
        sin_y = math.sin(self._base_yaw)
        vx_world = cos_y * cmd.vx - sin_y * cmd.vy
        vy_world = sin_y * cmd.vx + cos_y * cmd.vy

        extras: dict = {}
        if cmd.arm_delta:
            extras["arm_delta"] = cmd.arm_delta
        if cmd.extras:
            extras.update(cmd.extras)

        return Intent(
            t=float(t),
            base_yaw=float(self._base_yaw),
            base_height=float(self._body_height),
            base_lin_vel=np.array([vx_world, vy_world], dtype=np.float64),
            base_yaw_rate=float(cmd.yaw_rate),
            extras=extras,
        )

    def is_finished(self, t: float) -> bool:
        if self._stopped:
            return True
        if self._max_steps is not None and self._step_count >= self._max_steps:
            return True
        return False

    def close(self) -> None:
        """Close the input source."""
        if self._source_open:
            try:
                self._source.close()
            except Exception:
                pass
            self._source_open = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_yaw(state: RobotState) -> Optional[float]:
        """Extract yaw from state, preferring base_pose quaternion."""
        if state.base_pose is not None and len(state.base_pose) >= 7:
            return _yaw_from_quat_wxyz(np.asarray(state.base_pose[3:7], dtype=np.float64))
        if state.qpos is not None and len(state.qpos) >= 7:
            return _yaw_from_quat_wxyz(np.asarray(state.qpos[3:7], dtype=np.float64))
        return None
