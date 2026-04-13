"""Typed payloads exchanged across the deploy pipeline.

These dataclasses are the contract between every layer:

* :class:`RobotState` — what a :class:`RobotIO` produces every step
* :class:`Intent`     — what a :class:`TrajectoryFollower` produces every step
* :class:`ControlCommand` — what a :class:`Controller` produces every step
* :class:`StepInfo`   — telemetry / loss / metrics produced by tasks & observers

Design notes:

* Every field that depends on the robot has variable shape; consumers query
  the accompanying :class:`~genedynamics.robots.g1.G1RobotSpec` (or whatever
  ``RobotSpec`` the IO advertises) for indexing.
* Optional fields default to ``None`` rather than empty arrays so that
  controllers can branch on availability without comparing shapes.
* Arrays are typed as :class:`numpy.ndarray` in the protocol; concrete IO
  adapters may produce ``jax.Array`` instead — controllers transfer to host
  via their declared ``runtime`` backend when needed.
* Nothing in this module imports the concrete robot implementations or any
  third-party SDK; it stays light enough to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

__all__ = [
    "RobotState",
    "Intent",
    "LocoCommand",
    "ControlCommand",
    "StepInfo",
]


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------


@dataclass
class RobotState:
    """Snapshot of a robot's measured (or simulated) state at one time step.

    Attributes:
        t: Wall-clock or simulation time in seconds.
        qpos: Generalized coordinates (typically including the floating base if
            present), shape ``(nq,)``.
        qvel: Generalized velocities, shape ``(nv,)``.
        base_pose: Optional 7-vector ``(x, y, z, qw, qx, qy, qz)`` for the
            floating base in the world frame. Convenience accessor when ``qpos``
            does not begin with the base.
        base_twist: Optional 6-vector ``(vx, vy, vz, wx, wy, wz)`` for the
            floating base in the world frame.
        joint_torque: Optional measured joint torques, shape ``(num_actuated,)``.
            Real hardware only.
        contact: Optional dict mapping site / body name → 6-vector wrench
            ``(fx, fy, fz, tx, ty, tz)`` in the world frame.
        imu: Optional dict for raw IMU readings (accel / gyro / orientation).
            Real hardware only.
        extras: Free-form passthrough for backend-specific fields.
    """

    t: float
    qpos: np.ndarray
    qvel: np.ndarray
    base_pose: Optional[np.ndarray] = None
    base_twist: Optional[np.ndarray] = None
    joint_torque: Optional[np.ndarray] = None
    contact: Optional[Mapping[str, np.ndarray]] = None
    imu: Optional[Mapping[str, np.ndarray]] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectory follower output (high-level intent)
# ---------------------------------------------------------------------------


@dataclass
class Intent:
    """Robot-agnostic high-level command produced by a :class:`TrajectoryFollower`.

    A follower converts a planner's trajectory into per-step intent. Different
    controllers consume different subsets of the fields:

    * Sport-mode follower needs only ``base_lin_vel`` / ``base_yaw_rate``
      / ``base_height``.
    * RL controller may additionally use the ``contact_phase`` and
      ``arm_targets``.
    * Whole-body controller consumes the full intent including
      ``torso_yaw`` and per-foot targets via ``extras``.

    Attributes:
        t: Time in seconds.
        base_pos_xy: Desired base position in the world horizontal plane,
            shape ``(2,)``. May be ``None`` for purely velocity-driven intent.
        base_yaw: Desired base yaw angle in radians.
        base_height: Desired base height in meters.
        base_lin_vel: Desired base linear velocity in the base (or world) frame,
            shape ``(2,)`` (vx, vy).
        base_yaw_rate: Desired base yaw rate in rad/s.
        torso_yaw: Optional decoupled torso yaw angle (humanoids only).
        arm_targets: Optional joint-name → angle map for arm posture hints.
        contact_phase: Optional phase label (``"double_support"``,
            ``"left_swing"``, ``"right_swing"``, ``"flight"``, …).
        extras: Free-form passthrough (e.g. footstep targets, swing trajectories).
    """

    t: float
    base_yaw: float
    base_height: float
    base_lin_vel: np.ndarray
    base_yaw_rate: float
    base_pos_xy: Optional[np.ndarray] = None
    torso_yaw: Optional[float] = None
    arm_targets: Optional[Mapping[str, float]] = None
    contact_phase: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Control command (controller → IO)
# ---------------------------------------------------------------------------


@dataclass
class LocoCommand:
    """High-level locomotion command consumed by sport-mode style controllers."""

    vx: float
    vy: float
    yaw_rate: float
    body_height: Optional[float] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ControlCommand:
    """Command sent from a :class:`Controller` to a :class:`RobotIO`.

    A single :class:`ControlCommand` carries one of several payload kinds,
    discriminated by :attr:`kind`:

    ============ ============================================================
    ``kind``     payload fields
    ============ ============================================================
    ``joint_pos``  ``joint_pos`` (+ optional ``kp`` / ``kd`` / ``joint_vel``)
    ``joint_vel``  ``joint_vel``
    ``torque``     ``joint_torque``
    ``loco``       ``loco_cmd`` (high-level velocity command)
    ``mixed``      ``loco_cmd`` for legs + ``joint_pos`` for upper body
    ============ ============================================================

    A :class:`RobotIO` advertises which kinds it accepts; pipelines validate
    at startup. ``extras`` is free-form passthrough for backend-specific data
    (e.g. feedforward terms, contact masks, telemetry tags).
    """

    kind: str
    joint_pos: Optional[np.ndarray] = None
    joint_vel: Optional[np.ndarray] = None
    joint_torque: Optional[np.ndarray] = None
    kp: Optional[np.ndarray] = None
    kd: Optional[np.ndarray] = None
    loco_cmd: Optional[LocoCommand] = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"joint_pos", "joint_vel", "torque", "loco", "mixed"}
        if self.kind not in valid:
            raise ValueError(f"ControlCommand.kind={self.kind!r} not in {sorted(valid)}")


# ---------------------------------------------------------------------------
# Per-step info / telemetry
# ---------------------------------------------------------------------------


@dataclass
class StepInfo:
    """Per-step bookkeeping returned by tasks and forwarded to observers."""

    done: bool = False
    success: Optional[bool] = None
    reward: Optional[float] = None
    cost: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
