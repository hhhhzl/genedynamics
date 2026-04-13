"""Loco client protocol for sport-mode controllers.

A :class:`LocoClient` is a small interface that converts a high-level
:class:`LocoCommand` (vx, vy, yaw_rate, body_height) into per-leg joint
position targets at every control tick. The controller layer
(:class:`SportModeController`) is the same regardless of which client is
plugged in:

* :class:`SparkRLLocoClient` — wraps the bundled ``g1_motion.pt`` PPO
  policy. Default sim baseline; trained against G1 in IsaacGym/MJX.
* :class:`RealLocoClient`    — wraps Unitree LocoClient SDK on real G1.

The protocol intentionally does **not** speak in joint-space directly.
Clients return a ``Mapping[str, float]`` keyed by joint name so the
controller can merge the leg targets with upper-body targets without
worrying about index alignment.
"""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from genedynamics.deploy.interfaces.messages import LocoCommand, RobotState

__all__ = ["LocoClient"]


@runtime_checkable
class LocoClient(Protocol):
    """Translate :class:`LocoCommand` into leg joint targets.

    Implementations are stateful (they own a phase clock, swing schedule,
    foot anchors, etc.) and must be reset at the start of each episode.

    Attributes:
        nominal_step_period: Step period in seconds. Used by the controller
            for diagnostics and (optionally) by safety filters that want to
            know when the next foot transition will occur.
    """

    nominal_step_period: float

    def reset(
        self,
        pelvis_world: np.ndarray,
        pelvis_yaw: float = 0.0,
    ) -> None:
        """Anchor the client to the current pelvis pose.

        Args:
            pelvis_world: 3-vector pelvis position in the world frame.
            pelvis_yaw: Pelvis yaw in radians (z-axis rotation).
        """
        ...

    def step(
        self,
        cmd: LocoCommand,
        dt: float,
        pelvis_world: np.ndarray,
        pelvis_yaw: float,
        state: Optional[RobotState] = None,
    ) -> Mapping[str, float]:
        """Advance one control tick and return per-leg joint targets.

        Args:
            cmd: High-level (vx, vy, yaw_rate, body_height) command.
            dt: Control period in seconds.
            pelvis_world: 3-vector pelvis position in the world frame.
            pelvis_yaw: Pelvis yaw in radians.
            state: Optional full :class:`RobotState`. Required by RL-policy
                clients (which need joint qpos/qvel + base angular velocity
                + IMU/quaternion to build the obs vector). Template-walker
                clients ignore it; the parameter is optional so existing
                callers do not have to pass anything.

        Returns:
            Mapping ``{joint_name: angle}`` for the 12 leg joints (6 per leg).
            Joint names match :attr:`G1RobotSpec.left_leg_joints` and
            :attr:`G1RobotSpec.right_leg_joints` so the controller can splice
            them directly into a full 29-vector.
        """
        ...

    @property
    def swing_foot(self) -> Optional[str]:
        """``"left"`` / ``"right"`` / ``None`` — which foot is currently swinging."""
        ...

    @property
    def phase(self) -> float:
        """Phase progression of the current step in ``[0, 1)``."""
        ...
