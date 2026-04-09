"""Teleoperation input source protocol and shared command type.

An :class:`InputSource` is the only hardware-facing object in the teleop
stack — everything above it (follower, controller, safety) is unchanged from
the normal deploy loop.

``TeleopCommand`` carries the raw human intent in a robot-agnostic form:

* ``vx / vy / yaw_rate`` — base velocity commands (m/s, rad/s)
* ``body_height_delta`` — incremental height adjustment (m, per step)
* ``arm_delta`` — optional per-joint arm nudges keyed by joint name (rad)
* ``stop`` — user pressed emergency-stop / controller disconnected
* ``extras`` — free-form passthrough for source-specific signals

The :class:`InputSource` protocol is deliberately minimal: ``read()`` is
called once per control step and must never block for more than a few
milliseconds (real-time constraint). Implementations handle all buffering
internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, runtime_checkable

__all__ = ["TeleopCommand", "InputSource"]


@dataclass
class TeleopCommand:
    """Robot-agnostic teleoperation command produced by an :class:`InputSource`.

    Attributes:
        vx: Forward base velocity command in m/s (body frame).
        vy: Lateral base velocity command in m/s (body frame).
        yaw_rate: Yaw-rate command in rad/s.
        body_height_delta: Incremental height offset applied each step (m).
        arm_delta: Optional per-joint arm position nudge keyed by joint name
            (rad/step). Empty dict = no arm command.
        stop: When ``True`` the runner should freeze the robot and end the
            episode. Set on E-stop or source disconnection.
        extras: Free-form passthrough (e.g. gripper open/close, mode flags).
    """

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    body_height_delta: float = 0.0
    arm_delta: Dict[str, float] = field(default_factory=dict)
    stop: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InputSource(Protocol):
    """Non-blocking teleoperation input device protocol.

    Implementations must be safe to call at the control frequency (50–200 Hz).
    Any I/O (sockets, HID polling) must happen on a background thread, with
    :meth:`read` only reading from an in-process buffer.

    Attributes:
        name: Short identifier used in logs and observers.
    """

    name: str

    def open(self) -> None:
        """Open the device and start any background polling threads."""
        ...

    def read(self) -> TeleopCommand:
        """Return the latest command without blocking.

        Returns a zeroed :class:`TeleopCommand` (with ``stop=True`` if the
        device is disconnected) when no new data is available.
        """
        ...

    def close(self) -> None:
        """Release the device and stop background threads."""
        ...
