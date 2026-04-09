"""Teleoperation input layer for the deploy pipeline.

Provides the bridge between human input devices and the standard runner loop:

* :class:`TeleopCommand`   — robot-agnostic command struct
* :class:`InputSource`     — non-blocking device protocol
* :class:`GamepadSource`   — pygame gamepad (Xbox / DualShock)
* :class:`KeyboardSource`  — pynput keyboard (WASD, no GUI required)
* :class:`ROS2TwistSource` — ROS2 ``geometry_msgs/Twist`` subscriber
* :class:`TeleopFollower`  — :class:`TrajectoryFollower` that reads from
  any :class:`InputSource` and produces :class:`Intent` each step

Typical use::

    from genedynamics.deploy.teleop import TeleopFollower, GamepadSource

    follower = TeleopFollower(source=GamepadSource(), base_height_nominal=0.75)
    # ... plug follower into runner.run_preset(preset) or a hand-rolled loop
"""

from genedynamics.deploy.teleop.input_source import InputSource, TeleopCommand
from genedynamics.deploy.teleop.gamepad_source import GamepadSource
from genedynamics.deploy.teleop.keyboard_source import KeyboardSource
from genedynamics.deploy.teleop.ros2_source import ROS2TwistSource
from genedynamics.deploy.teleop.teleop_follower import TeleopFollower

__all__ = [
    "TeleopCommand",
    "InputSource",
    "GamepadSource",
    "KeyboardSource",
    "ROS2TwistSource",
    "TeleopFollower",
]
