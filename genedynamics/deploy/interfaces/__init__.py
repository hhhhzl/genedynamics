"""Deploy pipeline contracts.

This package defines the **only** types and protocols that cross component
boundaries in the deploy pipeline. Concrete implementations live elsewhere
(``deploy/io/``, ``deploy/controllers/``, ``deploy/followers/``,
``deploy/safety/``, ``deploy/observers/``) and depend on this package — never
the other way around.

The five protocols form a closed loop::

    TrajectoryFollower  ──Intent──▶  Controller  ──ControlCommand──▶  SafetyFilter
                                                                          │
                                                                          ▼
            Observer  ◀──StepInfo──  Pipeline  ◀───RobotState───  RobotIO
                                       │                              ▲
                                       └──────ControlCommand──────────┘

Adding a new robot, simulator backend, or runtime backend means writing
implementations that satisfy these protocols; the pipeline loop and the rest
of the components require no changes.
"""

from genedynamics.deploy.interfaces.controller import Controller
from genedynamics.deploy.interfaces.follower import TrajectoryFollower
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    LocoCommand,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.interfaces.observers import Observer
from genedynamics.deploy.interfaces.robot_io import RobotIO
from genedynamics.deploy.interfaces.safety import SafetyFilter, SafetyResult
from genedynamics.deploy.runtime_check import CommandKindError, RuntimeMismatchError

__all__ = [
    "CommandKindError",
    "Controller",
    "ControlCommand",
    "Intent",
    "LocoCommand",
    "Observer",
    "RobotIO",
    "RobotState",
    "RuntimeMismatchError",
    "SafetyFilter",
    "SafetyResult",
    "StepInfo",
    "TrajectoryFollower",
]
