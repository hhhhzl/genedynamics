"""
Robot registry for multi-backend deploy (quadruped, humanoid).

Register models, env classes, and specs for extensibility.
"""

from genedynamics.robots.registry import (
    RobotEntry,
    get_robot_registry,
    register_quadruped,
    register_humanoid,
)
from genedynamics.robots.profile import NamedElement, RobotBinding, RobotProfile

__all__ = [
    "RobotEntry",
    "get_robot_registry",
    "register_quadruped",
    "register_humanoid",
    "NamedElement",
    "RobotBinding",
    "RobotProfile",
]
