"""
Robot models for environments.

This module provides robot model abstractions for:
- Manipulators (robotic arms)
- Drones (quadrotors)
- Other robot types

All robot models implement the RobotModel protocol, providing:
- Forward/inverse kinematics
- Jacobian computation
- Forward/inverse dynamics
"""

from genedynamics.envs.robots.base import RobotModel

__all__ = ["RobotModel"]

# Import concrete implementations if available
try:
    from genedynamics.envs.robots.manipulator import ManipulatorModel
    __all__.append("ManipulatorModel")
except ImportError:
    pass

try:
    from genedynamics.envs.robots.drone import DroneModel
    __all__.append("DroneModel")
except ImportError:
    pass

try:
    from genedynamics.envs.robots.g1 import G1RobotModel
    __all__.append("G1RobotModel")
except ImportError:
    pass
