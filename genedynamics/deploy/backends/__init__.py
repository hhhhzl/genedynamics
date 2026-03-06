"""
Hardware backends for real robot deployment.

State backends: provide (qpos, qvel) from robot/SDK.
Control backends: send joint targets to robot.
"""

from genedynamics.deploy.backends.stub import StubStateBackend, StubControlBackend
from genedynamics.deploy.backends.localization_adapter import LocalizationStateBackend

__all__ = ["StubStateBackend", "StubControlBackend", "LocalizationStateBackend"]

UNITREE_AVAILABLE = False
try:
    from genedynamics.deploy.backends.unitree_go2 import (  # noqa: F401
        UnitreeGo2StateBackend,
        UnitreeGo2ControlBackend,
    )
    UNITREE_AVAILABLE = True
except ImportError:
    pass
