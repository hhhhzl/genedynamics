"""
Deploy pipeline — Phase 9 architecture.

Entry point:  ``genedynamics-deploy`` → :func:`genedynamics.deploy.runner.main`

Core modules
------------
* :mod:`runner`       — registry-driven run loop
* :mod:`registries`   — component registries (io, controller, safety, …)
* :mod:`config_schema`— :class:`DeployConfig` / :class:`ComponentConfig` (nested-class style)
* :mod:`presets`      — ready-made configs for G1 corridor

Protocol interfaces
-------------------
* :mod:`interfaces.messages`   — :class:`RobotState`, :class:`Intent`, :class:`ControlCommand`
* :mod:`interfaces.robot_io`   — :class:`RobotIO` protocol
* :mod:`interfaces.controller` — :class:`Controller` protocol

IO backends
-----------
* :mod:`io.mujoco_io`     — :class:`MujocoRobotIO`
* :mod:`io.mjx_io`        — :class:`MjxRobotIO`
* :mod:`io.unitree_g1_io` — :class:`UnitreeG1RobotIO`
* :mod:`io.stub_io`       — :class:`StubRobotIO`

Controllers
-----------
* :mod:`controllers.wbc`        — :class:`HumanoidWBCController`
* :mod:`controllers.sport_mode` — :class:`SportModeController` + :class:`SparkRLLocoClient`
* :mod:`controllers.rl`         — :class:`RLController`, :class:`UnitreeRLGymG1Controller`
"""

from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig
from genedynamics.deploy.runner import main as run

__all__ = [
    "ComponentConfig",
    "DeployConfig",
    "run",
]
