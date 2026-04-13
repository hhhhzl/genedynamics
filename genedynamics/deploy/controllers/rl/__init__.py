"""Reinforcement-learning controllers for the deploy pipeline.

This package implements §14 of the deploy refactor plan: a uniform
framework for plugging third-party RL policies (Unitree RL Gym, IsaacLab,
LeggedGym, custom) behind the :class:`Controller` protocol.

Public surface:

* :class:`PolicyArtifact`     — where weights live, how to load them
* :class:`ObsBuilder`         — RobotState + Intent → obs vector (Protocol)
* :class:`ActionMapper`       — policy output → ControlCommand (Protocol)
* :class:`RLController`       — generic decimating controller
* :class:`PassthroughRLController`  — callable-based reference / test adapter
* :class:`UnitreeRLGymG1Controller` — concrete adapter for unitree_rl_gym

Adding a new policy = write one ObsBuilder + one ActionMapper, optionally
subclass :class:`RLController` to bind them, and register the result.
"""

from genedynamics.deploy.controllers.rl.base import (
    ActionMapper,
    InferenceFn,
    ObsBuilder,
    PolicyArtifact,
    RLController,
)
from genedynamics.deploy.controllers.rl.passthrough import (
    JointPosActionMapper,
    PassthroughObsBuilder,
    PassthroughRLController,
)
from genedynamics.deploy.controllers.rl.unitree_rl_gym import (
    UnitreeRLGymActionMapper,
    UnitreeRLGymG1Controller,
    UnitreeRLGymObsBuilder,
    UnitreeRLGymObsCfg,
)

__all__ = [
    "ActionMapper",
    "InferenceFn",
    "JointPosActionMapper",
    "ObsBuilder",
    "PassthroughObsBuilder",
    "PassthroughRLController",
    "PolicyArtifact",
    "RLController",
    "UnitreeRLGymActionMapper",
    "UnitreeRLGymG1Controller",
    "UnitreeRLGymObsBuilder",
    "UnitreeRLGymObsCfg",
]
