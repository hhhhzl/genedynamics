"""Whole-body controller for humanoid robots.

Public surface:

* :class:`HumanoidWBCController` — implements the
  :class:`~genedynamics.deploy.interfaces.controller.Controller` protocol
* :class:`WBCConfig`              — top-level grouped config
* :class:`TaskWeightsConfig` / :class:`TaskGainsConfig` /
  :class:`LimitsConfig` / :class:`SolverConfig` — sub-configs
* :class:`WBCResult`              — diagnostic bundle attached to the emitted
  ``ControlCommand``

Lower-level pieces (``task_stack``, ``contact_blocks``, ``friction_cone``,
``qp_builder``, ``qp_solver_adapter``) are exposed as submodules for unit
testing and reuse but are not part of the recommended import surface.
"""

from genedynamics.deploy.controllers.wbc.config import (
    LimitsConfig,
    SolverConfig,
    TaskGainsConfig,
    TaskWeightsConfig,
    WBCConfig,
)
from genedynamics.deploy.controllers.wbc.controller import (
    HumanoidWBCController,
    WBCResult,
)

__all__ = [
    "HumanoidWBCController",
    "LimitsConfig",
    "SolverConfig",
    "TaskGainsConfig",
    "TaskWeightsConfig",
    "WBCConfig",
    "WBCResult",
]
