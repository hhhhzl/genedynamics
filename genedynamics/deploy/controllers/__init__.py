"""Controllers consumed by the deploy pipeline.

Each subpackage hosts an implementation of the
:class:`~genedynamics.deploy.interfaces.controller.Controller` protocol:

* :mod:`wbc`                — humanoid whole-body inverse-dynamics controller
* :mod:`sport_mode`         — LocoClient adapter (spark RL policy / Unitree SDK)
* :mod:`rl`                 — third-party RL policy framework
* :mod:`quadruped_stepping` — batch stepping-walk controller for Go2
"""

from genedynamics.deploy.controllers.quadruped_stepping import (
    MinimalFollowerConfig,
    QuadrupedSteppingController,
    SteppingWalkFollowerMinimal,
    load_stepping_plan_from_seed_dir,
)
from genedynamics.deploy.controllers.rl import (
    PassthroughRLController,
    PolicyArtifact,
    RLController,
    UnitreeRLGymG1Controller,
)
from genedynamics.deploy.controllers.sport_mode import (
    LocoClient,
    RealLocoClient,
    SparkRLLocoClient,
    SportModeController,
)
from genedynamics.deploy.controllers.wbc import (
    HumanoidWBCController,
    LimitsConfig,
    SolverConfig,
    TaskGainsConfig,
    TaskWeightsConfig,
    WBCConfig,
)

__all__ = [
    # WBC
    "HumanoidWBCController",
    "LimitsConfig",
    "SolverConfig",
    "TaskGainsConfig",
    "TaskWeightsConfig",
    "WBCConfig",
    # Sport-mode
    "LocoClient",
    "RealLocoClient",
    "SparkRLLocoClient",
    "SportModeController",
    # RL
    "PassthroughRLController",
    "PolicyArtifact",
    "RLController",
    "UnitreeRLGymG1Controller",
    # Quadruped stepping
    "QuadrupedSteppingController",
    "MinimalFollowerConfig",
    "SteppingWalkFollowerMinimal",
    "load_stepping_plan_from_seed_dir",
]
