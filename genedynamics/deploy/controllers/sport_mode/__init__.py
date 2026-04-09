"""Sport-mode controllers (mixed legs + upper-body PD).

Public surface:

* :class:`SportModeController` — :class:`Controller` protocol implementation
  combining a :class:`LocoClient` (legs) with the follower's
  :class:`HumanoidTaskSpec` joint hints (waist + arms).
* :class:`LocoClient`        — protocol for plug-in walkers.
* :class:`SparkRLLocoClient` — wraps the bundled ``g1_motion.pt``
  PPO walking policy. The default sim baseline.
* :class:`RealLocoClient`    — wraps Unitree SDK ``LocoClient`` for real G1.
"""

from genedynamics.deploy.controllers.sport_mode.controller import SportModeController
from genedynamics.deploy.controllers.sport_mode.loco_client import LocoClient
from genedynamics.deploy.controllers.sport_mode.real_loco_client import RealLocoClient
from genedynamics.deploy.controllers.sport_mode.spark_rl_loco_client import (
    SparkRLLocoClient,
)

__all__ = [
    "LocoClient",
    "RealLocoClient",
    "SparkRLLocoClient",
    "SportModeController",
]
