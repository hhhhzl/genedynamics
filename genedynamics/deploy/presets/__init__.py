"""Nested-class deploy config presets.

Each preset subclasses :class:`genedynamics.deploy.config_schema.DeployConfig`
and defines its components as nested ``ComponentConfig`` classes. Presets
are meant to be cheap to read and to copy when constructing a variant — see
:class:`G1CorridorMujocoSportModePreset` for the canonical example.

Switching robots, simulators or controllers is supposed to feel like
inheriting and overriding one inner class. The two paired presets below
demonstrate the sim → real transition that Phase 7 was supposed to make
trivial:

* :class:`G1CorridorMujocoSportModePreset` — sim baseline (MuJoCo + mock loco)
* :class:`G1CorridorRealSportModePreset`   — real G1 (Unitree IO + real loco)

The actual run logic lives in :mod:`genedynamics.deploy.runner`; presets
just declare what to build.
"""

from genedynamics.deploy.presets.g1_corridor_mujoco_sport_mode import (
    G1CorridorMujocoSportModePreset,
)
from genedynamics.deploy.presets.g1_corridor_mujoco_wbc import (
    G1CorridorMujocoWBCPreset,
)
from genedynamics.deploy.presets.g1_corridor_real_sport_mode import (
    G1CorridorRealSportModePreset,
)
from genedynamics.deploy.presets.go2_stepping_stones import (
    Go2SteppingStonesMujocoPreset,
)

__all__ = [
    "G1CorridorMujocoSportModePreset",
    "G1CorridorMujocoWBCPreset",
    "G1CorridorRealSportModePreset",
    "Go2SteppingStonesMujocoPreset",
]
