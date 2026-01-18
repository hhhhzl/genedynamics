import warnings

from .src import *

warnings.warn(
    "The environments.d3il.d3il_sim.sims.sl.teleoperation package is deprecated. Please use environments.d3il.d3il_sim.sims.sl.multibot_teleop instead",
    DeprecationWarning,
)
