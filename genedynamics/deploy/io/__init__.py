"""Robot IO adapters for the deploy pipeline.

Each adapter implements the :class:`RobotIO` protocol from
:mod:`genedynamics.deploy.interfaces` and exposes a small set of physics
query methods that controllers may consume directly. Sim/real switching is
one config-line change of ``class_name``; the rest of the pipeline is
agnostic.

* :class:`BaseRobotIO`   — abstract base, command latching, episode bookkeeping
* :class:`MujocoRobotIO` — numpy MuJoCo (CPU)
* :class:`MjxRobotIO`    — JAX MJX (GPU/TPU), default for new presets
"""

from genedynamics.deploy.io.base import BaseRobotIO
from genedynamics.deploy.io.mujoco_io import FootContactSnapshot, MujocoRobotIO
from genedynamics.deploy.io.stub_io import StubRobotIO, StubSpec

# Optional: MjxRobotIO needs jax / mujoco-mjx; gracefully degrade.
try:
    from genedynamics.deploy.io.mjx_io import MjxRobotIO
    _HAS_MJX = True
except ImportError:
    _HAS_MJX = False
    MjxRobotIO = None  # type: ignore[assignment]

# Optional: UnitreeG1RobotIO needs the Unitree SDK. The module itself imports
# fine without it; the SDK import is deferred until construction.
from genedynamics.deploy.io.unitree_g1_io import UnitreeG1RobotIO


# Optional: BraxRobotIO needs brax + jax; gracefully degrade.
try:
    from genedynamics.deploy.io.brax_io import BraxRobotIO
    _HAS_BRAX = True
except ImportError:
    _HAS_BRAX = False
    BraxRobotIO = None  # type: ignore[assignment]

# Optional: IsaacLabRobotIO needs omni.isaac.lab + torch; gracefully degrade.
try:
    from genedynamics.deploy.io.isaac_lab_io import IsaacLabRobotIO
    _HAS_ISAAC_LAB = True
except ImportError:
    _HAS_ISAAC_LAB = False
    IsaacLabRobotIO = None  # type: ignore[assignment]


__all__ = [
    "BaseRobotIO",
    "FootContactSnapshot",
    "MujocoRobotIO",
    "StubRobotIO",
    "StubSpec",
    "UnitreeG1RobotIO",
]
if _HAS_MJX:
    __all__.append("MjxRobotIO")
if _HAS_BRAX:
    __all__.append("BraxRobotIO")
if _HAS_ISAAC_LAB:
    __all__.append("IsaacLabRobotIO")
