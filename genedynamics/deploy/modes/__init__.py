"""
Execution modes for deploy pipeline.

Each mode (sim, real, shadow, replay) knows how to build and run its execution.
"""

from genedynamics.deploy.modes.base import ExecutionModeProtocol
from genedynamics.deploy.modes.sim import SimMode
from genedynamics.deploy.modes.shadow import ShadowMode
from genedynamics.deploy.modes.replay import ReplayMode
from genedynamics.deploy.modes.real import RealMode

__all__ = [
    "ExecutionModeProtocol",
    "get_mode",
    "SimMode",
    "ShadowMode",
    "ReplayMode",
    "RealMode",
]

_MODES = {
    "sim": SimMode(),
    "shadow": ShadowMode(),
    "replay": ReplayMode(),
    "real": RealMode(),
}


def get_mode(mode_name: str) -> ExecutionModeProtocol:
    """Get mode by name."""
    name = mode_name.lower().strip()
    if name not in _MODES:
        raise ValueError(f"Unknown mode: {mode_name}. Available: {list(_MODES.keys())}")
    return _MODES[name]
