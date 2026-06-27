"""
Schedule overlay system for geometry-aware parameter modulation.

Maps constraint penalty (rho) and margin through hardness, constraint,
and diffusion overlays to produce per-step solver parameters
(kappa, delta, sigma, theta, eta).

Auto-registers available backends on import.
"""

from genedynamics.genemetry.schedule.overlay import ScheduleOverlay
from genedynamics.genemetry.schedule.config import (
    OverlayConfig,
    resolve_overlay_config,
)

try:
    from genedynamics.genemetry.schedule.backends import overlay_jax  
except ImportError:
    pass

try:
    from genedynamics.genemetry.schedule.backends import overlay_numpy  
except ImportError:
    pass

__all__ = ["ScheduleOverlay", "OverlayConfig", "resolve_overlay_config"]
