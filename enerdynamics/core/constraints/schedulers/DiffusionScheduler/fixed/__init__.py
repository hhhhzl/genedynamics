"""
Fixed diffusion scheduler.

All diffusion parameters are constant regardless of diffusion step.
"""

# Import to trigger registration
from . import fixed  # noqa: F401
from . import backends  # noqa: F401
from .fixed import FixedDiffusionScheduler

__all__ = ["FixedDiffusionScheduler"]

