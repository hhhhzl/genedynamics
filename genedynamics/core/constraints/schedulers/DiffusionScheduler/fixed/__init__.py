"""
Fixed diffusion scheduler.

All diffusion parameters are constant regardless of diffusion step.
"""

# Import to trigger registration
from . import fixed  
from . import backends  
from .fixed import FixedDiffusionScheduler

__all__ = ["FixedDiffusionScheduler"]

