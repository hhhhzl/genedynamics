"""
2GO diffusion scheduler.

Coupled diffusion schedule for 2GO.
"""

# Import to trigger registration
from . import twogo
from . import backends
from .twogo import TwoGODiffusionScheduler

__all__ = ["TwoGODiffusionScheduler"]

