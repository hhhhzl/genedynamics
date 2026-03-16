"""
Diffusion schedulers.

Diffusion schedulers are responsible for diffusion-related parameters:
- M_k (sample size)
- T_k (temperature)
- Optional: s_k (beta scaling factor)
"""

# Import to trigger registration
from . import base  
from . import fixed  
from . import dualcontrol
from . import twogo

from .base import DiffusionScheduler
from .fixed import FixedDiffusionScheduler
from .dualcontrol import DualControlDiffusionScheduler
from .twogo import TwoGODiffusionScheduler

__all__ = [
    "DiffusionScheduler",
    "FixedDiffusionScheduler",
    "DualControlDiffusionScheduler",
    "TwoGODiffusionScheduler",
]

