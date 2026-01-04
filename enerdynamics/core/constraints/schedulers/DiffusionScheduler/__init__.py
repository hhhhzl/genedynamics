"""
Diffusion schedulers.

Diffusion schedulers are responsible for diffusion-related parameters:
- M_k (sample size)
- T_k (temperature)
- Optional: s_k (beta scaling factor)
"""

# Import to trigger registration
from . import base  # noqa: F401
from . import fixed  # noqa: F401
from . import dualcontrol  # noqa: F401

from .base import DiffusionScheduler
from .fixed import FixedDiffusionScheduler
from .dualcontrol import DualControlDiffusionScheduler

__all__ = [
    "DiffusionScheduler",
    "FixedDiffusionScheduler",
    "DualControlDiffusionScheduler",
]

