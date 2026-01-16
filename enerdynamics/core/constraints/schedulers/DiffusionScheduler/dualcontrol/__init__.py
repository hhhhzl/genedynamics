"""
Dual-control diffusion scheduler.

Adaptively adjusts diffusion parameters based on diversity (ESS) feedback
using dual variable λ^diff.
"""

# Import to trigger registration
from . import dual_control  
from . import backends  
from .dual_control import DualControlDiffusionScheduler

__all__ = ["DualControlDiffusionScheduler"]

