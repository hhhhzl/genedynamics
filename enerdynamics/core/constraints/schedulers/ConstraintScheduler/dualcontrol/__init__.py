"""
Dual-control constraint scheduler.

Adaptively adjusts constraint parameters based on feasibility feedback
using dual variable λ^con.
"""

# Import to trigger registration
from . import dual_control  # noqa: F401
from . import backends  # noqa: F401
from .dual_control import DualControlConstraintScheduler

__all__ = ["DualControlConstraintScheduler"]

