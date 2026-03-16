"""
2GO constraint scheduler.

Time-coupled feasibility schedule for 2GO.
"""

# Import to trigger registration
from . import twogo
from . import backends
from .twogo import TwoGOConstraintScheduler

__all__ = ["TwoGOConstraintScheduler"]

