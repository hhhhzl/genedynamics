"""
Emerging barrier constraint scheduler.

Fixed-form (for now) scheduler that exposes barrier parameters to consumers.
Designed to be future-extendable to adaptive variants.
"""

from .emergingbarrier import EmergingBarrierConstraintScheduler

__all__ = ["EmergingBarrierConstraintScheduler"]

