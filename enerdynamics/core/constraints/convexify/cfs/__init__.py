"""
CFS (Convex Feasible Set) convexifier.

Converts non-convex obstacle constraints into local convex corridors
using linearization around reference points.
"""

# Import to trigger registration
from . import cfs  # noqa: F401
from .cfs import CFSConvexifier

__all__ = ["CFSConvexifier"]

