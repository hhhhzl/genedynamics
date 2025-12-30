"""
CFS (Convex Feasible Set) convexifier.

Converts non-convex obstacle constraints into local convex corridors
using linearization around reference points.
"""

# Import to trigger registration
from . import cfs  # noqa: F401
from . import backends  # noqa: F401 - Import backends to trigger registration
from .cfs import CFSConvexifier

__all__ = ["CFSConvexifier"]

