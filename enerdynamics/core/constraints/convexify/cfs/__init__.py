"""
CFS (Convex Feasible Set) convexifier.

Converts non-convex obstacle constraints into local convex corridors
using linearization around reference points.
"""

# Import to trigger registration
from . import cfs  
from . import backends
from .cfs import CFSConvexifier
from .action import CFSActionConvexifier

__all__ = ["CFSConvexifier", "CFSActionConvexifier"]

