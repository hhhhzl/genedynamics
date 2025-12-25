"""
Convexification module: Convert constraints to convex form.

This module provides convexifiers that transform non-convex or nonlinear
constraints into convex representations (typically linear inequalities A x >= b).

Convexifiers:
- CFS: Convex Feasible Set (non-convex obstacles -> local convex corridors)
- CBF: Control Barrier Function (nonlinear safety -> linear inequalities in u)
"""

from .base import Convexifier
from .cfs import CFSConvexifier
from .cbf import CBFConvexifier

__all__ = ["Convexifier", "CFSConvexifier", "CBFConvexifier"]

