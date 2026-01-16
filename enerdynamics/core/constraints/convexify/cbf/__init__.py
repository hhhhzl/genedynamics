"""
CBF (Control Barrier Function) convexifier.

Converts nonlinear safety constraints into linear inequalities in control space.
"""

# Import to trigger registration
from . import cbf  
from .cbf import CBFConvexifier

__all__ = ["CBFConvexifier"]

