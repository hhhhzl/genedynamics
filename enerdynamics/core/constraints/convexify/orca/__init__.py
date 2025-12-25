"""
ORCA (Optimal Reciprocal Collision Avoidance) convexifier.

ORCA converts velocity obstacles into halfspace constraints for multi-agent
collision avoidance.
"""

from .orca import ORCAConvexifier

__all__ = ["ORCAConvexifier"]

