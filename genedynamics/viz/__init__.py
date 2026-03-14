"""
Unified robot motion visualization utilities.

This package provides a shared data model and renderer that can be reused by:
- deploy replay episodes (states.npy/actions.npy)
- experiment outputs (trajectory/trajectory.json)
"""

from genedynamics.viz.motion_episode import MotionEpisode
from genedynamics.viz.motion_renderer import MotionRenderer

__all__ = ["MotionEpisode", "MotionRenderer"]
