"""
Windowing policies for horizon-local geometry operations.

Auto-registers available backends on import.
"""

from genedynamics.genemetry.window.sliding import SlidingWindow

try:
    from genedynamics.genemetry.window.backends import multimodality_numpy  # noqa: F401
except ImportError:
    pass

__all__ = ["SlidingWindow"]
