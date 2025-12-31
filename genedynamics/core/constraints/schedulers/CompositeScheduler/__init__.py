"""
Composite scheduler: Combines multiple constraint and diffusion schedulers.
"""

# Import to trigger registration
from . import composite  # noqa: F401
from .composite import CompositeScheduler, MergeStrategy

__all__ = ["CompositeScheduler", "MergeStrategy"]

