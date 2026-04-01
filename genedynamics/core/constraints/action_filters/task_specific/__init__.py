"""
Task-specific constraint filter plugins.

These filters extend the generic ConstraintFilter interface with
domain-specific knowledge (e.g. per-foot stepping-stone geometry).
They are NOT part of the core action_filters package and should be
imported explicitly by the method plugin that needs them.
"""

from .cbf_stepping import SteppingCBFFilter

__all__ = ["SteppingCBFFilter"]
