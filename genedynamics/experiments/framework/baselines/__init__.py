"""
Pluggable baseline implementations for the experiment framework.

Register baselines here. Add new baselines by implementing BaselineProtocol.
"""

from __future__ import annotations

from .mrmfmbd_baseline import MRMFMBDBaseline
from .cmaes_baseline import CMAESBaseline
from .cem_baseline import CEMBaseline

from ..baseline_registry import register_baseline

# Auto-register baselines
register_baseline(MRMFMBDBaseline())
register_baseline(CMAESBaseline())
register_baseline(CEMBaseline())

__all__ = ["MRMFMBDBaseline", "CMAESBaseline", "CEMBaseline"]
