"""
Co-design baseline implementations.

Register baselines here. Add new baselines by implementing BaselineProtocol.
"""

from __future__ import annotations

from .mrmfmbd_baseline import MRMFMBDBaseline

from ..registry import register_baseline

# Auto-register MRMFMBD
register_baseline(MRMFMBDBaseline())

__all__ = ["MRMFMBDBaseline"]
