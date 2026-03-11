"""
Baseline algorithm implementations for comparison experiments.

Each baseline (mrmfmbd, cmaes, etc.) implements BaselineProtocol.
Register here to make baselines available to the platform.
"""

from __future__ import annotations

from .mrmfmbd import MRMFMBDBaseline

from ...framework.baseline_registry import register_baseline

# Auto-register MRMFMBD
register_baseline(MRMFMBDBaseline())

__all__ = ["MRMFMBDBaseline"]
