"""
Co-design experiment platform.

Baseline-agnostic interface: plug any baseline (MRMFMBD, ES, CMA-ES, etc.)
into the experiment runner. No baseline implementations yet - interface only.
"""

from __future__ import annotations

from .baseline import BaselineProtocol, BaselineResult, BaselineConfig
from .registry import BaselineRegistry, register_baseline, get_baseline, list_baselines
from .platform import CoDesignExperimentPlatform, CoDesignExperimentConfig
from .engineering import (
    CheckpointManager,
    CheckpointState,
    ExperimentLogger,
    config_hash,
    set_seed,
)

# Import baselines to trigger registration
try:
    from . import baselines  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BaselineProtocol",
    "BaselineResult",
    "BaselineConfig",
    "BaselineRegistry",
    "register_baseline",
    "get_baseline",
    "list_baselines",
    "CoDesignExperimentPlatform",
    "CoDesignExperimentConfig",
    "CheckpointManager",
    "CheckpointState",
    "ExperimentLogger",
    "config_hash",
    "set_seed",
]
