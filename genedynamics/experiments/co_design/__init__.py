"""
Co-design experiment platform — backward compatibility re-exports.

All functionality has been moved to genedynamics.experiments.framework.
This module re-exports for scripts that still import from co_design.
"""

from __future__ import annotations

# Re-export from framework
from genedynamics.experiments.framework.baseline import (
    BaselineProtocol,
    BaselineResult,
    BaselineConfig,
)
from genedynamics.experiments.framework.baseline_registry import (
    register_baseline,
    get_baseline,
    list_baselines,
    has_baseline,
    BaselineRegistry,
)
from genedynamics.experiments.framework.baseline_platform import (
    BaselineExperimentPlatform as CoDesignExperimentPlatform,
    BaselineExperimentConfig as CoDesignExperimentConfig,
)
from genedynamics.experiments.common.engineering import (
    CheckpointManager,
    CheckpointState,
    ExperimentLogger,
    config_hash_dict as config_hash,
    set_seed,
)

# Import baselines to trigger registration
try:
    from genedynamics.experiments.framework import baselines as _baselines  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BaselineProtocol",
    "BaselineResult",
    "BaselineConfig",
    "register_baseline",
    "get_baseline",
    "list_baselines",
    "has_baseline",
    "BaselineRegistry",
    "CoDesignExperimentPlatform",
    "CoDesignExperimentConfig",
    "CheckpointManager",
    "CheckpointState",
    "ExperimentLogger",
    "config_hash",
    "set_seed",
]
