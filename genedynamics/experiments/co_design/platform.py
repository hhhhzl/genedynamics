"""
Backward compatibility: re-exports from framework.baseline_platform.
"""

from genedynamics.experiments.framework.baseline_platform import (
    BaselineExperimentConfig as CoDesignExperimentConfig,
    BaselineExperimentPlatform as CoDesignExperimentPlatform,
)

__all__ = ["CoDesignExperimentConfig", "CoDesignExperimentPlatform"]
