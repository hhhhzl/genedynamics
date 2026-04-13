"""
Backward compatibility: re-exports from framework.baseline.
"""

from genedynamics.experiments.framework.baseline import (
    BaselineConfig,
    BaselineResult,
    BaselineProtocol,
)

__all__ = ["BaselineConfig", "BaselineResult", "BaselineProtocol"]
