"""
Backward compatibility: re-exports from framework.baseline_registry.
"""

from genedynamics.experiments.framework.baseline_registry import (
    register_baseline,
    get_baseline,
    list_baselines,
    has_baseline,
    BaselineRegistry,
)

__all__ = [
    "register_baseline",
    "get_baseline",
    "list_baselines",
    "has_baseline",
    "BaselineRegistry",
]
