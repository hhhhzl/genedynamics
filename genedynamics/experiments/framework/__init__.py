"""
Experimental framework for unified experiment execution.

This package provides a plugin-based framework for running experiments with
different methods, environments, metrics, and visualizations.
Also includes baseline comparison infrastructure (task-agnostic).
"""

from .base import (
    MethodPlugin,
    EnvironmentPlugin,
    MetricsPlugin,
    VisualizationPlugin,
    ObstacleGeneratorPlugin,
)
from .config import ExperimentConfig
from .experiment import ExperimentRunner
from .registry import PluginRegistry
from .baseline import BaselineProtocol, BaselineConfig, BaselineResult
from .baseline_registry import (
    BaselineRegistry,
    register_baseline,
    get_baseline,
    list_baselines,
)
from .baseline_platform import BaselineExperimentPlatform, BaselineExperimentConfig
from .task_domain_provider import (
    TaskDomainProvider,
    register_task_domain_provider,
    get_task_domain_provider,
    list_task_domains,
)

__all__ = [
    "MethodPlugin",
    "EnvironmentPlugin",
    "MetricsPlugin",
    "VisualizationPlugin",
    "ObstacleGeneratorPlugin",
    "ExperimentConfig",
    "ExperimentRunner",
    "PluginRegistry",
    "BaselineProtocol",
    "BaselineConfig",
    "BaselineResult",
    "BaselineRegistry",
    "register_baseline",
    "get_baseline",
    "list_baselines",
    "BaselineExperimentPlatform",
    "BaselineExperimentConfig",
    "TaskDomainProvider",
    "register_task_domain_provider",
    "get_task_domain_provider",
    "list_task_domains",
]

