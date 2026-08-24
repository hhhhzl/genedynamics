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
# `baseline.py` keeps the co-design optimizer interface (Config/Result/Protocol)
# used by the relocated optimizers in solvers/single/codesign_optimizers/.
from .baseline import BaselineProtocol, BaselineConfig, BaselineResult
from .codesign_runner import (
    run_codesign,
    register_codesign_solver,
    get_codesign_solver,
    list_codesign_solvers,
    CoDesignResult,
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
    "run_codesign",
    "register_codesign_solver",
    "get_codesign_solver",
    "list_codesign_solvers",
    "CoDesignResult",
    "BaselineExperimentPlatform",
    "BaselineExperimentConfig",
    "TaskDomainProvider",
    "register_task_domain_provider",
    "get_task_domain_provider",
    "list_task_domains",
]
