"""
Experimental framework for unified experiment execution.

This package provides a plugin-based framework for running experiments with
different methods, environments, metrics, and visualizations.
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

__all__ = [
    "MethodPlugin",
    "EnvironmentPlugin",
    "MetricsPlugin",
    "VisualizationPlugin",
    "ObstacleGeneratorPlugin",
    "ExperimentConfig",
    "ExperimentRunner",
    "PluginRegistry",
]
