"""Public exports loaded on demand so optional backends remain optional."""

from importlib import import_module

__all__ = ['MethodPlugin', 'EnvironmentPlugin', 'MetricsPlugin', 'VisualizationPlugin', 'ObstacleGeneratorPlugin', 'ExperimentConfig', 'ExperimentRunner', 'PluginRegistry']

_EXPORTS = {'MethodPlugin': ('genedynamics.experiments.framework.base', 'MethodPlugin'), 'EnvironmentPlugin': ('genedynamics.experiments.framework.base', 'EnvironmentPlugin'), 'MetricsPlugin': ('genedynamics.experiments.framework.base', 'MetricsPlugin'), 'VisualizationPlugin': ('genedynamics.experiments.framework.base', 'VisualizationPlugin'), 'ObstacleGeneratorPlugin': ('genedynamics.experiments.framework.base', 'ObstacleGeneratorPlugin'), 'ExperimentConfig': ('genedynamics.experiments.framework.config', 'ExperimentConfig'), 'ExperimentRunner': ('genedynamics.experiments.framework.experiment', 'ExperimentRunner'), 'PluginRegistry': ('genedynamics.experiments.framework.registry', 'PluginRegistry')}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))

_EXPORTS["BaseRunner"] = ("genedynamics.experiments.framework.base_runner", "BaseRunner")
__all__.append("BaseRunner")
