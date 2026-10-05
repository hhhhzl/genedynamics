# API surface

The V1 public surface is intentionally small:

- `genedynamics.experiments.framework.ExperimentConfig`
- `genedynamics.experiments.framework.ExperimentRunner`
- `genedynamics.experiments.framework.PluginRegistry`
- `MethodPlugin`, `EnvironmentPlugin`, `MetricsPlugin`,
  `VisualizationPlugin`, and `ObstacleGeneratorPlugin`
- core types and backend interfaces under `genedynamics.core`
- registered environment/robot factories under `genedynamics.envs` and
  `genedynamics.robots`
- deployment registries and configuration under `genedynamics.deploy`

Internal solver modules remain importable for research, but V1 compatibility is
defined at the plugin, configuration, and runner boundaries. Direct imports of
private helpers or simulator implementation classes may change before 1.0.

Use the [plugin guide](../guides/adding-a-plugin.md) for extension patterns and
the [compatibility matrix](../reference/compatibility.md) for backend coverage.
