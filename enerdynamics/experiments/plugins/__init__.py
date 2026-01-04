"""
Plugin implementations for the experimental framework.

This package contains implementations of various plugins:
- methods: Solver method plugins (EDOC, MPPI, CEM, etc.)
- environments: Environment plugins (single/double integrator, etc.)
- metrics: Metrics computation plugins (SSR, obstacle density, etc.)
- visualizations: Visualization plugins
- obstacles: Obstacle generation plugins
"""

# Import and export all plugins for convenient access
from .methods import (
    EDOCMethodPlugin,
)

from .environments import (
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    DroneEnvironmentPlugin,
    DroneBox3DPlugin,
    ManipulatorEnvironmentPlugin,
)

from .metrics import (
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
)

from .visualizations import (
    TrajectoryVisualizationPlugin,
    Trajectory3DVisualizationPlugin,
    DiffusionVisualizationPlugin,
    EnergyRewardVisualizationPlugin,
    StatesVisualizationPlugin,
    SchedulerParamsVisualizationPlugin,
)

from .obstacles import (
    Box2DObstacleGeneratorPlugin,
    Box3DObstacleGeneratorPlugin,
)

__all__ = [
    # Methods
    'EDOCMethodPlugin',
    # Environments
    'SingleIntegrator2DPlugin',
    'DoubleIntegrator2DPlugin',
    'DroneEnvironmentPlugin',
    'DroneBox3DPlugin',
    'ManipulatorEnvironmentPlugin',
    # Metrics
    'SSRMetricsPlugin',
    'ObstacleDensityMetricsPlugin',
    'NonconvexityMetricsPlugin',
    # Visualizations
    'TrajectoryVisualizationPlugin',
    'Trajectory3DVisualizationPlugin',
    'DiffusionVisualizationPlugin',
    'EnergyRewardVisualizationPlugin',
    'StatesVisualizationPlugin',
    'SchedulerParamsVisualizationPlugin',
    # Obstacles
    'Box2DObstacleGeneratorPlugin',
    'Box3DObstacleGeneratorPlugin',
]

