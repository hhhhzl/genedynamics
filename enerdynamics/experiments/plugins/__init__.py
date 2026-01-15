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
    EBMBDMethodPlugin,
    MBDMethodPlugin,
)

from .environments import (
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    DroneEnvironmentPlugin,
    DroneBox3DPlugin,
    DroneFull3DPlugin,
    DroneFull3DPhysicsPlugin,
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
    Diffusion3DVisualizationPlugin,
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
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    # Environments
    'SingleIntegrator2DPlugin',
    'DoubleIntegrator2DPlugin',
    'DroneEnvironmentPlugin',
    'DroneBox3DPlugin',
    'DroneFull3DPlugin',
    'DroneFull3DPhysicsPlugin',
    'ManipulatorEnvironmentPlugin',
    # Metrics
    'SSRMetricsPlugin',
    'ObstacleDensityMetricsPlugin',
    'NonconvexityMetricsPlugin',
    # Visualizations
    'TrajectoryVisualizationPlugin',
    'Trajectory3DVisualizationPlugin',
    'DiffusionVisualizationPlugin',
    'Diffusion3DVisualizationPlugin',
    'EnergyRewardVisualizationPlugin',
    'StatesVisualizationPlugin',
    'SchedulerParamsVisualizationPlugin',
    # Obstacles
    'Box2DObstacleGeneratorPlugin',
    'Box3DObstacleGeneratorPlugin',
]

