"""
Plugin implementations for the experimental framework.

This package contains implementations of various plugins:
- methods: Solver method plugins (MBD, MDOC, CFS-MBD, MPPI, etc.)
- environments: Environment plugins (single/double integrator, etc.)
- metrics: Metrics computation plugins (SSR, obstacle density, etc.)
- visualizations: Visualization plugins
- obstacles: Obstacle generation plugins
"""

# Import and export all plugins for convenient access
from .methods import (
    EBMBDMethodPlugin,
    MBDMethodPlugin,
    D3ILUnifiedMethodPlugin,
    MDOCMethodPlugin,
    CFSMBDMethodPlugin,
    CFSMBDFullMethodPlugin,
    MPPIMethodPlugin,
    TwoGOMethodPlugin,
    DPCCMethodPlugin,
    SafeDiffuserMethodPlugin,
    MGAMethodPlugin,
    ModelBasedOnlyMethodPlugin,
    DIALContactMethodPlugin,
    PegasusFlowContactMethodPlugin,
    ISSAContactMethodPlugin,
    ATACOMContactMethodPlugin,
    StandaloneRLMethodPlugin,
)

from .environments import (
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    QuadrupedFlatMjxPlugin,
    QuadrupedGo2MjxPlugin,
    QuadrupedGo2BraxPlugin,
    QuadrupedSteppingStones2DPlugin,
    HumanoidSimplifiedMjxPlugin,
    HumanoidG1MjxPlugin,
    HumanoidRunBraxPlugin,
    DroneEnvironmentPlugin,
    DroneBox3DPlugin,
    DroneFull3DPlugin,
    DroneFull3DPhysicsPlugin,
    ManipulatorEnvironmentPlugin,
    D3ILAvoidingPlugin,
    D3ILAvoiding9DPlugin,
    AvoidingPlanEnvironmentPlugin,
    HumanoidCorridor2DPlugin,
    ManipulatorSurfaceScanPlugin,
    ManipulatorPegInsertPlugin,
    HumanoidBoxPushPlugin,
)

from .metrics import (
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
    EpisodeOutcomeMetricsPlugin,
    SteppingStonesMetricsPlugin,
    CorridorMetricsPlugin,
)

from .visualizations import (
    TrajectoryVisualizationPlugin,
    Trajectory3DVisualizationPlugin,
    TrajectoryModesVisualizationPlugin,
    DiffusionVisualizationPlugin,
    Diffusion3DVisualizationPlugin,
    EnergyRewardVisualizationPlugin,
    StatesVisualizationPlugin,
    SchedulerParamsVisualizationPlugin,
    SteppingStonesTrajectoryVisualizationPlugin,
    SteppingStonesModesVisualizationPlugin,
    CorridorTrajectoryVisualizationPlugin,
)

from .obstacles import (
    Box2DObstacleGeneratorPlugin,
    Box3DObstacleGeneratorPlugin,
    D3ILAvoidingFixedGeneratorPlugin,
    SteppingStones2DObstacleGeneratorPlugin,
    Corridor2DObstacleGeneratorPlugin,
    EmptyObstacleGeneratorPlugin,
)

__all__ = [
    # Methods
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    'D3ILUnifiedMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'CFSMBDFullMethodPlugin',
    'MPPIMethodPlugin',
    'TwoGOMethodPlugin',
    'DPCCMethodPlugin',
    'SafeDiffuserMethodPlugin',
    'MGAMethodPlugin',
    'ModelBasedOnlyMethodPlugin',
    'DIALContactMethodPlugin',
    'PegasusFlowContactMethodPlugin',
    'ISSAContactMethodPlugin',
    'ATACOMContactMethodPlugin',
    'StandaloneRLMethodPlugin',
    # Environments
    'SingleIntegrator2DPlugin',
    'DoubleIntegrator2DPlugin',
    'QuadrupedFlatMjxPlugin',
    'QuadrupedGo2MjxPlugin',
    'QuadrupedGo2BraxPlugin',
    'QuadrupedSteppingStones2DPlugin',
    'HumanoidSimplifiedMjxPlugin',
    'HumanoidG1MjxPlugin',
    'HumanoidRunBraxPlugin',
    'DroneEnvironmentPlugin',
    'DroneBox3DPlugin',
    'DroneFull3DPlugin',
    'DroneFull3DPhysicsPlugin',
    'ManipulatorEnvironmentPlugin',
    'D3ILAvoidingPlugin',
    'D3ILAvoiding9DPlugin',
    'AvoidingPlanEnvironmentPlugin',
    'HumanoidCorridor2DPlugin',
    'ManipulatorSurfaceScanPlugin',
    'ManipulatorPegInsertPlugin',
    'HumanoidBoxPushPlugin',
    # Metrics
    'SSRMetricsPlugin',
    'ObstacleDensityMetricsPlugin',
    'NonconvexityMetricsPlugin',
    'EpisodeOutcomeMetricsPlugin',
    'SteppingStonesMetricsPlugin',
    'CorridorMetricsPlugin',
    # Visualizations
    'TrajectoryVisualizationPlugin',
    'Trajectory3DVisualizationPlugin',
    'TrajectoryModesVisualizationPlugin',
    'DiffusionVisualizationPlugin',
    'Diffusion3DVisualizationPlugin',
    'EnergyRewardVisualizationPlugin',
    'StatesVisualizationPlugin',
    'SchedulerParamsVisualizationPlugin',
    'SteppingStonesTrajectoryVisualizationPlugin',
    'SteppingStonesModesVisualizationPlugin',
    'CorridorTrajectoryVisualizationPlugin',
    # Obstacles
    'Box2DObstacleGeneratorPlugin',
    'Box3DObstacleGeneratorPlugin',
    'D3ILAvoidingFixedGeneratorPlugin',
    'SteppingStones2DObstacleGeneratorPlugin',
    'Corridor2DObstacleGeneratorPlugin',
    'EmptyObstacleGeneratorPlugin',
]
