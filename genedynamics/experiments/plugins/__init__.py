"""
Plugin implementations for the experimental framework.

This package contains implementations of various plugins:
- methods: Solver method plugins (EDOC, MPPI, CEM, etc.)
- environments: Environment plugins (single/double integrator, etc.)
- metrics: Metrics computation plugins (SSR, obstacle density, etc.)
- visualizations: Visualization plugins
- obstacles: Obstacle generation plugins
- baselines: Baseline algorithms for comparison (mrmfmbd, etc.)
- task_domains: Task domain providers (softzoo, etc.)
"""

# Import to trigger task domain and baseline registration
try:
    from . import task_domains  # noqa: F401
except ImportError:
    pass
try:
    from . import baselines  # noqa: F401
except ImportError:
    pass

# Import and export all plugins for convenient access
from .methods import (
    EBMBDMethodPlugin,
    MBDMethodPlugin,
    MBD3DMethodPlugin,
    MRMFMBDMethodPlugin,
    D3ILUnifiedMethodPlugin,
    MDOCMethodPlugin,
    CFSMBDMethodPlugin,
    CFSMBDFullMethodPlugin,
    TwoGOMethodPlugin,
    DPCCMethodPlugin,
    SafeDiffuserMethodPlugin,
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
    SoftZooEnvironmentPlugin,
    MujocoSceneMappingPlugin,
    NerfSynthetic3DGSPlugin,
)

from .metrics import (
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
    EpisodeOutcomeMetricsPlugin,
    SteppingStonesMetricsPlugin,
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
    GateDynamicsVisualizationPlugin,
)

from .obstacles import (
    Box2DObstacleGeneratorPlugin,
    Box3DObstacleGeneratorPlugin,
    D3ILAvoidingFixedGeneratorPlugin,
    SteppingStones2DObstacleGeneratorPlugin,
)

__all__ = [
    # Methods
    'EDOCMethodPlugin',
    'EBMBDMethodPlugin',
    'MBDMethodPlugin',
    'MBD3DMethodPlugin',
    'MRMFMBDMethodPlugin',
    'EDOCMPCMethodPlugin',
    'D3ILUnifiedMethodPlugin',
    'MDOCMethodPlugin',
    'CFSMBDMethodPlugin',
    'CFSMBDFullMethodPlugin',
    'TwoGOMethodPlugin',
    'DPCCMethodPlugin',
    'SafeDiffuserMethodPlugin',
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
    'SoftZooEnvironmentPlugin',
    'MujocoSceneMappingPlugin',
    'NerfSynthetic3DGSPlugin',
    # Metrics
    'SSRMetricsPlugin',
    'ObstacleDensityMetricsPlugin',
    'NonconvexityMetricsPlugin',
    'EpisodeOutcomeMetricsPlugin',
    'SteppingStonesMetricsPlugin',
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
    'GateDynamicsVisualizationPlugin',
    # Obstacles
    'Box2DObstacleGeneratorPlugin',
    'Box3DObstacleGeneratorPlugin',
    'D3ILAvoidingFixedGeneratorPlugin',
    'SteppingStones2DObstacleGeneratorPlugin',
]

