"""
Environment plugin implementations.
"""

from .single_integrator_2d import SingleIntegrator2DPlugin
from .double_integrator_2d import DoubleIntegrator2DPlugin
from .quadruped import (
    QuadrupedFlatMjxPlugin,
    QuadrupedGo2MjxPlugin,
    QuadrupedGo2BraxPlugin,
)
from .quadruped_stepping_stones_2d import QuadrupedSteppingStones2DPlugin
from .humanoid import (
    HumanoidSimplifiedMjxPlugin,
    HumanoidG1MjxPlugin,
    HumanoidRunBraxPlugin,
)
from .drone import (
    DroneEnvironmentPlugin,
    DroneBox3DPlugin,
    DroneFull3DPlugin,
    DroneFull3DPhysicsPlugin,
)
from .manipulator import ManipulatorEnvironmentPlugin
from .d3il_avoiding import D3ILAvoidingPlugin
from .d3il_avoiding_9d import D3ILAvoiding9DPlugin
from .avoiding_plan import AvoidingPlanEnvironmentPlugin
from .mujoco_scene_mapping import MujocoSceneMappingPlugin
from .nerf_synthetic_3dgs import NerfSynthetic3DGSPlugin
from .replica_3dgs import Replica3DGSPlugin
from .tum_rgbd_3dgs import TUM_RGBD_3DGSPlugin
from .mujoco_active_perception import MuJoCoActivePerceptionPlugin
from .humanoid_corridor import HumanoidCorridor2DPlugin
from .humanoid_box_push import HumanoidBoxPushPlugin
from .manipulator_peg_insert import ManipulatorPegInsertPlugin
from .manipulator_surface_scan import ManipulatorSurfaceScanPlugin

__all__ = [
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
    'MujocoSceneMappingPlugin',
    'NerfSynthetic3DGSPlugin',
    'Replica3DGSPlugin',
    'TUM_RGBD_3DGSPlugin',
    'MuJoCoActivePerceptionPlugin',
    'HumanoidCorridor2DPlugin',
    'ManipulatorSurfaceScanPlugin',
    'ManipulatorPegInsertPlugin',
    'HumanoidBoxPushPlugin',
]
