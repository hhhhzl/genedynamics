import gym
from gym.envs.registration import register

from .wrappers import MaskVelocityWrapper

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    rocket_lander_gym = None


# Register no vel envs
def create_no_vel_env(env_id: str):
    def make_env():
        env = gym.make(env_id)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )


### CUSTOM ###
import os
from attrdict import AttrDict
import numpy as np
import torch
import taichi as ti

from ... import designer as designer_module


DEFAULT_DEVICE = 'torch_cpu'
DEFAULT_OUT_DIR = '/tmp/tmp'

DEFAULT_TI_DEVICE_MEMORY_FRACTION = 0.6
DEFAULT_DESIGNER_KWARGS = AttrDict({
    'designer_lr': 0., # dummy
    'set_design_types': ['geometry', 'softness', 'actuator', 'actuator_direction'],
})
DEFAULT_DESIGNER_KWARGS_ANNOTATED_PCD = DEFAULT_DESIGNER_KWARGS + AttrDict({
    # Still need to specify `annotated_pcd_path`
    'designer_type': 'annotated_pcd',
    'annotated_pcd_n_voxels': 60,
    'annotated_pcd_passive_softness_mul': 10.,
    'annotated_pcd_passive_geometry_mul': 0.2,
})


class Env(gym.Env):
    def __init__(self, env_cls_name, env_kwargs, designer_kwargs, ti_device_memory_fraction, env_additional_kwargs=dict()):
        # Make env
        ti.init(arch=ti.cuda, device_memory_fraction=ti_device_memory_fraction)
        if env_cls_name == 'land_environment':
            from softzoo.envs.land_environment import LandEnvironment
            env_cls = LandEnvironment
        elif env_cls_name == 'aquatic_environment':
            from softzoo.envs.aquatic_environment import AquaticEnvironment
            env_cls = AquaticEnvironment
        elif env_cls_name == 'subterrain_environment':
            from softzoo.envs.subterrain_environment import SubterrainEnvironment
            env_cls = SubterrainEnvironment
        elif env_cls_name == 'dummy_env':
            from softzoo.envs.dummy_env import DummyEnv
            env_cls = DummyEnv
        else:
            raise NotImplementedError
        env_kwargs['cfg_kwargs'] = env_additional_kwargs
        self.env = env_cls(**env_kwargs)
        self.env.initialize()

        # Make designer
        torch_device = 'cpu'
        self.designer = designer_module.make(designer_kwargs, self.env, torch_device)
        self.designer_kwargs = designer_kwargs

        self.observation_space = self.env.observation_space

        self.use_reduced_action_space = True
        if self.use_reduced_action_space:
            n_valid_actuators = torch.where(self.designer.actuator.data.sum(1) > 0)[0].max()
            self.env.action_space.low = self.env.action_space.low[:n_valid_actuators]
            self.env.action_space.high = self.env.action_space.high[:n_valid_actuators]
            self.env.action_space.shape = (n_valid_actuators,)
        self.action_space = self.env.action_space

    def reset(self):
        self.designer.reset()
        designer_out = self.designer()
        design = dict()
        for design_type in self.designer_kwargs.set_design_types:
            if design_type == 'actuator_direction':
                assert getattr(self.designer, 'has_actuator_direction', False)
            design[design_type] = designer_out[design_type]

        obs = self.env.reset(design)

        return obs

    def step(self, action):
        if self.use_reduced_action_space:
            # NOTE: the zero'th actuator is passive
            if isinstance(action, torch.Tensor):
                action = torch.cat([torch.zeros((1,)).to(action), action])
            else:
                action = np.concatenate([np.zeros((1,)), action])
        return self.env.step(action)

    def render(self, mode):
        return self.env.render()

    def close(self):
        self.env.close()


all_env_names = ['Ground', 'Desert', 'Wetland', 'Clay', 'Ice', 'Snow', 'Shallow_Water', 'Ocean']
pcd_dir = os.path.join(os.path.dirname(__file__), '../../../softzoo/assets/meshes/pcd')
all_animal_names = [os.path.splitext(v)[0] for v in os.listdir(pcd_dir) if os.path.splitext(v)[-1] == '.pcd']
all_task_names = ['movement_speed', 'travel_distance', 'turning', 'velocity_tracking', 'waypoint_following']
for env_name in all_env_names:
    for animal_name in all_animal_names:
        for task_name in all_task_names:
            task_name_id = ''.join([v.capitalize() for v in task_name.split('_')])
            if env_name == 'Ocean':
                env_cls_name = 'aquatic_environment'
            else:
                env_cls_name = 'land_environment'
            register(
                id=f'{env_name}-{animal_name}-{task_name_id}-v0',
                entry_point=Env,
                kwargs={
                    'env_cls_name': env_cls_name,
                    'env_kwargs': {
                        'cfg_file': f'no_grad/{task_name}/{env_name.lower()}.yaml',
                        'out_dir': os.environ.get('SOFTZOO_OUT_DIR', DEFAULT_OUT_DIR),
                        'device': DEFAULT_DEVICE,
                    },
                    'env_additional_kwargs': dict(),
                    'designer_kwargs': DEFAULT_DESIGNER_KWARGS_ANNOTATED_PCD + AttrDict({
                        'annotated_pcd_path': f'./softzoo/assets/meshes/pcd/{animal_name}.pcd',
                    }),
                    'ti_device_memory_fraction': DEFAULT_TI_DEVICE_MEMORY_FRACTION,
                },
            )
