import os
from attrdict import AttrDict
import taichi as ti
import gym
from gym.envs.registration import register

from .. import designer as designer_module


DEFAULT_DEVICE = 'torch_cpu'
DEFAULT_OUT_DIR = '/tmp/tmp'

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
    def __init__(self, env_cls_name, env_kwargs, designer_kwargs, ti_device_memory_fraction):
        # Make env
        ti.init(arch=ti.cuda, device_memory_fraction=ti_device_memory_fraction)
        if env_cls_name == 'land_environment':
            from dsrc.envs.land_environment import LandEnvironment
            env_cls = LandEnvironment
        elif env_cls_name == 'aquatic_environment':
            from dsrc.envs.aquatic_environment import AquaticEnvironment
            env_cls = AquaticEnvironment
        elif env_cls_name == 'subterrain_environment':
            from dsrc.envs.subterrain_environment import SubterrainEnvironment
            env_cls = SubterrainEnvironment
        elif env_cls_name == 'dummy_env':
            from dsrc.envs.dummy_env import DummyEnv
            env_cls = DummyEnv
        else:
            raise NotImplementedError
        self.env = env_cls(**env_kwargs)
        self.env.initialize()

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Make designer
        torch_device = 'cpu'
        self.designer = designer_module.make(designer_kwargs, self.env, torch_device)
        self.designer_kwargs = designer_kwargs

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
        return self.env.step(action)


register(
    id='Ground-Caterpillar-v0',
    entry_point=Env,
    kwargs={
        'env_cls_name': 'land_environment',
        'env_kwargs': {
            'cfg_file': 'no_grad/ground.yaml',
            'out_dir': os.environ.get('DSRC_OUT_DIR', DEFAULT_OUT_DIR),
            'device': DEFAULT_DEVICE,
        },
        'designer_kwargs': DEFAULT_DESIGNER_KWARGS_ANNOTATED_PCD + AttrDict({
            'annotated_pcd_path': './dsrc/assets/meshes/pcd/Caterpillar.pcd',
        }),
        'ti_device_memory_fraction': 0.1, # TODO: make argument
    },
)
