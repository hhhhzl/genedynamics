import os
import argparse
import random
import tqdm
import numpy as np
import torch
import taichi as ti

from .. import designer as designer_module


def main():
    # Parse input arguments and general initialization
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.torch_seed)

    ti.init(arch=ti.cuda, device_memory_fraction=0.8)

    # Instantiate environment
    env = make_env(args)
    torch_device = 'cuda' if args.non_taichi_device == 'torch_gpu' else 'cpu'

    action_dim = env.action_space.shape[0]

    # Instantiate designer
    designer = designer_module.make(args, env, torch_device)

    # Run
    obs = env.reset()
    designer.reset()

    designer_out = designer()
    design = dict()
    for design_type in ['geometry', 'actuator']:
        design[design_type] = designer_out[design_type]
    if getattr(designer, 'has_actuator_direction', False):
        key = 'actuator_direction'
        design[key] = designer_out[key]
    env.design_space.set_design(design)

    window = env.renderer.window
    env.renderer.camera.track_user_inputs(window, movement_speed=1.0, hold_key=ti.ui.RMB)
    i = 0
    act = np.zeros((action_dim,))
    while window.running:
        if i % args.act_freq == 0:
            if window.event is not None:
                act = np.zeros((action_dim,))

            if window.get_event(ti.ui.PRESS):
                key_to_act(window, act) # in-place

        # print(act) # DEBUG
        act *= args.act_strength
        obs, reward, done, info = env.step(act)
        env.render()

        i = (i + 1) % 1e4
    env.close()


def key_to_act(window, act):
    pos_keys = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'z', 'c', 'b']
    neg_keys = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'Return', 'x', 'v', 'n']
    # pos_keys = ['r', 't', 'y', 'u', 'i', 'o', 'p', 'z', 'c', 'b']
    # neg_keys = ['f', 'g', 'h', 'j', 'k', 'l', 'Return', 'x', 'v', 'n']
    key = window.event.key
    if key in pos_keys:
        act[pos_keys.index(key)] = 1.
    if key in neg_keys:
        act[neg_keys.index(key)] = -1.


def make_parser():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--torch-seed', type=int, default=100)
    # Environment
    parser.add_argument('--out-dir', type=str, default='/tmp/tmp')
    parser.add_argument('--non-taichi-device', type=str, choices=['numpy', 'torch_cpu', 'torch_gpu'], default='torch_cpu')
    parser.add_argument('--env', type=str, default='land_environment',
                        choices=['land_environment', 'aquatic_environment', 'subterrain_environment', 'dummy_env'])
    parser.add_argument('--env-config-file', type=str, default='fixed_plain.yaml')
    parser.add_argument('--action-space', type=str, default='actuation',
                        choices=['actuation', 'particle_v', 'actuator_v'])
    parser.add_argument('--act-strength', type=float, default=1.)
    parser.add_argument('--act-freq', type=int, default=1)

    # Designer
    parser = designer_module.augment_parser(parser)
    
    return parser


def parse_args():
    parser = make_parser()
    args = parser.parse_args()

    return args


def make_env(args):
    if args.env == 'land_environment':
        from softzoo.envs.land_environment import LandEnvironment
        env_cls = LandEnvironment
    elif args.env == 'aquatic_environment':
        from softzoo.envs.aquatic_environment import AquaticEnvironment
        env_cls = AquaticEnvironment
    elif args.env == 'subterrain_environment':
        from softzoo.envs.subterrain_environment import SubterrainEnvironment
        env_cls = SubterrainEnvironment
    elif args.env == 'dummy_env':
        from softzoo.envs.dummy_env import DummyEnv
        env_cls = DummyEnv
    else:
        raise NotImplementedError

    cfg_kwargs = {
        # NOTE: make sure not using offscreen rendering
        'RENDERER.GGUI.offscreen_rendering': False,
        'RENDERER.GGUI.save_to_video': False,
        # make sure can run forever
        'SIMULATOR.needs_grad': False,
        'SIMULATOR.use_checkpointing': False,
        'ENVIRONMENT.objective': 'move_in_circles',
    }
    env_kwargs = dict(
        cfg_file=args.env_config_file,
        out_dir=args.out_dir,
        device=args.non_taichi_device,
        cfg_kwargs=cfg_kwargs,
    )
    env = env_cls(**env_kwargs)
    env.initialize()

    return env


if __name__ == '__main__':
    main()
