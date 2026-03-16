import argparse
import random
import json
import numpy as np
import pickle
import torch

from .tools.utils import construct_inputs
from .pytorch_neat.cppn import create_cppn
from .run import Optim


def main():
    args = parse_args()

    # Load arguments
    args_white_list = ['out_dir', 'seed', 'torch_seed', 'n_frames', 'env_config_file']
    with open(args.args_path, 'r') as fp:
        loaded_args = json.load(fp)
        inp_args = vars(args)
        for k, v in loaded_args.items():
            if k not in args_white_list:
                if k == 'env_config_file' and v is None:
                    continue
                inp_args[k] = v
        for k, v in inp_args.items():
            setattr(args, k, v)

    # Load genome data NOTE: make sure in the correct directory in case mesh path is not matched
    with open(args.genome_path, 'rb') as f:
        data = pickle.load(f)

    # Load controller
    model = Optim(args)

    if args.controller_path is not None:
        checkpoint = torch.load(args.controller_path)
        model.controller.load_state_dict(checkpoint['model_state_dict'])

    # Misc
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.torch_seed)

    if args.coords_inp[0] == 'None': args.coords_inp = []

    # Get design
    genome, config = data['genome'], data['config']
    design = model.genome_to_design(genome, config)

    # Run
    ep_reward = 0.
    obs = model.env.reset(design)
    model.controller.reset()
    for frame in range(args.n_frames):
        current_s = model.env.sim.solver.current_s
        act = model.controller(current_s, obs)
        obs, reward, done, info = model.env.step(act)
        ep_reward += reward
        if done:
            break

        if model.env.has_renderer:
            if hasattr(model.env.objective, 'render'):
                model.env.objective.render()

            model.env.render()

    print(f'Episode reward: {ep_reward}')


def parse_args():
    parser = argparse.ArgumentParser()
    # Load
    parser.add_argument('--args-path', type=str, required=True)
    parser.add_argument('--genome-path', type=str, required=True)
    parser.add_argument('--controller-path', type=str, default=None)
    # General
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--torch-seed', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='/tmp/tmp')
    # Environment
    parser.add_argument('--env-config-file', type=str, default=None)
    parser.add_argument('--n-frames', type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
