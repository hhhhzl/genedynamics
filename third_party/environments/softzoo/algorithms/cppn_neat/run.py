import os
import argparse
import random
import numpy as np
import torch
import torch.multiprocessing as multip
import neat
import open3d as o3d
import taichi as ti
import yaml
import json
import matplotlib as mpl

from softzoo.tools.general_utils import pcd_to_mesh

from .population import Population
from .parallel import ParallelEvaluator
from .reporter import SaveResultReporter
from .pytorch_neat.cppn import create_cppn
from .tools.utils import construct_inputs
from .. import controllers as controller_module


def main():
    # General initialization
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.torch_seed)

    if args.coords_inp[0] == 'None': args.coords_inp = []

    # Get configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    config.genome_config.num_inputs = len(args.coords_inp) + len(args.seed_object_mesh)
    config.genome_config.input_keys = [-v for v in range(1, config.genome_config.num_inputs + 1)]
    
    config.genome_config.num_outputs = 0
    if 'geometry' in args.optimize_design_types:
        config.genome_config.num_outputs += 2
    if 'actuator' in args.optimize_design_types:
        with open(os.path.join(f'softzoo/configs/env_configs', args.env_config_file), 'r') as f:
            env_config = yaml.safe_load(f)
        n_actuators = env_config['ENVIRONMENT']['design_space_config']['n_actuators']
        config.genome_config.num_outputs += n_actuators
    
    config.genome_config.output_keys = [v for v in range(config.genome_config.num_outputs)]

    args.pop_size = config.pop_size # sneak in population size for controller optimizer (CMAES)

    # Set up
    pop = Population(config)
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(
            args.out_dir,
            {'args': args},
        )
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    optim_kwargs = dict(args=args)
    evaluator = ParallelEvaluator(args.num_workers, Optim, optim_kwargs)

    # Save
    with open(os.path.join(args.out_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    # Run
    pop.run(evaluator.evaluate_fitness,
            evaluator.evaluate_constraint,
            n=args.max_evaluations)


def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--torch-seed', type=int, default=100)
    parser.add_argument('--out-dir', type=str, default='/tmp/tmp')
    # Optimization
    parser.add_argument('--config-path', type=str, default='./config/neat.cfg')
    parser.add_argument('--num-workers', type=int, default=0) # 0 for using local worker only (good for debugging)
    parser.add_argument('--max-evaluations', type=int, default=100)
    parser.add_argument('--coords-inp', nargs='+', type=str, default=['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    parser.add_argument('--seed-object-mesh', nargs='+', type=str, default=[])
    parser.add_argument('--optimize-design-types', type=str, nargs='+', default=['geometry', 'actuator'],
                        choices=['geometry', 'actuator']) # TODO: 'softness', 'actuator_direction'
    # Environment
    parser.add_argument('--ti-device-memory-fraction', type=float, default=0.1)
    parser.add_argument('--env', type=str, default='aquatic_environment',
                        choices=['land_environment', 'aquatic_environment', 'subterrain_environment', 'dummy_env'])
    parser.add_argument('--env-config-file', type=str, default='fixed_plain.yaml')
    parser.add_argument('--non-taichi-device', type=str, choices=['numpy', 'torch_cpu', 'torch_gpu'], default='torch_cpu')
    parser.add_argument('--n-frames', type=int, default=10)
    # Controller
    parser.add_argument('--update-controller', action='store_true', default=False)
    parser.add_argument('--cmaes-sigma', type=float, default=0.1)
    parser = controller_module.augment_parser(parser)

    args = parser.parse_args()

    assert args.action_space == 'actuation'

    return args


class Optim:
    def __init__(self, args):
        # Setup environment
        ti.init(arch=ti.cuda, device_memory_fraction=args.ti_device_memory_fraction)
        self.make_env(args)

        # Point coordinates
        points = self.env.design_space.get_x(s=0).float()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())

        # Prepare coordinate-related input
        points_std = (points - points.min(0)[0]) / (points.max(0)[0] - points.min(0)[0]) # NOTE: standardize to 0 to 1; will be later-on centered at 0
        inputs = construct_inputs(points_std, args.coords_inp, args.seed_object_mesh)

        # Set attribute and output directory
        self.points = points
        self.pcd = pcd # copy of the original shape
        self.inputs = inputs
        self.args = args

        self.ply_out_dir = os.path.join(self.args.out_dir, 'ply')
        os.makedirs(self.ply_out_dir, exist_ok=True)

        self.pcd_out_dir = os.path.join(self.args.out_dir, 'pcd')
        os.makedirs(self.pcd_out_dir, exist_ok=True)

        # Save base shape
        out_path = os.path.join(self.pcd_out_dir, 'base_shape.pcd')
        o3d.io.write_point_cloud(out_path, pcd)

        # Setup controller
        torch_device = 'cuda' if args.non_taichi_device == 'torch_gpu' else 'cpu'
        self.controller = controller_module.make(args, self.env, torch_device)

    def make_env(self, args):
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

        cfg_kwargs = dict()
        cfg_kwargs['ENVIRONMENT.use_renderer'] = True
        env_kwargs = dict(
            cfg_file=args.env_config_file,
            out_dir=args.out_dir,
            device=args.non_taichi_device,
            cfg_kwargs=cfg_kwargs,
        )
        self.env = env_cls(**env_kwargs)
        self.env.initialize()

    def evaluate_genome_fitness(self, genome, config, genome_id, generation, render=False):
        # Convert genome to design
        design = self.genome_to_design(genome, config)

        # Render / save to file
        if render:
            pcd = self.design_to_pcd(design)
            out_path = os.path.join(self.pcd_out_dir, f'gen{generation:04d}_genome{genome_id:04d}.pcd')
            o3d.io.write_point_cloud(out_path, pcd)

            try:
                mesh = pcd_to_mesh(pcd)
                out_path = os.path.join(self.ply_out_dir, f'gen{generation:04d}_genome{genome_id:04d}.ply')
                o3d.io.write_triangle_mesh(out_path, mesh)
            except:
                print('Fail to convert to mesh')

        # Reset
        obs = self.env.reset(design)
        self.controller.reset()

        # Run simulation
        ep_reward = 0.
        max_episode_steps = getattr(self.env.objective, 'max_episode_steps', self.args.n_frames)
        for frame in range(max_episode_steps):
            current_s = self.env.sim.solver.current_s
            act = self.controller(current_s, obs)
            obs, reward, done, info = self.env.step(act)
            ep_reward += reward
            if done:
                break

            if self.env.has_renderer and render:
                if hasattr(self.env.objective, 'render'):
                    self.env.objective.render()

                self.env.render()

        fitness = ep_reward

        return fitness

    def evaluate_genome_constraint(self, genome, config, genome_id, generation):
        design = self.genome_to_design(genome, config)

        # Check connectivity
        mask = design['geometry'] > 0
        if mask.float().mean() > 0:
            pcd = self.design_to_pcd(design)
            if True: # in point cloud
                labels = pcd.cluster_dbscan(eps=0.02, min_points=10)
                N = len(np.unique(labels))
            else: # in mesh
                mesh = pcd_to_mesh(pcd)
                cluster_tri_indices, cluster_n_tri, cluster_area = mesh.cluster_connected_triangles()
                N = len(cluster_n_tri)
        else:
            N = 0
        # is_connected = N == 1
        is_connected = N > 0 # DEBUG

        # Get validity
        validity = is_connected

        return validity

    def genome_to_design(self, genome, config):
        leaf_names = self.inputs.keys()
        design = dict()
        node_names = []
        if 'geometry' in self.args.optimize_design_types:
            node_names += ['geometry_empty', 'geometry_occupied']
            design['geometry'] = []
        if 'actuator' in self.args.optimize_design_types:
            node_names += [f'actuator_{v}' for v in range(self.env.design_space.n_actuators)]
            design['actuator'] = []
        nodes = create_cppn(genome, config, leaf_names=leaf_names, node_names=node_names)

        for node in nodes:
            node_out = node(**self.inputs)
            if 'geometry' in node.name:
                design['geometry'].append(node_out)
            elif 'actuator' in node.name:
                design['actuator'].append(node_out)
            else:
                raise NotImplementedError
        
        if 'geometry' in self.args.optimize_design_types:
            design['geometry'] = torch.stack(design['geometry'], dim=0)
            design['geometry'] = design['geometry'].argmax(dim=0) # assume channel-0 is empty and channel-1 is occupied

        if 'actuator' in self.args.optimize_design_types:
            design['actuator'] = torch.stack(design['actuator'], dim=0)
            max_idcs = design['actuator'].argmax(dim=0)
            actuator = torch.zeros_like(design['actuator'])
            max_idcs_rp = torch.stack([max_idcs] * design['actuator'].shape[0])
            actuator.scatter_(0, max_idcs_rp, torch.ones(max_idcs_rp.shape))
            design['actuator'] = actuator

        return design

    def design_to_pcd(self, design):
        try:
            mask = (design['geometry'] > 0)
            points = self.points[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pcd.estimate_normals()
            
            if 'actuator' in design.keys():
                actuator_idcs = design['actuator'][:,mask].argmax(0)
                base_colors = np.array(mpl.colormaps.get_cmap('tab20').colors + \
                                       mpl.colormaps.get_cmap('tab20b').colors + \
                                       mpl.colormaps.get_cmap('tab20c').colors)
                colors = base_colors[actuator_idcs]
                pcd.colors = o3d.utility.Vector3dVector(colors)
        except:
            pcd = o3d.geometry.PointCloud()

        return pcd


if __name__ == '__main__':
    multip.set_start_method('forkserver')
    main()
