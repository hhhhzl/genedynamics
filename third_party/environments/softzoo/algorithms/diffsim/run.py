import os
import argparse
import random
import tqdm
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import torch
import taichi as ti

from softzoo.envs import ENV_CONFIGS_DIR
from softzoo.tools.general_utils import Logger, save_pcd_to_mesh
from .loss import LossSet
from .. import controllers as controller_module
from .. import designer as designer_module


def main():
    # Parse input arguments and general initialization
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.torch_seed)

    ti.init(arch=ti.cuda, device_memory_fraction=0.8)

    if args.load_args:
        with open(args.load_args, 'r') as fp:
            loaded_args = json.load(fp)
        inp_args = vars(args)
        for k, v in loaded_args.items():
            if k not in ['load_args', 'load_controller', 'load_designer', 'out_dir', 'optimize_controller', 'optimize_designer', 'eval',
                         'env_config_file', 'n_iters', 'n_frames', 'save_every_iter', 'render_every_iter', 'log_reward', 'objective_reward_mode']:
                inp_args[k] = v
        for k, v in inp_args.items():
            setattr(args, k, v)

    if args.eval:
        args.optimize_controller = False
        args.optimize_designer = False

    # Instantiate environment
    env = make_env(args)
    torch_device = 'cuda' if args.non_taichi_device == 'torch_gpu' else 'cpu'

    # Define loss
    loss_set = make_loss(args, env, torch_device)

    # Instantiate designer
    designer = designer_module.make(args, env, torch_device)

    if args.load_designer:
        designer.load_checkpoint(args.load_designer)

    # Instantiate controller
    if args.load_rl_controller:
        assert not args.optimize_controller, 'Cannot optimize controller when using RL model'
        args.save_controller = False
        assert not args.load_controller, 'Can only load either Diffsim or RL controller'
        from stable_baselines3 import PPO
        from ..controllers.base import Base
        rl_model = PPO.load(args.load_rl_controller, device=torch_device)
        class NewControllerCls(Base):
            def __init__(self, _model, _n_actuators):
                super(NewControllerCls, self).__init__()
                self._model = _model
                self._n_actuators = _n_actuators

            def reset(self):
                pass

            def forward(self, s, inp):
                act, _ = self._model.predict(inp)
                act = np.concatenate([[0], act, np.zeros((self._n_actuators - act.shape[0] - 1))])
                return act
        controller = NewControllerCls(rl_model, env.design_space.n_actuators)
    else:
        controller = controller_module.make(args, env, torch_device)

        if args.load_controller:
            controller.load_checkpoint(args.load_controller)

    # Misc
    post_substep_grad_fn = []
    if args.action_space == 'actuator_v':
        # only suppport particle-based representation for now
        env.design_space.instantiate_v_buffer()
        post_substep_grad_fn.append(env.design_space.add_v_with_buffer.grad)

    if 'TrajectoryFollowingLoss' in args.loss_types:
        traj_len = int(args.goal[-1])
        traj = ti.Vector.field(3, env.renderer.f_dtype, shape=(traj_len))
        traj_loss = loss_set.losses[args.loss_types.index('TrajectoryFollowingLoss')]
        traj_loss.reset()
        traj.from_numpy(traj_loss.data['traj'].to_numpy().astype(env.renderer.f_dtype_np)[:traj_len])

    ckpt_root_dir = os.path.join(args.out_dir, 'ckpt')
    os.makedirs(ckpt_root_dir, exist_ok=True)
    if args.optimize_designer or args.save_designer:
        design_dir = os.path.join(args.out_dir, 'design')
        os.makedirs(design_dir, exist_ok=True)

        ckpt_dir_designer = os.path.join(ckpt_root_dir, 'designer')
        os.makedirs(ckpt_dir_designer, exist_ok=True)

    if args.optimize_controller or args.save_controller:
        ckpt_dir_controller = os.path.join(ckpt_root_dir, 'controller')
        os.makedirs(ckpt_dir_controller, exist_ok=True)

    with open(os.path.join(ckpt_root_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    env_config_path = os.path.join(ENV_CONFIGS_DIR, args.env_config_file)
    shutil.copy(env_config_path, os.path.join(ckpt_root_dir, 'env_config.yaml'))

    # Profiling
    logger = Logger(args.out_dir)
    def time_fn(func):
        def inner(*args, **kwargs):
            logger.tic(func.__qualname__)
            out = func(*args, **kwargs)
            logger.toc(func.__qualname__)
            return out
        return inner

    loss_set.compute_loss = time_fn(loss_set.compute_loss)
    controller.update = time_fn(controller.update)
    designer.update = time_fn(designer.update)

    logger.print(vars(args))

    # Plotting
    data_for_plots = dict(reward=[], loss=[])

    # Run
    for it in range(args.n_iters):
        try:
            logger.reset() # reset accumulated timer

            # Forward
            designer.reset()
            designer_out = designer()
            design = dict()
            for design_type in args.set_design_types:
                if design_type == 'actuator_direction':
                    assert getattr(designer, 'has_actuator_direction', False)
                design[design_type] = designer_out[design_type]

            obs = env.reset(design)
            controller.reset()
            ep_reward = 0.

            if args.optimize_designer and (it % args.render_every_iter == 0):
                if 'particle_based_representation' in str(env.design_space):
                    for design_type in args.optimize_design_types:
                        design_fpath = os.path.join(design_dir, f'{design_type}_{it:04d}.pcd')
                        design_pcd = designer.save_pcd(design_fpath, design, design_type)
                    save_pcd_to_mesh(os.path.join(design_dir, f'mesh_{it:04d}.ply'), design_pcd)
                elif 'voxel_based_representation' in str(env.design_space):
                    for design_type in args.optimize_design_types:
                        design_fpath = os.path.join(design_dir, f'{design_type}_{it:04d}.ply')
                        designer.save_voxel_grid(design_fpath, design, design_type)
                else:
                    raise NotImplementedError

            for frame in tqdm.tqdm(range(args.n_frames), desc=f'Forward #{it:04d}'):
                current_s = env.sim.solver.current_s
                current_s_local = env.sim.solver.get_cyclic_s(current_s)
                act = controller(current_s, obs)
                if args.action_space == 'particle_v':
                    env.design_space.add_to_v(current_s_local, act) # only add v to the first local substep since v accumulates
                    obs, reward, done, info = env.step(None)
                elif args.action_space == 'actuator_v':
                    env.design_space.set_v_buffer(current_s, act)
                    env.design_space.add_v_with_buffer(current_s, current_s_local)
                    obs, reward, done, info = env.step(None)
                else:
                    obs, reward, done, info = env.step(act)
                ep_reward += reward

                if env.has_renderer and (it % args.render_every_iter == 0):
                    if 'TrajectoryFollowingLoss' in args.loss_types: # plot trajectory
                        env.renderer.scene.particles(traj, radius=0.003)

                    if hasattr(env.objective, 'render'):
                        env.objective.render()

                    env.render()

                if (it % args.save_every_iter == 0):
                    if args.optimize_designer or args.save_designer:
                        designer.save_checkpoint(os.path.join(ckpt_dir_designer, f'iter_{it:04d}.ckpt'))
                    if args.optimize_controller or args.save_controller:
                        controller.save_checkpoint(os.path.join(ckpt_dir_controller, f'iter_{it:04d}.ckpt'))

                if done:
                    break

            # Backward
            loss_reset_kwargs = {k: {} for k in args.loss_types}
            grad_names = dict()
            if args.action_space == 'particle_v':
                grad_name_control = 'self.env.sim.solver.v'
            elif args.action_space == 'actuator_v':
                grad_name_control = 'self.env.design_space.v_buffer'
            else:
                grad_name_control = 'self.env.sim.solver.act_buffer'
            if args.optimize_controller:
                for s in controller.all_s:
                    if s not in grad_names.keys():
                        grad_names[s] = []
                    grad_names[s].append(grad_name_control)
            if args.optimize_designer:
                s = None
                if s not in grad_names.keys():
                    grad_names[s] = []
                for dsr_buffer_name in args.optimize_design_types:
                    grad_names[s].append(f'self.env.design_space.buffer.{dsr_buffer_name}')
            try:
                all_loss, grad = loss_set.compute_loss(loss_reset_kwargs, post_substep_grad_fn, compute_grad=len(grad_names) > 0, grad_names=grad_names)
            except: # HACK
                all_loss = np.zeros([1])
                grad = dict()

            # Optimize
            if args.optimize_controller:
                grad_control = []
                for s in controller.all_s:
                    grad_control.append(grad[s][grad_name_control])
                
                controller.update(grad_control)

            if args.optimize_designer:
                grad_design = dict()
                for dsr_buffer_name in args.optimize_design_types:
                    grad_design[dsr_buffer_name] = grad[None][f'self.env.design_space.buffer.{dsr_buffer_name}']
                designer.update(grad_design)

            # Logging
            if it % args.log_every_iter == 0:
                if args.log_reward:
                    logger.scalar(f'reward', ep_reward)

                total_loss = all_loss.sum()
                logger.scalar('total_loss', total_loss)

                for loss_name, loss in zip(loss_set.loss_names, loss_set.losses):
                    loss_stats = loss.get_loss_stats()
                    for k, v in loss_stats.items():
                        logger.scalar(f'loss/{loss_name}/{k}', v)
                
                for s_name in grad.keys():
                    for grad_key, v in grad[s_name].items():
                        logger.scalar(f'grad/{s_name}/{grad_key}', v.mean().item())

                if args.optimize_designer:
                    logger.print(designer.log_text())

                logger.write(it)

            data_for_plots['reward'].append(ep_reward)
            data_for_plots['loss'].append(total_loss)
        except KeyboardInterrupt:
            break

    env.close()
    logger.close()

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title('Reward')
    axes[0].plot(np.arange(len(data_for_plots['reward'])), data_for_plots['reward'])
    axes[1].set_title('Loss')
    axes[1].plot(np.arange(len(data_for_plots['loss'])), data_for_plots['loss'])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'results.png'))


def make_parser():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--torch-seed', type=int, default=100)
    parser.add_argument('--render-every-iter', type=int, default=10)
    parser.add_argument('--save-every-iter', type=int, default=10)
    parser.add_argument('--log-every-iter', type=int, default=1)
    parser.add_argument('--log-reward', action='store_true', default=False)

    parser.add_argument('--load-args', type=str, default=None)
    parser.add_argument('--load-controller', type=str, default=None)
    parser.add_argument('--load-rl-controller', type=str, default=None)
    parser.add_argument('--load-designer', type=str, default=None)

    parser.add_argument('--save-designer', action='store_true', default=False)
    parser.add_argument('--save-controller', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)

    # Environment
    parser.add_argument('--out-dir', type=str, default='/tmp/tmp')
    parser.add_argument('--non-taichi-device', type=str, choices=['numpy', 'torch_cpu', 'torch_gpu'], default='torch_cpu')
    parser.add_argument('--env', type=str, default='land_environment',
                        choices=['land_environment', 'aquatic_environment', 'dummy_env', 'manipulation_environment'])
    parser.add_argument('--env-config-file', type=str, default='fixed_plain.yaml')
    parser.add_argument('--objective-reward-mode', type=str, default=None)
    parser.add_argument('--dump-rendering-data', action='store_true', default=False)
    
    # Optimization
    parser.add_argument('--n-iters', type=int, default=1)
    parser.add_argument('--n-frames', type=int, default=10)
    parser.add_argument('--loss-types', type=str, nargs='+', default=['FinalStepCoMLoss'],
                        choices=['FinalStepCoMLoss', 'TrajectoryFollowingLoss', 'PerStepCoVLoss',
                                 'AnimatedEMDLoss', 'VelocityFollowingLoss', 'WaypointFollowingLoss',
                                 'RotationLoss', 'ThrowingObjectLoss', 'ObjectVelocityLoss'])
    parser.add_argument('--loss-coefs', type=float, nargs='+', default=[1.])
    parser.add_argument('--optimize-designer', action='store_true', default=False)
    parser.add_argument('--set-design-types', type=str, nargs='+', default=['geometry', 'softness', 'actuator', 'actuator_direction'],
                        choices=['geometry', 'softness', 'actuator', 'actuator_direction'])
    parser.add_argument('--optimize-design-types', type=str, nargs='+', default=['geometry', 'softness', 'actuator'],
                        choices=['geometry', 'softness', 'actuator', 'actuator_direction'])
    parser.add_argument('--optimize-controller', action='store_true', default=False)
    # Loss [Final step CoM loss]
    parser.add_argument('--x-mul', type=float, nargs='+', default=[1., 0., 0.])
    # Loss [Trajectory Following loss]
    parser.add_argument('--goal', type=float, nargs='+', default=[0.8, 0., 0., 1700]) # x, y, z, s
    # Loss [Per step CoV loss]
    parser.add_argument('--v-mul', type=float, nargs='+', default=[1., 0., 0.])
    # Loss [Animated EMD Loss]
    parser.add_argument('--mesh-dir', type=str, default='./local/meshes/fantasy_horse')
    parser.add_argument('--substep-freq', type=int, default=100)
    parser.add_argument('--mesh-num-points', type=int, default=5000)
    parser.add_argument('--final-target-idx', type=int, default=None)
    parser.add_argument('--recenter-mesh-target', action='store_true', default=False)
    # Loss [Velocity following loss]
    parser.add_argument('--v-following-v-mul', type=float, nargs='+', default=[1., 1., 1.])
    parser.add_argument('--v-following-mode', type=int, default=0)
    # Loss [Waypoint following loss]
    pass
    # Loss [Rotation loss]
    parser.add_argument('--rotation-up-direction', type=float, nargs='+', default=[0., 1., 0.])
    # Loss [ThrowingObjectLoss]
    parser.add_argument('--obj-x-mul', type=float, nargs='+', default=[1., 0., 0.])
    parser.add_argument('--obj-particle-id', type=int, default=2)
    # Loss [ObjectVelocityLoss]
    parser.add_argument('--obj-v-mul', type=float, nargs='+', default=[1., 0., 0.])
    # --obj-particle-id already exists

    # Designer
    parser = designer_module.augment_parser(parser)

    # Controller
    parser = controller_module.augment_parser(parser)

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
    elif args.env == 'manipulation_environment':
        from softzoo.envs.manipulation_environment import ManipulationEnvironment
        env_cls = ManipulationEnvironment
    else:
        raise NotImplementedError

    cfg_kwargs = dict()
    if args.render_every_iter > 0:
        cfg_kwargs['ENVIRONMENT.use_renderer'] = True
    if args.objective_reward_mode not in [None, 'None']:
        cfg_kwargs['ENVIRONMENT.objective_config.reward_mode'] = args.objective_reward_mode
    if args.dump_rendering_data:
        cfg_kwargs['RENDERER.GL.dump_data'] = True
    env_kwargs = dict(
        cfg_file=args.env_config_file,
        out_dir=args.out_dir,
        device=args.non_taichi_device,
        cfg_kwargs=cfg_kwargs,
    )
    env = env_cls(**env_kwargs)
    env.initialize()

    if args.render_every_iter > 0: assert env.has_renderer

    return env


def make_loss(args, env, torch_device):
    loss_configs = dict()
    loss_coefs = dict()
    for i, loss_type in enumerate(args.loss_types):
        if loss_type == 'FinalStepCoMLoss':
            loss_config = dict(x_mul=args.x_mul)
        elif loss_type == 'TrajectoryFollowingLoss':
            loss_config = dict(goal=args.goal)
        elif loss_type == 'PerStepCoVLoss':
            loss_config = dict(v_mul=args.v_mul)
        elif loss_type == 'AnimatedEMDLoss':
            loss_config = dict(mesh_dir=args.mesh_dir, substep_freq=args.substep_freq, 
                               mesh_num_points=args.mesh_num_points, final_target_idx=args.final_target_idx,
                               recenter_mesh_target=args.recenter_mesh_target)
        elif loss_type == 'VelocityFollowingLoss':
            loss_config = dict(v_mul=args.v_following_v_mul, mode=args.v_following_mode)
        elif loss_type == 'WaypointFollowingLoss':
            loss_config = dict()
        elif loss_type == 'RotationLoss':
            loss_config = dict(up_direction=args.rotation_up_direction)
        elif loss_type == 'ThrowingObjectLoss':
            loss_config = dict(x_mul=args.obj_x_mul, obj_particle_id=args.obj_particle_id)
        elif loss_type == 'ObjectVelocityLoss':
            loss_config = dict(v_mul=args.obj_v_mul, obj_particle_id=args.obj_particle_id)
        else:
            raise ValueError(f'Unrecognized loss type {loss_type}')
        loss_configs[loss_type] = loss_config
        loss_coefs[loss_type] = args.loss_coefs[i]
    loss_set = LossSet(env, loss_configs, loss_coefs)

    return loss_set


if __name__ == '__main__':
    main()
