from typing import Dict
import os
import numpy as np
import open3d as o3d
import geomloss
import taichi as ti
import torch

from softzoo.engine import I_DTYPE, F_DTYPE, NORM_EPS
from softzoo.tools.general_utils import load_points_from_mesh


@ti.data_oriented
class LossSet:
    def __init__(self,
                 env,
                 loss_configs: Dict[str, Dict],
                 loss_coefs: Dict[str, float]):
        self.env = env

        self.total_loss = ti.field(dtype=self.env.sim.solver.f_dtype,
                                   shape=(self.env.sim.solver.max_substeps),
                                   needs_grad=True)
        self.loss_names = loss_configs.keys()
        self.losses = [] # use list to be accessed in taichi kernel
        self.loss_coefs = []
        for k, v in loss_configs.items():
            self.losses.append(globals()[k](parent=self, env=env, **v))
            self.loss_coefs.append(loss_coefs[k])
        self.shared_data = dict() # TODO: use shared data to avoid recomputation

    def compute_loss(self, loss_reset_kwargs, post_substep_grad_fn=[], compute_grad=False, grad_names=dict()):
        # Instantiate a cache for desired gradients
        grad = {k: {vv: None for vv in v} for k, v in grad_names.items()}
        grad_s = grad_names.keys()
        
        # Reset intermediate data for loss computation and gradient
        self.reset(loss_reset_kwargs)

        # Forward
        latest_s = self.env.sim.solver.get_latest_s(consider_cycle=False)
        norm_factor = 1. # / latest_s # TODO: do we need normalization?!

        # Backward
        if compute_grad:
            for s in range(latest_s, 0, -1):
                def pre_grad_fn():
                    # Compute loss
                    for loss in self.losses:
                        if s == latest_s:
                            loss.compute_final_step_loss(s)
                        else:    
                            loss.compute_per_step_loss(s)
                    self.accumulate_loss(s, norm_factor)
                    
                    # Compute loss gradient
                    self.total_loss.grad[s] = 1.
                    self.accumulate_loss.grad(s, norm_factor)
                    for loss in self.losses:
                        if s == latest_s:
                            loss.compute_final_step_grad(s)
                        else:
                            loss.compute_per_step_grad(s)
                
                sm1 = s - 1 # NOTE: to obtain gradient at s, we need to call substep_grad(s-1)
                action_sm1 = self.env.sim.solver.checkpoint_cache['act_buffer'][sm1]
                dt_sm1 = self.env.sim.solver.checkpoint_cache['dt'][sm1]
                self.env.sim.solver.substep_grad(action_sm1, dt=dt_sm1, pre_grad_fn=pre_grad_fn)

                sm1_local = self.env.sim.solver.current_s_local # NOTE: substep_grad decrement s

                for grad_fn in post_substep_grad_fn:
                    grad_fn(sm1, sm1_local)

                if sm1 in grad_s:
                    assert sm1_local == self.env.sim.solver.get_cyclic_s(sm1)
                    for grad_k in grad[sm1].keys():
                        if grad_k[:20] == 'self.env.sim.solver.':
                            var_name = grad_k[20:]
                            grad[sm1][grad_k] = self.env.sim.device.clone(self.env.sim.apply('get', var_name + '.grad', s=sm1_local))
                        elif grad_k[:22] == 'self.env.design_space.':
                            var_name = grad_k[22:]
                            assert var_name == 'v_buffer'
                            grad[sm1][grad_k] = self.env.design_space.get_v_buffer_grad(s=sm1)
                        else:
                            raise ValueError(f'Unrecognized gradient name {grad_k}')

            # Compute gradient in design space
            self.env.design_space.set_design_grad(None)
            for s_none in grad.keys():
                if s_none is not None: # time-invariant data
                    continue
                for grad_k in grad[s_none].keys():
                    if grad_k[:20] == 'self.env.sim.solver.':
                        var_name = grad_k[20:]
                        grad[s_none][grad_k] = self.env.sim.device.clone(self.env.sim.apply('get', var_name + '.grad'))
                    elif grad_k[:29] == 'self.env.design_space.buffer.':
                        grad[s_none][grad_k] = self.env.sim.device.to_ext(eval(grad_k + '.grad'))
                    else:
                        raise ValueError(f'Unrecognized gradient name {grad_k}')
        else:
            for s in range(latest_s, 0, -1):
                def pre_grad_fn():
                    # Compute loss
                    for loss in self.losses:
                        if s == latest_s:
                            loss.compute_final_step_loss(s)
                        else:    
                            loss.compute_per_step_loss(s)
                    self.accumulate_loss(s, norm_factor)

                sm1 = s - 1 # NOTE: to obtain gradient at s, we need to call substep_grad(s-1)
                action_sm1 = self.env.sim.solver.checkpoint_cache['act_buffer'][sm1]
                dt_sm1 = self.env.sim.solver.checkpoint_cache['dt'][sm1]
                self.env.sim.solver.substep_grad(action_sm1, dt=dt_sm1, pre_grad_fn=pre_grad_fn, compute_grad=False)

        # print(latest_s, self.total_loss.to_numpy()[:latest_s+1].sum()) # DEBUG

        return self.total_loss.to_numpy()[:latest_s+1], grad

    def reset(self, loss_reset_kwargs):
        for k, v in zip(self.loss_names, self.losses):
            v.reset(**loss_reset_kwargs[k])

        self.total_loss.fill(0.)
        self.total_loss.grad.fill(0.)

    @ti.kernel
    def accumulate_loss(self, s: I_DTYPE, norm_factor: F_DTYPE):
        for i in ti.static(range(len(self.losses))):
            self.total_loss[s] += self.loss_coefs[i] * self.losses[i].data['loss'][s] * norm_factor


@ti.data_oriented
class Loss:
    def __init__(self, parent, env, **kwargs):
        self.parent = parent # NOTE: check self.parent.shared_data to avoid recomputation
        self.env = env

        self.data = dict()

    def reset(self, **kwargs):
        for v in self.data.values():
            v.fill(0.)
            v.grad.fill(0.)

    def compute_final_step_loss(self, s):
        pass

    def compute_final_step_grad(self, s):
        pass

    def compute_per_step_loss(self, s):
        pass

    def compute_per_step_grad(self, s):
        pass

    def get_s_local(self, s):
        return self.env.sim.solver.get_cyclic_s(s)

    def get_loss_stats(self): # used for logging
        stats = dict()
        stats['loss'] = self.data['loss'].to_numpy().mean()
        return stats

    ### Helper functions ###
    @ti.kernel
    def _compute_n_robot_particles(self):
        # Require self.data['n_robot_particles']
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            self.data['n_robot_particles'][None] += ti.cast(is_robot, F_DTYPE)

    @ti.kernel
    def _compute_x_avg(self, s: I_DTYPE, s_local: I_DTYPE):
        # Require self.data['x_avg']
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot: # assume uniform density
                self.data['x_avg'][s] += self.env.sim.solver.x[s_local, p] / self.data['n_robot_particles'][None]

    @ti.kernel
    def _compute_v_avg_simple(self, s: I_DTYPE, s_local: I_DTYPE):
        # Require self.data['v_avg'] NOTE: no projection on orientation
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                self.data['v_avg'][s] += self.env.sim.solver.v[s_local, p] / self.data['n_robot_particles'][None]
    
    @ti.ad.grad_replaced
    def _compute_v_avg(self, s, s_local):
        self.env.design_space.compute_orientation_kernel(s_local,
                                                         self.env.design_space.orientation_data['min_p'],
                                                         self.env.design_space.orientation_data['max_p'])
        self._compute_v_avg_kernel(s, s_local)

    @ti.ad.grad_for(_compute_v_avg)
    def _compute_v_avg_grad(self, s, s_local):
        self.env.design_space.reset_orientation_grad() # NOTE: reset gradient every time since we use shared orientation data
        self._compute_v_avg_kernel.grad(s, s_local)
        self.env.design_space.compute_orientation_kernel.grad(s_local,
                                                              self.env.design_space.orientation_data['min_p'],
                                                              self.env.design_space.orientation_data['max_p'])

    @ti.kernel
    def _compute_v_avg_kernel(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            is_robot = self.env.design_space.is_robot(id) and (self.env.sim.solver.p_rho[p] > 0)
            if is_robot:
                orientation = self.env.design_space.orientation_data['orientation'][None]
                v = self.env.sim.solver.v[s_local, p]
                proj_v = v.dot(orientation)
                self.data['v_avg'][s] += proj_v * orientation / self.data['n_robot_particles'][None]
    ########################


@ti.data_oriented
class FinalStepCoMLoss(Loss):
    def __init__(self, parent, env, x_mul=[1., 0., 0.]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )
        self.x_mul = x_mul

    def reset(self):
        super().reset()

    def compute_final_step_loss(self, s):
        self._compute_n_robot_particles()
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                x_mul = ti.Vector(self.x_mul, F_DTYPE) / self.data['n_robot_particles'][None]
                p_loss = -(self.env.sim.solver.x[s_local, p] * x_mul).sum()
                self.data['loss'][s] += p_loss # TODO: compute difference from initial position


@ti.data_oriented
class TrajectoryFollowingLoss(Loss):
    def __init__(self, parent, env, goal):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(env.sim.solver.max_substeps), needs_grad=True),
            traj=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(env.sim.solver.max_substeps), needs_grad=True),
        )
        self.goal = goal # (x, y, z, s)

    def reset(self):
        super().reset()
        s = 0
        s_local = self.get_s_local(s)
        self._compute_n_robot_particles()
        self._compute_x_avg(s, s_local) # get robot initial position as reference
        self._prepare_traj(s0=s)

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    def _prepare_traj(self, s0):
        self._terrain_info = [v for v in self.env.sim.solver.static_component_info if v['type'] == 'Static.Terrain'][0]
        self._set_traj(s0)

    @ti.kernel
    def _set_traj(self, s0: I_DTYPE):
        max_s = int(self.goal[3])
        for s in range(max_s):
            # get waypoints
            goal_xyz = ti.Vector(self.goal[:3], F_DTYPE)
            traj_xyz_at_s = goal_xyz * s / ti.cast(self.goal[3], F_DTYPE) + self.data['x_avg'][s0]
            
            # conversion from mpm resolution to terrain surface resolution
            base = traj_xyz_at_s * self.env.sim.solver.inv_dx - 0.5
            padding = ti.cast(self.env.sim.solver.padding, F_DTYPE)
            base = (base - padding) / (ti.cast(self.env.sim.solver.n_grid, F_DTYPE) - 2 * padding)
            I = ti.cast(base * self._terrain_info['resolution'], I_DTYPE)
            i = ti.min(ti.max(I[0], 0), self._terrain_info['resolution'] - 1)
            j = ti.min(ti.max(I[2], 0), self._terrain_info['resolution'] - 1)

            # set trajectory
            surface_point = self._terrain_info['polysurface_points'][i, j]
            # surface_normal = self._terrain_info['polysurface_normals'][i, j] # NOTE: use normal may lead to self-intersecting trajectory

            height = ti.cast(self.goal[1], F_DTYPE)
            self.data['traj'][s] = surface_point + height

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        self.data['loss'][s] = (self.data['x_avg'][s] - self.data['traj'][s]).norm()


@ti.data_oriented
class PerStepCoVLoss(Loss):
    def __init__(self, parent, env, v_mul):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )
        self.v_mul = v_mul

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                v_mul = ti.Vector(self.v_mul, F_DTYPE) / self.data['n_robot_particles'][None]
                p_loss = -(self.env.sim.solver.v[s_local, p] * v_mul).sum()
                self.data['loss'][s] += p_loss


@ti.data_oriented
class AnimatedEMDLoss(Loss):
    def __init__(self, parent, env, mesh_dir, substep_freq, mesh_num_points,
                 final_target_idx=None, recenter_mesh_target=True, no_reset_offset=False,
                 no_normalize_by_n_particles=False):
        super().__init__(parent, env)

        assert env.sim.device.name in ['TorchCPU', 'TorchGPU']

        self.substep_freq = substep_freq
        self.final_target_idx = final_target_idx
        self.recenter_mesh_target = recenter_mesh_target
        self.no_reset_offset = no_reset_offset
        self.no_normalize_by_n_particles = no_normalize_by_n_particles

        self.points = []
        for mesh_fname in sorted(os.listdir(mesh_dir)):
            file_ext = os.path.splitext(mesh_fname)[-1]
            mesh_fpath = os.path.abspath(os.path.join(mesh_dir, mesh_fname))
            if file_ext == '.obj':
                mesh_scale = self.env.design_space.cfg.base_shape.scale
                mesh_offset = self.env.design_space.cfg.base_shape.offset
                points_i = load_points_from_mesh(mesh_fpath, scale=mesh_scale, offset=mesh_offset, num_points=mesh_num_points)
            elif file_ext == '.pcd':
                pcd = o3d.io.read_point_cloud(mesh_fpath)
                points_i = np.asarray(pcd.points)
            else:
                continue
            
            points_i = env.sim.device.tensor(points_i)
            self.points.append(points_i)

        self.loss_F = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=0.01,
            debias=False,
            potentials=False)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()
        x0 = self.env.design_space.get_x(0)
        self.x0_mean = x0.mean(0)
        self._grad = dict()

    def compute_final_step_loss(self, s):
        if self.final_target_idx is not None:
            self._compute_step_loss(s, self.final_target_idx)

    def compute_final_step_grad(self, s):
        if self.final_target_idx is not None:
            self._compute_step_grad(s)

    def compute_per_step_loss(self, s):
        if (s % self.substep_freq == 0) and (self.final_target_idx is None):
            v_i = (s // self.substep_freq) % len(self.points)
            self._compute_step_loss(s, v_i)
    
    def compute_per_step_grad(self, s):
        if (s % self.substep_freq == 0) and (self.final_target_idx is None):
            self._compute_step_grad(s)

    def _compute_step_loss(self, s, v_i):
        x = self.env.design_space.get_x(s) # TODO: cannot handle geometry codesign for now
        x.requires_grad = True

        x_target = self.points[v_i]
        if self.recenter_mesh_target:
            offset = x.mean(0)
        else:
            offset = self.x0_mean
        if not self.no_reset_offset:
            x_target = x_target - x_target.mean(0) + offset # NOTE: do we need rotation?!

        L = self.loss_F(x, x_target)
        if not self.no_normalize_by_n_particles:
            L = L / self.data['n_robot_particles'][None]
        self.data['loss'][s] = L.item()

        if False:
            import numpy as np
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(x.data.cpu().numpy())
            pcd.paint_uniform_color((0., 1., 0.))
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(x_target.data.cpu().numpy())
            pcd_target.paint_uniform_color((1., 0., 0.))
            # o3d.visualization.draw_geometries([pcd, pcd_target])

            pcd_merged = o3d.geometry.PointCloud()
            pcd_merged.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points), np.asarray(pcd_target.points)]))
            pcd_merged.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.colors), np.asarray(pcd_target.colors)]))
            o3d.io.write_point_cloud('./local/tmp.pcd', pcd_merged)
            import pdb; pdb.set_trace()

        g_x, = torch.autograd.grad(L, [x])
        self._grad[s] = g_x

    def _compute_step_grad(self, s):
        # NOTE: assume shape not changing
        s_local = self.get_s_local(s)
        mask = self.env.design_space.get_particle_mask()
        n_active_particles = mask.sum().item()
        mask = mask[mask>0]
        mask = mask[:, None]
        grad = self.env.sim.device.create_f_tensor((n_active_particles, self.env.sim.solver.dim))
        grad = self._grad[s] * mask + grad * (1 - mask) # TODO: setting grad like this can be slow with many non-robot particles
        self.add_design_x_grad(s_local, grad)

    @ti.kernel
    def add_design_x_grad(self, s: I_DTYPE, grad: ti.types.ndarray()):
        # for p in range(self.env.sim.solver.n_particles[None]):
        for p in range(self.env.design_space.p_start, self.env.design_space.p_end):
            p_ext = p - self.env.design_space.p_start
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                for d in ti.static(range(self.env.sim.solver.dim)):
                    self.env.sim.solver.x.grad[s, p][d] += grad[p_ext, d]


@ti.data_oriented
class VelocityFollowingLoss(Loss):
    def __init__(self, parent, env, v_mul, mode):
        super().__init__(parent, env)

        self.v_mul = v_mul
        self.mode = mode

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            v_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )
        if self.mode == 1:
            self.data['loss_norm'] = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True)
            self.data['loss_dir'] = ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True)
        self.non_learnable_data = dict(
            v_tgt=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=())
        )

        self.v_avg_mode = 1

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['v_tgt'][None] = self.env.objective.get_v_tgt(s)
        if self.v_avg_mode == 0:
            self._compute_v_avg_simple(s, s_local)
        else:
            self._compute_v_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['v_tgt'][None] = self.env.objective.get_v_tgt(s)
        self._compute_loss.grad(s, s_local)
        if self.v_avg_mode == 0:
            self._compute_v_avg_simple.grad(s, s_local)
        else:
            self._compute_v_avg_grad(s, s_local)

    def get_loss_stats(self):
        stats = super().get_loss_stats()
        if self.mode == 1:
            stats['loss_norm'] = self.data['loss_norm'].to_numpy().mean()
            stats['loss_dir'] = self.data['loss_dir'].to_numpy().mean()
        return stats

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        v_mul = ti.Vector(self.v_mul, F_DTYPE)

        v_tgt = self.non_learnable_data['v_tgt'][None]
        v_avg = self.data['v_avg'][s]

        if ti.static(self.mode == 0): # squared difference of absolute velocity
            v_diff = (v_avg - v_tgt) * v_mul
            self.data['loss'][s] = v_diff.dot(v_diff)
        elif ti.static(self.mode == 1): # split into norm and direction
            v_tgt_norm = v_tgt.norm(NORM_EPS)
            v_avg_norm = v_avg.norm(NORM_EPS)

            v_tgt_dir = v_tgt * v_mul / (v_tgt_norm + 1e-6) # NOTE: only multiply v_mul for direction
            v_avg_dir = v_avg * v_mul / (v_avg_norm + 1e-6)

            loss_norm = ti.pow(v_tgt_norm - v_avg_norm, 2)
            self.data['loss_norm'][s] = loss_norm

            loss_dir = -v_tgt_dir.dot(v_avg_dir) # cosine distance
            self.data['loss_dir'][s] = loss_dir

            self.data['loss'][s] = loss_norm * ti.cast(self.env.objective.config['weight_norm'], F_DTYPE) + \
                                loss_dir * ti.cast(self.env.objective.config['weight_direction'], F_DTYPE)
        else:
            v_diff_pos = (v_avg - v_tgt) * v_mul
            v_diff_neg = (-v_avg - v_tgt) * v_mul
            self.data['loss'][s] = ti.min(v_diff_pos.dot(v_diff_pos), v_diff_neg.dot(v_diff_neg))


@ti.data_oriented
class WaypointFollowingLoss(Loss):
    def __init__(self, parent, env):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )
        self.non_learnable_data = dict(
            x_tgt=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=())
        )
        self.env.objective.draw_x = True

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['x_tgt'][None] = self.env.objective.get_x_tgt(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self.non_learnable_data['x_tgt'][None] = self.env.objective.get_x_tgt(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        x_tgt = self.non_learnable_data['x_tgt'][None]
        x_avg = self.data['x_avg'][s]

        x_diff = x_tgt - x_avg
        self.data['loss'][s] = x_diff.dot(x_diff)


@ti.data_oriented
class RotationLoss(Loss):
    def __init__(self, parent, env, up_direction):
        super().__init__(parent, env)

        self.up_direction = up_direction

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_robot_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
            x_avg=ti.Vector.field(env.sim.solver.dim, dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
        )

        self.env.objective.draw_x = True

    def reset(self):
        super().reset()
        self._compute_n_robot_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_x_avg(s, s_local)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)
        self._compute_x_avg.grad(s, s_local)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self.env.design_space.is_robot(id) and self.env.sim.solver.p_rho[p] > 0:
                up_direction = ti.Vector(self.up_direction, F_DTYPE)
                x_centered = self.env.sim.solver.x[s_local, p] - self.data['x_avg'][s]
                v_tan_dir = up_direction.cross(x_centered).normalized(NORM_EPS)
                p_loss = -v_tan_dir.dot(self.env.sim.solver.v[s_local, p])
                self.data['loss'][s] += p_loss / self.data['n_robot_particles'][None]


@ti.data_oriented
class ThrowingObjectLoss(Loss):
    def __init__(self, parent, env, obj_particle_id=2, x_mul=[1., 0., 0.]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_object_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )
        self.obj_particle_id = obj_particle_id
        self.x_mul = x_mul

    def reset(self):
        super().reset()
        self._compute_n_object_particles()

    def compute_final_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_final_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.func
    def _is_object(self, id):
        return id == self.obj_particle_id

    @ti.kernel
    def _compute_n_object_particles(self):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id):
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id) and self.env.sim.solver.p_rho[p] > 0:
                x_mul = ti.Vector(self.x_mul, F_DTYPE) / self.data['n_object_particles'][None]
                p_loss = -(self.env.sim.solver.x[s_local, p] * x_mul).sum()
                self.data['loss'][s] += p_loss # TODO: compute difference from initial position


@ti.data_oriented
class ObjectVelocityLoss(Loss):
    def __init__(self, parent, env, obj_particle_id=2, v_mul=[1., 0., 0.]):
        super().__init__(parent, env)

        self.data = dict(
            loss=ti.field(dtype=F_DTYPE, shape=(self.env.sim.solver.max_substeps), needs_grad=True),
            n_object_particles=ti.field(dtype=F_DTYPE, shape=(), needs_grad=True),
        )
        self.obj_particle_id = obj_particle_id
        self.v_mul = v_mul

    def reset(self):
        super().reset()
        self._compute_n_object_particles()

    def compute_per_step_loss(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss(s, s_local)

    def compute_per_step_grad(self, s):
        s_local = self.get_s_local(s)
        self._compute_loss.grad(s, s_local)

    @ti.func
    def _is_object(self, id):
        return id == self.obj_particle_id

    @ti.kernel
    def _compute_n_object_particles(self):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id):
                self.data['n_object_particles'][None] += ti.cast(1, F_DTYPE)

    @ti.kernel
    def _compute_loss(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.env.sim.solver.n_particles[None]):
            id = self.env.sim.solver.particle_ids[p]
            if self._is_object(id) and self.env.sim.solver.p_rho[p] > 0:
                v_mul = ti.Vector(self.v_mul, F_DTYPE) / self.data['n_object_particles'][None]
                p_loss = -(self.env.sim.solver.v[s_local, p] * v_mul).sum()
                self.data['loss'][s] += p_loss
