import os
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim

from .base import Base
from .wass_barycenter_utils.interp import WassersteinInterpolate
from softzoo.tools.general_utils import extract_part_pca


class WassersteinBarycenter(Base):
    def __init__(self,
                 env,
                 n_actuators,
                 voxel_resolution,
                 init_coefs_geometry,
                 init_coefs_actuator,
                 init_coefs_softness,
                 lr,
                 geometry_offset,
                 softness_offset,
                 passive_geometry_mul,
                 passive_softness_mul,
                 device='cpu'):
        super(WassersteinBarycenter, self).__init__(env)

        # Read processed base shape in voxel grid
        self.basis_names = ['Orca', 'GreatWhiteShark', 'Fish_2']
        data = dict()
        curr_dir = os.path.dirname(__file__)
        for k in self.basis_names:
            data[k] = torch.load(f'{curr_dir}/wass_barycenter_utils/mesh/{k}.pth')
        bases = torch.stack([v['basis'] for k, v in data.items()])
        n_basis = bases.shape[0]

        length = 60
        kernel_size = 30
        target_size = (60, 35, 26)
        voxelize_dx = 1.0 / length
        eps = 1e-7
        sigma = 0.5
        niter = 10

        # Extract actuator from annotated PCDs
        self.pcd_names = ['Orca', 'GreatWhiteShark', 'Fish_2']
        pcd = dict()
        for k in self.pcd_names:
            pcd_path = os.path.join(curr_dir, f'../../softzoo/assets/meshes/pcd/{k}.pcd')
            pcd[k] = o3d.io.read_point_cloud(pcd_path)

        bbox_bounds = []
        for k1, k2 in zip(self.basis_names, self.pcd_names):
            assert k1.split('_')[0] == k2.split('_')[0]
            bbox = pcd[k2].get_axis_aligned_bounding_box()
            bbox_bounds.append(bbox.max_bound - bbox.min_bound)
        self.pcd_voxel_size = np.array(bbox_bounds).max(1) / length
        
        pcd_voxel_grid = dict()
        for i, (k, v) in enumerate(pcd.items()):
            pcd_voxel_grid[k] = o3d.geometry.VoxelGrid.create_from_point_cloud(v, voxel_size=self.pcd_voxel_size[i])

        # Construct Wasserstein inteprolation
        self.shape = WassersteinInterpolate(bases, kernel_size, sigma, niter, eps, eps, None, 'max')
        if init_coefs_geometry is None:
            init_coefs_geometry = torch.from_numpy(np.random.uniform(0., 1., size=(n_basis,)))
        else:
            n_coefs = len(init_coefs_geometry)
            assert n_coefs == n_basis, f'Number of initial coefficients {n_coefs} not consistent with number of basis {n_basis}'
            init_coefs_geometry = torch.tensor(init_coefs_geometry)
        self.shape.alpha.data = init_coefs_geometry

        # Construct actuator and stiffness
        self.use_pcd_basis = True
        voxel_grid_size = tuple(voxel_resolution)

        if self.use_pcd_basis:
            act_mapping = { # HACK
                'Orca': [0, 1, 2, 3, 4],
                'GreatWhiteShark': [0, 4, 3, 2, 1],
                'Fish_2': [0, 2, 1, 4, 3],
            }
            passive_color = np.array([0, 0, 0])

            actuator_basis = []
            softness_basis = []
            for i, (k, v) in enumerate(pcd_voxel_grid.items()):
                part_pca_components, part_pca_singular_values, part_pc, part_colors = extract_part_pca(pcd[k], return_part_colors=True)

                passive_part_id = np.linalg.norm(part_colors - passive_color, axis=1).argmin(0)

                g_idx_mean_ref = torch.stack(torch.where(bases[i, 0] > 0), dim=-1).float().mean(0)
                vg_idcs = torch.stack([torch.tensor(voxel.grid_index) for voxel in v.get_voxels()])
                g_idx_offset = (g_idx_mean_ref - vg_idcs.float().mean(0)).int()

                actuator_basis_k = torch.zeros((n_actuators,) + voxel_grid_size)
                softness_basis_k = torch.zeros(voxel_grid_size)
                for voxel in v.get_voxels():
                    act_id = np.linalg.norm(part_colors - voxel.color, axis=1).argmin(0)
                    g_idx = torch.tensor(voxel.grid_index) + g_idx_offset
                    for ii in range(3): g_idx[ii] = torch.clamp(g_idx[ii], min=0, max=voxel_resolution[ii]) # HACK
                    actuator_basis_k[act_mapping[k][act_id], g_idx[0], g_idx[1], g_idx[2]] = 1

                    if act_id == passive_part_id:
                        softness_basis_k[g_idx[0], g_idx[1], g_idx[2]] = 1. # NOTE: 1 is passive
                
                actuator_basis.append(actuator_basis_k)
                softness_basis.append(softness_basis_k)
            self.actuator_basis = torch.stack(actuator_basis)
            self.softness_basis = torch.stack(softness_basis)

            self.actuator_ceofs = nn.Parameter(torch.ones((n_basis,)))
            if init_coefs_actuator is None:
                init_coefs_actuator = torch.from_numpy(np.random.uniform(0., 1., size=(n_basis,)))
            else:
                n_coefs = len(init_coefs_actuator)
                assert n_coefs == n_basis, f'Number of initial coefficients {n_coefs} not consistent with number of basis {n_basis}'
                init_coefs_actuator = torch.tensor(init_coefs_actuator)
            self.actuator_ceofs.data = init_coefs_actuator

            self.softness_ceofs = nn.Parameter(torch.ones((n_basis,)))
            if init_coefs_softness is None:
                init_coefs_softness = torch.from_numpy(np.random.uniform(0., 1., size=(n_basis,)))
            else:
                n_coefs = len(init_coefs_softness)
                assert n_coefs == n_basis, f'Number of initial coefficients {n_coefs} not consistent with number of basis {n_basis}'
                init_coefs_softness = torch.tensor(init_coefs_softness)
            self.softness_ceofs.data = init_coefs_softness

            self.use_coefs_act_fn = True
            self.shape.use_coefs_act_fn = self.use_coefs_act_fn
            def coefs_act_fn(_x):
                _x = torch.clamp(_x, min=0.001) # NOTE: make sure it can revive from 0
                return _x / _x.sum()
            self.coefs_act_fn = coefs_act_fn
        else:
            self.actuator = nn.Parameter(torch.ones((n_actuators,) + voxel_grid_size))

        # Others
        self.device = torch.device(device)
        self.shape.to(self.device)

        self.n_basis = n_basis
        self.n_actuators = n_actuators
        self.geometry_offset = geometry_offset
        self.softness_offset = softness_offset
        self.voxel_resolution = voxel_resolution
        self.target_size = target_size
        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul

        self.renorm_shape_alpha()

        self.optim = optim.Adam(self.parameters(), lr=lr)

    def reset(self):
        self.out_cache = dict(geometry=None)

    def forward(self, inp=None):
        geom_thresh = 0.5

        barycenter, v = self.shape()
        barycenter = barycenter[0, 0]
        geometry = torch.zeros(self.voxel_resolution).to(barycenter)
        geometry[:self.target_size[0], :self.target_size[1], :self.target_size[2]] = barycenter

        geometry = geometry * (geometry > geom_thresh).detach()

        # NOTE: use transport map (not working)
        if self.use_pcd_basis:
            if self.use_coefs_act_fn:
                actuator_ceofs = self.coefs_act_fn(self.actuator_ceofs)
                softness_ceofs = self.coefs_act_fn(self.softness_ceofs)
            else:
                actuator_ceofs = self.actuator_ceofs
                softness_ceofs = self.softness_ceofs
            actuator = (self.actuator_basis * actuator_ceofs[:, None, None, None, None]).sum(0)

            # Use passiveness
            passiveness = (softness_ceofs[:,None,None,None] * self.softness_basis).sum(0) # NOTE: use coefficients from softness

            # active_geometry = geometry # NOTE: otherwise will lead to holes in shape
            # passive_geometry = geometry * self.passive_geometry_mul
            # passiveness_for_geometry = (self.shape.alpha[:,None,None,None] * self.softness_basis).sum(0) # DEBUG passiveness.detach()
            # geometry = active_geometry * (1. - passiveness_for_geometry) + passive_geometry * passiveness_for_geometry

            mask_for_softness = geometry.detach() # NOTE: not update geometry
            active_softness = mask_for_softness
            passive_softness = mask_for_softness * self.passive_softness_mul
            softness = active_softness * (1. - passiveness) + passive_softness * passiveness
        else:
            actuator = torch.softmax(self.actuator, dim=0)
            softness = torch.sigmoid(self.softness) + self.softness_offset

        geometry = geometry * (geometry > geom_thresh).detach() # NOTE: thresholding occupancy

        geo_mask_st = torch.clamp(torch.sign(geometry - geom_thresh), 0.) # NOTE: gradient flow from actuator and softness to shape
        softness = softness * geo_mask_st
        actuator = actuator * geo_mask_st

        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness
        self.out_cache['actuator'] = actuator

        # NOTE: must use clone here otherwise tensor may be modified in-place in sim
        design = {k: v.clone() for k, v in self.out_cache.items()}

        return design

    def renorm_shape_alpha(self):
        if not self.use_coefs_act_fn:
            self.shape.alpha.data = torch.clamp(self.shape.alpha.data, min=1e-4)
            self.shape.alpha.data = self.shape.alpha.data / self.shape.alpha.data.sum()

            if self.use_pcd_basis:
                self.actuator_ceofs.data = torch.clamp(self.actuator_ceofs.data, min=1e-4)
                self.actuator_ceofs.data = self.actuator_ceofs.data / self.actuator_ceofs.data.sum()

    def log_text(self):
        text = f'[Geometry] {self.shape.alpha.data.cpu().numpy()} '
        if self.use_pcd_basis:
            text += f'[Actuator] {self.actuator_ceofs.data.cpu().numpy()}'
            if self.use_coefs_act_fn:
                actuator_coefs = self.coefs_act_fn(self.actuator_ceofs.data).cpu().numpy()
                text += f'[Actuator ACT] {actuator_coefs}'
            text += f'[Softness] {self.softness_ceofs.data.cpu().numpy()}'
            if self.use_coefs_act_fn:
                softness_ceofs = self.coefs_act_fn(self.softness_ceofs.data).cpu().numpy()
                text += f'[Softness ACT] {softness_ceofs}'
        return text

    def update(self, grad, retain_graph=False):
        super().update(grad, retain_graph)
        self.renorm_shape_alpha()

    def check_voxel_alignment(self, data, pcd_voxel_grid):
        basis_voxel_grid = dict()
        for i, (k, v) in enumerate(data.items()):
            mask = v['basis'][0].numpy() > 0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.stack(np.where(mask), axis=-1) * self.pcd_voxel_size[i])
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.pcd_voxel_size[i])
            basis_voxel_grid[k] = voxel_grid
        
        for k1, k2 in zip(self.basis_names, self.pcd_names):
            vg1 = basis_voxel_grid[k1]
            vg2 = pcd_voxel_grid[k2]
            # o3d.visualization.draw_geometries([vg1, vg2])
            # o3d.io.write_voxel_grid('./local/test1.ply', vg1)
            # o3d.io.write_voxel_grid('./local/test2.ply', vg2)
            import pdb; pdb.set_trace()

    def check_actuator_basis(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, self.n_basis, subplot_kw={'projection': '3d'}, figsize=(4.8*self.n_basis, 4.8*2))
        vis_colors = plt.get_cmap('tab20')(np.arange(self.n_actuators) / self.n_actuators)
        for i in range(self.n_basis):
            for j in range(self.n_actuators):
                v = self.actuator_basis[i, j]
                xyz = np.stack(np.where(v), axis=-1)
                axes[0, i].scatter(xyz[:,0], xyz[:,2], xyz[:,1], color=vis_colors[j]) # NOTE: taichi is y being up

            mask = (self.shape.bases[i, 0] > 0).data.cpu().numpy()
            xyz = np.stack(np.where(mask), axis=-1)
            axes[1, i].scatter(xyz[:,0], xyz[:,2], xyz[:,1])
        
        for ax in axes.flatten():
            ax.set_xlim(0, max(self.voxel_resolution))
            ax.set_ylim(0, max(self.voxel_resolution))
            ax.set_zlim(0, max(self.voxel_resolution))
            
        for ax in axes.flatten(): ax.view_init(0, 90) # view angle (0, 0), (0, 90), (90, 0)

        fig.tight_layout()
        fig.savefig('./local/test.png')
        import pdb; pdb.set_trace()

    def check_softness_basis(self):
        for i in range(self.n_basis):
            mask = self.softness_basis[i].data.numpy() > 0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.stack(np.where(mask), axis=-1) * self.pcd_voxel_size[i])
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.pcd_voxel_size[i])
            o3d.visualization.draw_geometries([voxel_grid])
