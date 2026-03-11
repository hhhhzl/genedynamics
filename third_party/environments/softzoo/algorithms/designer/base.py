import numpy as np
import matplotlib as mpl
import open3d as o3d

import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(self, env, *args, **kwargs):
        super(Base, self).__init__()
        self.env = env
        self.optim = None

    def reset(self):
        raise NotImplementedError

    def forward(self, inp=None):
        raise NotImplementedError

    def update(self, grad, retain_graph=False):
        self.optim.zero_grad()
        for i, k in enumerate(grad.keys()):
            retain_graph =  i != (len(grad) - 1)
            self.out_cache[k].backward(gradient=grad[k], retain_graph=retain_graph)
        self.optim.step()

    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def visualize(self, design, mode='geometry'):
        pcd = self.design_to_pcd(design, mode)
        o3d.visualization.draw_geometries([pcd])

    def save_pcd(self, filepath, design, mode='geometry'):
        pcd = self.design_to_pcd(design, mode)
        o3d.io.write_point_cloud(filepath, pcd)
        return pcd

    def design_to_pcd(self, design, mode):
        xyz, x_mask = self.env.design_space.get_x(0, keep_mask=True)
        xyz = xyz.data.cpu().numpy()
        geometry = design['geometry'].data.cpu().numpy()        
        mask = (geometry > self.env.design_space.cfg.p_rho_lower_bound_mul)
        assert np.allclose(mask.astype(np.float), x_mask) # sanity check
        xyz = xyz[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        cont_cmap = mpl.cm.get_cmap('viridis')
        if mode == 'geometry':
            geometry = geometry[mask]
            norm = mpl.colors.Normalize(vmin=geometry.min(), vmax=geometry.max())
            colors = cont_cmap(norm(geometry))[:,:3]
        elif mode == 'softness':
            softness = design['softness'].detach().cpu().numpy()[mask]
            norm = mpl.colors.Normalize(vmin=softness.min(), vmax=softness.max())
            colors = cont_cmap(norm(softness))[:,:3]
        elif mode == 'actuator':
            actuator = design['actuator'].detach().cpu().numpy()
            actuator_idcs = actuator[:,mask].argmax(0)
            base_colors = np.array(mpl.cm.get_cmap('tab20').colors + mpl.cm.get_cmap('tab20b').colors + mpl.cm.get_cmap('tab20c').colors)
            colors = base_colors[actuator_idcs]
        else:
            colors = np.zeros((xyz.shape[0], 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def save_voxel_grid(self, filepath, design, mode='geometry'):
        geometry = design['geometry'].data.cpu().numpy()
        mask = (geometry > self.env.design_space.cfg.p_rho_lower_bound_mul)
        if mode == 'geometry':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.stack(np.where(mask), axis=-1))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.)

            vox_mesh = o3d.geometry.TriangleMesh()
            voxels = voxel_grid.get_voxels()
            for v in voxels:
                cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
                cube.paint_uniform_color(v.color)
                cube.translate(v.grid_index, relative=False)
                vox_mesh += cube

            vox_mesh.merge_close_vertices(0.0000001)
            o3d.io.write_triangle_mesh(filepath, vox_mesh)
        elif mode == 'actuator':
            xyz = self.env.design_space.get_x(0).data.cpu().numpy()
            actuator = self.env.design_space.get_actuator(0)
            actuator_idcs = actuator.argmax(1)
            base_colors = np.array(mpl.cm.get_cmap('tab20').colors + mpl.cm.get_cmap('tab20b').colors + mpl.cm.get_cmap('tab20c').colors)
            colors = base_colors[actuator_idcs]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            filepath = filepath.replace('.ply', '.pcd') # save as point cloud
            o3d.io.write_point_cloud(filepath, pcd)

    def log_text(self):
        return ''
