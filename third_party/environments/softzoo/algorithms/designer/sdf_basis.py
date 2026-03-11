import open3d as o3d
import trimesh
import numpy as np
from scipy.optimize import linear_sum_assignment
import mesh_to_sdf
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d import transforms
import roma

from .base import Base
from softzoo.tools.general_utils import extract_part_pca


class SDFBasis(Base):
    def __init__(self,
                 env,
                 pcd_paths,
                 mesh_paths,
                 n_actuators,
                 lr,
                 n_voxels=20,
                 passive_geometry_mul=1,
                 passive_softness_mul=1,
                 use_global_coefs=False,
                 init_coefs_geometry=None,
                 init_coefs_softness=None,
                 init_coefs_actuator=None,
                 init_coefs_actuator_direction=None,
                 align_parts=True,
                 coefs_activation='linear',
                 actuator_mul=1.,
                 device='cpu'):
        super(SDFBasis, self).__init__(env)

        assert len(pcd_paths) > 0, 'Need to at least specify one pcd path'
        assert len(pcd_paths) == len(mesh_paths), f'Number of pcds {len(pcd_paths)} is not consistent with that of meshes {len(mesh_paths)}'
        n_basis = len(pcd_paths)

        # Load pcd
        all_pcds = []
        all_points = []
        for pcd_path in pcd_paths:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
            all_pcds.append(pcd)
            all_points.append(points)

        # Get calibrated points based on base points (coords from design space representation)
        coords = self.env.design_space.get_x(s=0, keep_mask=True)[0].float()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        coords_min, coords_mean, coords_max = coords.min(0), coords.mean(0), coords.max(0)
        self.coords = coords

        def calibrate_points(_pts, y_offset=0., _pts_min=None, _pts_mean=None, _pts_max=None):
            if _pts_min is None: _pts_min = _pts.min(0)
            if _pts_mean is None: _pts_mean = _pts.mean(0)
            if _pts_max is None: _pts_max = _pts.max(0)
            _pts_calibrated = _pts - _pts_mean # center
            _pts_calibrated = _pts_calibrated / max(_pts_max - _pts_min) * max(coords_max - coords_min) # rescale
            _pts_calibrated = _pts_calibrated + coords_mean # recenter
            _pts_calibrated = _pts_calibrated + np.clip(coords_min - _pts_calibrated.min(0), a_min=0, a_max=np.inf) # make sure within min-bound
            _pts_calibrated = _pts_calibrated - np.clip(_pts_calibrated.max(0) - coords_max, a_min=0, a_max=np.inf) # make sure within max-bound
            _pts_calibrated[:,1] = _pts_calibrated[:,1] + y_offset # align lower bound in y-axis
            return _pts_calibrated
        
        all_points_calibrated = []
        calibration_kwargs = []
        for points in all_points:
            points_calibrated = calibrate_points(points)
            y_offset = coords_min[1] - points_calibrated.min(0)[1]
            all_points_calibrated.append(points_calibrated)
            calibration_kwargs.append({
                'y_offset': y_offset,
                '_pts_min': points.min(0),
                '_pts_mean': points.mean(0),
                '_pts_max': points.max(0),
            })

        # Load mesh and calibrate based on base points
        all_meshes = []
        for i, mesh_path in enumerate(mesh_paths):
            mesh = trimesh.load(mesh_path)
            verts = np.asarray(mesh.vertices)
            mesh.vertices = calibrate_points(verts, **calibration_kwargs[i])
            all_meshes.append(mesh)

        # Get actuator group and direction
        all_part_pca_components = []
        all_part_pc = []
        all_part_colors = []
        for pcd in all_pcds:
            part_pca_components, part_pca_singular_values, part_pc, part_colors = extract_part_pca(pcd, return_part_colors=True)
            all_part_pca_components.append(part_pca_components)
            all_part_pc.append(part_pc)
            all_part_colors.append(part_colors)

        # Align parts
        if align_parts:
            ref_i = 0
            part_pc_ref = all_part_pc[ref_i]
            part_colors_ref = all_part_colors[ref_i]
            part_pca_components_ref = all_part_pca_components[ref_i]
            part_pc_ref_mean = {k: v.mean(0) for k, v in part_pc_ref.items()}

            pc_ref_mean = all_points[ref_i].mean(0)

            all_part_pc_new = []
            all_part_colors_new = []
            all_part_pca_components_new = []
            for i, (part_pc, part_colors, part_pca_components) in enumerate(zip(all_part_pc, all_part_colors, all_part_pca_components)):
                if i == ref_i: # don't do anything for reference
                    part_pc_new = part_pc_ref
                    part_colors_new = part_colors_ref
                    part_pca_components_new = part_pca_components_ref
                else:
                    part_pc_mean = {k: v.mean(0) for k, v in part_pc.items()}
                    n_parts_ref, n_parts_i = len(part_pc_ref_mean), len(part_pc_mean)
                    cost = np.zeros((n_parts_ref, n_parts_i))
                    assert n_parts_ref == n_parts_i, f'Number of parts not consistent {n_parts_ref} vs {n_parts_i}'
                    if True: # TODO: won't work sometimes
                        pc_i_mean = all_points[i].mean(0)
                        for k1, v1 in part_pc_ref_mean.items():
                            for k2, v2 in part_pc_mean.items():
                                if k1 == 0 and k2 == 0: # make sure passive parts are aligned (with index 0)
                                    cost[k1, k2] = -9999999.
                                else:
                                    dir1 = (v1 - pc_ref_mean) / np.linalg.norm(v1 - pc_ref_mean)
                                    dir2 = (v2 - pc_i_mean) / np.linalg.norm(v2 - pc_i_mean)

                                    cost[k1, k2] = -np.dot(dir1, dir2)
                    else:
                        for k1, v1 in part_pc_ref_mean.items():
                            for k2, v2 in part_pc_mean.items():
                                if k1 == 0 and k2 == 0: # make sure passive parts are aligned (with index 0)
                                    cost[k1, k2] = -9999999.
                                else:
                                    cost[k1, k2] = np.linalg.norm(v1 - v2)
                    row_ind, col_ind = linear_sum_assignment(cost)

                    part_pc_new, part_colors_new, part_pca_components_new = dict(), dict(), dict()
                    for row_i, col_i in zip(row_ind, col_ind):
                        part_pc_new[col_i] = part_pc[row_i]
                        part_colors_new[col_i] = part_colors[row_i]
                        part_pca_components_new[col_i] = part_pca_components[row_i]

                    part_colors_new_mat = np.zeros((len(part_colors_new), 3))
                    for _k, _v in part_colors_new.items():
                        part_colors_new_mat[_k] = _v
                    part_colors_new = part_colors_new_mat
                
                all_part_pc_new.append(part_pc_new)
                all_part_colors_new.append(part_colors_new)
                all_part_pca_components_new.append(part_pca_components_new)

            all_part_pc = all_part_pc_new
            all_part_colors = all_part_colors_new
            all_part_pca_components = all_part_pca_components_new

        # Get voxel grid
        all_voxel_size = []
        for points_calibrated in all_points_calibrated:
            voxel_size = max(points_calibrated.max(0) - points_calibrated.min(0)) / n_voxels
            all_voxel_size.append(voxel_size)

        all_part_voxel_grid = []
        for i, part_pc in enumerate(all_part_pc):
            voxel_size = all_voxel_size[i]
            
            part_voxel_grid = dict()
            for k, part_pc_k in part_pc.items():
                part_pc_k_calibrated = calibrate_points(part_pc_k, **calibration_kwargs[i])
                part_pcd_k = o3d.geometry.PointCloud()
                part_pcd_k.points = o3d.utility.Vector3dVector(part_pc_k_calibrated)
                part_voxel_grid[k] = o3d.geometry.VoxelGrid.create_from_point_cloud(part_pcd_k, voxel_size=voxel_size)
            all_part_voxel_grid.append(part_voxel_grid)

        # Construct actuator basis
        passive_color = np.array([0, 0, 0])
        all_is_passive = []
        all_actuator_basis = []
        for part_voxel_grid, part_colors in zip(all_part_voxel_grid, all_part_colors):
            passive_k = np.where(np.all(part_colors == passive_color, axis=1))[0][0]
            is_passive = np.zeros((coords.shape[0]), dtype=bool)

            actuator = np.zeros((n_actuators, coords.shape[0]), dtype=float)
            for k, part_voxel_grid_k in part_voxel_grid.items():
                is_part = part_voxel_grid_k.check_if_included(o3d.utility.Vector3dVector(coords))
                if k == passive_k:
                    is_passive = np.logical_or(is_passive, is_part)
                else:
                    actuator[k, is_part] = 1.

            for k in part_voxel_grid.keys(): # sanity check
                if k > 0:
                    has_actuator = np.any(actuator[:k, :] > 0, axis=0)
                    actuator[k, has_actuator] = 0
            assert actuator.sum(0).min() in [0, 1] and actuator.sum(0).max() in [0, 1]
            
            all_is_passive.append(is_passive)
            all_actuator_basis.append(actuator)
        self.all_is_passive = nn.Parameter(torch.from_numpy(np.array(all_is_passive)).float(), requires_grad=False)
        self.all_actuator_basis = nn.Parameter(torch.from_numpy(np.array(all_actuator_basis)), requires_grad=False)

        # Construct muscle direction basis
        use_canonical_dir = True # HACK align to canonical direction
        all_actuator_direction_basis = []
        for part_pca_components in all_part_pca_components:
            actuator_directions = np.zeros((n_actuators, self.env.sim.solver.dim))
            for k, part_pca_components_k in part_pca_components.items():
                dir = part_pca_components_k[0]
                if use_canonical_dir:
                    principle_dim = np.abs(dir).argmax()
                    if dir[principle_dim] < 0:
                        dir *= -1
                actuator_directions[k] = dir
            all_actuator_direction_basis.append(actuator_directions)
        self.all_actuator_direction_basis = nn.Parameter(torch.from_numpy(np.array(all_actuator_direction_basis)), requires_grad=False)

        # Construct geometry basis
        all_sdf_basis = []
        for mesh in all_meshes:
            sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                          coords,
                                          surface_point_method='scan',
                                          sign_method='depth', # NOTE: need to use depth otherwise lead to noise in point cloud
                                          bounding_radius=None,
                                          scan_count=100,
                                          scan_resolution=400,
                                          sample_point_count=10000000,
                                          normal_sample_count=11)
            all_sdf_basis.append(sdf)

        self.all_sdf_basis = nn.Parameter(torch.from_numpy(np.array(all_sdf_basis)), requires_grad=False)

        # Interpolation coefficients
        if coefs_activation == 'softmax':
            init_random_bound = [0., 10.]
        else:
            init_random_bound = [0., 1.]

        if init_coefs_geometry in [None, []]:
            init_coefs_geometry = np.random.uniform(*init_random_bound, size=(n_basis,))
        else:
            assert len(init_coefs_geometry) == n_basis
            init_coefs_geometry = np.array(init_coefs_geometry)
        if init_coefs_softness in [None, []]:
            init_coefs_softness = np.random.uniform(*init_random_bound, size=(n_basis,))
        else:
            assert len(init_coefs_softness) == n_basis
            init_coefs_softness = np.array(init_coefs_softness)
        if init_coefs_actuator in [None, []]:
            init_coefs_actuator = np.random.uniform(*init_random_bound, size=(n_basis,))
        else:
            assert len(init_coefs_actuator) == n_basis
            init_coefs_actuator = np.array(init_coefs_actuator)
        if init_coefs_actuator_direction in [None, []]:
            init_coefs_actuator_direction = np.random.uniform(*init_random_bound, size=(n_basis,))
        else:
            assert len(init_coefs_actuator_direction) == n_basis
            init_coefs_actuator_direction = np.array(init_coefs_actuator_direction)
        if use_global_coefs:
            self.coefs = nn.Parameter(torch.from_numpy(init_coefs_geometry), requires_grad=True)
        else:
            self.coefs_geometry = nn.Parameter(torch.from_numpy(init_coefs_geometry), requires_grad=True)
            self.coefs_softness = nn.Parameter(torch.from_numpy(init_coefs_softness), requires_grad=True)
            self.coefs_actuator = nn.Parameter(torch.from_numpy(init_coefs_actuator), requires_grad=True)
            self.coefs_actuator_direction= nn.Parameter(torch.from_numpy(init_coefs_actuator_direction), requires_grad=True)

        # Misc
        self.passive_geometry_mul = passive_geometry_mul
        self.passive_softness_mul = passive_softness_mul
        self.n_actuators = n_actuators
        self.use_global_coefs = use_global_coefs
        self.coefs_activation = coefs_activation
        self.actuator_mul = actuator_mul

        self.device = torch.device(device)
        self.to(self.device)

        trainable_params = [v for v in self.parameters() if v.requires_grad]
        self.optim = optim.Adam(trainable_params, lr=lr)

        if self.coefs_activation == 'linear':
            def coefs_act_fn(_x):
                _x = torch.clamp(_x, min=0.)
                return _x / _x.sum()
        else:
            coefs_act_fn = lambda _x: torch.softmax(_x, dim=0)
        self.coefs_act_fn = coefs_act_fn

    def reset(self):
        self.out_cache = dict(geometry=None, softness=None, actuator=None, actuator_direction=None)

    def forward(self, inp=None):
        if self.use_global_coefs:
            coefs_geometry = self.coefs_act_fn(self.coefs)
            coefs_softness = self.coefs_act_fn(self.coefs)
            coefs_actuator = self.coefs_act_fn(self.coefs)
            coefs_actuator_direction = self.coefs_act_fn(self.coefs)
        else:
            coefs_geometry = self.coefs_act_fn(self.coefs_geometry)
            coefs_softness = self.coefs_act_fn(self.coefs_softness)
            coefs_actuator = self.coefs_act_fn(self.coefs_actuator)
            coefs_actuator_direction = self.coefs_act_fn(self.coefs_actuator_direction)

        direction_average_method = ['naive', 'rotmat_avg'][1]
        if direction_average_method == 'naive':
            actuator_directions = (coefs_actuator_direction[:, None, None] * self.all_actuator_direction_basis).sum(0)
        else:
            convention = ['X', 'Y', 'Z']
            actuator_directions = []
            for i in range(self.n_actuators):
                rotmat = transforms.euler_angles_to_matrix(self.all_actuator_direction_basis[:,i], convention)
                rotmat_ws = torch.sum(coefs_actuator_direction[:,None, None] * rotmat, dim=0)
                rotmat_ws = roma.special_procrustes(rotmat_ws)
                dir_ws = transforms.matrix_to_euler_angles(rotmat_ws, convention)
                actuator_directions.append(dir_ws)
            actuator_directions = torch.stack(actuator_directions, dim=0)
        actuator_directions_matrix = []
        for i in range(self.n_actuators):
            AAt = actuator_directions[i].reshape(-1, 1) @ actuator_directions[i].reshape(-1, 1).T
            actuator_directions_matrix.append(AAt)
        actuator_directions_matrix = torch.stack(actuator_directions_matrix, dim=0)

        actuator = (coefs_actuator[:, None, None] * self.all_actuator_basis).sum(0)

        geom_thresh = 0.5
        sdf = (self.all_sdf_basis * coefs_geometry[:, None]).sum(0)
        sdf -= 0.001 # HACK NOTE: since (sdf <= 0.001) is closer to the original shape
        mask = torch.sigmoid(-sdf * 1000) # NOTE: (sdf < 0).float() has no grad flow. This is a mask of "occupied"
        mask = mask * (mask > geom_thresh).detach()
        
        passiveness = (coefs_softness[:,None] * self.all_is_passive).sum(0) # NOTE: use coefficients from softness

        active_geometry = mask
        passive_geometry = mask * self.passive_geometry_mul
        passiveness_for_geometry = passiveness.detach()
        geometry = active_geometry * (1. - passiveness_for_geometry) + passive_geometry * passiveness_for_geometry

        mask_for_softness = mask.detach() # NOTE: not update geometry
        active_softness = mask_for_softness
        passive_softness = mask_for_softness * self.passive_softness_mul
        softness = active_softness * (1. - passiveness) + passive_softness * passiveness

        geo_mask_st = torch.clamp(torch.sign(mask - geom_thresh), 0.) # NOTE: gradient flow from actuator and softness to shape
        softness = softness * geo_mask_st
        actuator = actuator * geo_mask_st

        self.out_cache['actuator_direction'] = actuator_directions_matrix
        self.out_cache['geometry'] = geometry
        self.out_cache['softness'] = softness
        self.out_cache['actuator'] = actuator * self.actuator_mul

        design = {k: v.clone() for k, v in self.out_cache.items()}

        return design

    def check_actuator(self, actuator, occupancy_mask=None, draw_mode=1):
        # Example usage: self.check_actuator(actuator, mask)
        import matplotlib.pyplot as plt
        labels = np.arange(self.n_actuators) / self.n_actuators
        vis_colors = plt.get_cmap('tab20')(labels)
        part_pcds = []
        if occupancy_mask is not None: # base points (drawn first)
            pcd = o3d.geometry.PointCloud()
            mask = occupancy_mask.data.cpu().numpy() > 0.1 # NOTE: thershold need to be smaller than geometry-passive-softness-mul
            pcd.points = o3d.utility.Vector3dVector(self.coords[mask])
            pcd.paint_uniform_color([0., 0., 0.])
            part_pcds += [pcd]
        if draw_mode == 0:
            actuator = actuator.data.bool().cpu().numpy()
            for i in range(self.n_actuators):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.coords[actuator[i]])
                pcd.paint_uniform_color(vis_colors[i,:3])
                part_pcds.append(pcd)
        else:
            actuator_max = actuator.max(0)[0]
            for i in range(self.n_actuators):
                actuator_mask = (actuator[i] == actuator_max) & (actuator[i] != 0.)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.coords[actuator_mask])
                pcd.paint_uniform_color(vis_colors[i,:3])
                part_pcds.append(pcd)
        o3d.visualization.draw_geometries(part_pcds)

    def check_geometry(self, coords, geometry):
        # Example usage: self.check_geometry(self.coords, geometry)
        import matplotlib as mpl
        geometry = geometry.data.cpu().numpy()

        colormap = mpl.cm.bwr
        # vmag_max = max(abs(geometry.min()), abs(geometry.max())) # make white being at zero
        # normalize = mpl.colors.Normalize(vmin=-vmag_max, vmax=vmag_max)
        # normalize = mpl.colors.Normalize(vmin=geometry.min(), vmax=geometry.max())
        normalize = mpl.colors.Normalize(vmin=0., vmax=1.)
        color = colormap(normalize(geometry))[:,:3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(color)

        o3d.visualization.draw_geometries([pcd])

    def check_sdf(self, coords, sdf_thresh=0.):
        # Example usage: self.coefs.data[0] = 1.; self.check_sdf(coords, sdf_thresh=0.001); or self.coefs.data = coefs_geometry; self.check_sdf(self.coords, sdf_thresh=0.001);
        sdf = (self.coefs[:, None] * self.all_sdf_basis).sum(0).data.cpu().numpy()
        mask = sdf <= sdf_thresh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords[mask])
        o3d.visualization.draw_geometries([pcd])

    def check_mesh_points_consistency(self, mesh, points, coords=None, sdf=None):
        # Example usage: self.check_mesh_points_consistency(all_meshes[0], all_points_calibrated[0])
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.paint_uniform_color([0., 1., 0.])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1., 0., 0.])

        if coords is None:
            o3d.visualization.draw_geometries([mesh_o3d, pcd])
        else:
            pcd_coords = o3d.geometry.PointCloud()
            pcd_coords.points = o3d.utility.Vector3dVector(coords)
            if sdf is not None:
                import matplotlib as mpl
                colormap = mpl.cm.bwr
                vmag_max = min(abs(sdf.min()), abs(sdf.max())) # make white being at zero
                normalize = mpl.colors.Normalize(vmin=-vmag_max, vmax=vmag_max)
                # normalize = mpl.colors.Normalize(vmin=sdf.min(), vmax=sdf.max())
                color = colormap(normalize(sdf))[:,:3]
                pcd_coords.colors = o3d.utility.Vector3dVector(color)
            else:
                pcd_coords.paint_uniform_color([0., 0., 1.])
            o3d.visualization.draw_geometries([mesh_o3d, pcd_coords])

    def visualization_dev(self, coords, points_calibrated, all_part_pc, all_part_voxel_grid, voxel_size):
        # Example usage: self.visualization_dev(coords, all_points_calibrated[0], all_part_pc[0], all_part_voxel_grid[0], voxel_size)
        import matplotlib.pyplot as plt

        coords_pcd = o3d.geometry.PointCloud()
        coords_pcd.points = o3d.utility.Vector3dVector(coords)

        points_calibrated_pcd = o3d.geometry.PointCloud()
        points_calibrated_pcd.points = o3d.utility.Vector3dVector(points_calibrated)
        points_calibrated_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(points_calibrated_pcd, voxel_size=voxel_size)
        o3d.visualization.draw_geometries([coords_pcd, points_calibrated_voxel_grid])
        # o3d.visualization.draw_geometries([coords_pcd, points_calibrated_pcd])
        # o3d.visualization.draw_geometries([points_calibrated_pcd])
        # o3d.visualization.draw_geometries_with_vertex_selection([points_calibrated_pcd])

        # is_occupied = points_calibrated_voxel_grid.check_if_included(o3d.utility.Vector3dVector(coords))

        labels = np.array(list(all_part_pc.keys()))
        max_label = labels.max()
        vis_colors = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))

        all_part_pcd_raw_vis = []
        for k, part_pc_raw in all_part_pc.items():
            part_pcd_raw_vis = o3d.geometry.PointCloud()
            part_pcd_raw_vis.points = o3d.utility.Vector3dVector(part_pc_raw)
            part_pcd_raw_vis.paint_uniform_color(vis_colors[k][:3])
            all_part_pcd_raw_vis.append(part_pcd_raw_vis)
        o3d.visualization.draw_geometries(all_part_pcd_raw_vis)

        all_part_pcd_vis = []
        for k, part_voxel_grid in all_part_voxel_grid.items():
            is_part = part_voxel_grid.check_if_included(o3d.utility.Vector3dVector(coords))
            # is_part = np.logical_and(is_occupied, is_part)

            part_pcd_vis = o3d.geometry.PointCloud()
            part_pcd_vis.points = o3d.utility.Vector3dVector(coords[is_part])
            part_pcd_vis.paint_uniform_color(vis_colors[k][:3])
            all_part_pcd_vis.append(part_pcd_vis)
        
        coords_pcd = o3d.geometry.PointCloud()
        coords_pcd.points = o3d.utility.Vector3dVector(coords)
        o3d.visualization.draw_geometries(all_part_pcd_vis)
        # o3d.visualization.draw_geometries(list(all_part_voxel_grid.values())) 

    def log_text(self):
        if self.use_global_coefs:
            coefs = self.coefs_act_fn(self.coefs)
        else:
            coefs_geometry = self.coefs_act_fn(self.coefs_geometry)
            coefs_softness = self.coefs_act_fn(self.coefs_softness)
            coefs_actuator = self.coefs_act_fn(self.coefs_actuator)
            coefs_actuator_direction = self.coefs_act_fn(self.coefs_actuator_direction)

        text = ''
        if self.use_global_coefs:
            text += f'[Global] {coefs.data.cpu().numpy()}  '
            text += f'[Global raw] {self.coefs.data.cpu().numpy()}  '
        else:
            text += f'[Geometry] {coefs_geometry.data.cpu().numpy()}  '
            text += f'[Softness] {coefs_softness.data.cpu().numpy()}  '
            text += f'[Actuator] {coefs_actuator.data.cpu().numpy()}  '
            text += f'[Direction] {coefs_actuator_direction.data.cpu().numpy()}  '
            text += f'[Geometry raw] {self.coefs_geometry.data.cpu().numpy()}  '
            text += f'[Softness raw] {self.coefs_softness.data.cpu().numpy()}  '
            text += f'[Actuator raw] {self.coefs_actuator.data.cpu().numpy()}  '
            text += f'[Direction raw] {self.coefs_actuator_direction.data.cpu().numpy()}  '
        
        return text

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        model_state_dict = dict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'coefs' in k: # to only consider coefficients; otherwise points in basis may be be consistent (due to random sampling of base shape particles)
                model_state_dict[k] = v
        self.load_state_dict(model_state_dict, strict=False)
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    @property
    def has_actuator_direction(self):
        return True


def test():
    import argparse
    from functools import partial
    import numpy as np
    import trimesh
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    import mesh_to_sdf
    import skimage

    class AppWindow:
        def __init__(self, query_points, n_meshes, all_sdf, num_points, mesh_names):
            self.to_mesh = True # ugly visualization

            self.window = gui.Application.instance.create_window(
                "SDF Interpolation", 2000, 1000)

            self.widget3d = gui.SceneWidget()
            self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

            self.query_points = query_points
            self.lit = rendering.MaterialRecord()
            self.lit.shader = "defaultLit"
            self.lit.point_size = 6.0 # default is 3.0
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(self.query_points)
            self.geometry_color = [0.5, 0.75, 1.0]
            self.geometry_name = "query_geometry"
            if self.to_mesh:
                mesh = self.pcd_to_mesh(self.pcd)
                mesh.paint_uniform_color(self.geometry_color)
                self.widget3d.scene.add_geometry(self.geometry_name, mesh, self.lit)
            else:
                self.pcd.paint_uniform_color(self.geometry_color)
                self.widget3d.scene.add_geometry(self.geometry_name, self.pcd, self.lit)
            bounds = self.widget3d.scene.bounding_box
            self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
            self.widget3d.scene.show_axes(True)

            em = self.window.theme.font_size
            margin = 0.5 * em
            self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
            self._slider_mesh_coefs = []
            for i in range(n_meshes):
                slider_mesh_coef = gui.Slider(gui.Slider.INT)
                slider_mesh_coef.set_limits(0, 100)
                slider_mesh_coef.set_on_value_changed(partial(self._on_slider_mesh_coef, i=i))
                slider_mesh_coef.int_value = 5
                self._slider_mesh_coefs.append(slider_mesh_coef)
                self.panel.add_child(gui.Label(f"Mesh #{i} ({mesh_names[i]})"))
                self.panel.add_child(slider_mesh_coef)
            
            self.window.set_on_layout(self._on_layout)
            self.window.add_child(self.widget3d)
            self.window.add_child(self.panel)

            self.all_sdf = all_sdf
            self.num_points = num_points
            self.unnormed_coefs = np.array([v.int_value for v in self._slider_mesh_coefs])
            self.update_pcd()

        def _on_layout(self, layout_context):
            r = self.window.content_rect
            self.widget3d.frame = r
            width = 17 * layout_context.theme.font_size
            height = min(
                r.height,
                self.panel.calc_preferred_size(
                    layout_context, gui.Widget.Constraints()).height)
            self.panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

        def _on_slider_mesh_coef(self, coef, i):
            self.unnormed_coefs[i] = coef
            self.update_pcd()
            
        def update_pcd(self):
            coefs = self.unnormed_coefs / (self.unnormed_coefs.sum() + 1e-8)
            
            merged_sdf = np.zeros((self.num_points,))
            for coef, sdf in zip(coefs, self.all_sdf):
                merged_sdf += coef * sdf
            mask = merged_sdf < 0

            merged_points = self.query_points[mask]
            self.pcd.points = o3d.utility.Vector3dVector(merged_points)
            geometry_name = self.geometry_name
            if self.widget3d.scene.has_geometry(geometry_name):
                self.widget3d.scene.remove_geometry(geometry_name)
            if self.to_mesh:
                mesh = self.pcd_to_mesh(self.pcd)
                mesh.paint_uniform_color(self.geometry_color)
                self.widget3d.scene.add_geometry(self.geometry_name, mesh, self.lit)
            else:
                self.pcd.paint_uniform_color(self.geometry_color)
                self.widget3d.scene.add_geometry(geometry_name, self.pcd, self.lit)

        def pcd_to_mesh(self, pcd):
            padding = 3
            # pts = self.query_points
            # pts = np.asarray(pcd.points)
            voxel_size = 0.026 # min(pts.max(0) - pts.min(0)) / 8 # 15 # good 0.025
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            voxels = voxel_grid.get_voxels()
            indices = np.stack(list(vx.grid_index for vx in voxels))
            indices_shape = indices.max(0) + 1
            voxel_grid_shape = indices_shape + 2 * padding
            voxels_np = np.ones(voxel_grid_shape)
            for idx in indices:
                idx += padding
                voxels_np[idx[0], idx[1], idx[2]] = -1.
            verts, faces, normals, values = skimage.measure.marching_cubes(voxels_np, level=0)
            voxel_grid_extents = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
            verts = (verts - padding) / indices_shape * voxel_grid_extents + voxel_grid.get_min_bound()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            # mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()
            return mesh

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-paths', type=str, nargs='+', default=[])
    parser.add_argument('--coefs', type=float, nargs='+', default=[])
    parser.add_argument('--num-points', type=int, default=50000)
    parser.add_argument('--interactive', action='store_true', default=False)
    args = parser.parse_args()

    n_meshes = len(args.mesh_paths)
    assert n_meshes > 0, f'There are only {n_meshes} meshes. Need to at least specify one mesh'
    if not args.interactive:
        assert len(args.coefs) == n_meshes, f'Number of meshes {n_meshes} not consistent with number of coefficients {len(args.coefs)}'
        assert sum(args.coefs) == 1, f'Coefficients should sum up to 1 not {sum(args.coefs)}'

    meshes = []
    for mesh_path in args.mesh_paths:
        mesh = trimesh.load(mesh_path)
        meshes.append(mesh)

    scale = max([max(mesh.extents) for mesh in meshes])
    for i in range(len(meshes)):
        vertices = np.asarray(meshes[i].vertices)
        vertices -= vertices.mean(0)
        vertices /= scale
        meshes[i].vertices = vertices

    mesh_span_ub, mesh_span_lb = [], []
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices)
        mesh_span_ub.append(vertices.max(0))
        mesh_span_lb.append(vertices.min(0))
    mesh_span_ub = np.max(mesh_span_ub, axis=0)
    mesh_span_lb = np.min(mesh_span_lb, axis=0)
    query_points = np.stack([
        np.random.uniform(mesh_span_lb[i], mesh_span_ub[i], size=args.num_points)
            for i in range(3)], axis=1)

    all_sdf = []
    for mesh in meshes:
        sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                      query_points,
                                      surface_point_method='scan',
                                      sign_method='depth', # NOTE: need to use depth otherwise lead to noise in point cloud
                                      bounding_radius=None,
                                      scan_count=100,
                                      scan_resolution=400,
                                      sample_point_count=10000000,
                                      normal_sample_count=11)
        all_sdf.append(sdf)

    if args.interactive:
        gui.Application.instance.initialize()
        mesh_names = [v.split('/')[-1].split('.')[0] for v in (args.mesh_paths)]
        window = AppWindow(query_points, n_meshes, all_sdf, args.num_points, mesh_names)
        gui.Application.instance.run()
    else:
        merged_sdf = np.zeros((args.num_points,))
        for coef, sdf in zip(args.coefs, all_sdf):
            merged_sdf += coef * sdf

        mask = merged_sdf < 0
        merged_points = query_points[mask]
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        o3d.io.write_point_cloud('./local/test.pcd', merged_pcd)


if __name__ == '__main__':
    test()
