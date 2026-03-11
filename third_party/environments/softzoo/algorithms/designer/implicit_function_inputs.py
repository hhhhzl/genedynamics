import numpy as np
import trimesh
import mesh_to_sdf
import torch


class ImplicitFunctionInputs:
    def __init__(self, env, coord_input_names, seed_meshes=[]):
        self.env = env
        if 'None' in coord_input_names:
            coord_input_names = []
        self.coord_input_names = coord_input_names
        self.seed_meshes = []
        for seed_mesh_path in seed_meshes:
            mesh = trimesh.load(seed_mesh_path)

            tr = np.eye(4, 4) # make size of 1-cube and span at [0,1]
            for i in range(3):
                tr[i, i] = 1. / mesh.extents[i]
            mesh.apply_transform(tr)
            tr = np.eye(4, 4)
            tr[:3, 3] = -mesh.bounds[0]
            mesh.apply_transform(tr)

            self.seed_meshes.append(mesh)
        self.input_names = []

    def construct_data(self):
        self.data = []

        # Get coordinate-related input
        coords = self.env.design_space.get_x(s=0).float()
        coords_min, coords_max = coords.min(0)[0], coords.max(0)[0]
        coords_centered = coords - coords.mean(0, keepdim=True)
        coords_normalized = (coords - coords_min) / (coords_max - coords_min) - 0.5

        if 'x' in self.coord_input_names:
            self.data.append(coords_normalized[:, 0])
        if 'y' in self.coord_input_names:
            self.data.append(coords_normalized[:, 1])
        if 'z' in self.coord_input_names:
            self.data.append(coords_normalized[:, 2])
        if 'd_xy' in self.coord_input_names:
            self.data.append(torch.sqrt(coords_centered[:, 0] ** 2 + coords_centered[:, 1] ** 2))
        if 'd_yz' in self.coord_input_names:
            self.data.append(torch.sqrt(coords_centered[:, 1] ** 2 + coords_centered[:, 2] ** 2))
        if 'd_xz' in self.coord_input_names:
            self.data.append(torch.sqrt(coords_centered[:, 0] ** 2 + coords_centered[:, 2] ** 2))
        if 'd_xyz' in self.coord_input_names:
            self.data.append(torch.linalg.norm(coords_centered, dim=1))

        self.input_names.extend(self.coord_input_names)

        # Seeding object
        for i, mesh in enumerate(self.seed_meshes):
            coords_0_to_1 = coords_normalized + 0.5
            query_points = coords_0_to_1.data.cpu().numpy()
            sdf = mesh_to_sdf.mesh_to_sdf(mesh, query_points)
            sdf = torch.from_numpy(sdf)
            self.data.append(sdf)
            self.input_names.append(f'd_s_{i}')

        self.data = torch.stack(self.data, dim=1)

    @property
    def dim(self):
        return len(self.seed_meshes) + len(self.coord_input_names)
