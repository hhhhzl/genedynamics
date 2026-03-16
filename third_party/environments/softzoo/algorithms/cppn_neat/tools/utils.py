import numpy as np
import torch
import trimesh
import mesh_to_sdf
import open3d as o3d
import skimage


def construct_inputs(points, coords_inp, seed_object_mesh):
    inputs = dict()

    center = points.mean(0, keepdim=True)
    points_centered = points - center

    if 'x' in coords_inp:
        inputs['x'] = points_centered[:, 0]
    if 'y' in coords_inp:
        inputs['y'] = points_centered[:, 1]
    if 'z' in coords_inp:
        inputs['z'] = points_centered[:, 2]

    if 'd_xy' in coords_inp:
        inputs['d_xy'] = torch.sqrt(points_centered[:, 0]**2 + points_centered[:, 1]**2)
    if 'd_yz' in coords_inp:
        inputs['d_yz'] = torch.sqrt(points_centered[:, 1]**2 + points_centered[:, 2]**2)
    if 'd_xz' in coords_inp:
        inputs['d_xz'] = torch.sqrt(points_centered[:, 0]**2 + points_centered[:, 2]**2)
    if 'd_xyz' in coords_inp:
        inputs['d_xyz'] = torch.linalg.norm(points_centered, dim=1)

    # Prepare seeding-object input
    if len(seed_object_mesh) > 0:
        for mesh_i, mesh_path in enumerate(seed_object_mesh):
            # load mesh
            mesh = trimesh.load_mesh(mesh_path)
            # assert mesh.is_watertight

            # normalize both query points and mesh to 0-1
            tr = np.eye(4, 4)
            for i in range(3):
                tr[i, i] = 1. / max(mesh.extents)
            mesh.apply_transform(tr)
            tr = np.eye(4, 4)
            tr[:3, 3] = -mesh.bounds[0]
            mesh.apply_transform(tr)

            points_max, points_min = points.max(0, keepdim=True)[0], points.min(0, keepdim=True)[0]
            query_points = (points - points_min) / (points_max - points_min)

            # convert to sdf
            sdf = mesh_to_sdf.mesh_to_sdf(mesh,
                                          query_points.cpu().numpy(),
                                          scan_count=100,
                                          normal_sample_count=11,
                                          sign_method='depth')
            sdf = torch.from_numpy(sdf)
            inputs[f'd_s_{mesh_i}'] = sdf

    print('CPPN Inputs: ', inputs.keys())

    return inputs


def o3d_mesh_to_trimesh(o3d_mesh):
    kwargs = dict(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles)
    )
    if len(o3d_mesh.vertex_normals) != 0:
        kwargs['vertex_normals'] = np.asarray(o3d_mesh.vertex_normals)
    if len(o3d_mesh.triangle_normals) != 0:
        kwargs['face_normals'] = np.asarray(o3d_mesh.triangle_normals)
    mesh = trimesh.Trimesh(**kwargs)

    return mesh
