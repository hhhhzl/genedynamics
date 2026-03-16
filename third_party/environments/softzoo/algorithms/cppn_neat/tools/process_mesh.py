import os
import argparse
import numpy as np
import trimesh
import open3d as o3d
import mesh_to_sdf
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--n-points', type=int, default=50000)
    parser.add_argument('--plot-mode', type=int, default=0)
    args = parser.parse_args()

    # Get query points
    points = np.random.uniform(0., 1., size=(args.n_points, 3))

    points_max, points_min = points.max(0)[None,:], points.min(0)[None,:]
    query_points = (points - points_min) / (points_max - points_min) # 0 - 1

    # Read mesh
    valid_ext = ['.obj', '.stl', '.ply']
    root_dir = os.path.abspath(args.root_dir)
    for fname in sorted(os.listdir(root_dir)):
        ext = os.path.splitext(fname)[-1]
        if ext not in valid_ext:
            print(f'Skip {fname}')
            continue
        
        fpath = os.path.join(root_dir, fname)
        mesh = trimesh.load_mesh(fpath)

        print(f'[{fname}]')
        print(f'Watertight: {mesh.is_watertight}')

        tr = np.eye(4, 4)
        tr[:3, 3] = -mesh.bounds[0]
        mesh.apply_transform(tr)
        tr = np.eye(4, 4)
        for i in range(3):
            tr[i, i] = 1. / max(mesh.extents)
        mesh.apply_transform(tr)

        sdf = mesh_to_sdf.mesh_to_sdf(mesh, query_points, scan_count=100, normal_sample_count=11, sign_method='depth')

        pcd = o3d.geometry.PointCloud()
        if args.plot_mode == 0:
            pcd.points = o3d.utility.Vector3dVector(points[sdf < 0])
        elif args.plot_mode == 1:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(float_to_rgba(sdf, abs_vmax=abs(sdf.min()))[:,:3])
        else:
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = np.zeros((points.shape[0], 3))
            colors[sdf < 0, 0] = 1.
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])


def float_to_rgba(val, abs_vmax=None, cmap='bwr'):
    if abs_vmax is None:
        norm = plt.Normalize(vmin=val.min(), vmax=val.max())
    else:
        norm = plt.Normalize(vmin=-abs_vmax, vmax=abs_vmax)
    cmap = mpl.cm.get_cmap(cmap)

    return cmap(norm(val))


if __name__ == '__main__':
    main()

