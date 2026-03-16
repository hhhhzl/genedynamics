import argparse
import pickle
import torch
import open3d as o3d

from softzoo.tools.general_utils import pcd_to_mesh

from .utils import construct_inputs
from ..pytorch_neat.cppn import create_cppn


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome-path', type=str, required=True)
    parser.add_argument('--use-optim-obj', action='store_true', default=False)
    parser.add_argument('--visualize', type=str, choices=['pcd', 'mesh'], default='mesh')
    parser.add_argument('--mode', type=str, choices=['default', 'shape_gen'], default='default')
    args = parser.parse_args()

    if args.mode == 'shape_gen':
        from ..run_shape_gen import Optim
    else:
        from ..run import Optim

    # Load genome data NOTE: make sure in the correct directory in case mesh path is not matched
    with open(args.genome_path, 'rb') as f:
        data = pickle.load(f)

    if args.use_optim_obj:
        optim = Optim(data['args'])
        design = optim.genome_to_design(data['genome'], data['config'])
        pcd = optim.design_to_pcd(design)
        points = pcd.points
    else:
        points = torch.rand(size=(data['args'].n_points, 3))

        inputs = construct_inputs(points, data['args'].coords_inp, data['args'].seed_object_mesh)

        leaf_names = inputs.keys()
        node_names = ['geometry_empty', 'geometry_occupied']
        nodes = create_cppn(data['genome'], data['config'], leaf_names=leaf_names, node_names=node_names)

        design = dict(geometry=[])
        for node in nodes:
            node_out = node(**inputs)
            if 'geometry' in node.name:
                design['geometry'].append(node_out)
            else:
                raise NotImplementedError
        
        design['geometry'] = torch.stack(design['geometry'], dim=0)
        design['geometry'] = design['geometry'].argmax(dim=0)

        mask = (design['geometry'] > 0)
        points = points[mask].cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    if args.visualize == 'pcd':
        o3d.visualization.draw_geometries([pcd])
    elif args.visualize == 'mesh':
        mesh = pcd_to_mesh(pcd)
        o3d.visualization.draw_geometries([mesh])
    else:
        raise ValueError(f'Unrecognized visualization mode {args.visualize}')


if __name__ == '__main__':
    main()
