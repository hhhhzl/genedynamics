""" Interface with designers. """


def augment_parser(parser):
    # Designer [General]
    parser.add_argument('--designer-type', type=str, default='mlp')
    parser.add_argument('--designer-lr', type=float, default=0.003)
    parser.add_argument('--designer-geometry-offset', type=float, default=0.5)
    parser.add_argument('--designer-softness-offset', type=float, default=0.5)
    # Designer [MLP]
    parser.add_argument('--mlp-coord-input-names', type=str, nargs='+', default=['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    parser.add_argument('--mlp-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--mlp-activation', type=str, default='Tanh')
    parser.add_argument('--mlp-seed-meshes', type=str, nargs='+', default=[])
    # Designer [Diff-CPPN]
    parser.add_argument('--cppn-coord-input-names', type=str, nargs='+', default=['x', 'y', 'z', 'd_xy', 'd_yz', 'd_xz', 'd_xyz'])
    parser.add_argument('--cppn-seed-meshes', type=str, nargs='+', default=[])
    parser.add_argument('--cppn-n-hiddens', type=int, default=3)
    parser.add_argument('--cppn-activation-repeat', type=int, default=10)
    parser.add_argument('--cppn-activation-options', type=str, nargs='+', default=['sin', 'sigmoid'])
    # Designer [Annotated-PCD]
    parser.add_argument('--annotated-pcd-path', type=str, default=None)
    parser.add_argument('--annotated-pcd-n-voxels', type=int, default=60)
    parser.add_argument('--annotated-pcd-passive-softness-mul', type=float, default=10)
    parser.add_argument('--annotated-pcd-passive-geometry-mul', type=float, default=0.5)
    # Designer [SDF Basis]
    parser.add_argument('--sdf-basis-pcd-paths', type=str, nargs='+', default=[])
    parser.add_argument('--sdf-basis-mesh-paths', type=str, nargs='+', default=[])
    parser.add_argument('--sdf-basis-passive-softness-mul', type=float, default=10)
    parser.add_argument('--sdf-basis-passive-geometry-mul', type=float, default=0.5)
    parser.add_argument('--sdf-basis-init-coefs-geometry', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-softness', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-actuator', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-init-coefs-actuator-direction', type=float, nargs='+', default=None)
    parser.add_argument('--sdf-basis-use-global-coefs', action='store_true', default=False)
    parser.add_argument('--sdf-basis-n-voxels', type=int, default=60)
    parser.add_argument('--sdf-basis-coefs-activation', type=str, default='linear')
    parser.add_argument('--sdf-basis-actuator-mul', type=float, default=1.)
    # Designer [Particle-based Representation]
    pass
    # Designer [Voxel-based Representation]
    pass
    # Designer [Wasserstein Barycenter]
    parser.add_argument('--wass-barycenter-init-coefs-geometry', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-init-coefs-actuator', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-init-coefs-softness', type=float, nargs='+', default=None)
    parser.add_argument('--wass-barycenter-passive-softness-mul', type=float, default=10)
    parser.add_argument('--wass-barycenter-passive-geometry-mul', type=float, default=0.5)
    # Designer [Loss Lanscape Voxel-based Representation] NOTE: used for study diff-physics
    parser.add_argument('--loss-landscape-vbr-grid-index', type=int, nargs='+', default=[0, 0, 0])
    parser.add_argument('--loss-landscape-vbr-value-range', type=float, nargs='+', default=[0., 1.])
    parser.add_argument('--loss-landscape-vbr-n-trials', type=int, default=10)
    parser.add_argument('--loss-landscape-vbr-trial-type', type=str, default='geometry')

    return parser


def make(args, env, torch_device):
    n_actuators = env.design_space.n_actuators
    if hasattr(args, 'n_actuators'):
        n_actuators = args.n_actuators
    designer_kwargs = dict(
        env=env,
        n_actuators=n_actuators,
        lr=args.designer_lr,
        device=torch_device,
    )
    if args.designer_type == 'mlp':
        from ..designer.mlp import MLP
        designer_cls = MLP
        designer_kwargs['coord_input_names'] = args.mlp_coord_input_names
        designer_kwargs['filters'] = args.mlp_filters
        designer_kwargs['activation'] = args.mlp_activation
        designer_kwargs['seed_meshes'] = args.mlp_seed_meshes
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'diff_cppn':
        from ..designer.diff_cppn import DiffCPPN
        designer_cls = DiffCPPN
        designer_kwargs['coord_input_names'] = args.cppn_coord_input_names
        designer_kwargs['seed_meshes'] = args.cppn_seed_meshes
        designer_kwargs['n_hiddens'] = args.cppn_n_hiddens
        designer_kwargs['activation_repeat'] = args.cppn_activation_repeat
        designer_kwargs['activation_options'] = args.cppn_activation_options
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'annotated_pcd':
        from ..designer.annotated_pcd import AnnotatedPCD
        designer_cls = AnnotatedPCD
        designer_kwargs['pcd_path'] = args.annotated_pcd_path
        designer_kwargs['n_voxels'] = args.annotated_pcd_n_voxels
        designer_kwargs['passive_geometry_mul'] = args.annotated_pcd_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.annotated_pcd_passive_softness_mul
    elif args.designer_type == 'sdf_basis':
        from ..designer.sdf_basis import SDFBasis
        designer_cls = SDFBasis
        designer_kwargs['pcd_paths'] = args.sdf_basis_pcd_paths
        designer_kwargs['mesh_paths'] = args.sdf_basis_mesh_paths
        designer_kwargs['passive_geometry_mul'] = args.sdf_basis_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.sdf_basis_passive_softness_mul
        designer_kwargs['init_coefs_geometry'] = args.sdf_basis_init_coefs_geometry
        designer_kwargs['init_coefs_softness'] = args.sdf_basis_init_coefs_softness
        designer_kwargs['init_coefs_actuator'] = args.sdf_basis_init_coefs_actuator
        designer_kwargs['init_coefs_actuator_direction'] = args.sdf_basis_init_coefs_actuator_direction
        designer_kwargs['use_global_coefs'] = args.sdf_basis_use_global_coefs
        designer_kwargs['n_voxels'] = args.sdf_basis_n_voxels
        designer_kwargs['coefs_activation'] = args.sdf_basis_coefs_activation
        designer_kwargs['actuator_mul'] = args.sdf_basis_actuator_mul
    elif args.designer_type == 'pbr':
        from ..designer.particle_based_repr import ParticleBasedRepresentation
        designer_cls = ParticleBasedRepresentation
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
    elif args.designer_type == 'vbr':
        from ..designer.voxel_based_repr import VoxelBasedRepresentation
        designer_cls = VoxelBasedRepresentation
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
    elif args.designer_type == 'wass_barycenter':
        from ..designer.wass_barycenter import WassersteinBarycenter
        designer_cls = WassersteinBarycenter
        designer_kwargs['init_coefs_geometry'] = args.wass_barycenter_init_coefs_geometry
        designer_kwargs['init_coefs_actuator'] = args.wass_barycenter_init_coefs_actuator
        designer_kwargs['init_coefs_softness'] = args.wass_barycenter_init_coefs_softness
        designer_kwargs['geometry_offset'] = args.designer_geometry_offset
        designer_kwargs['softness_offset'] = args.designer_softness_offset
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
        designer_kwargs['passive_geometry_mul'] = args.wass_barycenter_passive_geometry_mul
        designer_kwargs['passive_softness_mul'] = args.wass_barycenter_passive_softness_mul
    elif args.designer_type == 'loss_landscape_vbr':
        from ..designer.loss_landscape_vbr import LossLandscapeVBR
        designer_cls = LossLandscapeVBR
        designer_kwargs['voxel_resolution'] = env.design_space.voxel_resolution
        designer_kwargs['grid_index'] = args.loss_landscape_vbr_grid_index
        designer_kwargs['value_range'] = args.loss_landscape_vbr_value_range
        designer_kwargs['n_trials'] = args.loss_landscape_vbr_n_trials
        designer_kwargs['trial_type'] = args.loss_landscape_vbr_trial_type
    else:
        raise ValueError(f'Unrecognized designer type {args.designer_type}')
    designer = designer_cls(**designer_kwargs)

    return designer
