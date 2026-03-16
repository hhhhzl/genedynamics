""" Interface with controllers. """


def augment_parser(parser):
    # Controller [General]
    parser.add_argument('--action-space', type=str, default='actuation',
                        choices=['actuation', 'particle_v', 'actuator_v'])
    parser.add_argument('--action-v-strength', type=float, default=1.)
    parser.add_argument('--controller-type', type=str, default='sin_wave_open_loop',
                        choices=['sin_wave_open_loop', 'pure_sin_wave_open_loop', 'random', 'trajopt',
                                 'mlp', 'sin_wave_closed_loop'])
    parser.add_argument('--controller-lr', type=float, default=0.003)
    # Controller [Sine wave]
    parser.add_argument('--n-sin-waves', type=int, default=4)
    parser.add_argument('--actuation-omega', type=float, nargs='+', default=[30.])
    # Controller [Trajectory optimization]
    parser.add_argument('--actuation-activation', type=str, default='linear',
                        choices=['tanh', 'softmax', 'linear'])
    # Controller [Pure Sine wave]
    parser.add_argument('--sin-omega-mul', type=float, default=10)
    # Controller [MLP]
    parser.add_argument('--controller-obs-names', type=str, nargs='+', default=['com', 'objective'])
    parser.add_argument('--controller-mlp-hidden-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--controller-mlp-activation', type=str, default='Tanh')
    parser.add_argument('--controller-mlp-final-activation', type=str, default=None)
    # Controller [Closed-loop Sine wave]
    parser.add_argument('--closed-loop-n-sin-waves', type=int, default=4)
    parser.add_argument('--closed-loop-actuation-omega', type=float, nargs='+', default=[30.])
    parser.add_argument('--closed-loop-sinwave-obs-names', type=str, nargs='+', default=['com', 'objective'])
    parser.add_argument('--closed-loop-sinwave-hidden-filters', type=int, nargs='+', default=[32, 32])
    parser.add_argument('--closed-loop-sinwave-activation', type=str, default='Tanh')

    return parser


def make(args, env, torch_device):
    if args.action_space == 'particle_v':
        n_actuators = env.design_space.compute_n_particles()
        actuation_strength = args.action_v_strength
        actuation_dim = env.sim.solver.dim
    elif args.action_space == 'actuator_v':
        n_actuators = env.design_space.n_actuators
        actuation_strength = args.action_v_strength
        actuation_dim = env.sim.solver.dim
    else:
        n_actuators = env.design_space.n_actuators
        actuation_strength = env.cfg.ENVIRONMENT.actuation_strength
        actuation_dim = 1
    if hasattr(args, 'n_actuators'):
        n_actuators = args.n_actuators
    controller_kwargs = dict(
        n_actuators=n_actuators,
        actuation_strength=actuation_strength,
        lr=args.controller_lr,
        device=torch_device,
    )
    if args.controller_type == 'sin_wave_open_loop':
        from .sinwave import SinWaveOpenLoop
        controller_cls = SinWaveOpenLoop
        controller_kwargs['n_sin_waves'] = args.n_sin_waves
        controller_kwargs['actuation_omega'] = args.actuation_omega
    elif args.controller_type == 'pure_sin_wave_open_loop':
        from .sinwave import PureSinWaveOpenLoop
        controller_cls = PureSinWaveOpenLoop
        controller_kwargs['omega_mul'] = args.sin_omega_mul
    elif args.controller_type == 'sin_wave_closed_loop':
        from .sinwave import SinWaveClosedLoop
        controller_cls = SinWaveClosedLoop
        controller_kwargs['obs_space'] = env.observation_space
        controller_kwargs['n_sin_waves'] = args.closed_loop_n_sin_waves
        controller_kwargs['actuation_omega'] = args.closed_loop_actuation_omega
        controller_kwargs['obs_names'] = args.closed_loop_sinwave_obs_names
        controller_kwargs['hidden_filters'] = args.closed_loop_sinwave_hidden_filters
        controller_kwargs['activation'] = args.closed_loop_sinwave_activation
    elif args.controller_type == 'random':
        from .random import Random
        controller_cls = Random
        controller_kwargs['env'] = env
    elif args.controller_type == 'trajopt':
        from .trajopt import TrajOpt
        controller_cls = TrajOpt
        controller_kwargs['max_steps'] = env.sim.solver.max_substeps
        controller_kwargs['actuation_dim'] = actuation_dim
    elif args.controller_type == 'mlp':
        from .mlp import MLP
        controller_cls = MLP
        controller_kwargs['obs_space'] = env.observation_space
        controller_kwargs['obs_names'] = args.controller_obs_names
        controller_kwargs['hidden_filters'] = args.controller_mlp_hidden_filters
        controller_kwargs['activation'] = args.controller_mlp_activation
        controller_kwargs['actuation_dim'] = actuation_dim
        controller_kwargs['final_activation'] = args.controller_mlp_final_activation
    else:
        raise ValueError(f'Unrecognized controller type {args.controller_type}')
    controller = controller_cls(**controller_kwargs)

    return controller
