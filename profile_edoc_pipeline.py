# profile_edoc_pipeline.py
"""
Profile EDOC with new HighPerformanceConstraintPipeline.

This profile script uses the new constraint pipeline architecture instead of
the legacy ConstraintManager, allowing for performance comparison.
"""
import cProfile
import pstats
import io
import numpy as np
from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_env, make_energy
from enerdynamics.experiments.common.constraints import create_constraint_pipeline
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.envs.obstacles import ObstacleManager
from enerdynamics.experiments.common.obstacle_generation import generate_box2d_obstacles

# Import to trigger registration of all constraint components
from enerdynamics.core.constraints.convexify import cfs  # noqa: F401
from enerdynamics.core.constraints.operators import qp  # noqa: F401
from enerdynamics.core.constraints.schedulers import cosine_anneal  # noqa: F401
from enerdynamics.core.constraints.solvers import jaxopt_osqp_solver  # noqa: F401

def profile_edoc_pipeline():
    """Profile EDOC with new constraint pipeline."""
    # Force backend (can be "numpy" or "jax")
    backend_name = "jax"  # Change to "jax" to test JAX backend
    RuntimeBackendManager.set_backend(backend_name, device="cpu")
    
    # Setup environment (adjust parameters as needed)
    env = make_env("single_integrator_box_2d")
    env.dt = 0.05
    env.horizon = 64
    env.control_limit = 1.0
    
    energy = make_energy("single_integrator_box_2d")
    
    # Create obstacles (similar to level 6)
    start_pos = np.array([-0.2, -1.5], dtype=np.float32)
    target_pos = np.array([0.0, 0.0], dtype=np.float32)
    
    obstacle_config = {
        'robot_radius': 0.05,
        'obstacle_radius_scale': 1.3,
        'p_max': 2.0,
        'map_bounds': {
            'x_min': -1.5,
            'x_max': 1.0,
            'y_min': -2.0,
            'y_max': 0.5
        },
        'enable_connectivity_check': True,
        'enable_nonconvexity_check': True,
    }
    
    obstacles = generate_box2d_obstacles(
        level=6,
        seed=0,
        start_pos=start_pos,
        target_pos=target_pos,
        config=obstacle_config,
    )
    
    # Build SDF texture for acceleration (critical for CFS performance)
    if len(obstacles) > 0:
        obstacles.build_sdf_texture_2d(
            x_min=-1.5,
            x_max=1.0,
            y_min=-2.0,
            y_max=0.5,
            res=0.01,
            force_rebuild=True,
        )
    
    # Create constraint configuration (matching legacy config)
    constraint_config = {
        'soft_constraint': {
            'enabled': True,
            'alpha': 1.0,
            'beta': 10.0,
        },
        'hard_constraint': {
            'enabled': True,
            'clearance': 0.1,
        },
        'schedule': {
            'enabled': True,
            'type': 'soft_to_hard',
            'soft_alpha_start': 1.0,
            'soft_alpha_end': 0.0,
            'hard_clearance_start': 0.5,
            'hard_clearance_end': 0.1,
            'schedule_type': 'linear',
            'reverse_mode': True,
        },
        'cfs': {
            'enabled': True,
            'use_trajectory_qp': True,  # Use per-step QP instead
            'max_iterations': 30,
            'use_late_stage_only': True,
            'late_stage_ratio': 0.2,
            'reconstruct_velocity': False,
            'smoothness_weight': 1.0,
            'constraint_margin': 0.25,
            'max_constraints_per_point': 8,
        },
        'action_constraint_type': 'speed',
    }
    
    # Create constraint pipeline using new architecture
    constraint_pipeline = create_constraint_pipeline(
        obstacles=obstacles,
        level=6,
        env=env,
        config=constraint_config,
        backend_name=backend_name,
        obstacle_config=obstacle_config,
    )
    
    if constraint_pipeline is None:
        print("ERROR: Failed to create constraint pipeline")
        return
    
    print(f"[Profile] Created constraint pipeline with backend: {backend_name}")
    print(f"[Profile] Pipeline config: backend={constraint_pipeline.config.backend}, "
          f"JIT={constraint_pipeline.config.use_jit}, batch={constraint_pipeline.config.use_batch}")
    
    # Create planner with constraint pipeline
    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=64,
        dt=0.05,
        action_diffuse_steps=100,
        action_nsample=64,
        action_score_mode="energy",
        constraint_pipeline=constraint_pipeline,  # Use new pipeline
        constraint_manager=None,  # Explicitly set to None to use only pipeline
        terminal_energy_weight=50.0,
    )
    
    # Initial state
    rng = 42
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = planner.plan(rng)
        print(f"[Profile] Planning completed successfully")
        states = result.get('states', None)
        if states is not None and len(states) > 0:
            final_state = states[-1] if isinstance(states, (list, tuple)) else states[-1] if hasattr(states, '__getitem__') else 'N/A'
            print(f"[Profile] Final state: {final_state}")
        else:
            print(f"[Profile] Final state: N/A")
    except Exception as e:
        print(f"Error during planning: {e}")
        import traceback
        traceback.print_exc()
        profiler.disable()
        return
    
    profiler.disable()
    
    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("=" * 80)
    print(f"Top 30 functions by cumulative time (backend: {backend_name}):")
    print("=" * 80)
    print(s.getvalue())
    
    # Also print by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(30)
    
    print("=" * 80)
    print(f"Top 30 functions by total time (excluding subcalls) (backend: {backend_name}):")
    print("=" * 80)
    print(s2.getvalue())
    
    # Print summary statistics
    print("=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    total_time = sum(stat[2] for stat in ps.stats.values())
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Total function calls: {ps.total_calls}")
    print(f"Primitive calls: {ps.prim_calls}")

if __name__ == "__main__":
    profile_edoc_pipeline()

