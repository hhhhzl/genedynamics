# profile_edoc_numpy.py
"""
Profile EDOC NumPy backend to identify performance bottlenecks.
"""
import cProfile
import pstats
import io
import numpy as np
from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_env, make_energy
from enerdynamics.core.constraints import (
    ConstraintManager,
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
    CFSProjection,
    ConstraintScheduleManager,
)
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.experiments.common.constraints import SpeedConstraint
from enerdynamics.envs.obstacles import ObstacleManager
from enerdynamics.experiments.common.obstacle_generation import generate_box2d_obstacles

def profile_edoc_numpy():
    """Profile EDOC NumPy backend."""
    # Force JAX backend before creating planner
    RuntimeBackendManager.set_backend("jax", device="cpu")
    
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
    
    # Build SDF texture for JAX acceleration (critical for CFS performance)
    # This matches what the actual experiment framework does
    if len(obstacles) > 0:
        obstacles.build_sdf_texture_2d(
            x_min=-1.5,
            x_max=1.0,
            y_min=-2.0,
            y_max=0.5,
            res=0.01,
            force_rebuild=True,
        )
    
    # Create constraint manager with CFS (matching real config)
    soft_constraint = ObstacleSoftConstraint(
        obstacles=obstacles,
        alpha=1.0,
        beta=10.0,
    )
    
    hard_constraint = ObstacleHardConstraint(
        obstacles=obstacles,
        clearance=0.1,
    )
    
    speed_constraint = SpeedConstraint(
        u_max=1.0,
    )
    
    schedule_manager = ConstraintScheduleManager.create_soft_to_hard(
        soft_alpha_start=1.0,
        soft_alpha_end=0.0,
        hard_clearance_start=0.5,
        hard_clearance_end=0.1,
        schedule_type="linear",
        reverse_mode=True,
    )
    
    feasibility_op = CFSProjection(
        obstacles=obstacles,
        schedule_manager=schedule_manager,
        use_late_stage_only=True,
        late_stage_ratio=0.2,
        use_trajectory_qp=False,
        smoothness_weight=0.0,
        reconstruct_velocity=False,
        velocity_dt=None,
        max_iterations=5,
    )
    
    constraint_manager = ConstraintManager(
        soft_constraints=[soft_constraint],
        hard_constraints=[hard_constraint, speed_constraint],
        feasibility_operator=feasibility_op,
        schedule_manager=schedule_manager,
    )
    
    # Create planner with constraint manager
    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=64,
        dt=0.05,
        action_diffuse_steps=100,
        action_nsample=64,
        action_score_mode="energy",
        constraint_manager=constraint_manager,
        terminal_energy_weight=50.0,
    )
    
    # Initial state (already defined above)
    rng = 42
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = planner.plan(rng)
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
    print("Top 30 functions by cumulative time:")
    print("=" * 80)
    print(s.getvalue())
    
    # Also print by total time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(30)
    
    print("=" * 80)
    print("Top 30 functions by total time (excluding subcalls):")
    print("=" * 80)
    print(s2.getvalue())

if __name__ == "__main__":
    profile_edoc_numpy()