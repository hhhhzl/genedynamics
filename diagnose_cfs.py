"""
Diagnostic script to check CFS projection setup and performance issues.
"""
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

def diagnose_cfs():
    """Diagnose CFS projection setup."""
    print("=" * 80)
    print("CFS Projection Diagnostic")
    print("=" * 80)
    
    # Setup
    RuntimeBackendManager.set_backend("jax", device="cpu")
    env = make_env("single_integrator_box_2d")
    env.dt = 0.05
    env.horizon = 64
    env.control_limit = 1.0
    
    energy = make_energy("single_integrator_box_2d")
    
    # Generate obstacles
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
    
    print(f"\n1. Obstacle Information:")
    print(f"   Total obstacles: {len(obstacles)}")
    
    # Check obstacle types and JAX support
    obstacle_types = {}
    jax_support_count = 0
    for i, obs in enumerate(obstacles):
        obs_type = type(obs).__name__
        obstacle_types[obs_type] = obstacle_types.get(obs_type, 0) + 1
        has_jax_sdf = hasattr(obs, "jax_sdf")
        if has_jax_sdf:
            jax_support_count += 1
        if i < 5:  # Print first 5
            print(f"   Obstacle {i}: {obs_type}, has_jax_sdf: {has_jax_sdf}")
    
    print(f"   Obstacle types: {obstacle_types}")
    print(f"   Obstacles with jax_sdf: {jax_support_count}/{len(obstacles)}")
    
    # Build SDF texture
    print(f"\n2. SDF Texture:")
    if len(obstacles) > 0:
        obstacles.build_sdf_texture_2d(
            x_min=-1.5,
            x_max=1.0,
            y_min=-2.0,
            y_max=0.5,
            res=0.01,
            force_rebuild=True,
        )
        has_texture = obstacles.get_sdf_texture_2d() is not None
        has_sample_method = hasattr(obstacles, 'sample_sdf_and_grad_2d')
        print(f"   SDF texture built: {has_texture}")
        print(f"   Has sample_sdf_and_grad_2d: {has_sample_method}")
    
    # Create constraint manager
    soft_constraint = ObstacleSoftConstraint(
        obstacles=obstacles,
        alpha=1.0,
        beta=10.0,
    )
    
    hard_constraint = ObstacleHardConstraint(
        obstacles=obstacles,
        clearance=0.1,
    )
    
    speed_constraint = SpeedConstraint(u_max=1.0)
    
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
    
    print(f"\n3. CFS Projection Setup:")
    print(f"   CFSProjection created: {feasibility_op is not None}")
    print(f"   use_late_stage_only: {feasibility_op.use_late_stage_only}")
    print(f"   late_stage_ratio: {feasibility_op.late_stage_ratio}")
    print(f"   max_iterations: {feasibility_op.max_iterations}")
    
    # Try to create JAX projector
    print(f"\n4. JAX Projector Creation:")
    try:
        jax_projector = feasibility_op.make_jax_projector()
        if jax_projector is not None:
            print(f"   ✓ JAX projector created successfully")
            print(f"   Projector type: {type(jax_projector)}")
            
            # Test with dummy data
            import jax.numpy as jnp
            dummy_positions = jnp.zeros((64, 2), dtype=jnp.float32)
            dummy_clearance = jnp.asarray(0.1, dtype=jnp.float32)
            try:
                result = jax_projector(dummy_positions, dummy_clearance)
                print(f"   ✓ Test projection successful, output shape: {result.shape}")
            except Exception as e:
                print(f"   ✗ Test projection failed: {e}")
        else:
            print(f"   ✗ JAX projector is None (fallback to NumPy)")
    except Exception as e:
        print(f"   ✗ Failed to create JAX projector: {e}")
        import traceback
        traceback.print_exc()
    
    constraint_manager = ConstraintManager(
        soft_constraints=[soft_constraint],
        hard_constraints=[hard_constraint, speed_constraint],
        feasibility_operator=feasibility_op,
        schedule_manager=schedule_manager,
    )
    
    # Create planner and check backend
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
    
    print(f"\n5. Planner Backend:")
    print(f"   Backend type: {type(planner._backend_impl).__name__}")
    if hasattr(planner._backend_impl, '_jax_cfs_projector'):
        cfs_proj = planner._backend_impl._jax_cfs_projector
        print(f"   _jax_cfs_projector: {cfs_proj is not None}")
        if cfs_proj is not None:
            print(f"   Projector type: {type(cfs_proj)}")
    else:
        print(f"   _jax_cfs_projector attribute not found")
    
    print(f"\n6. Schedule Check:")
    if schedule_manager is not None:
        total_steps = 100 - 2  # Ndiffuse - 2
        print(f"   Total diffusion steps: 100")
        print(f"   Late stage steps (last 20%): {int(100 * 0.2)}")
        print(f"   Checking hard constraint activation:")
        for step in [80, 85, 90, 95, 99]:
            is_active = schedule_manager.is_hard_active(step, total_steps)
            clearance = schedule_manager.get_hard_clearance(default=0.0, step=step, total_steps=total_steps)
            print(f"     Step {step}: active={is_active}, clearance={clearance:.3f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    diagnose_cfs()

