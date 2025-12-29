"""
Test script to verify CFS optimization improvements.
"""
import numpy as np
import time
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
from enerdynamics.experiments.common.obstacle_generation import generate_box2d_obstacles

def test_cfs_performance():
    """Test CFS performance with optimizations."""
    print("=" * 80)
    print("Testing CFS Optimization")
    print("=" * 80)
    
    # Test both backends
    for backend_name in ["jax", "numpy"]:
        print(f"\n{'='*80}")
        print(f"Testing {backend_name.upper()} backend")
        print(f"{'='*80}")
        
        RuntimeBackendManager.set_backend(backend_name, device="cpu")
        
        # Setup
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
        
        # Build SDF texture (critical for JAX performance)
        if len(obstacles) > 0:
            obstacles.build_sdf_texture_2d(
                x_min=-1.5,
                x_max=1.0,
                y_min=-2.0,
                y_max=0.5,
                res=0.01,
                force_rebuild=True,
            )
        
        # Create constraints
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
        
        # Check if JAX projector was created
        if backend_name == "jax":
            jax_projector = feasibility_op.make_jax_projector()
            print(f"  JAX projector created: {jax_projector is not None}")
            if jax_projector is not None:
                print(f"  Projector type: {type(jax_projector).__name__}")
        
        constraint_manager = ConstraintManager(
            soft_constraints=[soft_constraint],
            hard_constraints=[hard_constraint, speed_constraint],
            feasibility_operator=feasibility_op,
            schedule_manager=schedule_manager,
        )
        
        # Create planner
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
        
        # Check backend
        print(f"  Planner backend: {type(planner._backend_impl).__name__}")
        if hasattr(planner._backend_impl, '_jax_cfs_projector'):
            cfs_proj = planner._backend_impl._jax_cfs_projector
            print(f"  CFS projector in backend: {cfs_proj is not None}")
        
        # Run planning with timing
        rng = 42
        print(f"\n  Running planning...")
        start_time = time.time()
        
        try:
            result = planner.plan(rng)
            planning_time = time.time() - start_time
            
            print(f"  ✓ Planning completed in {planning_time:.2f} seconds")
            
            # Check if trajectory avoids obstacles
            states = result.get("states", [])
            if len(states) > 0:
                # Check final state distance to target
                final_state = np.asarray(states[-1])
                target = np.array([0.0, 0.0])
                distance = np.linalg.norm(final_state - target)
                print(f"  Final distance to target: {distance:.4f}")
                
                # Check if any state is too close to obstacles
                min_sdf = float('inf')
                for state in states[-10:]:  # Check last 10 states
                    pos = np.asarray(state[:2], dtype=np.float32)
                    sdf = obstacles.sdf(pos)
                    min_sdf = min(min_sdf, float(sdf))
                print(f"  Minimum SDF in last 10 states: {min_sdf:.4f}")
                if min_sdf < 0.05:
                    print(f"  ⚠ Warning: Some states are very close to obstacles (SDF < 0.05)")
                else:
                    print(f"  ✓ Trajectory maintains safe clearance")
        except Exception as e:
            print(f"  ✗ Planning failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_cfs_performance()

