"""
Test script to verify CFS optimization improvements and compare numerical differences.
"""
import numpy as np
import time
from genedynamics.envs.factories import make_env
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.types import Trajectory
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.experiments.common.constraints import create_constraint_pipeline
from genedynamics.experiments.common.obstacle_generation import generate_box2d_obstacles

def analyze_trajectory(obstacles, states, actions, backend_name):
    """Analyze trajectory and return detailed statistics."""
    # Convert to NumPy arrays (handle JAX arrays)
    # For states, convert each element if needed
    states_list = []
    for s in states:
        s_np = np.asarray(s)
        # If it's a JAX array, convert to NumPy
        try:
            import jax
            if isinstance(s_np, jax.Array):
                s_np = np.array(jax.device_get(s_np))
        except (ImportError, AttributeError):
            pass
        states_list.append(s_np)
    states_np = np.array(states_list)
    
    # Handle actions (may be JAX array or None)
    actions_np = None
    if actions is not None:
        try:
            # Check if it's a JAX array by trying to get length
            # JAX arrays can't be used in boolean context
            _ = len(actions)  # This will work for both NumPy and JAX arrays
            actions_np = np.asarray(actions)
            # If it's a JAX array, convert explicitly
            try:
                import jax
                if isinstance(actions_np, jax.Array):
                    actions_np = np.array(jax.device_get(actions_np))
            except (ImportError, AttributeError, TypeError):
                pass
        except (TypeError, ValueError):
            actions_np = None
    
    stats = {
        'backend': backend_name,
        'num_states': len(states_np),
        'states': states_np,
        'actions': actions_np,
    }
    
    if len(states_np) == 0:
        return stats
    
    # Extract positions (handle both list of arrays and 2D array)
    if states_np.ndim == 2:
        # If states is already a 2D array, take first 2 columns
        positions = states_np[:, :2].astype(np.float32)
    else:
        # If states is a list of arrays, extract first 2 elements from each
        positions = np.array([np.asarray(s)[:2] for s in states_np], dtype=np.float32)
    stats['positions'] = positions
    
    # Compute SDF for each position
    sdf_values = []
    for pos in positions:
        sdf = obstacles.sdf(pos)
        sdf_values.append(float(sdf) if np.isscalar(sdf) else float(sdf[0]))
    stats['sdf_values'] = np.array(sdf_values)
    
    # Statistics
    stats['min_sdf'] = float(np.min(sdf_values))
    stats['max_sdf'] = float(np.max(sdf_values))
    stats['mean_sdf'] = float(np.mean(sdf_values))
    stats['std_sdf'] = float(np.std(sdf_values))
    
    # Check clearance violations (SDF < robot_radius)
    robot_radius = 0.05
    violations = np.array(sdf_values) < robot_radius
    stats['num_violations'] = int(np.sum(violations))
    stats['violation_indices'] = np.where(violations)[0].tolist()
    
    # Final state analysis
    final_pos = positions[-1]
    target = np.array([0.0, 0.0])
    stats['final_distance_to_target'] = float(np.linalg.norm(final_pos - target))
    stats['final_sdf'] = float(sdf_values[-1])
    
    # Trajectory length
    if len(positions) > 1:
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        stats['trajectory_length'] = float(np.sum(segment_lengths))
        stats['mean_segment_length'] = float(np.mean(segment_lengths))
    else:
        stats['trajectory_length'] = 0.0
        stats['mean_segment_length'] = 0.0
    
    return stats

def compare_results(jax_stats, numpy_stats):
    """Compare results from JAX and NumPy backends."""
    print("\n" + "=" * 80)
    print("Detailed Numerical Comparison")
    print("=" * 80)
    
    # Basic statistics
    print("\n[Basic Statistics]")
    print(f"  Trajectory length:")
    print(f"    JAX:   {jax_stats['trajectory_length']:.4f}")
    print(f"    NumPy: {numpy_stats['trajectory_length']:.4f}")
    print(f"    Diff:  {abs(jax_stats['trajectory_length'] - numpy_stats['trajectory_length']):.4f}")
    
    print(f"\n  Final distance to target:")
    print(f"    JAX:   {jax_stats['final_distance_to_target']:.4f}")
    print(f"    NumPy: {numpy_stats['final_distance_to_target']:.4f}")
    print(f"    Diff:  {abs(jax_stats['final_distance_to_target'] - numpy_stats['final_distance_to_target']):.4f}")
    
    # SDF analysis
    print(f"\n[SDF Analysis]")
    print(f"  Minimum SDF:")
    print(f"    JAX:   {jax_stats['min_sdf']:.4f}")
    print(f"    NumPy: {numpy_stats['min_sdf']:.4f}")
    print(f"    Diff:  {abs(jax_stats['min_sdf'] - numpy_stats['min_sdf']):.4f}")
    
    print(f"  Mean SDF:")
    print(f"    JAX:   {jax_stats['mean_sdf']:.4f}")
    print(f"    NumPy: {numpy_stats['mean_sdf']:.4f}")
    print(f"    Diff:  {abs(jax_stats['mean_sdf'] - numpy_stats['mean_sdf']):.4f}")
    
    print(f"  Final SDF:")
    print(f"    JAX:   {jax_stats['final_sdf']:.4f}")
    print(f"    NumPy: {numpy_stats['final_sdf']:.4f}")
    print(f"    Diff:  {abs(jax_stats['final_sdf'] - numpy_stats['final_sdf']):.4f}")
    
    # Violations
    print(f"\n[Clearance Violations (SDF < robot_radius=0.05)]")
    print(f"  JAX:   {jax_stats['num_violations']} violations")
    if jax_stats['num_violations'] > 0:
        print(f"    Indices: {jax_stats['violation_indices'][:10]}{'...' if len(jax_stats['violation_indices']) > 10 else ''}")
    print(f"  NumPy: {numpy_stats['num_violations']} violations")
    if numpy_stats['num_violations'] > 0:
        print(f"    Indices: {numpy_stats['violation_indices'][:10]}{'...' if len(numpy_stats['violation_indices']) > 10 else ''}")
    
    # Position differences
    if len(jax_stats['positions']) == len(numpy_stats['positions']):
        pos_diff = np.linalg.norm(jax_stats['positions'] - numpy_stats['positions'], axis=1)
        print(f"\n[Position Differences]")
        print(f"  Max difference:  {np.max(pos_diff):.6f}")
        print(f"  Mean difference: {np.mean(pos_diff):.6f}")
        print(f"  Std difference:  {np.std(pos_diff):.6f}")
        
        # Find points with largest differences
        max_diff_idx = np.argmax(pos_diff)
        print(f"\n  Largest difference at index {max_diff_idx}:")
        print(f"    JAX:   {jax_stats['positions'][max_diff_idx]}")
        print(f"    NumPy: {numpy_stats['positions'][max_diff_idx]}")
        print(f"    Diff:  {pos_diff[max_diff_idx]:.6f}")
        print(f"    SDF (JAX):   {jax_stats['sdf_values'][max_diff_idx]:.4f}")
        print(f"    SDF (NumPy): {numpy_stats['sdf_values'][max_diff_idx]:.4f}")
    
    # SDF value differences
    if len(jax_stats['sdf_values']) == len(numpy_stats['sdf_values']):
        sdf_diff = np.abs(jax_stats['sdf_values'] - numpy_stats['sdf_values'])
        print(f"\n[SDF Value Differences]")
        print(f"  Max difference:  {np.max(sdf_diff):.6f}")
        print(f"  Mean difference: {np.mean(sdf_diff):.6f}")
        print(f"  Std difference:  {np.std(sdf_diff):.6f}")
        
        # Find points with largest SDF differences
        max_sdf_diff_idx = np.argmax(sdf_diff)
        print(f"\n  Largest SDF difference at index {max_sdf_diff_idx}:")
        print(f"    Position (JAX):   {jax_stats['positions'][max_sdf_diff_idx]}")
        print(f"    Position (NumPy): {numpy_stats['positions'][max_sdf_diff_idx]}")
        print(f"    SDF (JAX):   {jax_stats['sdf_values'][max_sdf_diff_idx]:.4f}")
        print(f"    SDF (NumPy): {numpy_stats['sdf_values'][max_sdf_diff_idx]:.4f}")
        print(f"    Diff:  {sdf_diff[max_sdf_diff_idx]:.6f}")

def test_cfs_performance():
    """Test CFS performance with optimizations and compare numerical differences."""
    print("=" * 80)
    print("Testing CFS Optimization")
    print("=" * 80)
    
    # Store results for comparison
    results = {}
    
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
        
        # Build constraint pipeline (new architecture)
        constraint_config = {
            "cfs": {
                "enabled": True,
                "use_trajectory_qp": False,
                "max_iterations": 5,
                "constraint_margin": 0.25,
            },
            "schedule": {
                "enabled": True,
                "type": "soft_to_hard",
                "hard_clearance_start": 0.5,
                "hard_clearance_end": 0.1,
                "rho_start": 0.1,
                "rho_end": 10.0,
            },
        }
        method_params = {
            "cfs_qp_mode": "state_traj",
            "cfs_use_trajectory_qp": False,
        }
        pipeline = create_constraint_pipeline(
            obstacles=obstacles,
            level=6,
            env=env,
            config=constraint_config,
            backend_name=backend_name,
            obstacle_config=obstacle_config,
            method_params=method_params,
        )
        if pipeline is None:
            print("  ✗ Pipeline not created (no constraints).")
            continue

        # Build a nominal straight-line trajectory to project.
        num_steps = int(env.horizon)
        t = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
        start = np.asarray(start_pos, dtype=np.float32).reshape(1, -1)
        goal = np.asarray(target_pos, dtype=np.float32).reshape(1, -1)
        nominal_states = start + t[:, None] * (goal - start)
        nominal_actions = np.diff(nominal_states, axis=0) / float(env.dt)
        nominal_actions = np.clip(nominal_actions, -env.control_limit, env.control_limit)
        nominal = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in nominal_states],
            actions=[np.asarray(a, dtype=np.float32) for a in nominal_actions],
            info={},
        )

        # Apply constraint pipeline
        print(f"\n  Applying constraint pipeline...")
        start_time = time.time()
        try:
            repaired, pipe_info = pipeline.apply(
                nominal=nominal,
                ref=nominal,
                state=ScheduleState(k=0, K=1),
            )
            planning_time = time.time() - start_time
            print(f"  ✓ Pipeline applied in {planning_time:.2f} seconds")

            states = repaired.states
            actions = repaired.actions
            if len(states) > 0:
                stats = analyze_trajectory(obstacles, states, actions, backend_name)
                results[backend_name] = {
                    'stats': stats,
                    'planning_time': planning_time,
                    'pipeline_info': pipe_info,
                }

                # Print summary
                print(f"  Final distance to target: {stats['final_distance_to_target']:.4f}")
                print(f"  Minimum SDF: {stats['min_sdf']:.4f}")
                print(f"  Mean SDF: {stats['mean_sdf']:.4f}")
                print(f"  Trajectory length: {stats['trajectory_length']:.4f}")

                if stats['num_violations'] > 0:
                    print(f"  ⚠ Warning: {stats['num_violations']} states violate clearance (SDF < 0.05)")
                    print(f"    Violation indices: {stats['violation_indices'][:10]}{'...' if len(stats['violation_indices']) > 10 else ''}")
                else:
                    print(f"  ✓ Trajectory maintains safe clearance")
        except Exception as e:
            print(f"  ✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results if both backends succeeded
    if 'jax' in results and 'numpy' in results:
        compare_results(results['jax']['stats'], results['numpy']['stats'])

if __name__ == "__main__":
    test_cfs_performance()

