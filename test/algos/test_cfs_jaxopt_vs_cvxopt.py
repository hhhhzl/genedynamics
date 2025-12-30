"""
Test script to compare CFS projection results using jaxopt.OSQP vs cvxopt.

This script tests the numerical differences between jaxopt.OSQP and cvxopt QP solvers
when used for CFS projection, to ensure they produce similar results.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

from enerdynamics.core.constraints import CFSProjection
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import BoxObstacle
from enerdynamics.core.types import Trajectory


def test_cfs_with_solver(solver_name: str, use_jit: bool = False):
    """
    Test CFS projection with a specific solver.
    
    Args:
        solver_name: 'jaxopt' or 'cvxopt' (jaxopt uses OSQP solver)
        use_jit: Whether to use JIT compilation (only for jaxopt)
    
    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Testing CFS with {solver_name.upper()} (JIT={use_jit})")
    print(f"{'='*80}")
    
    # Set backend
    if solver_name == 'jaxopt' or solver_name == 'qpax':  # Support both names for backward compatibility
        RuntimeBackendManager.set_backend("jax", device="cpu")
    else:
        RuntimeBackendManager.set_backend("numpy", device="cpu")
    
    # Create simple obstacles between start and target
    start_pos = np.array([-1.0, -2.0], dtype=np.float32)
    target_pos = np.array([0.0, 0.0], dtype=np.float32)
    
    # Create two box obstacles in the path between start and target
    obstacles = ObstacleManager()
    
    # Obstacle 1: near the middle of the path
    obs1_center = np.array([0.5, 0.5], dtype=np.float32)
    obs1_half_extents = np.array([0.2, 0.2], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs1_center, half_extents=obs1_half_extents, name="obstacle1"))
    
    # Obstacle 2: closer to target
    obs2_center = np.array([-0.3, -0.3], dtype=np.float32)
    obs2_half_extents = np.array([0.15, 0.15], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs2_center, half_extents=obs2_half_extents, name="obstacle2"))
    
    # Build SDF texture for JAX acceleration
    obstacles.build_sdf_texture_2d(
        x_min=-1.5,
        x_max=1.5,
        y_min=-2.5,
        y_max=0.5,
        res=0.01,
        force_rebuild=True,
    )
    
    # Create CFS projection with fixed clearance schedule
    clearance_value = 0.1
    clearance_schedule = lambda step, total_steps: clearance_value
    
    if solver_name == 'jaxopt' or solver_name == 'qpax':  # Support both names
        # Force non-JIT mode for jaxopt.OSQP to compare with cvxopt
        # We need to ensure obstacles don't all have jax_sdf to force non-JIT mode
        # Or we can manually set use_jit after creation
        feasibility_op = CFSProjection(
            obstacles=obstacles,
            clearance_schedule=clearance_schedule,
            robot_radius=0.05,
            max_iterations=30,
            smoothness_weight=1.0,
            convergence_tol=1e-6,
            max_constraints_per_point=8,
            constraint_margin=0.25,
            use_trajectory_qp=True,
            force_python_backend=False,  # Use JAX backend
            fix_initial_state=True,
            monitor_objective=True,
        )
        # Force non-JIT mode by setting it after backend creation
        if hasattr(feasibility_op, '_backend_impl') and feasibility_op._backend_impl is not None:
            feasibility_op._backend_impl.use_jit = use_jit
            print(f"[CFS] Forced use_jit={use_jit} for comparison (using jaxopt.OSQP)")
    else:
        # NumPy backend uses cvxopt
        feasibility_op = CFSProjection(
            obstacles=obstacles,
            clearance_schedule=clearance_schedule,
            smoothness_weight=1.0,
            robot_radius=0.05,
            max_iterations=30,
            convergence_tol=1e-6,
            max_constraints_per_point=8,
            constraint_margin=0.25,
            use_trajectory_qp=True,
            force_python_backend=True,  # Use NumPy backend
            fix_initial_state=True,
            monitor_objective=True,
        )
    
    # Create a test trajectory (straight line through obstacles)
    num_points = 50
    start = np.array([-1.0, -1.0], dtype=np.float32)
    goal = np.array([1.0, 1.0], dtype=np.float32)
    
    # Generate initial trajectory (straight line)
    t = np.linspace(0, 1, num_points)
    positions = start[None, :] + t[:, None] * (goal - start)[None, :]
    
    # Create trajectory object
    states = [np.concatenate([pos, np.zeros(2)]) for pos in positions]  # Add velocity
    # Actions should be one less than states (T states, T-1 actions)
    actions = [np.zeros(2) for _ in range(len(states) - 1)]  # Dummy actions
    trajectory = Trajectory(states=states, actions=actions)
    
    # Run CFS projection
    start_time = time.time()
    projected_trajectory = feasibility_op.project(trajectory)
    elapsed_time = time.time() - start_time
    
    # Extract positions
    projected_positions = np.array([s[:2] for s in projected_trajectory.states], dtype=np.float32)
    
    # Compute statistics
    sdf_values = []
    for pos in projected_positions:
        sdf = obstacles.sdf(pos)
        sdf_values.append(float(sdf) if np.isscalar(sdf) else float(sdf[0]))
    sdf_values = np.array(sdf_values)
    
    # Check violations
    robot_radius = 0.05
    violations = sdf_values < robot_radius
    num_violations = int(np.sum(violations))
    
    # Trajectory length
    if len(projected_positions) > 1:
        diffs = np.diff(projected_positions, axis=0)
        trajectory_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        trajectory_length = 0.0
    
    results = {
        'solver': solver_name,
        'use_jit': use_jit,
        'positions': projected_positions,
        'sdf_values': sdf_values,
        'min_sdf': float(np.min(sdf_values)),
        'mean_sdf': float(np.mean(sdf_values)),
        'max_sdf': float(np.max(sdf_values)),
        'num_violations': num_violations,
        'violation_indices': np.where(violations)[0].tolist(),
        'trajectory_length': trajectory_length,
        'elapsed_time': elapsed_time,
        'initial_positions': positions,
    }
    
    print(f"  ✓ Projection completed in {elapsed_time:.4f} seconds")
    print(f"  ✓ Minimum SDF: {results['min_sdf']:.6f}")
    print(f"  ✓ Mean SDF: {results['mean_sdf']:.6f}")
    print(f"  ✓ Trajectory length: {results['trajectory_length']:.6f}")
    print(f"  ✓ Violations (SDF < {robot_radius}): {num_violations}")
    if num_violations > 0:
        print(f"    Violation indices: {results['violation_indices']}")
    
    return results


def compare_results(jaxopt_results: dict, cvxopt_results: dict):
    """Compare results from jaxopt.OSQP and cvxopt."""
    print(f"\n{'='*80}")
    print("Detailed Comparison: JAXOPT.OSQP vs CVXOPT")
    print(f"{'='*80}")
    
    jaxopt_pos = jaxopt_results['positions']
    cvxopt_pos = cvxopt_results['positions']
    
    # Position differences
    pos_diffs = np.linalg.norm(jaxopt_pos - cvxopt_pos, axis=1)
    max_pos_diff = float(np.max(pos_diffs))
    mean_pos_diff = float(np.mean(pos_diffs))
    std_pos_diff = float(np.std(pos_diffs))
    
    print(f"\n[Position Differences]")
    print(f"  Max difference:  {max_pos_diff:.6f}")
    print(f"  Mean difference: {mean_pos_diff:.6f}")
    print(f"  Std difference:  {std_pos_diff:.6f}")
    
    # Find largest difference
    max_idx = int(np.argmax(pos_diffs))
    print(f"\n  Largest difference at index {max_idx}:")
    print(f"    JAXOPT: {jaxopt_pos[max_idx]}")
    print(f"    CVXOPT: {cvxopt_pos[max_idx]}")
    print(f"    Diff:   {pos_diffs[max_idx]:.6f}")
    
    # SDF differences
    sdf_diffs = np.abs(jaxopt_results['sdf_values'] - cvxopt_results['sdf_values'])
    max_sdf_diff = float(np.max(sdf_diffs))
    mean_sdf_diff = float(np.mean(sdf_diffs))
    
    print(f"\n[SDF Value Differences]")
    print(f"  Max difference:  {max_sdf_diff:.6f}")
    print(f"  Mean difference: {mean_sdf_diff:.6f}")
    
    # Statistics comparison
    print(f"\n[Statistics Comparison]")
    print(f"  Trajectory length:")
    print(f"    JAXOPT: {jaxopt_results['trajectory_length']:.6f}")
    print(f"    CVXOPT: {cvxopt_results['trajectory_length']:.6f}")
    print(f"    Diff:   {abs(jaxopt_results['trajectory_length'] - cvxopt_results['trajectory_length']):.6f}")
    
    print(f"  Minimum SDF:")
    print(f"    JAXOPT: {jaxopt_results['min_sdf']:.6f}")
    print(f"    CVXOPT: {cvxopt_results['min_sdf']:.6f}")
    print(f"    Diff:   {abs(jaxopt_results['min_sdf'] - cvxopt_results['min_sdf']):.6f}")
    
    print(f"  Mean SDF:")
    print(f"    JAXOPT: {jaxopt_results['mean_sdf']:.6f}")
    print(f"    CVXOPT: {cvxopt_results['mean_sdf']:.6f}")
    print(f"    Diff:   {abs(jaxopt_results['mean_sdf'] - cvxopt_results['mean_sdf']):.6f}")
    
    print(f"  Violations:")
    print(f"    JAXOPT: {jaxopt_results['num_violations']}")
    print(f"    CVXOPT: {cvxopt_results['num_violations']}")
    
    print(f"  Elapsed time:")
    print(f"    JAXOPT: {jaxopt_results['elapsed_time']:.4f} seconds")
    print(f"    CVXOPT: {cvxopt_results['elapsed_time']:.4f} seconds")
    print(f"    Speedup: {cvxopt_results['elapsed_time'] / jaxopt_results['elapsed_time']:.2f}x")
    
    return {
        'max_pos_diff': max_pos_diff,
        'mean_pos_diff': mean_pos_diff,
        'max_sdf_diff': max_sdf_diff,
        'mean_sdf_diff': mean_sdf_diff,
    }


def visualize_comparison(jaxopt_results: dict, cvxopt_results: dict, obstacles: ObstacleManager, output_dir: Path):
    """Visualize the comparison between jaxopt.OSQP and cvxopt results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: JAXOPT.OSQP result
    ax1 = axes[0]
    jaxopt_pos = jaxopt_results['positions']
    initial_pos = jaxopt_results['initial_positions']
    
    # Plot obstacles
    for obs in obstacles:
        if hasattr(obs, 'center') and hasattr(obs, 'half_extents'):
            # Box obstacle
            center = obs.center[:2]
            half_ext = obs.half_extents[:2]
            rect = plt.Rectangle(
                (center[0] - half_ext[0], center[1] - half_ext[1]),
                2 * half_ext[0], 2 * half_ext[1],
                fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax1.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            # Sphere obstacle
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1)
            ax1.add_patch(circle)
    
    # Plot trajectories
    ax1.plot(initial_pos[:, 0], initial_pos[:, 1], 'b--', alpha=0.5, label='Initial', linewidth=1)
    ax1.plot(jaxopt_pos[:, 0], jaxopt_pos[:, 1], 'g-', label='JAXOPT.OSQP', linewidth=2, marker='o', markersize=3)
    ax1.scatter(jaxopt_pos[0, 0], jaxopt_pos[0, 1], c='blue', s=100, marker='s', label='Start', zorder=5)
    ax1.scatter(jaxopt_pos[-1, 0], jaxopt_pos[-1, 1], c='red', s=100, marker='*', label='Goal', zorder=5)
    
    # Highlight violations
    violations = jaxopt_results['violation_indices']
    if len(violations) > 0:
        ax1.scatter(jaxopt_pos[violations, 0], jaxopt_pos[violations, 1], 
                   c='orange', s=150, marker='x', label=f'Violations ({len(violations)})', zorder=6, linewidths=3)
    
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title(f'JAXOPT.OSQP Result (Time: {jaxopt_results["elapsed_time"]:.4f}s)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    
    # Plot 2: CVXOPT result
    ax2 = axes[1]
    cvxopt_pos = cvxopt_results['positions']
    
    # Plot obstacles (same as above)
    for obs in obstacles:
        if hasattr(obs, 'center') and hasattr(obs, 'half_extents'):
            center = obs.center[:2]
            half_ext = obs.half_extents[:2]
            rect = plt.Rectangle(
                (center[0] - half_ext[0], center[1] - half_ext[1]),
                2 * half_ext[0], 2 * half_ext[1],
                fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax2.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1)
            ax2.add_patch(circle)
    
    # Plot trajectories
    ax2.plot(initial_pos[:, 0], initial_pos[:, 1], 'b--', alpha=0.5, label='Initial', linewidth=1)
    ax2.plot(cvxopt_pos[:, 0], cvxopt_pos[:, 1], 'm-', label='CVXOPT', linewidth=2, marker='o', markersize=3)
    ax2.scatter(cvxopt_pos[0, 0], cvxopt_pos[0, 1], c='blue', s=100, marker='s', label='Start', zorder=5)
    ax2.scatter(cvxopt_pos[-1, 0], cvxopt_pos[-1, 1], c='red', s=100, marker='*', label='Goal', zorder=5)
    
    # Highlight violations
    violations = cvxopt_results['violation_indices']
    if len(violations) > 0:
        ax2.scatter(cvxopt_pos[violations, 0], cvxopt_pos[violations, 1], 
                   c='orange', s=150, marker='x', label=f'Violations ({len(violations)})', zorder=6, linewidths=3)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title(f'CVXOPT Result (Time: {cvxopt_results["elapsed_time"]:.4f}s)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'cfs_jaxopt_vs_cvxopt_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved comparison plot to: {output_path}")
    plt.close()
    
    # Plot difference
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot obstacles
    for obs in obstacles:
        if hasattr(obs, 'center') and hasattr(obs, 'half_extents'):
            center = obs.center[:2]
            half_ext = obs.half_extents[:2]
            rect = plt.Rectangle(
                (center[0] - half_ext[0], center[1] - half_ext[1]),
                2 * half_ext[0], 2 * half_ext[1],
                fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, color='red', alpha=0.3, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
    
    # Plot both trajectories
    ax.plot(jaxopt_pos[:, 0], jaxopt_pos[:, 1], 'g-', label='JAXOPT.OSQP', linewidth=2, alpha=0.7)
    ax.plot(cvxopt_pos[:, 0], cvxopt_pos[:, 1], 'm-', label='CVXOPT', linewidth=2, alpha=0.7)
    
    # Draw lines connecting corresponding points
    pos_diffs = np.linalg.norm(jaxopt_pos - cvxopt_pos, axis=1)
    max_diff_idx = int(np.argmax(pos_diffs))
    for i in range(0, len(jaxopt_pos), 5):  # Every 5th point
        ax.plot([jaxopt_pos[i, 0], cvxopt_pos[i, 0]], 
                [jaxopt_pos[i, 1], cvxopt_pos[i, 1]], 
                'k--', alpha=0.3, linewidth=1)
    
    # Highlight largest difference
    ax.plot([jaxopt_pos[max_diff_idx, 0], cvxopt_pos[max_diff_idx, 0]], 
            [jaxopt_pos[max_diff_idx, 1], cvxopt_pos[max_diff_idx, 1]], 
            'r-', linewidth=3, label=f'Max diff: {pos_diffs[max_diff_idx]:.6f}')
    ax.scatter([jaxopt_pos[max_diff_idx, 0], cvxopt_pos[max_diff_idx, 0]], 
               [jaxopt_pos[max_diff_idx, 1], cvxopt_pos[max_diff_idx, 1]], 
               c='red', s=200, marker='o', zorder=5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('JAXOPT.OSQP vs CVXOPT Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'cfs_jaxopt_vs_cvxopt_difference.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved difference plot to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("="*80)
    print("CFS JAXOPT.OSQP vs CVXOPT Comparison Test")
    print("="*80)
    
    # Test jaxopt.OSQP (non-JIT mode)
    jaxopt_results = test_cfs_with_solver('jaxopt', use_jit=False)
    
    # Test cvxopt (NumPy backend)
    cvxopt_results = test_cfs_with_solver('cvxopt', use_jit=False)
    
    # Compare results
    comparison = compare_results(jaxopt_results, cvxopt_results)
    
    # Visualize - recreate obstacles for visualization
    output_dir = Path("test_results/cfs_comparison")
    obstacles = ObstacleManager()
    obs1_center = np.array([0.5, 0.5], dtype=np.float32)
    obs1_half_extents = np.array([0.2, 0.2], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs1_center, half_extents=obs1_half_extents, name="obstacle1"))
    obs2_center = np.array([-0.3, -0.3], dtype=np.float32)
    obs2_half_extents = np.array([0.15, 0.15], dtype=np.float32)
    obstacles.add(BoxObstacle(center=obs2_center, half_extents=obs2_half_extents, name="obstacle2"))
    visualize_comparison(jaxopt_results, cvxopt_results, obstacles, output_dir)
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")
    
    # Summary
    print(f"\n[Summary]")
    print(f"  Max position difference: {comparison['max_pos_diff']:.6f}")
    print(f"  Mean position difference: {comparison['mean_pos_diff']:.6f}")
    print(f"  Max SDF difference: {comparison['max_sdf_diff']:.6f}")
    print(f"  Mean SDF difference: {comparison['mean_sdf_diff']:.6f}")
    
    if comparison['max_pos_diff'] < 1e-3:
        print(f"\n  ✓ Results are very similar (max diff < 1e-3)")
    elif comparison['max_pos_diff'] < 1e-2:
        print(f"\n  ⚠ Results are similar but have some differences (max diff < 1e-2)")
    else:
        print(f"\n  ✗ Results differ significantly (max diff >= 1e-2)")


if __name__ == "__main__":
    main()

