"""
Test script to compare CFS pipeline results using jaxopt.OSQP vs cvxopt.

This script tests the numerical differences between jaxopt.OSQP and cvxopt QP solvers
when used in the new HighPerformanceConstraintPipeline architecture (CFSConvexifier + Operator),
to ensure they produce similar results to the legacy CFSProjection.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

from genedynamics.core.constraints.core import HighPerformanceConstraintPipeline, PipelineConfig
from genedynamics.core.constraints.convexify import CFSConvexifier
from genedynamics.core.constraints.operators.qp import TrajQPFilter
from genedynamics.core.constraints.schedulers import CosineAnnealScheduler
# Import solvers to ensure they are registered
from genedynamics.core.constraints.solvers import JAXOPTOsqpSolver, OSQPSolver, CVXOPTSolver
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.envs.obstacles.base import ObstacleManager
from genedynamics.envs.obstacles.convex import BoxObstacle
from genedynamics.core.types import Trajectory
from genedynamics.core.constraints.core.types import ScheduleState


def test_cfs_pipeline_with_solver(solver_name: str, use_jit: bool = False):
    """
    Test CFS pipeline with a specific solver.
    
    Args:
        solver_name: 'jaxopt' or 'cvxopt' (jaxopt uses OSQP solver)
        use_jit: Whether to use JIT compilation (only for jaxopt)
    
    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Testing CFS Pipeline with {solver_name.upper()} (JIT={use_jit})")
    print(f"{'='*80}")
    
    # Set backend
    if solver_name == 'jaxopt' or solver_name == 'qpax':
        backend_name = "jax"
        RuntimeBackendManager.set_backend("jax", device="cpu")
    else:
        backend_name = "numpy"
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
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        backend=backend_name,
        use_jit=use_jit,
        use_batch=True,
        cache_constraints=True,
        cache_params=True,
        verbose=True,
    )
    
    # Create scheduler (fixed margin for comparison)
    margin_value = 0.1
    scheduler = CosineAnnealScheduler(
        margin_start=margin_value,
        margin_end=margin_value,
        rho_start=1.0,
        rho_end=1.0,
    )
    
    # Create pipeline with new architecture
    # For CFS, we can use either:
    # 1. 'projection' operator: iterative projection (no QP solver, works for both backends)
    # 2. 'traj_qp' operator: trajectory-level QP (allows comparing jaxopt vs cvxopt)
    # 
    # For CFS constraints comparison, we use 'traj_qp' operator to compare QP solvers
    # TrajQPFilter now supports state-space constraints by extracting position constraints
    # and mapping them to action space, then updating states from optimized actions
    operator_name = "traj_qp"
    # Map solver names to TrajQPFilter's expected backend names
    # TrajQPFilter expects: "jax" for jaxopt/qpax, "numpy" for cvxopt
    if solver_name == 'jaxopt' or solver_name == 'qpax':
        traj_qp_backend = "jax"
    else:
        traj_qp_backend = "numpy"
    operator_kwargs = {
        'use_slack': True,
        'solver_backend': traj_qp_backend,  # 'jax' or 'numpy'
        'max_iterations': 30,  # Match legacy: max_iterations=30
        'convergence_tol': 1e-6,  # Match legacy: convergence_tol=1e-6
        'smoothness_weight': 1.0,  # Match legacy: smoothness_weight=1.0
    }
    
    pipeline = HighPerformanceConstraintPipeline(
        convexifier_name="cfs",
        operator_name=operator_name,
        scheduler_name="cosine_anneal",
        config=pipeline_config,
        obstacles=obstacles,
        position_extractor=None,  # Use default
        max_constraints_per_point=8,
        constraint_margin=0.25,
        robot_radius=0.05,
        use_jit=use_jit,  # For CFSJAXConvexifier
        **operator_kwargs,  # Pass operator-specific kwargs
    )
    
    # Override scheduler with fixed margin
    pipeline.scheduler = scheduler
    
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
    
    # Create schedule state (fixed step for comparison)
    schedule_state = ScheduleState(k=0, K=100)
    
    # Run CFS pipeline projection
    start_time = time.time()
    projected_trajectory, info = pipeline.apply(trajectory, trajectory, schedule_state)
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
        'operator_info': info,
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
    print("Detailed Comparison: JAXOPT.OSQP vs CVXOPT (New Pipeline)")
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
    if jaxopt_results['elapsed_time'] > 0:
        speedup = cvxopt_results['elapsed_time'] / jaxopt_results['elapsed_time']
        print(f"    Speedup: {speedup:.2f}x")
    
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
            center = obs.center[:2]
            half_ext = obs.half_extents[:2]
            rect = plt.Rectangle(
                (center[0] - half_ext[0], center[1] - half_ext[1]),
                2 * half_ext[0], 2 * half_ext[1],
                fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax1.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1)
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
                fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax2.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1)
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
    output_path = output_dir / 'cfs_pipeline_jaxopt_vs_cvxopt_comparison.png'
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
                fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
        elif hasattr(obs, 'center') and hasattr(obs, 'radius'):
            circle = plt.Circle(obs.center[:2], obs.radius, fill=True, facecolor='red', alpha=0.3, edgecolor='black', linewidth=1)
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
    ax.set_title('JAXOPT.OSQP vs CVXOPT Trajectory Comparison (New Pipeline)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    output_path = output_dir / 'cfs_pipeline_jaxopt_vs_cvxopt_difference.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved difference plot to: {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("="*80)
    print("CFS Pipeline JAXOPT.OSQP vs CVXOPT Comparison Test")
    print("Using new architecture: CFSConvexifier + TrajQPFilter")
    print("="*80)
    
    # Test jaxopt.OSQP (non-JIT mode)
    jaxopt_results = test_cfs_pipeline_with_solver('jaxopt', use_jit=False)
    
    # Test cvxopt (NumPy backend)
    cvxopt_results = test_cfs_pipeline_with_solver('cvxopt', use_jit=False)
    
    # Compare results
    comparison = compare_results(jaxopt_results, cvxopt_results)
    
    # Visualize - recreate obstacles for visualization
    output_dir = Path("test_results/cfs_pipeline_comparison")
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

