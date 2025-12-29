"""
Diagnostic script to identify differences between trajectory QP and pointwise projection.

This script compares:
1. NumPy with trajectory QP (use_trajectory_qp=True) vs pointwise (use_trajectory_qp=False)
2. JAX with pointwise projection (JAX always uses pointwise, ignores use_trajectory_qp)
3. NumPy trajectory QP vs JAX pointwise (actual experiment scenario)

This helps identify if trajectory QP is the main source of differences.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import cvxopt
    from cvxopt import matrix, spmatrix, solvers
    print(f"✓ cvxopt available: {cvxopt is not None}")
    print(f"✓ cvx_solvers available: {solvers is not None}")
except Exception as e:
    print(f"✗ cvxopt not available: {e}")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.legacy.projections.cfs import CFSProjection
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
from enerdynamics.core.types import Trajectory


def create_test_obstacles():
    """Create obstacles similar to experiment setup."""
    obstacles = ObstacleManager()
    # Add various obstacles
    obstacles.add(SphereObstacle(center=np.array([0.0, 0.0]), radius=0.3))
    obstacles.add(SphereObstacle(center=np.array([0.5, 0.5]), radius=0.2))
    obstacles.add(BoxObstacle(center=np.array([-0.3, 0.3]), half_extents=np.array([0.15, 0.15])))
    obstacles.add(SphereObstacle(center=np.array([-0.5, -0.5]), radius=0.25))
    return obstacles


def create_test_trajectory(horizon=20):
    """Create a test trajectory that violates obstacles (similar to EDOC output)."""
    # Create a trajectory that goes through obstacles
    states = []
    for i in range(horizon + 1):
        # Create a path that violates obstacles
        t = i / horizon
        x = -0.6 + t * 1.1  # From -0.6 to 0.5
        y = -0.6 + t * 1.1  # From -0.6 to 0.5
        # Add some noise to make it more realistic
        state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        states.append(state)
    
    actions = [np.zeros(2, dtype=np.float32) for _ in range(horizon)]
    return Trajectory(states=states, actions=actions)


def compare_projections(cfs_traj, cfs_point, trajectory, clearance, name_prefix=""):
    """Compare two CFS projection configurations."""
    print(f"\n{name_prefix}Comparison:")
    print("=" * 80)
    
    # Check cvxopt availability
    try:
        import cvxopt
        from cvxopt import matrix, spmatrix, solvers
        print(f"\n[DEBUG] cvxopt check:")
        print(f"  cvxopt available: {cvxopt is not None}")
        print(f"  solvers available: {solvers is not None}")
        print(f"  matrix available: {matrix is not None}")
        print(f"  spmatrix available: {spmatrix is not None}")
    except Exception as e:
        print(f"\n[DEBUG] cvxopt check: NOT AVAILABLE - {e}")
    
    # NumPy trajectory QP
    print("\n--- NumPy Trajectory QP (use_trajectory_qp=True) ---")
    result_traj = cfs_traj.project(trajectory, step=0, total_steps=100)
    
    # Check debug info
    if hasattr(cfs_traj, '_debug_trajectory_qp'):
        print("\n[DEBUG] Trajectory QP availability:")
        for info in cfs_traj._debug_trajectory_qp:
            print(f"  {info}")
    
    if hasattr(cfs_traj, '_debug_trajectory_qp_attempts'):
        print("\n[DEBUG] Trajectory QP attempts:")
        for attempt in cfs_traj._debug_trajectory_qp_attempts:
            print(f"  Iteration {attempt.get('iteration', '?')}: {attempt.get('status', 'unknown')}")
            if attempt.get('status') == 'failed':
                print(f"    Error: {attempt.get('error_type', '?')}: {attempt.get('error', '?')}")
                print(f"    n_vars={attempt.get('N', '?') * 2}, n_cons={attempt.get('n_cons', '?')}")
    
    positions_traj = np.stack([s[:2] for s in result_traj.states], axis=0)
    
    # NumPy pointwise
    print("--- NumPy Pointwise (use_trajectory_qp=False) ---")
    result_point = cfs_point.project(trajectory, step=0, total_steps=100)
    positions_point = np.stack([s[:2] for s in result_point.states], axis=0)
    
    # Compare
    diff = positions_traj - positions_point
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    per_step_diff = np.max(np.abs(diff), axis=1)
    
    print(f"\nNumPy Trajectory QP vs Pointwise:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Per-step max differences: {per_step_diff[:5]}... (showing first 5)")
    
    # Check smoothness (trajectory QP should be smoother)
    def compute_smoothness(positions):
        """Compute trajectory smoothness (lower is smoother)."""
        if len(positions) < 2:
            return 0.0
        diffs = np.diff(positions, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        return np.std(norms)  # Standard deviation of step sizes
    
    smoothness_traj = compute_smoothness(positions_traj)
    smoothness_point = compute_smoothness(positions_point)
    
    print(f"\nSmoothness (std of step sizes, lower is smoother):")
    print(f"  Trajectory QP: {smoothness_traj:.6f}")
    print(f"  Pointwise: {smoothness_point:.6f}")
    print(f"  Difference: {abs(smoothness_traj - smoothness_point):.6f}")
    
    return {
        'positions_traj': positions_traj,
        'positions_point': positions_point,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'smoothness_traj': smoothness_traj,
        'smoothness_point': smoothness_point,
    }


def compare_jax_vs_numpy(cfs_traj, trajectory, clearance, obstacles):
    """Compare JAX (pointwise) vs NumPy (trajectory QP)."""
    print("\n" + "=" * 80)
    print("JAX vs NumPy Comparison (Actual Experiment Scenario)")
    print("=" * 80)
    print("NumPy: use_trajectory_qp=True -> Trajectory QP")
    print("JAX: Always uses pointwise (ignores use_trajectory_qp)")
    
    # NumPy trajectory QP
    print("\n--- NumPy Trajectory QP ---")
    result_numpy = cfs_traj.project(trajectory, step=0, total_steps=100)
    positions_numpy = np.stack([s[:2] for s in result_numpy.states], axis=0)
    
    # JAX pointwise
    print("--- JAX Pointwise ---")
    jax_projector = cfs_traj.make_jax_projector()
    
    if jax_projector is None:
        print("WARNING: JAX projector not available. Cannot compare.")
        return None
    
    positions_initial = np.stack([s[:2] for s in trajectory.states], axis=0)
    positions_jax = jnp.asarray(positions_initial, dtype=jnp.float32)
    clearance_jax = jnp.asarray(clearance, dtype=jnp.float32)
    
    try:
        projected_positions_jax = jax_projector(positions_jax, clearance_jax)
        positions_jax_result = np.asarray(projected_positions_jax)
        
        # Compare
        diff = positions_numpy - positions_jax_result
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        per_step_diff = np.max(np.abs(diff), axis=1)
        
        print(f"\nNumPy Trajectory QP vs JAX Pointwise:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Per-step max differences: {per_step_diff[:5]}... (showing first 5)")
        
        # Smoothness comparison
        def compute_smoothness(positions):
            if len(positions) < 2:
                return 0.0
            diffs = np.diff(positions, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            return np.std(norms)
        
        smoothness_numpy = compute_smoothness(positions_numpy)
        smoothness_jax = compute_smoothness(positions_jax_result)
        
        print(f"\nSmoothness:")
        print(f"  NumPy Trajectory QP: {smoothness_numpy:.6f}")
        print(f"  JAX Pointwise: {smoothness_jax:.6f}")
        print(f"  Difference: {abs(smoothness_numpy - smoothness_jax):.6f}")
        
        # Check feasibility
        print("\n--- Feasibility Check ---")
        for i in [0, len(positions_numpy)//2, len(positions_numpy)-1]:
            sdf_numpy = obstacles.sdf(positions_numpy[i])
            sdf_jax = obstacles.sdf(positions_jax_result[i])
            print(f"  Point {i}:")
            print(f"    NumPy SDF: {sdf_numpy:.6f}, feasible: {sdf_numpy >= clearance}")
            print(f"    JAX SDF: {sdf_jax:.6f}, feasible: {sdf_jax >= clearance}")
        
        return {
            'positions_numpy': positions_numpy,
            'positions_jax': positions_jax_result,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'smoothness_numpy': smoothness_numpy,
            'smoothness_jax': smoothness_jax,
        }
    except Exception as e:
        print(f"ERROR: JAX projection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run diagnostic comparison."""
    print("=" * 80)
    print("Trajectory QP vs Pointwise Projection Diagnostic")
    print("=" * 80)
    
    # Create test setup
    obstacles = create_test_obstacles()
    
    # Build SDF texture for JAX (if needed)
    print("\nBuilding SDF texture for JAX support...")
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    print("SDF texture built.")
    
    clearance = 0.1
    horizon = 20
    
    # Create trajectory
    trajectory = create_test_trajectory(horizon=horizon)
    print(f"\nTest trajectory: {len(trajectory.states)} states")
    print(f"Initial positions (first 3):")
    for i in range(min(3, len(trajectory.states))):
        pos = trajectory.states[i][:2]
        sdf = obstacles.sdf(pos)
        print(f"  State {i}: {pos}, SDF: {sdf:.6f}, feasible: {sdf >= clearance}")
    
    # Create CFS projections with different configurations
    cfs_trajectory = CFSProjection(
        obstacles=obstacles,
        clearance_schedule=lambda step, total: clearance,
        max_iterations=5,
        convergence_tol=1e-6,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_trajectory_qp=True,  # NumPy will use trajectory QP
    )
    
    cfs_pointwise = CFSProjection(
        obstacles=obstacles,
        clearance_schedule=lambda step, total: clearance,
        max_iterations=5,
        convergence_tol=1e-6,
        max_constraints_per_point=8,
        constraint_margin=0.25,
        use_trajectory_qp=False,  # Both will use pointwise
    )
    
    # Test 1: NumPy trajectory QP vs NumPy pointwise
    print("\n" + "=" * 80)
    print("Test 1: NumPy Trajectory QP vs NumPy Pointwise")
    print("=" * 80)
    result1 = compare_projections(
        cfs_trajectory, cfs_pointwise, trajectory, clearance,
        name_prefix=""
    )
    
    # Test 2: JAX vs NumPy (actual experiment scenario)
    if JAX_AVAILABLE:
        result2 = compare_jax_vs_numpy(
            cfs_trajectory, trajectory, clearance, obstacles
        )
    else:
        print("\nJAX not available, skipping JAX comparison.")
        result2 = None
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. NumPy Trajectory QP vs Pointwise:")
    if result1:
        print(f"   - Max difference: {result1['max_diff']:.6f}")
        print(f"   - Smoothness difference: {abs(result1['smoothness_traj'] - result1['smoothness_point']):.6f}")
        if result1['max_diff'] > 1e-3:
            print("Significant difference detected!")
        else:
            print("Differences are small")
    
    if result2:
        print(f"\n2. NumPy Trajectory QP vs JAX Pointwise (Experiment Scenario):")
        print(f"   - Max difference: {result2['max_diff']:.6f}")
        print(f"   - Smoothness difference: {abs(result2['smoothness_numpy'] - result2['smoothness_jax']):.6f}")
        if result2['max_diff'] > 1e-3:
            print("Significant difference detected!")
            print("   → This is likely the source of differences in your experiments.")
        else:
            print("Differences are small")
    
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)
    print("If Test 1 shows large differences:")
    print("  → Trajectory QP vs pointwise projection produces different results")
    print("  → This is expected: different algorithms")
    print("\nIf Test 2 shows large differences:")
    print("  → JAX (pointwise) vs NumPy (trajectory QP) mismatch")
    print("  → Solution: Set use_trajectory_qp=False in experiments for alignment")
    print("  → Or: Implement trajectory QP in JAX version")
    print("=" * 80)


if __name__ == "__main__":
    main()

