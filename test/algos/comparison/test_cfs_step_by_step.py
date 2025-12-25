"""
Step-by-step comparison of CFS projection components.

This test breaks down CFS projection into individual steps and compares
each step between NumPy and JAX versions to identify where differences occur.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle
from enerdynamics.core.constraints.projections.cfs import CFSProjection


def create_test_obstacles():
    """Create test obstacles."""
    obstacles = ObstacleManager()
    obstacles.add(SphereObstacle(center=np.array([0.0, 0.0]), radius=0.3))
    obstacles.add(SphereObstacle(center=np.array([0.5, 0.5]), radius=0.2))
    return obstacles


def test_step1_sdf_computation():
    """Step 1: Compare SDF computation."""
    print("=" * 80)
    print("Step 1: SDF Computation Comparison")
    print("=" * 80)
    
    obstacles = create_test_obstacles()
    
    # Build SDF texture for JAX
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    # Test points
    test_points = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.5, 0.5],
        [-0.3, 0.3],
    ], dtype=np.float32)
    
    print(f"Testing {len(test_points)} points")
    
    # NumPy version: direct SDF computation
    print("\n--- NumPy SDF (direct) ---")
    sdf_numpy = obstacles.sdf(test_points)
    print(f"SDF values: {sdf_numpy}")
    
    # JAX version: SDF texture
    if JAX_AVAILABLE:
        print("\n--- JAX SDF (texture) ---")
        sdf_jax, _ = obstacles.sample_sdf_and_grad_2d(test_points, backend="jax")
        sdf_jax = np.asarray(sdf_jax)
        print(f"SDF values: {sdf_jax}")
        
        # Compare
        diff = np.abs(sdf_numpy - sdf_jax)
        print(f"\nDifference:")
        print(f"  Max: {np.max(diff):.6f}")
        print(f"  Mean: {np.mean(diff):.6f}")
        print(f"  Per-point: {diff}")
        
        if np.max(diff) > 0.01:
            print("WARNING: Significant SDF difference detected!")
            print("  This may be due to SDF texture approximation.")
    
    print("\n" + "=" * 80)


def test_step2_gradient_computation():
    """Step 2: Compare gradient computation."""
    print("\n" + "=" * 80)
    print("Step 2: Gradient Computation Comparison")
    print("=" * 80)
    
    obstacles = create_test_obstacles()
    
    # Build SDF texture
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    # Test points
    test_points = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.5, 0.5],
    ], dtype=np.float32)
    
    print(f"Testing {len(test_points)} points")
    
    # NumPy version: finite difference or obstacle.gradient
    print("\n--- NumPy Gradients ---")
    obstacles_list = list(obstacles)
    if len(obstacles_list) == 0:
        print("No obstacles available")
        return
    
    grads_numpy = []
    for point in test_points:
        # Get gradient for first obstacle
        obs = obstacles_list[0]
        if hasattr(obs, "gradient"):
            try:
                grad = obs.gradient(point)
            except Exception:
                # Use finite difference
                grad = CFSProjection._finite_difference_gradient(obs, point)
        else:
            grad = CFSProjection._finite_difference_gradient(obs, point)
        grads_numpy.append(grad)
    grads_numpy = np.stack(grads_numpy, axis=0)
    print(f"Gradients shape: {grads_numpy.shape}")
    print(f"Gradients:\n{grads_numpy}")
    
    # JAX version: SDF texture gradient
    if JAX_AVAILABLE:
        print("\n--- JAX Gradients (texture) ---")
        _, grads_jax = obstacles.sample_sdf_and_grad_2d(test_points, backend="jax")
        grads_jax = np.asarray(grads_jax)
        print(f"Gradients shape: {grads_jax.shape}")
        print(f"Gradients:\n{grads_jax}")
        
        # Compare
        diff = np.abs(grads_numpy - grads_jax)
        print(f"\nDifference:")
        print(f"  Max: {np.max(diff):.6f}")
        print(f"  Mean: {np.mean(diff):.6f}")
        print(f"  Per-point max: {np.max(diff, axis=1)}")
        
        # Normalize and compare directions
        grads_numpy_norm = grads_numpy / (np.linalg.norm(grads_numpy, axis=1, keepdims=True) + 1e-8)
        grads_jax_norm = grads_jax / (np.linalg.norm(grads_jax, axis=1, keepdims=True) + 1e-8)
        dot_products = np.sum(grads_numpy_norm * grads_jax_norm, axis=1)
        print(f"  Direction similarity (dot product): {dot_products}")
        
        if np.min(dot_products) < 0.95:
            print("WARNING: Gradient directions differ significantly!")
    
    print("\n" + "=" * 80)


def test_step3_constraint_building():
    """Step 3: Compare constraint building."""
    print("\n" + "=" * 80)
    print("Step 3: Constraint Building Comparison")
    print("=" * 80)
    
    obstacles = create_test_obstacles()
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    cfs = CFSProjection(obstacles=obstacles)
    
    # Test point
    x_ref = np.array([0.1, 0.1], dtype=np.float32)
    clearance = 0.1
    
    print(f"Reference point: {x_ref}")
    print(f"Clearance: {clearance}")
    
    # Get SDF values for each obstacle individually
    obstacles_list = list(obstacles)
    sdf_vals = np.array([obs.sdf(x_ref) for obs in obstacles_list], dtype=np.float32)
    print(f"SDF values for each obstacle: {sdf_vals}")
    
    # NumPy version: build constraints
    obstacles_list = list(obstacles)
    threshold = clearance + cfs.constraint_margin
    cand_mask = sdf_vals < threshold
    cand_indices = np.where(cand_mask)[0]
    
    if len(cand_indices) == 0:
        k = min(cfs.max_constraints_per_point, len(obstacles_list))
        cand_indices = np.argsort(sdf_vals)[:k]
    else:
        k = min(cfs.max_constraints_per_point, len(cand_indices))
        cand_indices = cand_indices[np.argsort(sdf_vals[cand_indices])[:k]]
    
    print(f"\nSelected {len(cand_indices)} candidate obstacles: {cand_indices}")
    
    # Build constraints using the internal method
    # We need to pass d0_all which is the SDF values for all obstacles at x_ref
    d0_all = sdf_vals
    A_numpy, b_numpy = cfs._build_linearized_halfspaces_from_candidates(
        x_ref, clearance, obstacles_list, cand_indices, d0_all
    )
    
    print(f"\n--- NumPy Constraints ---")
    print(f"A shape: {A_numpy.shape}")
    print(f"b shape: {b_numpy.shape}")
    print(f"A:\n{A_numpy}")
    print(f"b: {b_numpy}")
    
    # JAX version: would build similar constraints
    # This is mainly to verify the logic is the same
    if JAX_AVAILABLE:
        print(f"\n--- JAX Constraints (should be similar) ---")
        # The constraint building logic should be the same
        # The difference would be in numerical precision
        print("Constraint building logic is the same, differences come from:")
        print("  1. SDF computation (texture vs direct)")
        print("  2. Gradient computation (texture vs finite difference)")
        print("  3. Numerical precision (float32 vs float64)")
    
    print("\n" + "=" * 80)


def test_step4_qp_solving():
    """Step 4: Compare QP solving (already covered in test_qp_solver_comparison)."""
    print("\n" + "=" * 80)
    print("Step 4: QP Solving Comparison")
    print("=" * 80)
    print("See test_qp_solver_comparison.py for detailed QP solver comparison.")
    print("=" * 80)


def test_step5_iteration_loop():
    """Step 5: Compare iteration behavior."""
    print("\n" + "=" * 80)
    print("Step 5: Iteration Loop Comparison")
    print("=" * 80)
    
    obstacles = create_test_obstacles()
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    cfs = CFSProjection(
        obstacles=obstacles,
        max_iterations=10,
        convergence_tol=1e-6,
    )
    
    # Test positions
    positions = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.5, 0.5],
    ], dtype=np.float32)
    clearance = 0.1
    
    print(f"Testing {len(positions)} positions")
    print(f"Max iterations: {cfs.max_iterations}")
    print(f"Convergence tolerance: {cfs.convergence_tol}")
    
    # NumPy version: can break early
    print("\n--- NumPy Version (can break early) ---")
    result_numpy = cfs._project_cfs_batch(positions, clearance, step=0)
    print(f"Result shape: {result_numpy.shape}")
    print(f"Result:\n{result_numpy}")
    
    # Check how many iterations were actually used
    # (We can't easily measure this without modifying code, but we know it can break early)
    print("Note: NumPy version can break early when all points are feasible")
    
    # JAX version: runs all iterations
    if JAX_AVAILABLE:
        print("\n--- JAX Version (runs all iterations) ---")
        jax_projector = cfs.make_jax_projector()
        if jax_projector is not None:
            positions_jax = jnp.asarray(positions, dtype=jnp.float32)
            clearance_jax = jnp.asarray(clearance, dtype=jnp.float32)
            result_jax = jax_projector(positions_jax, clearance_jax)
            result_jax = np.asarray(result_jax)
            print(f"Result shape: {result_jax.shape}")
            print(f"Result:\n{result_jax}")
            
            diff = np.abs(result_numpy - result_jax)
            print(f"\nDifference:")
            print(f"  Max: {np.max(diff):.6f}")
            print(f"  Mean: {np.mean(diff):.6f}")
            
            print("\nNote: JAX version always runs max_iterations iterations")
            print("  This may cause over-iteration or different convergence behavior")
    
    print("\n" + "=" * 80)


def test_step_by_step_comparison():
    """Run all step-by-step comparisons."""
    print("=" * 80)
    print("CFS Projection Step-by-Step Comparison")
    print("=" * 80)
    
    test_step1_sdf_computation()
    test_step2_gradient_computation()
    test_step3_constraint_building()
    test_step4_qp_solving()
    test_step5_iteration_loop()
    
    print("\n" + "=" * 80)
    print("Summary of Potential Differences:")
    print("=" * 80)
    print("1. SDF Computation:")
    print("   - NumPy: Direct obstacle.sdf() (exact)")
    print("   - JAX: SDF texture (approximation, depends on resolution)")
    print()
    print("2. Gradient Computation:")
    print("   - NumPy: obstacle.gradient() or finite difference")
    print("   - JAX: SDF texture gradient (approximation)")
    print()
    print("3. QP Solving:")
    print("   - NumPy: Full enumeration (optimal)")
    print("   - JAX: JAXOpt or simplified (may be suboptimal)")
    print()
    print("4. Iteration Control:")
    print("   - NumPy: Can break early")
    print("   - JAX: Always runs max_iterations")
    print()
    print("5. Trajectory-level QP:")
    print("   - NumPy: Supported (cvxopt)")
    print("   - JAX: Not implemented")
    print("=" * 80)


if __name__ == "__main__":
    test_step_by_step_comparison()

