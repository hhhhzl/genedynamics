"""
Test script to compare NumPy vs JAX CFS projection directly.

This test isolates CFS projection differences by comparing the projection
results on the same input trajectory.
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

from enerdynamics.core.constraints.projections.cfs import CFSProjection
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle
from enerdynamics.core.types import Trajectory


def create_test_obstacles():
    """Create a simple obstacle configuration for testing."""
    obstacles = ObstacleManager()
    # Add a few obstacles
    obstacles.add(SphereObstacle(center=np.array([0.0, 0.0]), radius=0.3))
    obstacles.add(SphereObstacle(center=np.array([0.5, 0.5]), radius=0.2))
    obstacles.add(SphereObstacle(center=np.array([-0.3, 0.3]), radius=0.15))
    return obstacles


def create_test_trajectory():
    """Create a test trajectory that violates obstacles."""
    states = [
        np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float32),  # Near obstacle
        np.array([0.2, 0.2, 0.0, 0.0], dtype=np.float32),  # Inside obstacle
        np.array([0.3, 0.3, 0.0, 0.0], dtype=np.float32),  # Near obstacle
        np.array([0.4, 0.4, 0.0, 0.0], dtype=np.float32),  # Safe
        np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),  # Inside obstacle
    ]
    actions = [np.zeros(2, dtype=np.float32) for _ in range(len(states) - 1)]
    return Trajectory(states=states, actions=actions)


def test_cfs_projection_comparison():
    """Compare NumPy and JAX CFS projection on the same input."""
    print("=" * 80)
    print("CFS Projection Comparison Test")
    print("=" * 80)
    
    # Create obstacles
    obstacles = create_test_obstacles()
    
    # Build SDF texture for JAX (required for JAX version)
    print("\nBuilding SDF texture for JAX support...")
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    print("SDF texture built.")
    
    # Create CFS projection
    cfs = CFSProjection(
        obstacles=obstacles,
        max_iterations=10,
        convergence_tol=1e-6,
        max_constraints_per_point=8,
        constraint_margin=0.25,
    )
    
    # Create test trajectory
    trajectory = create_test_trajectory()
    clearance = 0.1
    
    print(f"\nTest trajectory: {len(trajectory.states)} states")
    print("Initial positions:")
    for i, state in enumerate(trajectory.states):
        pos = state[:2]
        print(f"  State {i}: {pos}")
    
    # NumPy version
    print("\n--- Running NumPy version ---")
    result_numpy = cfs.project(trajectory, step=0, total_steps=100)
    positions_numpy = np.stack([s[:2] for s in result_numpy.states], axis=0)
    print("NumPy projected positions:")
    for i, pos in enumerate(positions_numpy):
        print(f"  State {i}: {pos}")
    
    # JAX version
    print("\n--- Running JAX version ---")
    jax_projector = cfs.make_jax_projector()
    
    if jax_projector is None:
        print("WARNING: JAX projector not available. Cannot compare.")
        return
    
    positions_initial = np.stack([s[:2] for s in trajectory.states], axis=0)
    positions_jax = jnp.asarray(positions_initial, dtype=jnp.float32)
    clearance_jax = jnp.asarray(clearance, dtype=jnp.float32)
    
    try:
        projected_positions_jax = jax_projector(positions_jax, clearance_jax)
        positions_jax_result = np.asarray(projected_positions_jax)
        
        print("JAX projected positions:")
        for i, pos in enumerate(positions_jax_result):
            print(f"  State {i}: {pos}")
        
        # Compare results
        print("\n--- Comparison ---")
        diff = positions_numpy - positions_jax_result
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))
        
        print(f"Position differences:")
        for i, d in enumerate(diff):
            print(f"  State {i}: {d} (norm: {np.linalg.norm(d):.6f})")
        
        print(f"\nMax difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Max relative difference: {max_diff / (np.max(np.abs(positions_numpy)) + 1e-8):.6f}")
        
        # Check feasibility
        print("\n--- Feasibility Check ---")
        for i, pos in enumerate(positions_numpy):
            sdf_val = obstacles.sdf(pos)
            print(f"NumPy state {i}: SDF = {sdf_val:.6f}, feasible = {sdf_val >= clearance}")
        
        for i, pos in enumerate(positions_jax_result):
            sdf_val = obstacles.sdf(pos)
            print(f"JAX state {i}: SDF = {sdf_val:.6f}, feasible = {sdf_val >= clearance}")
        
    except Exception as e:
        print(f"ERROR: JAX projection failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


def test_cfs_projection_single_point():
    """Compare single point projection (simpler case)."""
    print("\n" + "=" * 80)
    print("Single Point CFS Projection Comparison")
    print("=" * 80)
    
    obstacles = create_test_obstacles()
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    cfs = CFSProjection(obstacles=obstacles)
    
    # Single point inside obstacle
    test_points = [
        np.array([0.1, 0.1], dtype=np.float32),  # Inside obstacle
        np.array([0.25, 0.25], dtype=np.float32),  # Inside obstacle
        np.array([0.6, 0.6], dtype=np.float32),  # Inside obstacle
    ]
    
    clearance = 0.1
    
    for point in test_points:
        print(f"\n--- Testing point: {point} ---")
        
        # Create single-point trajectory
        state = np.concatenate([point, np.zeros(2, dtype=np.float32)])
        trajectory = Trajectory(states=[state], actions=[])
        
        # NumPy version
        result_numpy = cfs.project(trajectory, step=0, total_steps=100)
        pos_numpy = result_numpy.states[0][:2]
        print(f"NumPy result: {pos_numpy}")
        
        # JAX version
        jax_projector = cfs.make_jax_projector()
        if jax_projector is not None:
            point_jax = jnp.asarray(point[None, :], dtype=jnp.float32)
            clearance_jax = jnp.asarray(clearance, dtype=jnp.float32)
            result_jax = jax_projector(point_jax, clearance_jax)
            pos_jax = np.asarray(result_jax[0])
            print(f"JAX result: {pos_jax}")
            print(f"Difference: {np.linalg.norm(pos_numpy - pos_jax):.6f}")
        else:
            print("JAX projector not available")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_cfs_projection_comparison()
    test_cfs_projection_single_point()

