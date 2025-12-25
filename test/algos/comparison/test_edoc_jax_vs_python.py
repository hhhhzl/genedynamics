"""
Test script to compare EDOC JAX loop vs Python loop.

This test isolates EDOC differences by comparing results with and without
JAX CFS projection (which forces Python loop vs JAX loop).
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

from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle
from enerdynamics.core.constraints import (
    ConstraintManager,
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
    CFSProjection,
    ConstraintScheduleManager,
)
from enerdynamics.solvers.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_energy


def create_test_environment():
    """Create a test environment with obstacles."""
    env = DoubleIntegratorBox2DEnv(
        dt=0.1,
        horizon=20,
        p_max=2.0,
        v_max=2.0,
    )
    
    # Create obstacles
    obstacles = ObstacleManager()
    obstacles.add(SphereObstacle(center=np.array([0.0, 0.0]), radius=0.3))
    obstacles.add(SphereObstacle(center=np.array([0.5, 0.5]), radius=0.2))
    
    return env, obstacles


def test_edoc_without_cfs():
    """Test EDOC without CFS (should use JAX loop)."""
    print("=" * 80)
    print("EDOC without CFS (JAX loop)")
    print("=" * 80)
    
    env, obstacles = create_test_environment()
    energy = make_energy("double_integrator_box_2d")
    
    # Create EDOC without CFS
    edoc = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=env.horizon,
        dt=env.dt,
        constraint_manager=None,  # No constraints
        action_space=True,
        action_diffuse_steps=20,
        action_nsample=10,
        action_score_mode="energy",
    )
    
    if not JAX_AVAILABLE:
        print("JAX not available, skipping test")
        return None
    
    x0 = np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    rng = jax.random.PRNGKey(42)
    
    print(f"Initial state: {x0}")
    print("Running EDOC (JAX loop, no CFS)...")
    
    try:
        result = edoc.plan(rng)
        print(f"Success! Final actions shape: {result['actions'].shape}")
        print(f"Final trajectory length: {len(result['states'])}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edoc_with_numpy_cfs():
    """Test EDOC with NumPy CFS (forces Python loop)."""
    print("\n" + "=" * 80)
    print("EDOC with NumPy CFS (Python loop)")
    print("=" * 80)
    
    env, obstacles = create_test_environment()
    energy = make_energy("double_integrator_box_2d")
    
    # Create CFS projection (will use NumPy version)
    # Don't build SDF texture to force NumPy fallback
    cfs = CFSProjection(
        obstacles=obstacles,
        max_iterations=10,
    )
    
    # Create constraint manager
    hard_constraint = ObstacleHardConstraint(obstacles=obstacles, clearance=0.1)
    constraint_manager = ConstraintManager(
        hard_constraints=[hard_constraint],
        feasibility_operator=cfs,
    )
    
    # Create EDOC with CFS
    edoc = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=env.horizon,
        dt=env.dt,
        constraint_manager=constraint_manager,
        action_space=True,
        action_diffuse_steps=20,
        action_nsample=10,
        action_score_mode="energy",
    )
    
    if not JAX_AVAILABLE:
        print("JAX not available, skipping test")
        return None
    
    x0 = np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    rng = jax.random.PRNGKey(42)
    
    print(f"Initial state: {x0}")
    print("Running EDOC (Python loop, NumPy CFS)...")
    
    try:
        result = edoc.plan(rng)
        print(f"Success! Final actions shape: {result['actions'].shape}")
        print(f"Final trajectory length: {len(result['states'])}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edoc_with_jax_cfs():
    """Test EDOC with JAX CFS (JAX loop with CFS)."""
    print("\n" + "=" * 80)
    print("EDOC with JAX CFS (JAX loop)")
    print("=" * 80)
    
    env, obstacles = create_test_environment()
    energy = make_energy("double_integrator_box_2d")
    
    # Build SDF texture to enable JAX CFS
    obstacles.build_sdf_texture_2d(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        res=0.01,
        force_rebuild=True,
    )
    
    # Create CFS projection (should use JAX version now)
    cfs = CFSProjection(
        obstacles=obstacles,
        max_iterations=10,
    )
    
    # Verify JAX projector is available
    jax_projector = cfs.make_jax_projector()
    if jax_projector is None:
        print("WARNING: JAX projector not available, will use NumPy version")
    else:
        print("JAX projector available, will use JAX loop")
    
    # Create constraint manager
    hard_constraint = ObstacleHardConstraint(obstacles=obstacles, clearance=0.1)
    constraint_manager = ConstraintManager(
        hard_constraints=[hard_constraint],
        feasibility_operator=cfs,
    )
    
    # Create EDOC with CFS
    edoc = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=env.horizon,
        dt=env.dt,
        constraint_manager=constraint_manager,
        action_space=True,
        action_diffuse_steps=20,
        action_nsample=10,
        action_score_mode="energy",
    )
    
    if not JAX_AVAILABLE:
        print("JAX not available, skipping test")
        return None
    
    x0 = np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    rng = jax.random.PRNGKey(42)
    
    print(f"Initial state: {x0}")
    print("Running EDOC (JAX loop, JAX CFS)...")
    
    try:
        result = edoc.plan(rng)
        print(f"Success! Final actions shape: {result['actions'].shape}")
        print(f"Final trajectory length: {len(result['states'])}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(result1, result2, name1, name2):
    """Compare two EDOC results."""
    if result1 is None or result2 is None:
        print(f"Cannot compare: one or both results are None")
        return
    
    print(f"\n--- Comparing {name1} vs {name2} ---")
    
    # Compare actions
    actions1 = result1['actions']
    actions2 = result2['actions']
    
    if actions1.shape != actions2.shape:
        print(f"Action shapes differ: {actions1.shape} vs {actions2.shape}")
    else:
        diff = np.abs(actions1 - actions2)
        print(f"Actions difference:")
        print(f"  Max: {np.max(diff):.6f}")
        print(f"  Mean: {np.mean(diff):.6f}")
        print(f"  Per-step max: {np.max(diff, axis=1)}")
    
    # Compare final states
    states1 = result1['states']
    states2 = result2['states']
    
    if len(states1) != len(states2):
        print(f"State sequence lengths differ: {len(states1)} vs {len(states2)}")
    else:
        states1_arr = np.stack(states1)
        states2_arr = np.stack(states2)
        diff = np.abs(states1_arr - states2_arr)
        print(f"States difference:")
        print(f"  Max: {np.max(diff):.6f}")
        print(f"  Mean: {np.mean(diff):.6f}")
        print(f"  Final state diff: {np.linalg.norm(states1_arr[-1] - states2_arr[-1]):.6f}")


def test_edoc_comparison():
    """Run all EDOC comparison tests."""
    print("=" * 80)
    print("EDOC JAX vs Python Loop Comparison")
    print("=" * 80)
    
    # Test 1: EDOC without CFS (baseline)
    result_no_cfs = test_edoc_without_cfs()
    
    # Test 2: EDOC with NumPy CFS
    result_numpy_cfs = test_edoc_with_numpy_cfs()
    
    # Test 3: EDOC with JAX CFS
    result_jax_cfs = test_edoc_with_jax_cfs()
    
    # Compare results
    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    if result_no_cfs is not None and result_numpy_cfs is not None:
        compare_results(result_no_cfs, result_numpy_cfs, "No CFS", "NumPy CFS")
    
    if result_numpy_cfs is not None and result_jax_cfs is not None:
        compare_results(result_numpy_cfs, result_jax_cfs, "NumPy CFS", "JAX CFS")
    
    if result_no_cfs is not None and result_jax_cfs is not None:
        compare_results(result_no_cfs, result_jax_cfs, "No CFS", "JAX CFS")
    
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  - No CFS vs NumPy CFS: Shows effect of CFS projection")
    print("  - NumPy CFS vs JAX CFS: Verifies CFS implementation consistency (should be ~0)")
    print("  - No CFS vs JAX CFS: Shows combined EDOC+CFS differences")
    print("=" * 80)


if __name__ == "__main__":
    test_edoc_comparison()

