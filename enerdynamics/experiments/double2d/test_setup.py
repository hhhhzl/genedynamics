"""
Quick test to verify experiment setup is correct.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
    print("✓ DoubleIntegratorBox2DEnv imported")
except Exception as e:
    print(f"✗ Failed to import DoubleIntegratorBox2DEnv: {e}")

try:
    from enerdynamics.envs.obstacles.base import ObstacleManager
    from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
    from enerdynamics.envs.obstacles.nonconvex import UnionObstacle
    print("✓ Obstacles imported")
except Exception as e:
    print(f"✗ Failed to import obstacles: {e}")

try:
    from enerdynamics.core.constraints import (
        ConstraintManager,
        ObstacleSoftConstraint,
        ObstacleHardConstraint,
        CFSProjection,
        ConstraintScheduleManager,
    )
    print("✓ Constraints imported")
except Exception as e:
    print(f"✗ Failed to import constraints: {e}")

try:
    from enerdynamics.solvers.edoc import EDOCPlanner
    print("✓ EDOCPlanner imported")
except Exception as e:
    print(f"✗ Failed to import EDOCPlanner: {e}")

try:
    from enerdynamics.envs.factories import make_energy
    print("✓ Energy factory imported")
except Exception as e:
    print(f"✗ Failed to import energy factory: {e}")

try:
    import jax
    import jax.numpy as jnp
    print("✓ JAX imported")
except Exception as e:
    print(f"✗ Failed to import JAX: {e}")

print("\nSetup test complete!")
