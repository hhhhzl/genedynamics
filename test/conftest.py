"""
Pytest configuration and shared fixtures.

This module provides common fixtures used across all tests.
"""

import pytest
import numpy as np
import os
from typing import Generator, Optional, Any

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

# Optional dependencies
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    from omni.isaac.core import World
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

try:
    import gymnasium
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

try:
    import brax
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

try:
    import pybullet
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

try:
    import pinocchio
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False

try:
    JAX_GPU_AVAILABLE = JAX_AVAILABLE and bool(jax.devices("gpu"))
except Exception:
    JAX_GPU_AVAILABLE = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator for tests."""
    return np.random.default_rng(42)


@pytest.fixture
def jax_rng() -> Optional[Any]:
    """JAX random key for tests."""
    if JAX_AVAILABLE:
        return jax.random.PRNGKey(42)
    return None


@pytest.fixture
def tolerance() -> float:
    """Numerical tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def state_dim() -> int:
    """Default state dimension for tests."""
    return 6


@pytest.fixture
def act_dim() -> int:
    """Default action dimension for tests."""
    return 3


@pytest.fixture
def sample_state(state_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample state vector for tests."""
    return rng.standard_normal(state_dim).astype(np.float32)


@pytest.fixture
def sample_action(act_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample action vector for tests."""
    return rng.standard_normal(act_dim).astype(np.float32)


@pytest.fixture
def sample_obstacle_center(rng: np.random.Generator) -> np.ndarray:
    """Sample obstacle center for tests."""
    return rng.standard_normal(3).astype(np.float32)


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit test")
    config.addinivalue_line("markers", "integration: Integration test")
    config.addinivalue_line("markers", "slow: Slow running test")
    config.addinivalue_line("markers", "requires_jax: Test requires JAX")
    config.addinivalue_line("markers", "requires_mujoco: Test requires MuJoCo")
    config.addinivalue_line("markers", "requires_isaac: Test requires Isaac Sim")
    config.addinivalue_line("markers", "requires_gymnasium: Test requires Gymnasium")
    config.addinivalue_line("markers", "requires_brax: Test requires Brax")
    config.addinivalue_line("markers", "requires_mujoco: Test requires MuJoCo")
    config.addinivalue_line("markers", "requires_isaac: Test requires Isaac Sim")
    config.addinivalue_line("markers", "requires_trimesh: Test requires trimesh")
    config.addinivalue_line("markers", "requires_scipy: Test requires SciPy")
    config.addinivalue_line("markers", "requires_pybullet: Test requires PyBullet")
    config.addinivalue_line("markers", "requires_pinocchio: Test requires Pinocchio")
    config.addinivalue_line("markers", "performance: Timing/scalability test")
    config.addinivalue_line("markers", "docker: Test requires a project Docker image")
    config.addinivalue_line("markers", "gpu: Test requires a supported accelerator")
    config.addinivalue_line("markers", "brax: Test requires Brax (deploy suite)")
    config.addinivalue_line("markers", "isaac_lab: Test requires Isaac Lab (deploy suite)")
    config.addinivalue_line("markers", "optional: Test covers an optional integration")


def pytest_collection_modifyitems(config, items):
    """Skip optional capability tests when that capability is unavailable."""
    del config
    capabilities = {
        "requires_jax": (JAX_AVAILABLE, "JAX is not installed"),
        "requires_mujoco": (MUJOCO_AVAILABLE, "MuJoCo is not installed"),
        "requires_isaac": (ISAAC_AVAILABLE, "Isaac Sim is not installed"),
        "requires_gymnasium": (GYMNASIUM_AVAILABLE, "Gymnasium is not installed"),
        "requires_brax": (BRAX_AVAILABLE, "Brax is not installed"),
        "requires_trimesh": (TRIMESH_AVAILABLE, "trimesh is not installed"),
        "requires_scipy": (SCIPY_AVAILABLE, "SciPy is not installed"),
        "requires_pybullet": (PYBULLET_AVAILABLE, "PyBullet is not installed"),
        "requires_pinocchio": (PINOCCHIO_AVAILABLE, "Pinocchio is not installed"),
        "brax": (BRAX_AVAILABLE, "Brax is not installed"),
        "isaac_lab": (ISAAC_AVAILABLE, "Isaac Lab is not installed"),
        "gpu": (JAX_GPU_AVAILABLE, "a JAX GPU device is not available"),
        "docker": (
            os.environ.get("GENEDYNAMICS_TEST_DOCKER") == "1",
            "set GENEDYNAMICS_TEST_DOCKER=1 inside a project image",
        ),
    }
    for item in items:
        for marker, (available, reason) in capabilities.items():
            if item.get_closest_marker(marker) is not None and not available:
                item.add_marker(pytest.mark.skip(reason=reason))
