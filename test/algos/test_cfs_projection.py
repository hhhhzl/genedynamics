# tests/test_cfs_projection.py
import sys
import types
import importlib.util
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pytest


# -----------------------------
# Local minimal Trajectory type
# -----------------------------
@dataclass
class Trajectory:
    # Keep this intentionally minimal: CFSProjection only needs .states and .actions
    states: List[np.ndarray]
    actions: Any = None


# -----------------------------
# Minimal stubs to load cfs.py
# -----------------------------
def _install_enerdynamics_stubs():
    """
    Your cfs.py imports:
      - enerdynamics.core.constraints.base.FeasibilityOperator
      - enerdynamics.core.types.Trajectory, State
      - enerdynamics.envs.obstacles.base.ObstacleManager
    This installs minimal stubs into sys.modules so cfs.py can be imported standalone.
    """
    stubbed_names = [
        "enerdynamics",
        "enerdynamics.core",
        "enerdynamics.core.constraints",
        "enerdynamics.core.constraints.base",
        "enerdynamics.core.types",
        "enerdynamics.envs",
        "enerdynamics.envs.obstacles",
        "enerdynamics.envs.obstacles.base",
    ]

    # Snapshot existing modules so we can restore later (avoid poisoning the process)
    prev_modules = {name: sys.modules.get(name) for name in stubbed_names}

    # Package chain: enerdynamics, enerdynamics.core, enerdynamics.core.constraints, ...
    def ensure_mod(name: str):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    ensure_mod("enerdynamics")
    ensure_mod("enerdynamics.core")
    ensure_mod("enerdynamics.core.constraints")
    ensure_mod("enerdynamics.envs")
    ensure_mod("enerdynamics.envs.obstacles")

    base_constraints = ensure_mod("enerdynamics.core.constraints.base")
    types_mod = ensure_mod("enerdynamics.core.types")
    obstacles_base = ensure_mod("enerdynamics.envs.obstacles.base")

    class FeasibilityOperator:
        pass

    State = np.ndarray

    class ObstacleManager:
        pass

    base_constraints.FeasibilityOperator = FeasibilityOperator
    types_mod.Trajectory = Trajectory
    types_mod.State = State
    obstacles_base.ObstacleManager = ObstacleManager
    return prev_modules


# -----------------------------
# Mock obstacles / manager
# -----------------------------
class CircleObstacle:
    """Signed distance: positive outside, negative inside."""
    def __init__(self, center, radius):
        self.c = np.asarray(center, dtype=np.float32).reshape(1, -1)
        self.r = float(radius)

    def sdf(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        d = np.linalg.norm(x - self.c, axis=1) - self.r
        return d.astype(np.float32)

    def gradient(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        v = x - self.c.reshape(-1)
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (v / n).astype(np.float32)


class MockObstacleManager:
    """
    Iterable over obstacles, plus a union sdf(points) used by your CFSProjection's stagnation check.
    """
    def __init__(self, obstacles):
        self._obs = list(obstacles)

    def __iter__(self):
        return iter(self._obs)

    def sdf(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if len(self._obs) == 0:
            return np.full((x.shape[0],), np.inf, dtype=np.float32)
        sdfs = np.stack([o.sdf(x) for o in self._obs], axis=0)
        return np.min(sdfs, axis=0).astype(np.float32)


# -----------------------------
# Helper: import your cfs.py
# -----------------------------
def import_cfs_module(path_to_cfs_py: str):
    prev = _install_enerdynamics_stubs()

    def _restore_modules():
        for name, old in prev.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    try:
        spec = importlib.util.spec_from_file_location("cfs_under_test", path_to_cfs_py)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    finally:
        _restore_modules()


# -----------------------------
# Tests
# -----------------------------
@pytest.fixture(scope="module")
def cfs():
    # Change this path if needed.
    # If your cfs.py lives inside your repo, point to that file instead.
    path = "enerdynamics/core/constraints/projections/cfs.py"
    return import_cfs_module(path)


def test_circle_inside_point_projects_to_clearance_shell(cfs):
    """
    A point inside a circle should be pushed out to satisfy sdf >= clearance.

    This test WILL FAIL with your current implementation if you normalize grad
    but don't scale the RHS (b) consistently.
    """
    R = 1.0
    clearance = 0.2
    obs = CircleObstacle(center=(0.0, 0.0), radius=R)
    mgr = MockObstacleManager([obs])

    proj = cfs.CFSProjection(
        obstacles=mgr,
        max_iterations=10,
        convergence_tol=1e-10,
        max_constraints_per_point=1,
        constraint_margin=0.0,
    )

    x0 = np.array([[0.5, 0.0]], dtype=np.float32)  # inside
    x1 = proj._project_cfs_batch(x0, clearance=clearance, step=None)

    # Must satisfy clearance
    sdf1 = obs.sdf(x1)[0]
    assert sdf1 >= clearance - 1e-4, f"sdf after projection = {sdf1}, expected >= {clearance}"

    # For this symmetric case (x-axis), the closest feasible point is at radius+clearance on x-axis.
    target = R + clearance
    r1 = float(np.linalg.norm(x1[0]))
    assert abs(r1 - target) < 1e-3, f"radius after projection {r1} != {target}"
    assert abs(float(x1[0, 1])) < 1e-3, f"y after projection {x1[0, 1]} expected ~0"


def test_circle_feasible_point_is_unchanged(cfs):
    R = 1.0
    clearance = 0.2
    obs = CircleObstacle(center=(0.0, 0.0), radius=R)
    mgr = MockObstacleManager([obs])

    proj = cfs.CFSProjection(
        obstacles=mgr,
        max_iterations=5,
        convergence_tol=1e-10,
        max_constraints_per_point=1,
        constraint_margin=0.0,
    )

    x0 = np.array([[1.5, 0.0]], dtype=np.float32)  # sdf = 0.5 >= 0.2
    x1 = proj._project_cfs_batch(x0, clearance=clearance, step=None)

    assert np.linalg.norm(x1 - x0) < 1e-6, f"feasible point moved by {np.linalg.norm(x1-x0)}"


def test_late_stage_only_gating_blocks_early_projection(cfs):
    """
    If use_late_stage_only=True, early steps should not modify the trajectory.

    This test will FAIL until you call should_apply() inside project().
    """
    R = 1.0
    clearance = 0.2
    obs = CircleObstacle(center=(0.0, 0.0), radius=R)
    mgr = MockObstacleManager([obs])

    proj = cfs.CFSProjection(
        obstacles=mgr,
        max_iterations=10,
        convergence_tol=1e-10,
        max_constraints_per_point=1,
        constraint_margin=0.0,
        use_late_stage_only=True,
        late_stage_ratio=0.3,  # apply only in last 30%
    )

    # Build a trajectory with a violating point
    traj = Trajectory(states=[np.array([0.5, 0.0], dtype=np.float32)], actions=None)

    # Early step => should not apply
    out = proj.project(traj, step=1, total_steps=10)

    moved = np.linalg.norm(out.states[0][:2] - traj.states[0][:2])
    assert moved < 1e-6, f"late-stage gating violated: moved {moved} in early step"
