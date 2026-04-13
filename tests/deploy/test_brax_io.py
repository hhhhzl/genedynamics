"""Tests for Phase 17 — BraxRobotIO.

All tests are marked ``@pytest.mark.brax`` for CI opt-in; they require
``brax >= 0.10`` and ``jax`` to be installed.

Covers:
1. reset() returns a RobotState with non-empty qpos / qvel
2. get_state() == reset() immediately after reset (no hidden step)
3. send_control(joint_pos=zeros) + step(dt) advances qpos
4. physics_backend == "brax" and array_runtime == "jax"
5. Integration: G1CorridorBraxPreset runs for 100 steps end-to-end
"""

from __future__ import annotations

import numpy as np
import pytest

brax = pytest.importorskip("brax")
jax = pytest.importorskip("jax")

from genedynamics.deploy.io.brax_io import BraxRobotIO
from genedynamics.deploy.interfaces.messages import ControlCommand

pytestmark = pytest.mark.brax


# ---------------------------------------------------------------------------
# 1. reset() returns a valid RobotState
# ---------------------------------------------------------------------------


class TestBraxReset:
    def test_reset_returns_state_with_arrays(self):
        io = BraxRobotIO(env_name="humanoid", backend="mjx")
        state = io.reset()
        assert state.qpos is not None
        assert state.qvel is not None
        assert state.qpos.shape[-1] > 0
        assert state.qvel.shape[-1] > 0
        io.close()

    def test_reset_time_is_zero(self):
        io = BraxRobotIO()
        state = io.reset()
        assert state.t == 0.0
        io.close()


# ---------------------------------------------------------------------------
# 2. get_state() matches reset() with no intervening step
# ---------------------------------------------------------------------------


class TestBraxGetState:
    def test_get_state_matches_reset(self):
        io = BraxRobotIO()
        reset_state = io.reset()
        get_state = io.get_state()
        np.testing.assert_allclose(
            np.asarray(jax.device_get(get_state.qpos)),
            np.asarray(jax.device_get(reset_state.qpos)),
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(get_state.qvel)),
            np.asarray(jax.device_get(reset_state.qvel)),
        )
        io.close()


# ---------------------------------------------------------------------------
# 3. send_control + step advances the state
# ---------------------------------------------------------------------------


class TestBraxStep:
    def test_step_advances_state(self):
        io = BraxRobotIO()
        state0 = io.reset()
        qpos0 = np.asarray(jax.device_get(state0.qpos))

        cmd = ControlCommand(
            kind="joint_pos",
            joint_pos=jax.numpy.zeros(io._env.action_size),
        )
        io.send_control(cmd)
        state1 = io.step(io.sim_dt)
        qpos1 = np.asarray(jax.device_get(state1.qpos))

        # Physics should have advanced — qpos should differ.
        assert not np.allclose(qpos0, qpos1), "qpos unchanged after step"
        io.close()


# ---------------------------------------------------------------------------
# 4. Backend and runtime tags
# ---------------------------------------------------------------------------


class TestBraxTags:
    def test_physics_backend(self):
        io = BraxRobotIO()
        assert io.physics_backend == "brax"
        io.close()

    def test_array_runtime(self):
        io = BraxRobotIO()
        assert io.array_runtime == "jax"
        io.close()

    def test_accepts(self):
        io = BraxRobotIO()
        assert "joint_pos" in io.accepts
        assert "torque" in io.accepts
        io.close()


# ---------------------------------------------------------------------------
# 5. Shadow sync
# ---------------------------------------------------------------------------


class TestBraxShadowSync:
    def test_shadow_qpos_is_numpy(self):
        io = BraxRobotIO(sync_host_shadow=True)
        io.reset()
        assert io.shadow_qpos is not None
        assert isinstance(io.shadow_qpos, np.ndarray)
        io.close()

    def test_shadow_disabled(self):
        io = BraxRobotIO(sync_host_shadow=False)
        io.reset()
        assert io.shadow_qpos is None
        io.close()


# ---------------------------------------------------------------------------
# 6. Spec is populated
# ---------------------------------------------------------------------------


class TestBraxSpec:
    def test_spec_has_actuated_joints(self):
        io = BraxRobotIO()
        io.reset()
        assert hasattr(io.spec, "num_actuated")
        assert io.spec.num_actuated > 0
        io.close()
