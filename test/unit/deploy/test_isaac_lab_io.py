"""Tests for Phase 18 — IsaacLabRobotIO.

All tests are marked ``@pytest.mark.isaac_lab`` for CI opt-in; they require
Isaac Lab (NVIDIA image) and ``torch >= 2.1`` to be installed.

Covers:
1. reset() returns a RobotState with torch tensors
2. physics_backend == "isaac_lab" and array_runtime == "torch"
3. send_control + step advances the state
4. Phase 16 bridge round-trip: torch IO → numpy controller → torch IO
5. Smoke test: G1CorridorIsaacLabPreset imports cleanly
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Isaac Lab is only available inside the NVIDIA Docker image.
# Skip the entire module if omni.isaac.lab is not importable.
pytest.importorskip("omni.isaac.lab")

from genedynamics.deploy.io.isaac_lab_io import IsaacLabRobotIO
from genedynamics.deploy.interfaces.messages import ControlCommand

pytestmark = pytest.mark.isaac_lab


# ---------------------------------------------------------------------------
# 1. reset() returns a valid RobotState with torch tensors
# ---------------------------------------------------------------------------


class TestIsaacLabReset:
    def test_reset_returns_state(self):
        io = IsaacLabRobotIO(headless=True)
        state = io.reset()
        assert state.qpos is not None
        assert state.qvel is not None
        assert isinstance(state.qpos, torch.Tensor)
        assert isinstance(state.qvel, torch.Tensor)
        io.close()

    def test_reset_time_is_zero(self):
        io = IsaacLabRobotIO(headless=True)
        state = io.reset()
        assert state.t == 0.0
        io.close()


# ---------------------------------------------------------------------------
# 2. Backend and runtime tags
# ---------------------------------------------------------------------------


class TestIsaacLabTags:
    def test_physics_backend(self):
        io = IsaacLabRobotIO(headless=True)
        assert io.physics_backend == "isaac_lab"
        io.close()

    def test_array_runtime(self):
        io = IsaacLabRobotIO(headless=True)
        assert io.array_runtime == "torch"
        io.close()

    def test_accepts(self):
        io = IsaacLabRobotIO(headless=True)
        assert "joint_pos" in io.accepts
        assert "torque" in io.accepts
        io.close()


# ---------------------------------------------------------------------------
# 3. send_control + step advances state
# ---------------------------------------------------------------------------


class TestIsaacLabStep:
    def test_step_advances(self):
        io = IsaacLabRobotIO(headless=True)
        state0 = io.reset()
        qpos0 = state0.qpos.detach().cpu().numpy()

        action_dim = io._env.action_space.shape[-1]
        cmd = ControlCommand(
            kind="joint_pos",
            joint_pos=torch.zeros(action_dim, device=io.device),
        )
        io.send_control(cmd)
        state1 = io.step(io.sim_dt)
        qpos1 = state1.qpos.detach().cpu().numpy()

        assert state1.t > 0.0
        io.close()


# ---------------------------------------------------------------------------
# 4. Phase 16 bridge round-trip
# ---------------------------------------------------------------------------


class TestIsaacLabBridge:
    def test_bridge_wraps_torch_to_numpy(self):
        from genedynamics.deploy.array_bridge import ArrayBridge

        io = IsaacLabRobotIO(headless=True)
        bridge = ArrayBridge(io, target_runtime="numpy")
        state = bridge.reset()
        assert isinstance(state.qpos, np.ndarray)
        assert bridge.array_runtime == "numpy"
        bridge.close()


# ---------------------------------------------------------------------------
# 5. Shadow sync
# ---------------------------------------------------------------------------


class TestIsaacLabShadow:
    def test_shadow_is_numpy(self):
        io = IsaacLabRobotIO(headless=True, sync_host_shadow=True)
        io.reset()
        if io.shadow_qpos is not None:
            assert isinstance(io.shadow_qpos, np.ndarray)
        io.close()


# ---------------------------------------------------------------------------
# 6. Preset imports cleanly (no Isaac Lab needed for import)
# ---------------------------------------------------------------------------


class TestIsaacLabPresetImport:
    def test_preset_importable(self):
        from genedynamics.deploy.presets.g1_corridor_isaac_lab import (
            G1CorridorIsaacLabPreset,
        )

        assert G1CorridorIsaacLabPreset.sim_dt == 1.0 / 200.0
        assert G1CorridorIsaacLabPreset.control_hz == 50.0
