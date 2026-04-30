"""Tests for Phase 16 — runtime compatibility validation and array bridge.

Covers:
1. Identity conversion (numpy → numpy)
2. numpy → jax conversion
3. torch → numpy conversion
4. ArrayBridge wrapping an IO
5. validate_runtimes on matching presets
6. validate_runtimes inserting a bridge on mismatch
7. validate_runtimes raising RuntimeMismatchError on incompatible runtimes
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from genedynamics.deploy.array_bridge import ArrayBridge, convert, _convert_state, _convert_command
from genedynamics.deploy.config_schema import BuiltComponents
from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.runtime_check import RuntimeMismatchError, validate_runtimes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides: Any) -> RobotState:
    defaults = dict(
        t=0.0,
        qpos=np.zeros(11, dtype=np.float64),
        qvel=np.zeros(10, dtype=np.float64),
        base_pose=np.zeros(7, dtype=np.float64),
        base_twist=np.zeros(6, dtype=np.float64),
        joint_torque=np.ones(4, dtype=np.float64),
    )
    defaults.update(overrides)
    return RobotState(**defaults)


def _make_command(**overrides: Any) -> ControlCommand:
    defaults = dict(
        kind="joint_pos",
        joint_pos=np.zeros(4, dtype=np.float64),
        kp=np.ones(4, dtype=np.float64) * 100.0,
        kd=np.ones(4, dtype=np.float64) * 4.0,
    )
    defaults.update(overrides)
    return ControlCommand(**defaults)


def _make_io_mock(array_runtime: str = "numpy", physics_backend: str = "stub") -> MagicMock:
    io = MagicMock()
    io.array_runtime = array_runtime
    io.physics_backend = physics_backend
    io.accepts = ("joint_pos", "torque", "mixed")
    io.spec = MagicMock()
    io.get_state.return_value = _make_state()
    io.reset.return_value = _make_state()
    io.step.return_value = _make_state()
    return io


def _make_controller_mock(runtime: str = "numpy") -> MagicMock:
    ctl = MagicMock()
    ctl.runtime = runtime
    ctl.produces = ("joint_pos",)
    return ctl


def _make_safety_mock(runtime: str = "numpy") -> MagicMock:
    sf = MagicMock()
    sf.runtime = runtime
    return sf


# ---------------------------------------------------------------------------
# 1. Identity round-trip: numpy → numpy
# ---------------------------------------------------------------------------


class TestConvertIdentity:
    def test_numpy_to_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = convert(arr, "numpy")
        assert result is arr  # no copy

    def test_none_passthrough(self):
        assert convert(None, "numpy") is None
        assert convert(None, "jax") is None


# ---------------------------------------------------------------------------
# 2. numpy → jax
# ---------------------------------------------------------------------------


class TestConvertNumpyToJax:
    @pytest.fixture(autouse=True)
    def _skip_if_no_jax(self):
        pytest.importorskip("jax")

    def test_conversion(self):
        import jax.numpy as jnp

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = convert(arr, "jax")
        assert hasattr(result, "devices")  # jax.Array attribute
        np.testing.assert_allclose(np.asarray(result), arr)

    def test_round_trip(self):
        arr = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        jax_arr = convert(arr, "jax")
        back = convert(jax_arr, "numpy")
        assert isinstance(back, np.ndarray)
        np.testing.assert_allclose(back, arr)


# ---------------------------------------------------------------------------
# 3. torch → numpy
# ---------------------------------------------------------------------------


class TestConvertTorchToNumpy:
    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_conversion(self):
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        result = convert(t, "numpy")
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_round_trip(self):
        import torch

        arr = np.array([7.0, 8.0], dtype=np.float32)
        t = convert(arr, "torch")
        assert isinstance(t, torch.Tensor)
        back = convert(t, "numpy")
        np.testing.assert_allclose(back, arr)


# ---------------------------------------------------------------------------
# 4. ArrayBridge wrapping an IO
# ---------------------------------------------------------------------------


class TestArrayBridge:
    def test_get_state_converts(self):
        io = _make_io_mock(array_runtime="numpy")
        bridge = ArrayBridge(io, target_runtime="numpy")
        state = bridge.get_state()
        assert isinstance(state.qpos, np.ndarray)

    def test_runtime_attribute(self):
        io = _make_io_mock(array_runtime="jax")
        bridge = ArrayBridge(io, target_runtime="numpy")
        assert bridge.array_runtime == "numpy"
        assert bridge.physics_backend == io.physics_backend

    def test_send_control_converts_back(self):
        io = _make_io_mock(array_runtime="numpy")
        bridge = ArrayBridge(io, target_runtime="numpy")
        cmd = _make_command()
        bridge.send_control(cmd)
        io.send_control.assert_called_once()

    def test_reset_delegates(self):
        io = _make_io_mock(array_runtime="numpy")
        bridge = ArrayBridge(io, target_runtime="numpy")
        bridge.reset("ep_001")
        io.reset.assert_called_once_with("ep_001")

    def test_close_delegates(self):
        io = _make_io_mock()
        bridge = ArrayBridge(io, target_runtime="numpy")
        bridge.close()
        io.close.assert_called_once()


# ---------------------------------------------------------------------------
# 5. validate_runtimes — matching preset (no changes)
# ---------------------------------------------------------------------------


class TestValidateRuntimesMatch:
    def test_all_numpy(self):
        built = BuiltComponents(
            io=_make_io_mock("numpy"),
            controller=_make_controller_mock("numpy"),
            safety=_make_safety_mock("numpy"),
            observers=[],
        )
        result = validate_runtimes(built)
        # IO should NOT be wrapped.
        assert not isinstance(result.io, ArrayBridge)

    def test_missing_io_skips(self):
        built = BuiltComponents(io=None, controller=_make_controller_mock(), observers=[])
        result = validate_runtimes(built)
        assert result.io is None

    def test_safety_any_is_compatible(self):
        built = BuiltComponents(
            io=_make_io_mock("numpy"),
            controller=_make_controller_mock("numpy"),
            safety=_make_safety_mock("any"),
            observers=[],
        )
        result = validate_runtimes(built)
        assert not isinstance(result.io, ArrayBridge)


# ---------------------------------------------------------------------------
# 6. validate_runtimes — mismatch inserts bridge
# ---------------------------------------------------------------------------


class TestValidateRuntimesBridge:
    def test_jax_io_numpy_controller_bridges(self):
        built = BuiltComponents(
            io=_make_io_mock("jax"),
            controller=_make_controller_mock("numpy"),
            safety=_make_safety_mock("numpy"),
            observers=[],
        )
        result = validate_runtimes(built)
        assert isinstance(result.io, ArrayBridge)
        assert result.io.array_runtime == "numpy"

    def test_numpy_io_torch_controller_bridges(self):
        built = BuiltComponents(
            io=_make_io_mock("numpy"),
            controller=_make_controller_mock("torch"),
            observers=[],
        )
        result = validate_runtimes(built)
        assert isinstance(result.io, ArrayBridge)
        assert result.io.array_runtime == "torch"


# ---------------------------------------------------------------------------
# 7. validate_runtimes — raises on incompatible
# ---------------------------------------------------------------------------


class TestValidateRuntimesError:
    def test_strict_mode_rejects_mismatch(self):
        built = BuiltComponents(
            io=_make_io_mock("jax"),
            controller=_make_controller_mock("numpy"),
            observers=[],
        )
        with pytest.raises(RuntimeMismatchError) as exc_info:
            validate_runtimes(built, strict=True)
        assert "jax" in str(exc_info.value)
        assert "numpy" in str(exc_info.value)

    def test_unbridgeable_runtime_raises(self):
        io = _make_io_mock("numpy")
        io.array_runtime = "rust"  # not in _BRIDGEABLE
        built = BuiltComponents(
            io=io,
            controller=_make_controller_mock("numpy"),
            observers=[],
        )
        # rust != numpy → mismatch, but rust not bridgeable.
        # Actually io=rust, ctl=numpy → not matching, and "rust" not in _BRIDGEABLE.
        with pytest.raises(RuntimeMismatchError):
            validate_runtimes(built)


# ---------------------------------------------------------------------------
# State / Command conversion helpers
# ---------------------------------------------------------------------------


class TestConvertStateCommand:
    def test_state_round_trip(self):
        state = _make_state()
        converted = _convert_state(state, "numpy")
        np.testing.assert_array_equal(converted.qpos, state.qpos)
        np.testing.assert_array_equal(converted.joint_torque, state.joint_torque)
        assert converted.t == state.t

    def test_command_round_trip(self):
        cmd = _make_command()
        converted = _convert_command(cmd, "numpy")
        np.testing.assert_array_equal(converted.joint_pos, cmd.joint_pos)
        assert converted.kind == cmd.kind

    def test_state_with_contact(self):
        contact = {"left_foot": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        state = _make_state(contact=contact)
        converted = _convert_state(state, "numpy")
        np.testing.assert_array_equal(
            converted.contact["left_foot"], contact["left_foot"]
        )
