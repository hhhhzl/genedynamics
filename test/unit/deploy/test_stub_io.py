"""Unit tests for :class:`StubRobotIO`."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.deploy.interfaces.messages import ControlCommand
from genedynamics.deploy.io.stub_io import StubRobotIO, StubSpec


def test_stub_io_default_construction():
    io = StubRobotIO()
    assert io.spec.num_actuated == 4
    assert io.physics_backend == "stub"
    assert io.array_runtime == "numpy"
    assert "joint_pos" in io.accepts and "loco" in io.accepts


def test_stub_io_reset_returns_initial_state():
    io = StubRobotIO(initial_qpos=np.array([1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4]))
    state = io.reset()
    assert io.reset_count == 1
    np.testing.assert_array_equal(state.qpos[:7], [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(state.base_pose, state.qpos[:7])


def test_stub_io_step_advances_clock_and_records():
    io = StubRobotIO()
    io.reset()
    io.send_control(ControlCommand(kind="joint_pos", joint_pos=np.zeros(4)))
    state = io.step(0.02)
    assert io.step_called_count == 1
    assert state.t == pytest.approx(0.02)
    assert len(io.sent_commands) == 1


def test_stub_io_records_command_history():
    io = StubRobotIO()
    io.reset()
    for v in [0.1, 0.2, 0.3]:
        io.send_control(ControlCommand(kind="joint_pos", joint_pos=np.full(4, v)))
        io.step(0.01)
    assert len(io.sent_commands) == 3
    assert io.sent_commands[1].joint_pos[0] == pytest.approx(0.2)


def test_stub_io_close_idempotent():
    io = StubRobotIO()
    io.reset()
    io.close()
    io.close()
    with pytest.raises(RuntimeError):
        io.reset()


def test_stub_spec_has_qpos_indices():
    s = StubSpec(num_actuated=6)
    np.testing.assert_array_equal(
        s.actuated_qpos_indices, np.arange(7, 13, dtype=np.int32)
    )
    assert len(s.actuated_joints) == 6
