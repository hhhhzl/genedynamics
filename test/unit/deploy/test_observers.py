"""Unit tests for the Phase 8 logger and recorder observers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.observers.logger import LoggerObserver
from genedynamics.deploy.observers.recorder import RecorderObserver


def _make_intent(t: float) -> Intent:
    return Intent(
        t=t,
        base_yaw=0.0,
        base_height=0.7,
        base_lin_vel=np.array([0.5, 0.0]),
        base_yaw_rate=0.0,
        contact_phase="double_support",
    )


def _make_state(t: float) -> RobotState:
    return RobotState(
        t=t,
        qpos=np.arange(11, dtype=np.float64),
        qvel=np.zeros(10),
    )


def _make_cmd() -> ControlCommand:
    return ControlCommand(
        kind="joint_pos",
        joint_pos=np.array([0.1, 0.2, 0.3, 0.4]),
        joint_torque=np.array([1.0, 2.0, 3.0, 4.0]),
    )


# ---------------------------------------------------------------------------
# LoggerObserver
# ---------------------------------------------------------------------------


def test_logger_writes_jsonl_lines(tmp_path: Path):
    obs = LoggerObserver(tmp_path, flush_every=1)
    obs.on_episode_start("ep_001", {"controller": "wbc", "robot": "g1"})
    for i in range(3):
        info = StepInfo(done=(i == 2), success=(True if i == 2 else None), reward=float(i))
        obs.on_step(float(i) * 0.02, _make_state(i * 0.02), _make_intent(i * 0.02), _make_cmd(), info)
    obs.on_episode_end({"final_progress": 0.95})

    log_path = tmp_path / "ep_001.jsonl"
    summary_path = tmp_path / "ep_001_summary.json"
    assert log_path.exists()
    assert summary_path.exists()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 5  # start + 3 steps + end
    start = json.loads(lines[0])
    assert start["_kind"] == "episode_start"
    assert start["episode_id"] == "ep_001"
    assert start["metadata"]["controller"] == "wbc"

    step0 = json.loads(lines[1])
    assert step0["_kind"] == "step"
    assert step0["t"] == pytest.approx(0.0)
    assert step0["intent"]["base_lin_vel"] == [0.5, 0.0]
    assert step0["intent"]["contact_phase"] == "double_support"

    end = json.loads(lines[-1])
    assert end["_kind"] == "episode_end"
    assert end["summary"]["final_progress"] == pytest.approx(0.95)

    summary = json.loads(summary_path.read_text())
    assert summary["final_progress"] == pytest.approx(0.95)


def test_logger_handles_no_episode_start_gracefully(tmp_path: Path):
    obs = LoggerObserver(tmp_path)
    info = StepInfo(done=False)
    # Should be a no-op since on_episode_start was never called.
    obs.on_step(0.0, _make_state(0.0), _make_intent(0.0), _make_cmd(), info)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# RecorderObserver
# ---------------------------------------------------------------------------


def test_recorder_dumps_npz(tmp_path: Path):
    obs = RecorderObserver(tmp_path)
    obs.on_episode_start("ep_002", {})
    for i in range(4):
        info = StepInfo(done=(i == 3), success=(i == 3) or None, reward=float(i), cost=0.1)
        obs.on_step(float(i) * 0.02, _make_state(i * 0.02), _make_intent(i * 0.02), _make_cmd(), info)
    obs.on_episode_end({})

    path = tmp_path / "ep_002.npz"
    assert path.exists()
    data = np.load(path, allow_pickle=True)
    assert data["t"].shape == (4,)
    assert data["qpos"].shape == (4, 11)
    assert data["cmd_joint_pos"].shape == (4, 4)
    assert data["cmd_joint_torque"].shape == (4, 4)
    np.testing.assert_array_equal(data["reward"], np.array([0.0, 1.0, 2.0, 3.0]))
    assert list(data["cmd_kind"]) == ["joint_pos"] * 4


def test_recorder_skip_state(tmp_path: Path):
    obs = RecorderObserver(tmp_path, include_state=False)
    obs.on_episode_start("ep_003", {})
    obs.on_step(0.0, _make_state(0.0), _make_intent(0.0), _make_cmd(), StepInfo())
    obs.on_episode_end({})
    data = np.load(tmp_path / "ep_003.npz", allow_pickle=True)
    assert "qpos" not in data.files
    assert "cmd_joint_pos" in data.files


def test_recorder_pads_variable_width(tmp_path: Path):
    obs = RecorderObserver(tmp_path)
    obs.on_episode_start("ep_004", {})
    cmd_a = ControlCommand(kind="joint_pos", joint_pos=np.array([0.1, 0.2]))
    cmd_b = ControlCommand(kind="joint_pos", joint_pos=np.array([0.1, 0.2, 0.3, 0.4]))
    obs.on_step(0.0, _make_state(0.0), _make_intent(0.0), cmd_a, StepInfo())
    obs.on_step(0.02, _make_state(0.02), _make_intent(0.02), cmd_b, StepInfo())
    obs.on_episode_end({})
    data = np.load(tmp_path / "ep_004.npz", allow_pickle=True)
    cmd_jp = data["cmd_joint_pos"]
    assert cmd_jp.shape == (2, 4)
    assert np.isnan(cmd_jp[0, 2])
    assert cmd_jp[1, 3] == pytest.approx(0.4)


def test_recorder_no_op_without_steps(tmp_path: Path):
    obs = RecorderObserver(tmp_path)
    obs.on_episode_start("ep_empty", {})
    obs.on_episode_end({})
    assert not (tmp_path / "ep_empty.npz").exists()
