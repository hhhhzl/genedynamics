"""Unit tests for the deploy/tasks/ ExecutionTask layer."""

from __future__ import annotations

import numpy as np
import pytest

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.io.stub_io import StubRobotIO
from genedynamics.deploy.tasks.base import BaseExecutionTask
from genedynamics.deploy.tasks.corridor_follow import CorridorFollowTask


def _state(x: float, y: float) -> RobotState:
    qpos = np.zeros(11)
    qpos[0] = x
    qpos[1] = y
    qpos[3] = 1.0  # quat w
    return RobotState(t=0.0, qpos=qpos, qvel=np.zeros(10), base_pose=qpos[:7].copy())


def _intent() -> Intent:
    return Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=0.7,
        base_lin_vel=np.zeros(2),
        base_yaw_rate=0.0,
    )


def _cmd() -> ControlCommand:
    return ControlCommand(kind="joint_pos", joint_pos=np.zeros(4))


# ---------------------------------------------------------------------------
# BaseExecutionTask
# ---------------------------------------------------------------------------


def test_base_task_never_done():
    task = BaseExecutionTask()
    task.reset(io=None)
    for i in range(5):
        info = task.step(_state(0, 0), _intent(), _cmd())
        assert not info.done
    assert task.summary()["steps"] == 5


# ---------------------------------------------------------------------------
# CorridorFollowTask
# ---------------------------------------------------------------------------


def test_corridor_follow_signals_done_at_goal():
    task = CorridorFollowTask(goal_xy=(2.0, 0.0), goal_tolerance_m=0.1, max_steps=100)
    io = StubRobotIO(initial_qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]))
    io.reset()
    task.reset(io)
    # Step away from goal — not done
    info = task.step(_state(0.5, 0.0), _intent(), _cmd())
    assert not info.done
    assert info.metrics["distance_to_goal"] == pytest.approx(1.5)
    # Step into goal sphere — done + success
    info = task.step(_state(2.05, 0.0), _intent(), _cmd())
    assert info.done
    assert info.success is True
    summary = task.summary()
    assert summary["succeeded"] is True
    assert summary["best_distance_m"] < 0.1


def test_corridor_follow_max_steps_timeout():
    task = CorridorFollowTask(goal_xy=(10.0, 0.0), goal_tolerance_m=0.1, max_steps=3)
    io = StubRobotIO()
    io.reset()
    task.reset(io)
    info1 = task.step(_state(0, 0), _intent(), _cmd())
    info2 = task.step(_state(0, 0), _intent(), _cmd())
    info3 = task.step(_state(0, 0), _intent(), _cmd())
    assert not info1.done
    assert not info2.done
    assert info3.done
    assert info3.success is False


def test_corridor_follow_progress_monotone_in_distance_decrease():
    task = CorridorFollowTask(goal_xy=(4.0, 0.0), goal_tolerance_m=0.1, max_steps=100)
    io = StubRobotIO(initial_qpos=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]))
    io.reset()
    task.reset(io)
    p1 = task.step(_state(1.0, 0.0), _intent(), _cmd()).metrics["progress"]
    p2 = task.step(_state(2.0, 0.0), _intent(), _cmd()).metrics["progress"]
    p3 = task.step(_state(3.0, 0.0), _intent(), _cmd()).metrics["progress"]
    assert p1 < p2 < p3
    assert p3 == pytest.approx(0.75, abs=0.01)


def test_corridor_follow_set_goal_late_binding():
    task = CorridorFollowTask(goal_tolerance_m=0.1, max_steps=10)
    io = StubRobotIO()
    io.reset()
    task.reset(io)
    task.set_goal((1.0, 0.0))
    info = task.step(_state(0.95, 0.0), _intent(), _cmd())
    assert info.done
    assert info.success is True


def test_corridor_follow_no_goal_yields_zero_progress():
    task = CorridorFollowTask(max_steps=10)
    io = StubRobotIO()
    io.reset()
    task.reset(io)
    info = task.step(_state(1.0, 0.0), _intent(), _cmd())
    assert info.metrics["progress"] == 0.0
    assert "distance_to_goal" not in info.metrics
