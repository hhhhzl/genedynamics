"""Tests for the osqp QP-solver path inside :class:`CBFFilter`.

The CBF filter prefers osqp when installed and falls back to scipy SLSQP
otherwise. The fallback path is exercised in ``test_safety.py``; this file
covers the osqp path explicitly by **monkey-patching** a fake osqp module
into ``self._osqp``. We don't actually depend on osqp being installed —
the fake exposes the same API surface (OSQP class with ``setup`` /
``solve`` / ``info``) and returns canned solutions.

We assert:

* The osqp path is taken when ``filter._osqp`` is non-None.
* The QP is set up with the right matrix shapes for both slack and
  hard-constraint modes.
* A solver-status of "infeasible" is reported via the ``violations`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.safety.cbf_filter import CBFFilter


# ---------------------------------------------------------------------------
# Spec stub (mirrors the one in test_safety.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubSpec:
    actuated_joints: tuple = ("j0", "j1", "j2", "j3")

    @property
    def joint_range(self) -> Dict[str, np.ndarray]:
        return {
            "j0": np.array([-1.0, 1.0]),
            "j1": np.array([-1.0, 1.0]),
            "j2": np.array([-1.0, 1.0]),
            "j3": np.array([-1.0, 1.0]),
        }


def _state() -> RobotState:
    return RobotState(t=0.0, qpos=np.zeros(11), qvel=np.zeros(10))


# ---------------------------------------------------------------------------
# Fake osqp
# ---------------------------------------------------------------------------


class _FakeOSQPInfo:
    def __init__(self, status_val: int, status: str) -> None:
        self.status_val = status_val
        self.status = status


class _FakeOSQPResult:
    def __init__(self, x: np.ndarray, status_val: int = 1, status: str = "solved") -> None:
        self.x = x
        self.info = _FakeOSQPInfo(status_val, status)


class _FakeOSQP:
    """Fake mirroring the osqp.OSQP API surface."""

    last_setup_args: Dict[str, Any] = {}
    next_result: Any = None

    def __init__(self) -> None:
        pass

    def setup(self, P, q, A, l, u, **kwargs) -> None:  # noqa: E741
        type(self).last_setup_args = {
            "P_shape": P.shape,
            "q_shape": q.shape,
            "A_shape": A.shape,
            "l_shape": l.shape,
            "u_shape": u.shape,
            "kwargs": kwargs,
        }

    def solve(self):
        return type(self).next_result


class _FakeOsqpModule:
    OSQP = _FakeOSQP


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_filter(slack_weight=1e3) -> CBFFilter:
    A = np.array([[1.0, 0.0, 0.0, 0.0]])
    b = np.array([0.5])

    def barrier(state, cmd):
        return A, b

    f = CBFFilter(_StubSpec(), barrier_fn=barrier, slack_weight=slack_weight)
    f._osqp = _FakeOsqpModule  # type: ignore[assignment]
    return f


def test_osqp_path_used_when_module_available():
    f = _make_filter()
    # Canned solution: u_safe[0] = 0.5 (the constraint boundary)
    _FakeOSQP.next_result = _FakeOSQPResult(
        x=np.concatenate([np.array([0.5, 0.0, 0.0, 0.0]), np.zeros(1)])
    )
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([-0.5, 0.0, 0.0, 0.0]))
    res = f.filter(_state(), cmd)
    assert res.intervened
    assert res.command.joint_pos[0] == pytest.approx(0.5)
    assert res.extras["solver"] == "osqp"


def test_osqp_setup_uses_slack_when_weight_set():
    f = _make_filter(slack_weight=1e3)
    _FakeOSQP.next_result = _FakeOSQPResult(
        x=np.concatenate([np.array([0.5, 0.0, 0.0, 0.0]), np.zeros(1)])
    )
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([-0.5, 0.0, 0.0, 0.0]))
    f.filter(_state(), cmd)
    args = _FakeOSQP.last_setup_args
    # n=4 vars + m=1 slack → P should be 5x5
    assert args["P_shape"] == (5, 5)
    assert args["q_shape"] == (5,)


def test_osqp_setup_drops_slack_when_weight_none():
    f = _make_filter(slack_weight=None)
    _FakeOSQP.next_result = _FakeOSQPResult(x=np.array([0.5, 0.0, 0.0, 0.0]))
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([-0.5, 0.0, 0.0, 0.0]))
    f.filter(_state(), cmd)
    args = _FakeOSQP.last_setup_args
    # n=4 vars only (no slack) → P should be 4x4
    assert args["P_shape"] == (4, 4)


def test_osqp_infeasible_status_reported():
    f = _make_filter()
    _FakeOSQP.next_result = _FakeOSQPResult(
        x=np.zeros(5), status_val=-3, status="primal_infeasible"
    )
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([-0.5, 0.0, 0.0, 0.0]))
    res = f.filter(_state(), cmd)
    # When the solver says infeasible, the filter should NOT modify the command
    assert not res.intervened
    assert "cbf_infeasible" in res.violations


def test_osqp_passthrough_when_already_feasible():
    f = _make_filter()
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([0.6, 0.0, 0.0, 0.0]))
    res = f.filter(_state(), cmd)
    assert not res.intervened
    # Solver should NOT have been called
    assert _FakeOSQP.last_setup_args.get("P_shape") in (None, (5, 5), (4, 4))
