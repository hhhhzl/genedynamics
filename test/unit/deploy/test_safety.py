"""Unit tests for the Phase 8 safety filters.

These tests build a synthetic spec stand-in (no MuJoCo required) so they
run on any host. The integration counterpart under
``test/integration/`` exercises the same filters against a real
:class:`MujocoRobotIO` and the G1 MJCF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pytest

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    LocoCommand,
    RobotState,
)
from genedynamics.deploy.safety import (
    CompositeSafetyFilter,
    JointLimitFilter,
    SelfCollisionFilter,
    TorqueLimitFilter,
)
from genedynamics.deploy.safety.cbf_filter import CBFFilter


# ---------------------------------------------------------------------------
# Synthetic spec
# ---------------------------------------------------------------------------


@dataclass
class _StubSpec:
    actuated_joints: Tuple[str, ...] = ("j0", "j1", "j2", "j3")
    joint_range: Dict[str, np.ndarray] = field(
        default_factory=lambda: {
            "j0": np.array([-1.0, 1.0]),
            "j1": np.array([-0.5, 0.5]),
            "j2": np.array([-2.0, 2.0]),
            "j3": np.array([-3.14, 3.14]),
        }
    )
    torque_limit: Dict[str, float] = field(
        default_factory=lambda: {"j0": 10.0, "j1": 5.0, "j2": 20.0, "j3": 15.0}
    )

    @property
    def actuated_qpos_indices(self) -> np.ndarray:
        return np.arange(7, 7 + len(self.actuated_joints), dtype=np.int32)


def _state(qpos: np.ndarray) -> RobotState:
    return RobotState(t=0.0, qpos=qpos, qvel=np.zeros(qpos.size))


# ---------------------------------------------------------------------------
# JointLimitFilter
# ---------------------------------------------------------------------------


def test_joint_limit_passthrough_when_in_range():
    spec = _StubSpec()
    f = JointLimitFilter(spec)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([0.5, 0.0, 1.0, 1.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert not res.intervened
    np.testing.assert_array_equal(res.command.joint_pos, cmd.joint_pos)


def test_joint_limit_clips_and_records_violation():
    spec = _StubSpec()
    f = JointLimitFilter(spec)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([1.5, -0.6, 0.0, 0.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    np.testing.assert_array_equal(
        res.command.joint_pos, np.array([1.0, -0.5, 0.0, 0.0])
    )
    assert res.violations["joint_limit_max_rad"] == pytest.approx(0.5)


def test_joint_limit_margin_shrinks_range():
    spec = _StubSpec()
    f = JointLimitFilter(spec, margin=0.1)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([0.95, 0.0, 0.0, 0.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    assert res.command.joint_pos[0] == pytest.approx(0.9)


def test_joint_limit_skips_non_position_commands():
    spec = _StubSpec()
    f = JointLimitFilter(spec)
    cmd = ControlCommand(kind="loco", loco_cmd=LocoCommand(0.1, 0, 0))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert not res.intervened
    assert res.command is cmd


# ---------------------------------------------------------------------------
# TorqueLimitFilter
# ---------------------------------------------------------------------------


def test_torque_limit_clips_to_force_range():
    spec = _StubSpec()
    f = TorqueLimitFilter(spec, safety_margin=1.0)
    cmd = ControlCommand(kind="torque", joint_torque=np.array([12.0, -6.0, 5.0, 100.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    np.testing.assert_array_equal(
        res.command.joint_torque, np.array([10.0, -5.0, 5.0, 15.0])
    )
    assert res.violations["torque_max_nm"] == pytest.approx(85.0)


def test_torque_limit_safety_margin():
    spec = _StubSpec()
    f = TorqueLimitFilter(spec, safety_margin=0.5)
    cmd = ControlCommand(kind="torque", joint_torque=np.array([10.0, 0.0, 0.0, 0.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    assert res.command.joint_torque[0] == pytest.approx(5.0)


def test_torque_limit_invalid_margin_raises():
    spec = _StubSpec()
    with pytest.raises(ValueError):
        TorqueLimitFilter(spec, safety_margin=1.5)


# ---------------------------------------------------------------------------
# SelfCollisionFilter
# ---------------------------------------------------------------------------


class _StubRobot:
    """Returns the configured distance for any pair lookup."""

    def __init__(self, dist: float) -> None:
        self.dist = dist
        self.calls: list[Tuple[str, str]] = []

    def body_pair_distance(self, a: str, b: str) -> float:
        self.calls.append((a, b))
        return self.dist


def test_self_collision_passthrough_without_robot():
    spec = _StubSpec()
    f = SelfCollisionFilter(spec, robot=None)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.zeros(4))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert not res.intervened


def test_self_collision_no_intervention_when_clear():
    spec = _StubSpec()
    f = SelfCollisionFilter(
        spec,
        robot=_StubRobot(dist=0.10),
        pairs=[("a", "b")],
        min_clearance=0.05,
        retreat_factor=0.5,
    )
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([0.4, 0.4, 0.0, 0.0]))
    res = f.filter(_state(np.array([0.0] * 7 + [0.0, 0.0, 0.0, 0.0])), cmd)
    assert not res.intervened


def test_self_collision_retreats_toward_current_qpos():
    spec = _StubSpec()
    f = SelfCollisionFilter(
        spec,
        robot=_StubRobot(dist=0.005),
        pairs=[("a", "b")],
        min_clearance=0.03,
        retreat_factor=0.6,
    )
    qpos = np.array([0.0] * 7 + [0.0, 0.0, 0.0, 0.0])
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([1.0, 1.0, 0.0, 0.0]))
    res = f.filter(_state(qpos), cmd)
    assert res.intervened
    expected = 0.6 * 0.0 + 0.4 * 1.0
    assert res.command.joint_pos[0] == pytest.approx(expected)
    assert res.command.joint_pos[1] == pytest.approx(expected)
    assert "self_collision_min_dist_m" in res.violations


# ---------------------------------------------------------------------------
# CBFFilter
# ---------------------------------------------------------------------------


def test_cbf_filter_passthrough_when_feasible():
    spec = _StubSpec()
    # Constraint: u[0] >= 0.0  (i.e. A=[1,0,0,0], b=[0])
    A = np.array([[1.0, 0.0, 0.0, 0.0]])
    b = np.array([0.0])

    def barrier(state, cmd):
        return A, b

    f = CBFFilter(spec, barrier_fn=barrier)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([0.5, 0.0, 0.0, 0.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert not res.intervened


def test_cbf_filter_projects_when_infeasible():
    spec = _StubSpec()
    A = np.array([[1.0, 0.0, 0.0, 0.0]])
    b = np.array([0.2])

    def barrier(state, cmd):
        return A, b

    f = CBFFilter(spec, barrier_fn=barrier)
    cmd = ControlCommand(kind="joint_pos", joint_pos=np.array([-0.5, 0.0, 0.0, 0.0]))
    res = f.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    assert res.command.joint_pos[0] >= 0.2 - 1e-3


def test_cbf_filter_requires_barrier_fn():
    spec = _StubSpec()
    with pytest.raises(ValueError):
        CBFFilter(spec, barrier_fn=None)


# ---------------------------------------------------------------------------
# CompositeSafetyFilter
# ---------------------------------------------------------------------------


def test_composite_chains_filters_in_order():
    spec = _StubSpec()
    composite = CompositeSafetyFilter(
        spec,
        filters=[JointLimitFilter(spec), TorqueLimitFilter(spec)],
    )
    cmd = ControlCommand(
        kind="joint_pos",
        joint_pos=np.array([5.0, 0.0, 0.0, 0.0]),
        joint_torque=np.array([100.0, 0.0, 0.0, 0.0]),
    )
    res = composite.filter(_state(np.zeros(11)), cmd)
    assert res.intervened
    assert res.command.joint_pos[0] == pytest.approx(1.0)
    assert res.command.joint_torque[0] == pytest.approx(10.0)
    assert "JointLimitFilter.joint_limit_max_rad" in res.violations
    assert "TorqueLimitFilter.torque_max_nm" in res.violations


def test_composite_reset_propagates():
    spec = _StubSpec()

    class _Counting(JointLimitFilter):
        def __init__(self):
            super().__init__(spec)
            self.resets = 0

        def reset(self):
            self.resets += 1

    a = _Counting()
    b = _Counting()
    composite = CompositeSafetyFilter(spec, filters=[a, b])
    composite.reset()
    assert a.resets == 1
    assert b.resets == 1
