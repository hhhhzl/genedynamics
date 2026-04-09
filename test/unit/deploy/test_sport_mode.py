"""Unit tests for the sport-mode controller.

These tests use synthetic stand-ins for the heavy dependencies (MuJoCo,
:class:`G1RobotModel`, :class:`MujocoRobotIO`) so they run in any
environment, including the host fedguide env where mujoco can't import.
The corresponding integration tests under ``test/integration/`` exercise
the same code paths against a real MuJoCo G1 model.

Coverage:

* :class:`SportModeController` correctly delegates to a :class:`LocoClient`,
  passes the live :class:`RobotState` through, and merges the leg targets
  with upper-body joint hints from either ``extras["humanoid_tasks"]``
  (legacy) or ``extras["upper_body_targets"]`` (new path used by the
  spark-RL diagnose).
* Kp/Kd vectors are split by sub-assembly (legs vs waist+arms).
* The controller still works when no upper-body source is provided
  (falls back to the spec's stand pose).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pytest

from genedynamics.deploy.controllers.sport_mode import (
    LocoClient,
    SportModeController,
)
from genedynamics.deploy.followers.humanoid.task_spec import (
    ArmJointTask,
    FootTask,
    HumanoidTaskSpec,
    PelvisTask,
)
from genedynamics.deploy.followers.common.plan_schema import CorridorPlanFrame
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    LocoCommand,
    RobotState,
)


# ---------------------------------------------------------------------------
# Synthetic spec / IO — minimal subsets of the real classes
# ---------------------------------------------------------------------------


_LEFT_LEG = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
_RIGHT_LEG = tuple(name.replace("left", "right") for name in _LEFT_LEG)
_WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
_LEFT_ARM = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)
_RIGHT_ARM = tuple(name.replace("left", "right") for name in _LEFT_ARM)
_ACTUATED = _LEFT_LEG + _RIGHT_LEG + _WAIST + _LEFT_ARM + _RIGHT_ARM


@dataclass
class _FakeSpec:
    actuated_joints: tuple = _ACTUATED
    left_leg_joints: tuple = _LEFT_LEG
    right_leg_joints: tuple = _RIGHT_LEG
    waist_joints: tuple = _WAIST
    left_arm_joints: tuple = _LEFT_ARM
    right_arm_joints: tuple = _RIGHT_ARM
    stand_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(len(_ACTUATED)))

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joints)

    def clip_to_joint_limits(self, vec: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64)


@dataclass
class _FakeIO:
    spec: _FakeSpec = field(default_factory=_FakeSpec)
    _state: Optional[RobotState] = None

    def get_state(self) -> RobotState:
        return self._state or RobotState(
            t=0.0,
            qpos=np.zeros(36),
            qvel=np.zeros(35),
            base_pose=np.array([0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]),
            base_twist=np.zeros(6),
        )


@dataclass
class _ScriptedLocoClient:
    """LocoClient that records what it was asked and returns fixed angles."""

    nominal_step_period: float = 0.6
    targets: dict[str, float] = field(default_factory=dict)
    reset_calls: list = field(default_factory=list)
    step_calls: list = field(default_factory=list)
    _phase: float = 0.0
    _swing: Optional[str] = None

    def reset(self, pelvis_world: np.ndarray, pelvis_yaw: float = 0.0) -> None:
        self.reset_calls.append((pelvis_world.copy(), float(pelvis_yaw)))
        self._phase = 0.0
        self._swing = "left"

    def step(
        self,
        cmd: LocoCommand,
        dt: float,
        pelvis_world: np.ndarray,
        pelvis_yaw: float,
        state: Optional[RobotState] = None,
    ) -> Mapping[str, float]:
        self.step_calls.append((cmd, dt, pelvis_world.copy(), float(pelvis_yaw), state))
        if not self.targets:
            self.targets = {
                **{name: 0.1 for name in _LEFT_LEG},
                **{name: -0.1 for name in _RIGHT_LEG},
            }
        return self.targets

    @property
    def swing_foot(self):
        return self._swing

    @property
    def phase(self):
        return self._phase


# ---------------------------------------------------------------------------
# Intent factories
# ---------------------------------------------------------------------------


def _make_intent_with_humanoid_tasks() -> Intent:
    """Legacy path: upper-body hints come from a HumanoidTaskSpec."""
    base = Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=0.79,
        base_lin_vel=np.array([0.3, 0.0]),
        base_yaw_rate=0.0,
        base_pos_xy=np.zeros(2),
        extras={"dt": 0.02},
    )
    frame = CorridorPlanFrame(
        time_sec=0.0, raw_state=np.zeros(14),
        x=0.0, y=0.0, psi=0.0, h=0.79, psi_torso=0.5,
        a_left=0.7, a_right=0.7, p_left=0.0, p_right=0.0,
        v_x=0.3, v_y=0.0, omega=0.0, h_dot=0.0, psi_dot_torso=0.0,
    )
    tasks = HumanoidTaskSpec(
        plan_frame=frame,
        pelvis=PelvisTask(position_world=np.array([0.0, 0.0, 0.79]), yaw_world=0.0),
        torso_yaw=0.5,
        left_foot=FootTask(position_world=np.array([0.0, 0.1, 0.0]), velocity_world=np.zeros(3),
                           yaw_world=0.0, in_contact=True),
        right_foot=FootTask(position_world=np.array([0.0, -0.1, 0.0]), velocity_world=np.zeros(3),
                            yaw_world=0.0, in_contact=True),
        left_arm=ArmJointTask(joint_targets={}),
        right_arm=ArmJointTask(joint_targets={}),
        joint_hints={
            "waist_yaw_joint": 0.5,
            "left_shoulder_roll_joint": -0.4,
            "left_elbow_joint": 1.5,
            "right_shoulder_roll_joint": 0.4,
            "right_elbow_joint": 1.5,
        },
        extras={},
    )
    base.extras["humanoid_tasks"] = tasks
    return base


def _make_intent_with_upper_body_targets() -> Intent:
    """New path: upper-body hints come from a lightweight ``UpperBodyTargets``."""
    base = Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=0.79,
        base_lin_vel=np.array([0.3, 0.0]),
        base_yaw_rate=0.0,
        base_pos_xy=np.zeros(2),
        extras={"dt": 0.02},
    )

    @dataclass
    class _StubUpperBodyTargets:
        joint_hints: dict

    base.extras["upper_body_targets"] = _StubUpperBodyTargets(
        joint_hints={
            "waist_yaw_joint": 0.7,
            "left_elbow_joint": 1.2,
            "right_elbow_joint": 1.2,
        }
    )
    return base


def _make_intent_bare() -> Intent:
    return Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=0.79,
        base_lin_vel=np.array([0.3, 0.0]),
        base_yaw_rate=0.0,
        base_pos_xy=np.zeros(2),
        extras={"dt": 0.02},
    )


# ---------------------------------------------------------------------------
# SportModeController — composes scripted LocoClient + fake IO
# ---------------------------------------------------------------------------


def test_sport_mode_controller_calls_loco_client_with_intent_velocity_and_state():
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    ctrl = SportModeController(io, loco_client=loco)
    ctrl.reset()

    intent = _make_intent_with_humanoid_tasks()
    state = io.get_state()
    cmd = ctrl.act(state, intent)

    assert len(loco.step_calls) == 1
    sent_cmd, sent_dt, _, _, sent_state = loco.step_calls[0]
    assert sent_cmd.vx == pytest.approx(0.3)
    assert sent_dt == pytest.approx(0.02)
    # Crucial: the controller forwards the live RobotState so RL clients
    # can build their observation vector from joint qpos/qvel.
    assert sent_state is state
    assert isinstance(cmd, ControlCommand)
    assert cmd.kind == "joint_pos"
    assert cmd.joint_pos.shape == (len(_ACTUATED),)


def test_sport_mode_controller_merges_leg_targets_with_humanoid_tasks_hints():
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    loco.targets = {
        "left_hip_pitch_joint": 0.42,
        "right_hip_pitch_joint": -0.42,
    }
    ctrl = SportModeController(io, loco_client=loco)
    ctrl.reset()
    cmd = ctrl.act(io.get_state(), _make_intent_with_humanoid_tasks())

    spec = io.spec
    q = cmd.joint_pos
    assert q[spec.actuated_joints.index("left_hip_pitch_joint")] == pytest.approx(0.42)
    assert q[spec.actuated_joints.index("right_hip_pitch_joint")] == pytest.approx(-0.42)
    assert q[spec.actuated_joints.index("waist_yaw_joint")] == pytest.approx(0.5)
    assert q[spec.actuated_joints.index("left_shoulder_roll_joint")] == pytest.approx(-0.4)
    assert q[spec.actuated_joints.index("left_elbow_joint")] == pytest.approx(1.5)
    assert q[spec.actuated_joints.index("right_elbow_joint")] == pytest.approx(1.5)


def test_sport_mode_controller_merges_leg_targets_with_upper_body_targets_hints():
    """The new path used by diagnose_sport_mode_corridor.py — direct
    UpperBodyTargets in extras['upper_body_targets']."""
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    ctrl = SportModeController(io, loco_client=loco)
    ctrl.reset()
    cmd = ctrl.act(io.get_state(), _make_intent_with_upper_body_targets())

    spec = io.spec
    q = cmd.joint_pos
    assert q[spec.actuated_joints.index("waist_yaw_joint")] == pytest.approx(0.7)
    assert q[spec.actuated_joints.index("left_elbow_joint")] == pytest.approx(1.2)
    assert q[spec.actuated_joints.index("right_elbow_joint")] == pytest.approx(1.2)


def test_sport_mode_controller_kp_kd_are_split_by_subassembly():
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    ctrl = SportModeController(
        io,
        loco_client=loco,
        leg_kp=120.0,
        leg_kd=5.0,
        upper_body_kp=60.0,
        upper_body_kd=2.0,
    )
    ctrl.reset()
    cmd = ctrl.act(io.get_state(), _make_intent_with_humanoid_tasks())

    spec = io.spec
    leg_idx = spec.actuated_joints.index("left_hip_pitch_joint")
    waist_idx = spec.actuated_joints.index("waist_yaw_joint")
    arm_idx = spec.actuated_joints.index("left_elbow_joint")

    assert cmd.kp[leg_idx] == pytest.approx(120.0)
    assert cmd.kp[waist_idx] == pytest.approx(60.0)
    assert cmd.kp[arm_idx] == pytest.approx(60.0)
    assert cmd.kd[leg_idx] == pytest.approx(5.0)
    assert cmd.kd[arm_idx] == pytest.approx(2.0)


def test_sport_mode_controller_works_without_upper_body_source_in_intent():
    """Sanity: with neither hint source, upper-body joints stay at stand."""
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    ctrl = SportModeController(io, loco_client=loco)
    ctrl.reset()
    cmd = ctrl.act(io.get_state(), _make_intent_bare())
    # Stand pose is zeros for our fake spec.
    assert cmd.joint_pos[io.spec.actuated_joints.index("waist_yaw_joint")] == 0.0
    assert cmd.joint_pos[io.spec.actuated_joints.index("left_elbow_joint")] == 0.0


def test_sport_mode_controller_extras_carry_diagnostics():
    io = _FakeIO()
    loco = _ScriptedLocoClient()
    ctrl = SportModeController(io, loco_client=loco)
    ctrl.reset()
    cmd = ctrl.act(io.get_state(), _make_intent_with_humanoid_tasks())
    assert "loco_cmd" in cmd.extras
    assert "swing_foot" in cmd.extras
    assert "phase" in cmd.extras
    assert isinstance(cmd.extras["loco_cmd"], LocoCommand)
