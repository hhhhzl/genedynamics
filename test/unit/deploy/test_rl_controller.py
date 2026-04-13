"""Unit tests for the Phase 6 RL controller framework.

These tests use callable-based inference closures so they don't depend on
torch / onnx / jax. They cover:

* :class:`RLController` decimation behavior
* :class:`RLController` last-action latching
* :class:`RLController` telemetry attachment
* :class:`PassthroughObsBuilder` layout
* :class:`JointPosActionMapper` round-trip
* :class:`UnitreeRLGymG1Controller` permutation logic and obs/action layout
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from genedynamics.deploy.controllers.rl import (
    PassthroughRLController,
    RLController,
    UnitreeRLGymG1Controller,
    UnitreeRLGymObsCfg,
)
from genedynamics.deploy.controllers.rl.passthrough import (
    JointPosActionMapper,
    PassthroughObsBuilder,
)
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
)


# ---------------------------------------------------------------------------
# Synthetic G1-shaped spec (matches the spec interface our controllers use)
# ---------------------------------------------------------------------------


_LEFT_LEG = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
)
_RIGHT_LEG = tuple(n.replace("left", "right") for n in _LEFT_LEG)
_ACTUATED = _LEFT_LEG + _RIGHT_LEG


@dataclass
class _MiniSpec:
    """Bare-minimum spec subset that the RL controllers need."""

    actuated_joints: tuple = _ACTUATED
    stand_ctrl: np.ndarray = field(default_factory=lambda: np.zeros(len(_ACTUATED)))

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joints)

    @property
    def actuated_qpos_indices(self) -> np.ndarray:
        # Pretend the floating base occupies qpos[0:7] and actuated joints
        # follow at qpos[7:7+N]
        return np.arange(7, 7 + self.num_actuated, dtype=np.int32)

    @property
    def actuated_dof_indices(self) -> np.ndarray:
        return np.arange(6, 6 + self.num_actuated, dtype=np.int32)

    def clip_to_joint_limits(self, vec: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        return np.asarray(vec, dtype=np.float64)


def _make_state(spec: _MiniSpec, *, q_offset: float = 0.0) -> RobotState:
    nq = 7 + spec.num_actuated
    nv = 6 + spec.num_actuated
    qpos = np.zeros(nq, dtype=np.float64)
    qpos[2] = 0.79  # base z
    qpos[3] = 1.0   # quat w
    qpos[7:] = q_offset
    qvel = np.zeros(nv, dtype=np.float64)
    return RobotState(
        t=0.0,
        qpos=qpos,
        qvel=qvel,
        base_pose=qpos[:7].copy(),
        base_twist=np.zeros(6),
    )


def _make_intent() -> Intent:
    return Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=0.79,
        base_lin_vel=np.array([0.5, 0.0]),
        base_yaw_rate=0.0,
        extras={"dt": 0.005},
    )


# ---------------------------------------------------------------------------
# PassthroughObsBuilder layout
# ---------------------------------------------------------------------------


def test_passthrough_obs_builder_layout():
    spec = _MiniSpec()
    builder = PassthroughObsBuilder(spec)
    n = spec.num_actuated
    expected = 3 + 3 + 3 + 3 + n + n + n
    assert builder.obs_dim == expected
    builder.reset()
    obs = builder.build(_make_state(spec), _make_intent())
    assert obs.shape == (expected,)
    assert obs.dtype == np.float32


def test_passthrough_obs_builder_history_stacking():
    spec = _MiniSpec()
    builder = PassthroughObsBuilder(spec, history_len=4)
    builder.reset()
    obs = builder.build(_make_state(spec), _make_intent())
    n = spec.num_actuated
    single = 3 + 3 + 3 + 3 + n + n + n
    assert obs.shape == (single * 4,)
    # First call duplicates the single frame 4 times
    assert np.allclose(obs[:single], obs[single : 2 * single])


def test_passthrough_obs_builder_remembers_last_action():
    spec = _MiniSpec()
    builder = PassthroughObsBuilder(spec)
    builder.reset()
    builder.remember_last_action(np.full(spec.num_actuated, 0.42, dtype=np.float32))
    obs = builder.build(_make_state(spec), _make_intent())
    n = spec.num_actuated
    last_action_slice = obs[-n:]
    assert np.allclose(last_action_slice, 0.42)


# ---------------------------------------------------------------------------
# JointPosActionMapper
# ---------------------------------------------------------------------------


def test_joint_pos_action_mapper_round_trip():
    spec = _MiniSpec()
    mapper = JointPosActionMapper(spec, action_scale=0.5, kp=80.0, kd=2.0)
    action = np.full(spec.num_actuated, 0.2, dtype=np.float32)
    cmd = mapper.map(action)
    assert isinstance(cmd, ControlCommand)
    assert cmd.kind == "joint_pos"
    # q_target = stand + 0.2 * 0.5 = 0.1
    assert np.allclose(cmd.joint_pos, 0.1)
    assert np.allclose(cmd.kp, 80.0)
    assert np.allclose(cmd.kd, 2.0)


def test_joint_pos_action_mapper_rejects_wrong_shape():
    spec = _MiniSpec()
    mapper = JointPosActionMapper(spec)
    with pytest.raises(ValueError, match="action_dim"):
        mapper.map(np.zeros(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# RLController decimation + telemetry
# ---------------------------------------------------------------------------


def test_rl_controller_decimation_calls_policy_at_decimation_boundary():
    spec = _MiniSpec()
    call_count = {"n": 0}

    def policy(obs):
        call_count["n"] += 1
        return np.zeros(spec.num_actuated, dtype=np.float32)

    ctrl = PassthroughRLController(
        spec=spec,
        inference_fn=policy,
        policy_hz=50.0,
        control_hz=200.0,  # decimation = 4
    )
    ctrl.reset()

    state = _make_state(spec)
    intent = _make_intent()
    for _ in range(10):
        ctrl.act(state, intent)
    # Decimation = 4 → policy called at ticks 0, 4, 8 → 3 times
    assert call_count["n"] == 3


def test_rl_controller_no_decimation_when_policy_hz_equals_control_hz():
    spec = _MiniSpec()
    call_count = {"n": 0}

    def policy(obs):
        call_count["n"] += 1
        return np.zeros(spec.num_actuated, dtype=np.float32)

    ctrl = PassthroughRLController(
        spec=spec, inference_fn=policy, policy_hz=200.0, control_hz=200.0
    )
    ctrl.reset()
    for _ in range(5):
        ctrl.act(_make_state(spec), _make_intent())
    assert call_count["n"] == 5


def test_rl_controller_latches_command_between_decimated_steps():
    spec = _MiniSpec()
    call_count = {"n": 0}

    def policy(obs):
        # Return a different action each time
        v = float(call_count["n"])
        call_count["n"] += 1
        return np.full(spec.num_actuated, v, dtype=np.float32)

    ctrl = PassthroughRLController(
        spec=spec, inference_fn=policy, policy_hz=50.0, control_hz=200.0
    )
    ctrl.reset()

    cmds = [ctrl.act(_make_state(spec), _make_intent()) for _ in range(8)]
    # Ticks 0,1,2,3 share the policy output from tick 0; ticks 4,5,6,7 share the next
    # action 0 → q_target = 0.0 * 0.25 = 0.0
    # action 1 → q_target = 1.0 * 0.25 = 0.25
    assert np.allclose(cmds[0].joint_pos, 0.0)
    assert np.allclose(cmds[3].joint_pos, 0.0)
    assert np.allclose(cmds[4].joint_pos, 0.25)
    assert np.allclose(cmds[7].joint_pos, 0.25)


def test_rl_controller_attaches_telemetry_to_extras():
    spec = _MiniSpec()
    ctrl = PassthroughRLController(
        spec=spec,
        inference_fn=lambda obs: np.zeros(spec.num_actuated, dtype=np.float32),
        policy_hz=50.0,
        control_hz=200.0,
    )
    ctrl.reset()
    cmd = ctrl.act(_make_state(spec), _make_intent())
    assert "rl" in cmd.extras
    rl = cmd.extras["rl"]
    assert "obs" in rl
    assert "action" in rl
    assert "decimation" in rl
    assert rl["decimation"] == 4


def test_rl_controller_validates_action_shape():
    spec = _MiniSpec()

    def bad_policy(obs):
        return np.zeros(7, dtype=np.float32)  # wrong size

    ctrl = PassthroughRLController(
        spec=spec, inference_fn=bad_policy, policy_hz=50.0, control_hz=200.0
    )
    ctrl.reset()
    with pytest.raises(ValueError, match="expected"):
        ctrl.act(_make_state(spec), _make_intent())


def test_rl_controller_requires_artifact_or_inference_fn():
    spec = _MiniSpec()
    with pytest.raises(ValueError, match="artifact or an inference_fn"):
        RLController(
            spec=spec,
            obs_builder=PassthroughObsBuilder(spec),
            action_mapper=JointPosActionMapper(spec),
            policy_hz=50.0,
            control_hz=200.0,
        )


# ---------------------------------------------------------------------------
# Unitree RL Gym adapter — permutation + obs/action layout
# ---------------------------------------------------------------------------


def _make_unitree_cfg(spec: _MiniSpec, scrambled: bool = True) -> UnitreeRLGymObsCfg:
    """Build a synthetic training cfg with a non-trivial DoF permutation."""
    if scrambled:
        # Reorder the policy DoF names so robot_idx ≠ policy_idx
        policy_dof_names = (_RIGHT_LEG + _LEFT_LEG)  # right first, then left
    else:
        policy_dof_names = tuple(spec.actuated_joints)
    n = len(policy_dof_names)
    return UnitreeRLGymObsCfg(
        history_len=1,
        lin_vel_scale=2.0,
        ang_vel_scale=0.25,
        dof_pos_scale=1.0,
        dof_vel_scale=0.05,
        action_scale=0.25,
        default_dof_pos=np.full(n, 0.1, dtype=np.float32),
        policy_dof_names=policy_dof_names,
        kp=np.full(n, 80.0, dtype=np.float64),
        kd=np.full(n, 2.0, dtype=np.float64),
    )


def test_unitree_rlgym_permutation_inverts_correctly():
    spec = _MiniSpec()
    cfg = _make_unitree_cfg(spec, scrambled=True)
    perm = UnitreeRLGymG1Controller._build_dof_permutation(spec, cfg.policy_dof_names)
    # round-trip: name → policy_idx → robot_idx → name should match the start
    for policy_idx, name in enumerate(cfg.policy_dof_names):
        robot_idx = perm[policy_idx]
        assert spec.actuated_joints[robot_idx] == name


def test_unitree_rlgym_raises_on_unknown_policy_joint():
    spec = _MiniSpec()
    bad_names = tuple(_LEFT_LEG) + ("nonexistent_joint",)
    with pytest.raises(ValueError, match="not found in robot spec"):
        UnitreeRLGymG1Controller._build_dof_permutation(spec, bad_names)


def test_unitree_rlgym_action_mapper_permutes_back_to_robot_order():
    spec = _MiniSpec()
    cfg = _make_unitree_cfg(spec, scrambled=True)
    ctrl = UnitreeRLGymG1Controller(
        spec=spec,
        cfg=cfg,
        inference_fn=lambda obs: np.zeros(spec.num_actuated, dtype=np.float32),
        policy_hz=50.0,
        control_hz=200.0,
    )
    # Hand-craft an action where each entry is its policy index, so we can
    # verify it gets placed at the correct robot slot.
    action = np.arange(spec.num_actuated, dtype=np.float32)
    cmd = ctrl.action_mapper.map(action)
    # q_target_robot[robot_idx] should equal default + policy_idx * action_scale
    perm = ctrl.action_mapper.perm_policy_to_robot
    for policy_idx, robot_idx in enumerate(perm):
        expected = 0.1 + policy_idx * 0.25
        assert cmd.joint_pos[robot_idx] == pytest.approx(expected, abs=1e-6)


def test_unitree_rlgym_obs_layout_matches_dim():
    spec = _MiniSpec()
    cfg = _make_unitree_cfg(spec)
    ctrl = UnitreeRLGymG1Controller(
        spec=spec,
        cfg=cfg,
        inference_fn=lambda obs: np.zeros(spec.num_actuated, dtype=np.float32),
        policy_hz=50.0,
        control_hz=200.0,
    )
    n = spec.num_actuated
    expected = (3 + 3 + 3 + 3 + n + n + n) * cfg.history_len
    assert ctrl.obs_builder.obs_dim == expected
    obs = ctrl.obs_builder.build(_make_state(spec), _make_intent())
    assert obs.shape == (expected,)


def test_unitree_rlgym_end_to_end_with_callable_policy():
    spec = _MiniSpec()
    cfg = _make_unitree_cfg(spec)

    captured = {"obs": None}

    def policy(obs):
        captured["obs"] = obs.copy()
        # Return a constant non-zero action
        return np.full(spec.num_actuated, 0.3, dtype=np.float32)

    ctrl = UnitreeRLGymG1Controller(
        spec=spec, cfg=cfg, inference_fn=policy, policy_hz=200.0, control_hz=200.0
    )
    ctrl.reset()
    cmd = ctrl.act(_make_state(spec), _make_intent())

    # Policy received the right-shape obs
    n = spec.num_actuated
    assert captured["obs"].shape == ((3 + 3 + 3 + 3 + n + n + n),)
    # Output is a joint_pos command with default + 0.3 * 0.25 = 0.175
    assert isinstance(cmd, ControlCommand)
    assert cmd.kind == "joint_pos"
    assert np.allclose(cmd.joint_pos, 0.175, atol=1e-6)
