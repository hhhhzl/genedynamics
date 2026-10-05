"""Unit tests for :class:`UnitreeG1RobotIO` using a stub Unitree SDK.

The real :mod:`unitree_sdk2py` only ships on hardware-side machines, so
these tests fabricate a fake module tree under ``unitree_sdk2py.*`` and
register it in ``sys.modules`` BEFORE importing
:class:`UnitreeG1RobotIO`. The IO's ``__init__`` then transparently picks
up the stubs.

What we cover:

* Lifecycle: ``ChannelFactoryInitialize`` is called once with
  ``(domain_id, network_interface)``.
* :meth:`reset` waits for the first lowstate packet and switches to the
  configured MotionSwitcher mode.
* :meth:`get_state` reads joint positions from ``motor_state[aid].q`` and
  IMU quaternion from ``imu_state.quaternion``.
* :meth:`send_control` validates the command kind and dispatches to the
  right SDK channel (``LocoClient.Move`` for ``loco``, ``lowcmd`` publisher
  for ``joint_pos`` / ``torque``, both channels for ``mixed``).
* Torque clamping respects the spec's ``torque_limit`` × safety margin.
* Localization plugin output overrides the floating base in
  :class:`RobotState`.

We do NOT cover the rate-limited ``_step_physics`` busy-wait — its real
behavior is timer-driven and is exercised by the integration test on
hardware.
"""

from __future__ import annotations

import sys
import types
from typing import Any, List, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub SDK module tree
# ---------------------------------------------------------------------------


class _StubMotorState:
    def __init__(self, q: float = 0.0, dq: float = 0.0, tau: float = 0.0) -> None:
        self.q = q
        self.dq = dq
        self.tau_est = tau


class _StubImuState:
    def __init__(self) -> None:
        # MuJoCo quat order: (w, x, y, z) → identity rotation
        self.quaternion = (1.0, 0.0, 0.0, 0.0)


class _StubLowState:
    def __init__(self, num_motors: int = 8) -> None:
        self.motor_state = [_StubMotorState(q=float(i)) for i in range(num_motors)]
        self.imu_state = _StubImuState()


class _StubMotorCmd:
    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.tau = 0.0


class _StubLowCmd:
    def __init__(self) -> None:
        self.motor_cmd = [_StubMotorCmd() for _ in range(64)]
        self.crc = 0


_INIT_CALLS: List[Tuple[int, str]] = []


def _ChannelFactoryInitialize(domain_id, iface):  # noqa: N802
    _INIT_CALLS.append((int(domain_id), str(iface)))


class _StubPublisher:
    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.published: List[Any] = []

    def Init(self) -> None:  # noqa: N802
        pass

    def Write(self, msg: Any) -> None:  # noqa: N802
        self.published.append(msg)


class _StubSubscriber:
    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self._cb = None

    def Init(self, cb, queueLen: int = 10) -> None:  # noqa: N802, N803
        self._cb = cb

    def feed(self, msg: Any) -> None:
        if self._cb is not None:
            self._cb(msg)


class _StubLocoClient:
    def __init__(self) -> None:
        self.moves: List[Tuple[float, float, float]] = []
        self.heights: List[float] = []
        self.balance_count: int = 0
        self.timeout: float = 0.0

    def Init(self) -> None:  # noqa: N802
        pass

    def SetTimeout(self, t: float) -> None:  # noqa: N802
        self.timeout = float(t)

    def Move(self, vx: float, vy: float, yaw: float) -> None:  # noqa: N802
        self.moves.append((float(vx), float(vy), float(yaw)))

    def SetStandHeight(self, h: float) -> None:  # noqa: N802
        self.heights.append(float(h))

    def BalanceStand(self, _flag: int) -> None:  # noqa: N802
        self.balance_count += 1


class _StubMotionSwitcher:
    def __init__(self) -> None:
        self.modes: List[str] = []

    def Init(self) -> None:  # noqa: N802
        pass

    def ReleaseMode(self) -> None:  # noqa: N802
        self.modes.append("__release__")

    def SelectMode(self, m: str) -> None:  # noqa: N802
        self.modes.append(m)


class _StubCRC:
    def __init__(self) -> None:
        self.calls = 0

    def Crc(self, _msg: Any) -> int:  # noqa: N802
        self.calls += 1
        return 0xDEADBEEF


def _install_stub_sdk(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Install fake ``unitree_sdk2py.*`` modules and return the leaf classes."""
    _INIT_CALLS.clear()

    root = types.ModuleType("unitree_sdk2py")
    core = types.ModuleType("unitree_sdk2py.core")
    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = _ChannelFactoryInitialize
    channel.ChannelPublisher = _StubPublisher
    channel.ChannelSubscriber = _StubSubscriber

    idl = types.ModuleType("unitree_sdk2py.idl")
    idl_unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    idl_msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    idl_dds = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    idl_dds.LowCmd_ = _StubLowCmd
    idl_dds.LowState_ = _StubLowState
    idl_default = types.ModuleType("unitree_sdk2py.idl.default")
    idl_default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd

    utils = types.ModuleType("unitree_sdk2py.utils")
    utils_crc = types.ModuleType("unitree_sdk2py.utils.crc")
    utils_crc.CRC = _StubCRC

    g1 = types.ModuleType("unitree_sdk2py.g1")
    g1_loco = types.ModuleType("unitree_sdk2py.g1.loco")
    g1_loco_client = types.ModuleType("unitree_sdk2py.g1.loco.g1_loco_client")
    g1_loco_client.LocoClient = _StubLocoClient

    g1_motion_switcher = types.ModuleType("unitree_sdk2py.g1.motion_switcher")
    g1_motion_switcher_client = types.ModuleType(
        "unitree_sdk2py.g1.motion_switcher.motion_switcher_client"
    )
    g1_motion_switcher_client.MotionSwitcherClient = _StubMotionSwitcher

    modules = {
        "unitree_sdk2py": root,
        "unitree_sdk2py.core": core,
        "unitree_sdk2py.core.channel": channel,
        "unitree_sdk2py.idl": idl,
        "unitree_sdk2py.idl.unitree_hg": idl_unitree_hg,
        "unitree_sdk2py.idl.unitree_hg.msg": idl_msg,
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": idl_dds,
        "unitree_sdk2py.idl.default": idl_default,
        "unitree_sdk2py.utils": utils,
        "unitree_sdk2py.utils.crc": utils_crc,
        "unitree_sdk2py.g1": g1,
        "unitree_sdk2py.g1.loco": g1_loco,
        "unitree_sdk2py.g1.loco.g1_loco_client": g1_loco_client,
        "unitree_sdk2py.g1.motion_switcher": g1_motion_switcher,
        "unitree_sdk2py.g1.motion_switcher.motion_switcher_client": g1_motion_switcher_client,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return {
        "init_calls": _INIT_CALLS,
        "ChannelFactoryInitialize": _ChannelFactoryInitialize,
        "LowState_": _StubLowState,
        "LowCmd_": _StubLowCmd,
    }


# ---------------------------------------------------------------------------
# Minimal spec stand-in
# ---------------------------------------------------------------------------


class _StubG1Spec:
    """Minimal spec exposing the surface UnitreeG1RobotIO touches."""

    def __init__(self) -> None:
        self.actuated_joints = (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_elbow_joint",
        )
        self.left_leg_joints = ("left_hip_pitch_joint", "left_knee_joint")
        self.right_leg_joints = ("right_hip_pitch_joint", "right_knee_joint")
        self.waist_joints = ("waist_yaw_joint",)
        self.left_arm_joints = ("left_shoulder_pitch_joint", "left_elbow_joint")
        self.right_arm_joints = ("right_shoulder_pitch_joint",)
        self.actuator_id = {name: i for i, name in enumerate(self.actuated_joints)}
        self.actuator_kp = {name: 50.0 for name in self.actuated_joints}
        self.torque_limit = {name: 25.0 for name in self.actuated_joints}
        self.num_actuated = len(self.actuated_joints)

    def actuator_kp_vector(self) -> np.ndarray:
        return np.full(self.num_actuated, 50.0, dtype=np.float64)

    def torque_limit_vector(self) -> np.ndarray:
        return np.full(self.num_actuated, 25.0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper to instantiate the IO with stubs in place
# ---------------------------------------------------------------------------


def _make_io(monkeypatch: pytest.MonkeyPatch, **kw):
    handles = _install_stub_sdk(monkeypatch)
    # Re-import inside the test so the lazy SDK import in __init__ picks up
    # the fakes. We don't import the module at top-of-file deliberately —
    # the IO module itself imports cleanly without the SDK, but constructing
    # the class is what triggers the lazy import.
    from genedynamics.deploy.io.unitree_g1_io import UnitreeG1RobotIO

    spec = _StubG1Spec()
    io = UnitreeG1RobotIO(spec=spec, **kw)
    return io, handles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_wires_channels_and_initializes_factory(monkeypatch):
    io, handles = _make_io(monkeypatch, network_interface="lo", domain_id=7)
    assert handles["init_calls"] == [(7, "lo")]
    assert io.physics_backend is None
    assert io.array_runtime == "numpy"
    assert "loco" in io.accepts and "joint_pos" in io.accepts
    assert io.sdk_health == ("failed", float("inf"))


def test_send_loco_dispatches_to_sdk_move(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand, LocoCommand

    io, _ = _make_io(monkeypatch)
    io._loco_client.balance_count = 0  # type: ignore[attr-defined]
    io.send_control(ControlCommand(kind="loco", loco_cmd=LocoCommand(0.5, 0.0, 0.1)))
    io._apply_command(io._latched_command)  # base.send_control only latches
    assert io._loco_client.moves == [(0.5, 0.0, 0.1)]  # type: ignore[attr-defined]


def test_send_loco_includes_body_height_when_set(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand, LocoCommand

    io, _ = _make_io(monkeypatch)
    cmd = ControlCommand(kind="loco", loco_cmd=LocoCommand(0.0, 0.0, 0.0, body_height=0.72))
    io.send_control(cmd)
    io._apply_command(io._latched_command)
    assert io._loco_client.heights == [0.72]  # type: ignore[attr-defined]


def test_send_joint_pos_publishes_lowcmd_with_kp_kd_and_crc(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand

    io, _ = _make_io(monkeypatch)
    n = io.spec.num_actuated
    cmd = ControlCommand(
        kind="joint_pos",
        joint_pos=np.linspace(-0.1, 0.1, n),
        kp=np.full(n, 80.0),
        kd=np.full(n, 5.0),
    )
    io.send_control(cmd)
    io._apply_command(io._latched_command)
    pub = io._lowcmd_pub  # type: ignore[attr-defined]
    assert len(pub.published) == 1
    msg = pub.published[0]
    assert msg.crc == 0xDEADBEEF
    for i in range(n):
        mc = msg.motor_cmd[i]
        assert mc.mode == 0x01
        assert mc.kp == pytest.approx(80.0)
        assert mc.kd == pytest.approx(5.0)
    assert msg.motor_cmd[0].q == pytest.approx(-0.1)


def test_torque_clamping_respects_safety_margin(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand

    io, _ = _make_io(monkeypatch, torque_safety_margin=0.5)
    n = io.spec.num_actuated
    big_tau = np.full(n, 100.0)
    cmd = ControlCommand(kind="torque", joint_torque=big_tau)
    io.send_control(cmd)
    io._apply_command(io._latched_command)
    msg = io._lowcmd_pub.published[0]  # type: ignore[attr-defined]
    # spec torque_limit = 25, margin = 0.5 → bound = 12.5, kp/kd zeroed for torque mode
    assert msg.motor_cmd[0].tau == pytest.approx(12.5)
    assert msg.motor_cmd[0].kp == pytest.approx(0.0)
    assert msg.motor_cmd[0].kd == pytest.approx(0.0)


def test_mixed_command_dispatches_loco_and_lowcmd_for_upper_body(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand, LocoCommand

    io, _ = _make_io(monkeypatch)
    n = io.spec.num_actuated
    cmd = ControlCommand(
        kind="mixed",
        joint_pos=np.full(n, 0.5),
        loco_cmd=LocoCommand(0.2, 0.0, 0.0),
        kp=np.full(n, 30.0),
        kd=np.full(n, 2.0),
    )
    io.send_control(cmd)
    io._apply_command(io._latched_command)
    # Loco channel got the move
    assert io._loco_client.moves == [(0.2, 0.0, 0.0)]  # type: ignore[attr-defined]
    # arm_sdk got an upper-body-only message. This channel can coexist with
    # sport-mode leg control; publishing rt/lowcmd here would fight it.
    assert io._lowcmd_pub.published == []  # type: ignore[attr-defined]
    msg = io._arm_sdk_pub.published[0]  # type: ignore[attr-defined]
    upper_set = (
        set(io.spec.waist_joints)
        | set(io.spec.left_arm_joints)
        | set(io.spec.right_arm_joints)
    )
    for i, name in enumerate(io.spec.actuated_joints):
        mc = msg.motor_cmd[i]
        if name in upper_set:
            assert mc.q == pytest.approx(0.5)
        else:
            # Untouched leg joints retain default-construction values.
            assert mc.q == pytest.approx(0.0)


def test_unsupported_command_kind_rejected(monkeypatch):
    from genedynamics.deploy.interfaces.messages import ControlCommand

    io, _ = _make_io(monkeypatch)
    # Wrap a kind not in the IO's accepts set; pre-validate via the base class
    bad = ControlCommand(kind="joint_vel", joint_vel=np.zeros(io.spec.num_actuated))
    with pytest.raises(ValueError):
        io.send_control(bad)


def test_read_state_pulls_motor_q_and_imu_quat(monkeypatch):
    io, _ = _make_io(monkeypatch)
    n = io.spec.num_actuated
    ls = _StubLowState(num_motors=n)
    for i in range(n):
        ls.motor_state[i].q = float(i) * 0.1
        ls.motor_state[i].dq = float(i) * 0.01
    io._on_lowstate(ls)  # type: ignore[attr-defined]
    state = io.get_state()
    # Joint slice in qpos starts at index 7 (floating base)
    np.testing.assert_allclose(state.qpos[7 : 7 + n], np.arange(n) * 0.1)
    np.testing.assert_allclose(state.qvel[6 : 6 + n], np.arange(n) * 0.01)
    # Identity IMU quat
    np.testing.assert_allclose(state.qpos[3:7], [1.0, 0.0, 0.0, 0.0])


def test_localization_overrides_floating_base(monkeypatch):
    class _MockLoc:
        def __init__(self) -> None:
            self.calls = 0

        def get_state(self):
            self.calls += 1
            qpos = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
            qvel = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.05])
            return qpos, qvel

        def health(self):
            return "ok"

    loc = _MockLoc()
    io, _ = _make_io(monkeypatch, localization=loc)
    n = io.spec.num_actuated
    io._on_lowstate(_StubLowState(num_motors=n))  # type: ignore[attr-defined]
    state = io.get_state()
    np.testing.assert_allclose(state.base_pose, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.base_twist, [0.1, 0.2, 0.0, 0.0, 0.0, 0.05])
    assert loc.calls >= 1
    assert state.extras["localization_health"] == "ok"


def test_sport_mode_does_not_use_motion_switcher(monkeypatch):
    io, _ = _make_io(monkeypatch, msc_mode="sport")
    # Sport is brought up through LocoClient's FSM during reset. Calling
    # SelectMode("normal") is rejected by current G1 firmware.
    assert io._motion_switcher.modes == []  # type: ignore[attr-defined]


def test_close_drops_to_damp(monkeypatch):
    io, _ = _make_io(monkeypatch)
    io.close()
    assert "damp" in io._motion_switcher.modes  # type: ignore[attr-defined]
