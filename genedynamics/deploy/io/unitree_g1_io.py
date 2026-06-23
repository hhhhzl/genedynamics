"""Real Unitree G1 RobotIO.

Wraps the Unitree G1 SDK (``unitree_sdk2py``) behind the same
:class:`RobotIO` contract used by :class:`MujocoRobotIO` / :class:`MjxRobotIO`.
The pipeline does not branch on sim vs real — switching is one config-line
change of the IO class name. Localization for the floating base comes from
any plugin under :mod:`genedynamics.deploy.localization` (mock / ROS2 /
Vicon SHM).

The SDK is imported lazily inside :meth:`__init__` so this module can be
imported on machines without the Unitree SDK installed (CI, dev laptops);
constructing the class without the SDK raises a clear ``ImportError``.

Supported :class:`ControlCommand` kinds:

* ``"loco"``       — high-level velocity command (delegated to the SDK
  ``LocoClient``; sport-mode equivalent of :class:`SparkRLLocoClient`).
* ``"joint_pos"``  — direct low-level joint targets via the
  ``MotionSwitcherClient`` low-level interface (Kp / Kd vectors required).
* ``"torque"``     — pure torque command (Kp = Kd = 0 internally).
* ``"mixed"``      — ``loco_cmd`` for legs + ``joint_pos`` for upper body
  (the upper-body slice is published on the dedicated arm SDK channel).

This is the **only** module under ``deploy/`` that talks to ``unitree_sdk2py``.
Everything else stays SDK-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from genedynamics.deploy.interfaces.messages import ControlCommand, RobotState
from genedynamics.deploy.io.base import BaseRobotIO
from genedynamics.robots.g1 import G1RobotSpec, g1_scene_path

__all__ = ["UnitreeG1RobotIO"]


# ---------------------------------------------------------------------------
# SDK channel / topic constants — kept here so the rest of deploy/ stays SDK-free.
# ---------------------------------------------------------------------------

_DEFAULT_NETWORK_INTERFACE = "eth0"
_DEFAULT_DOMAIN_ID = 0
_LOWSTATE_TOPIC = "rt/lowstate"
_LOWCMD_TOPIC = "rt/lowcmd"
_ARM_LOWSTATE_TOPIC = "rt/lowstate"  # arm joints share the lowstate channel on G1

# G1 sport-mode FSM ids (this firmware's convention, per CMU SPARK's g1_real_agent).
# NOTE: the pip SDK's *named* LocoClient methods use DIFFERENT ids
# (``Start()`` -> 500, ``Squat2StandUp()`` -> 706) which this G1 rejects as
# "Invalid FSM ID" — so the bring-up drives the generic ``SetFsmId()`` with these.
_G1_FSM_DAMP = 1
_G1_FSM_LOCK_STAND = 4     # stand up and lock (the stand-up step)
_G1_FSM_MAIN_MODE = 200    # walk-ready main control mode (LocoClient.Move works here)

# G1 arm control while sport mode runs the legs: publish LowCmd_ to ``rt/arm_sdk``
# (coexists with sport, unlike ``rt/lowcmd`` which would fight it). A weight on the
# special kNotUsedJoint slot (index 29 for G1-29dof) blends arm_sdk authority over
# the arms; 1.0 = arm_sdk fully owns them. Mirrors CMU SPARK's g1_real_agent.
_ARM_SDK_TOPIC = "rt/arm_sdk"
_ARM_SDK_WEIGHT_IDX = 29
_ARM_SDK_WEIGHT = 1.0


class UnitreeG1RobotIO(BaseRobotIO):
    """Real-hardware :class:`RobotIO` for the Unitree G1.

    Args:
        network_interface: Ethernet interface bound by ``ChannelFactory``.
            Defaults to ``"eth0"``. Set to ``"lo"`` for in-machine SDK
            simulators.
        domain_id: DDS domain id. ``0`` for stock G1.
        spec: Pre-built :class:`G1RobotSpec`. When omitted the spec is
            constructed by introspecting the bundled G1 MJCF — used purely
            for actuator naming / Kp / torque limits, not for kinematics.
        localization: Optional :class:`BaseLocalizationPlugin` instance. When
            present, the floating base in :class:`RobotState` is taken from
            the plugin (e.g. Vicon, T265, AMCL); otherwise the SDK IMU is
            used and the world XY is reported as zero.
        control_period_s: Nominal control period. :meth:`step` blocks (using
            a small busy-wait) until this period has elapsed since the last
            invocation, so the user-facing pipeline loop can stay identical
            to the sim case.
        default_kp: Fallback per-joint Kp when a :class:`ControlCommand`
            does not provide one. Defaults to the spec's actuator gains.
        default_kd: Fallback per-joint Kd. Defaults to ``2.0``.
        torque_safety_margin: Multiplied with the spec torque limits when
            clamping the commanded torque vector. ``1.0`` means trust the
            MJCF; lower values give a software safety margin.
        msc_mode: ``MotionSwitcher`` operating mode requested at startup.
            ``"low_level"`` for direct joint control, ``"sport"`` for stock
            sport-mode + (optional) low-level arm control.
    """

    physics_backend = None  # real hardware
    array_runtime = "numpy"
    accepts = ("loco", "joint_pos", "torque", "mixed")

    def __init__(
        self,
        *,
        network_interface: str = _DEFAULT_NETWORK_INTERFACE,
        domain_id: int = _DEFAULT_DOMAIN_ID,
        spec: Optional[G1RobotSpec] = None,
        localization: Optional[Any] = None,
        control_period_s: float = 0.02,
        default_kp: Optional[np.ndarray] = None,
        default_kd: float = 2.0,
        torque_safety_margin: float = 0.9,
        msc_mode: str = "low_level",
        model_xml_path: Optional[str | Path] = None,
    ) -> None:
        # Lazy SDK import — raises a clear error on dev laptops without SDK.
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as HgLowCmd,
                LowState_ as HgLowState,
            )
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__LowCmd_ as DefaultHgLowCmd,
            )
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient as SdkLocoClient
            try:
                # Current SDK layout (unitree_sdk2_python ≥ 1.0.1).
                from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                    MotionSwitcherClient,
                )
            except ImportError:
                # Older SDK exposed it under g1.* (e.g. the robot's unitree_sdk2-main).
                from unitree_sdk2py.g1.motion_switcher.motion_switcher_client import (
                    MotionSwitcherClient,
                )
        except ImportError as exc:  # pragma: no cover - exercised only on dev boxes
            raise ImportError(
                "UnitreeG1RobotIO requires the Unitree SDK (unitree_sdk2py). "
                "Install it via `pip install unitree_sdk2py` on a machine that "
                "actually talks to the robot."
            ) from exc

        # Build a spec from the bundled MJCF for actuator metadata.
        if spec is None:
            import mujoco

            xml_path = g1_scene_path(model_xml_path)
            model = mujoco.MjModel.from_xml_path(xml_path)
            spec = G1RobotSpec.from_mujoco_model(model, model_xml_path=xml_path)
        super().__init__(spec=spec)

        self._sdk_modules = {
            "ChannelFactoryInitialize": ChannelFactoryInitialize,
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "HgLowCmd": HgLowCmd,
            "HgLowState": HgLowState,
            "CRC": CRC(),
            "DefaultHgLowCmd": DefaultHgLowCmd,
            "SdkLocoClient": SdkLocoClient,
            "MotionSwitcherClient": MotionSwitcherClient,
        }

        self.network_interface = network_interface
        self.domain_id = int(domain_id)
        self.control_period_s = float(control_period_s)
        self.localization = localization
        self.torque_safety_margin = float(torque_safety_margin)
        self.msc_mode = msc_mode

        n = spec.num_actuated
        self._default_kp = (
            np.asarray(default_kp, dtype=np.float64).reshape(n)
            if default_kp is not None
            else spec.actuator_kp_vector()
        )
        self._default_kd = np.full(n, float(default_kd), dtype=np.float64)
        self._torque_limit = spec.torque_limit_vector() * self.torque_safety_margin

        # ------------------------------------------------------------------
        # Wire up DDS channels + clients.
        # ------------------------------------------------------------------
        ChannelFactoryInitialize(self.domain_id, self.network_interface)

        self._lowstate: Optional[Any] = None  # latest HgLowState
        self._lowstate_t: float = 0.0
        self._lowstate_sub = ChannelSubscriber(_LOWSTATE_TOPIC, HgLowState)
        self._lowstate_sub.Init(self._on_lowstate, queueLen=10)

        self._lowcmd_pub = ChannelPublisher(_LOWCMD_TOPIC, HgLowCmd)
        self._lowcmd_pub.Init()

        # Arm control channel (coexists with sport-mode leg control); see _send_arm_sdk.
        self._arm_sdk_pub = ChannelPublisher(_ARM_SDK_TOPIC, HgLowCmd)
        self._arm_sdk_pub.Init()

        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.Init()

        self._loco_client = SdkLocoClient()
        self._loco_client.Init()
        self._loco_client.SetTimeout(10.0)

        self._switch_to_mode(msc_mode)

        self._last_step_t = time.monotonic()

    # ------------------------------------------------------------------
    # SDK callbacks
    # ------------------------------------------------------------------

    def _on_lowstate(self, msg: Any) -> None:
        """Subscriber callback. Stash the freshest LowState into ``self``."""
        self._lowstate = msg
        self._lowstate_t = time.monotonic()

    def _switch_to_mode(self, mode: str) -> None:
        """Request a MotionSwitcher mode (low_level / sport / damp / zero)."""
        try:
            if mode == "low_level":
                self._motion_switcher.ReleaseMode()
                self._motion_switcher.SelectMode("ai")  # G1 low-level mode tag
            elif mode == "sport":
                # SPARK's high-level path does NOT touch MotionSwitcher — the sport
                # ("loco") service is driven directly via the LocoClient FSM. On this
                # G1, SelectMode("normal") returns code 7004; skip it. The actual
                # stand-up happens in _bring_up_sport() via SetFsmId (LockStand 4 ->
                # MainMode 200) when the IO is reset.
                pass
            elif mode == "damp":
                self._motion_switcher.SelectMode("damp")
            else:
                raise ValueError(f"Unknown msc_mode: {mode!r}")
        except Exception as exc:  # pragma: no cover
            # Don't crash on first connect — surfaces as degraded health.
            print(f"[UnitreeG1RobotIO] WARN: switch to {mode!r} failed: {exc}")

    def _bring_up_sport(self) -> None:
        """Stand the G1 up into walk-ready MainMode via the sport FSM.

        Sequence (mirrors CMU SPARK's ``g1_real_agent`` bring-up):
        ``<current> -> LockStand(4) -> MainMode(200)``, driven through the
        generic ``SetFsmId`` (the pip SDK's ``Start()``/``Squat2StandUp()`` use
        wrong ids for this firmware, returning "Invalid FSM ID"). After MainMode
        the legs accept ``LocoClient.Move``.

        NOTE: matches SPARK but is UNVERIFIED on hardware — validate on the robot
        (it should stand+lock on ``SetFsmId(4)``, then accept ``Move`` in MainMode).
        If ``SetFsmId(4)`` is rejected by the firmware, put the robot into
        LockStand via the remote/App first; this then advances it to MainMode.
        """
        loco = self._loco_client
        try:
            code, cur = loco.GetFsmId()
            print(f"[UnitreeG1RobotIO] sport bring-up: FSM before = {cur} (code={code})")
        except Exception as exc:  # pragma: no cover
            print(f"[UnitreeG1RobotIO] WARN: GetFsmId failed: {exc}")
        for target in (_G1_FSM_LOCK_STAND, _G1_FSM_MAIN_MODE):
            try:
                ret = loco.SetFsmId(target)
                print(f"[UnitreeG1RobotIO] sport bring-up: SetFsmId({target}) -> {ret}")
            except Exception as exc:  # pragma: no cover
                print(f"[UnitreeG1RobotIO] WARN: SetFsmId({target}) failed: {exc}")
            time.sleep(1.5)
        try:
            loco.BalanceStand(0)  # static balance once in MainMode
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # BaseRobotIO hooks
    # ------------------------------------------------------------------

    def _reset_robot(self) -> None:
        """Wait for the first lowstate packet and zero internal clocks.

        Hardware never truly "resets"; we instead make sure we have a fresh
        sensor packet, and ask the SDK to enter a balanced standing pose if
        we're in sport mode.
        """
        deadline = time.monotonic() + 5.0
        while self._lowstate is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._lowstate is None:
            raise RuntimeError(
                "UnitreeG1RobotIO: no lowstate packets received within 5s. "
                "Check network_interface / SDK / power."
            )
        if self.msc_mode == "sport":
            self._bring_up_sport()
        self._last_step_t = time.monotonic()

    def _read_state(self, t: float) -> RobotState:
        spec = self.spec
        n = spec.num_actuated
        qpos = np.zeros(7 + n, dtype=np.float64)
        qvel = np.zeros(6 + n, dtype=np.float64)
        joint_torque = np.zeros(n, dtype=np.float64)

        ls = self._lowstate
        if ls is not None:
            # Joints — SDK exposes them in a flat array indexed by joint id.
            for i, name in enumerate(spec.actuated_joints):
                aid = spec.actuator_id.get(name)
                if aid is None:
                    continue
                # SDK indexing: motor_state[aid] for HG / hg robots.
                ms = ls.motor_state[int(aid)]
                qpos[7 + i] = float(ms.q)
                qvel[6 + i] = float(ms.dq)
                joint_torque[i] = float(ms.tau_est)
            # IMU → base orientation (quat_wxyz). Position needs an external
            # localization source.
            imu = getattr(ls, "imu_state", None)
            if imu is not None:
                quat = np.asarray(imu.quaternion, dtype=np.float64).reshape(4)
                qpos[3:7] = quat  # SDK uses (w, x, y, z) — matches MuJoCo

        base_pose: Optional[np.ndarray] = None
        base_twist: Optional[np.ndarray] = None
        loc = self.localization
        if loc is not None:
            try:
                packet = loc.get_state()
            except Exception:  # pragma: no cover
                packet = None
            if packet is not None:
                pose, twist = packet
                qpos[0:7] = np.asarray(pose, dtype=np.float64).reshape(7)
                qvel[0:6] = np.asarray(twist, dtype=np.float64).reshape(6)
                base_pose = qpos[0:7].copy()
                base_twist = qvel[0:6].copy()
        if base_pose is None and qpos.shape[0] >= 7:
            base_pose = qpos[0:7].copy()
        if base_twist is None and qvel.shape[0] >= 6:
            base_twist = qvel[0:6].copy()

        return RobotState(
            t=float(t),
            qpos=qpos,
            qvel=qvel,
            base_pose=base_pose,
            base_twist=base_twist,
            joint_torque=joint_torque,
            imu={"raw": ls.imu_state} if ls is not None and hasattr(ls, "imu_state") else None,
            extras={
                "step": self._step_count,
                "lowstate_age_s": float(time.monotonic() - self._lowstate_t),
                "localization_health": getattr(loc, "health", lambda: "n/a")(),
            },
        )

    def _apply_command(self, cmd: ControlCommand) -> None:
        if cmd.kind == "loco":
            self._send_loco(cmd)
        elif cmd.kind in ("joint_pos", "torque"):
            self._send_lowcmd(cmd)
        elif cmd.kind == "mixed":
            # Legs go through the high-level loco client (sport mode); the upper
            # body goes via the rt/arm_sdk channel (coexists with sport), NOT
            # rt/lowcmd — which would fight sport-mode arm control.
            if cmd.loco_cmd is not None:
                self._send_loco(cmd)
            if cmd.joint_pos is not None or cmd.joint_torque is not None:
                self._send_arm_sdk(cmd)
        else:  # pragma: no cover
            raise ValueError(f"UnitreeG1RobotIO does not handle kind={cmd.kind!r}")

    def _step_physics(self, dt: float) -> None:
        """No physics; just rate-limit so the user loop matches ``dt``."""
        target = self._last_step_t + max(dt, self.control_period_s)
        while True:
            now = time.monotonic()
            if now >= target:
                break
            time.sleep(min(target - now, 0.001))
        self._last_step_t = time.monotonic()

    def _on_close(self) -> None:
        try:
            self._motion_switcher.SelectMode("damp")
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Command path internals
    # ------------------------------------------------------------------

    def _send_loco(self, cmd: ControlCommand) -> None:
        loco = cmd.loco_cmd
        if loco is None:
            raise ValueError("ControlCommand(kind='loco') requires loco_cmd")
        try:
            self._loco_client.Move(float(loco.vx), float(loco.vy), float(loco.yaw_rate))
            if loco.body_height is not None:
                self._loco_client.SetStandHeight(float(loco.body_height))
        except Exception as exc:  # pragma: no cover
            print(f"[UnitreeG1RobotIO] WARN: loco Move failed: {exc}")

    def _send_lowcmd(self, cmd: ControlCommand, *, upper_body_only: bool = False) -> None:
        spec = self.spec
        n = spec.num_actuated
        crc = self._sdk_modules["CRC"]

        # Default factory: the IDL LowCmd_ requires all fields, so a no-arg
        # ``HgLowCmd()`` raises on this SDK version. rt/lowcmd needs mode_machine.
        msg = self._sdk_modules["DefaultHgLowCmd"]()
        msg.mode_pr = 0
        ls = self._lowstate
        msg.mode_machine = int(getattr(ls, "mode_machine", 0)) if ls is not None else 0

        q_target = np.zeros(n, dtype=np.float64)
        if cmd.joint_pos is not None:
            jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
            q_target[: jp.size] = jp[:n]
        kp = (
            np.asarray(cmd.kp, dtype=np.float64).reshape(-1)
            if cmd.kp is not None
            else self._default_kp
        )
        kd = (
            np.asarray(cmd.kd, dtype=np.float64).reshape(-1)
            if cmd.kd is not None
            else self._default_kd
        )
        tau_ff = np.zeros(n, dtype=np.float64)
        if cmd.joint_torque is not None:
            tau = np.asarray(cmd.joint_torque, dtype=np.float64).reshape(-1)
            tau_ff[: tau.size] = tau[:n]
        if cmd.kind == "torque":
            kp = np.zeros(n, dtype=np.float64)
            kd = np.zeros(n, dtype=np.float64)

        # Clamp torques against the safety-margined limits.
        tau_ff = np.clip(tau_ff, -self._torque_limit, self._torque_limit)

        # Restrict to upper body when in mixed mode (legs are handled by sport mode).
        upper_set = set(spec.waist_joints) | set(spec.left_arm_joints) | set(spec.right_arm_joints)

        for i, name in enumerate(spec.actuated_joints):
            aid = spec.actuator_id.get(name)
            if aid is None:
                continue
            if upper_body_only and name not in upper_set:
                continue
            mc = msg.motor_cmd[int(aid)]
            mc.mode = 0x01  # PD mode
            mc.q = float(q_target[i])
            mc.dq = 0.0
            mc.kp = float(kp[i] if i < kp.size else 0.0)
            mc.kd = float(kd[i] if i < kd.size else 0.0)
            mc.tau = float(tau_ff[i])

        msg.crc = crc.Crc(msg)
        try:
            self._lowcmd_pub.Write(msg)
        except Exception as exc:  # pragma: no cover
            print(f"[UnitreeG1RobotIO] WARN: lowcmd publish failed: {exc}")

    def _send_arm_sdk(self, cmd: ControlCommand) -> None:
        """Publish upper-body (waist + arms) targets via the G1 ``rt/arm_sdk``
        channel, which coexists with sport-mode leg control.

        Mirrors CMU SPARK's ``g1_real_agent``: a weight on the special
        ``kNotUsedJoint`` slot (index 29) blends arm_sdk authority over the arms
        (1.0 = arm_sdk fully controls them); the per-joint q/kp/kd set the
        targets. Published to ``rt/arm_sdk`` (NOT ``rt/lowcmd``) so it does not
        fight sport mode; no CRC / mode_machine needed on this channel.
        """
        spec = self.spec
        n = spec.num_actuated
        msg = self._sdk_modules["DefaultHgLowCmd"]()
        # arm_sdk authority weight on the kNotUsedJoint slot.
        msg.motor_cmd[_ARM_SDK_WEIGHT_IDX].q = float(_ARM_SDK_WEIGHT)

        q_target = np.zeros(n, dtype=np.float64)
        if cmd.joint_pos is not None:
            jp = np.asarray(cmd.joint_pos, dtype=np.float64).reshape(-1)
            q_target[: jp.size] = jp[:n]
        kp = (
            np.asarray(cmd.kp, dtype=np.float64).reshape(-1)
            if cmd.kp is not None else self._default_kp
        )
        kd = (
            np.asarray(cmd.kd, dtype=np.float64).reshape(-1)
            if cmd.kd is not None else self._default_kd
        )

        upper_set = set(spec.waist_joints) | set(spec.left_arm_joints) | set(spec.right_arm_joints)
        for i, name in enumerate(spec.actuated_joints):
            if name not in upper_set:
                continue
            aid = spec.actuator_id.get(name)
            if aid is None:
                continue
            mc = msg.motor_cmd[int(aid)]
            # SPARK leaves motor_cmd.mode at default for arm_sdk (the weight slot
            # gates authority). Set mode=1 here if your firmware requires it.
            mc.q = float(q_target[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = float(kp[i] if i < kp.size else 0.0)
            mc.kd = float(kd[i] if i < kd.size else 0.0)
        try:
            self._arm_sdk_pub.Write(msg)
        except Exception as exc:  # pragma: no cover
            print(f"[UnitreeG1RobotIO] WARN: arm_sdk publish failed: {exc}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def sdk_health(self) -> Tuple[str, float]:
        """``(status, lowstate_age_s)`` — ``"ok"`` / ``"stale"`` / ``"failed"``."""
        if self._lowstate is None:
            return "failed", float("inf")
        age = time.monotonic() - self._lowstate_t
        if age > 0.2:
            return "stale", age
        return "ok", age
