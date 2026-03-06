"""
Unitree Go2 hardware backend via unitree_sdk2_python.

Requires: unitree_sdk2_python, CycloneDDS.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

UNITREE_SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    pass


def _require_unitree() -> None:
    if not UNITREE_SDK_AVAILABLE:
        raise ImportError(
            "Unitree Go2 backend requires unitree_sdk2_python. "
            "Install: pip install unitree_sdk2_python"
        )


class UnitreeGo2StateBackend:
    """State from Unitree Go2 LowState_ topic."""

    def __init__(
        self,
        network_interface: str = "",
        timeout_sec: float = 2.0,
    ) -> None:
        _require_unitree()
        ChannelFactoryInitialize(network_interface)
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._callback, 10)
        self._qpos: Optional[np.ndarray] = None
        self._qvel: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._timeout = timeout_sec

    def _callback(self, msg: "LowState_") -> None:
        import time
        imu = msg.imu_state
        q = np.array([
            imu.quaternion[1], imu.quaternion[2], imu.quaternion[3], imu.quaternion[0],
        ], dtype=np.float64)
        pos = np.array([0, 0, 0], dtype=np.float64)
        self._qpos = np.concatenate([pos, q, np.array(msg.motor_state[:12].q, dtype=np.float64)])
        self._qvel = np.concatenate([
            np.array(imu.gyroscope, dtype=np.float64),
            np.array(msg.motor_state[:12].dq, dtype=np.float64),
        ])
        self._timestamp = time.monotonic()

    def get_qpos_qvel(self) -> tuple:
        import time
        if self._qpos is None:
            raise RuntimeError("No state received from robot")
        if time.monotonic() - self._timestamp > self._timeout:
            raise RuntimeError("State timeout - robot connection lost?")
        return self._qpos.copy(), self._qvel.copy()

    def get_timestamp(self) -> float:
        return self._timestamp


class UnitreeGo2ControlBackend:
    """Control via Unitree Go2 LowCmd_ topic."""

    def __init__(
        self,
        network_interface: str = "",
        pub_dt: float = 0.002,
    ) -> None:
        _require_unitree()
        ChannelFactoryInitialize(network_interface)
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._pub_dt = pub_dt
        self._stand_posture: Optional[np.ndarray] = None
        self._stopped = False

    def send_joint_positions(self, positions: np.ndarray) -> None:
        if not UNITREE_SDK_AVAILABLE:
            return
        msg = LowCmd_()
        for i in range(min(12, len(positions))):
            msg.motor_cmd[i].q = float(positions[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].kp = 40.0
            msg.motor_cmd[i].kd = 1.0
            msg.motor_cmd[i].tau = 0.0
        if self._stopped and self._stand_posture is not None:
            for i in range(min(12, len(self._stand_posture))):
                msg.motor_cmd[i].q = float(self._stand_posture[i])
        self._pub.Write(msg)

    def set_stand_posture(self, positions: np.ndarray) -> None:
        self._stand_posture = np.asarray(positions, dtype=np.float64).copy()

    def emergency_stop(self) -> None:
        self._stopped = True
        if self._stand_posture is not None:
            self.send_joint_positions(self._stand_posture)
