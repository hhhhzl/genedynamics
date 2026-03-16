"""
Vicon shared-memory localization plugin.

Reads pose/velocity from shared memory written by external Vicon process.
Format: struct q13d = int64 utime + 13 float64 (pos3, quat4, vel3, omega3).
Requires: pyvicon_datastream (optional, for ViconDemo writer).
"""

from __future__ import annotations

import struct
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.deploy.localization.base_plugin import BaseLocalizationPlugin

VICON_AVAILABLE = False
try:
    from pyvicon_datastream import tools
    VICON_AVAILABLE = True
except ImportError:
    pass


class ViconShmPlugin(BaseLocalizationPlugin):
    """
    Read pose/velocity from shared memory (mocap_state_shm).

    Expects external process writing: struct q13d = int64 utime + 13 float64
    (pos[3], quat_xyzw[4], vel[3], omega[3]).
    """

    SHM_NAME = "mocap_state_shm"
    SHM_SIZE = 8 + 13 * 8
    STRUCT_FMT = "q13d"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._shm_name = config.get("shm_name", self.SHM_NAME)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._last_time: Optional[float] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name,
                create=False,
                size=self.SHM_SIZE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Shared memory '{self._shm_name}' not found. "
                "Start Vicon writer process first."
            )

    def get_state(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._shm is None:
            return None
        try:
            data = struct.unpack_from(self.STRUCT_FMT, self._shm.buf, 0)
            utime = data[0]
            position = np.array(data[1:4], dtype=np.float64)
            quat_xyzw = np.array(data[4:8], dtype=np.float64)
            quat_wxyz = np.roll(quat_xyzw, 1)
            vel = np.array(data[8:11], dtype=np.float64)
            omega = np.array(data[11:14], dtype=np.float64)
            qpos = np.concatenate([position, quat_wxyz])
            qvel = np.concatenate([vel, omega])
            self._last_time = utime * 1e-6
            return qpos, qvel
        except Exception:
            return None

    def get_last_update_time(self) -> Optional[float]:
        return self._last_time

    def close(self) -> None:
        if self._shm:
            self._shm.close()
            self._shm = None


class ViconDemoWriter:
    """
    Standalone Vicon DataStream writer that publishes to shared memory.

    Run in separate process. Requires pyvicon_datastream.
    """

    def __init__(
        self,
        vicon_tracker_ip: str,
        vicon_object_name: str,
        vicon_z_offset: float = 0.0,
        shm_name: str = "mocap_state_shm",
        fs: float = 100.0,
    ) -> None:
        if not VICON_AVAILABLE:
            raise ImportError("ViconDemoWriter requires pyvicon_datastream")
        self.tracker = tools.ObjectTracker(vicon_tracker_ip)
        if not self.tracker.is_connected:
            raise RuntimeError(f"Failed to connect to Vicon at {vicon_tracker_ip}")
        self.object_name = vicon_object_name
        self.z_offset = vicon_z_offset
        self.shm_name = shm_name
        self.fs = fs
        self._shm = shared_memory.SharedMemory(name=shm_name, create=True, size=ViconShmPlugin.SHM_SIZE)

    def run(self) -> None:
        import time
        from scipy.spatial.transform import Rotation as R
        prev_time = prev_pos = prev_quat = None
        while True:
            pos_data = self.tracker.get_position(self.object_name)
            if not pos_data:
                time.sleep(0.01)
                continue
            t = time.time()
            obj = pos_data[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            position = np.array([x, y, z]) / 1000.0
            position[2] += self.z_offset
            quat = R.from_euler("XYZ", [roll, pitch, yaw]).as_quat()
            vel = np.zeros(3)
            omega = np.zeros(3)
            if prev_time and prev_pos is not None:
                dt = t - prev_time
                if dt > 0:
                    vel = (position - prev_pos) / dt
                    dq = R.from_quat(quat) * R.from_quat(prev_quat).inv()
                    omega = dq.as_rotvec() / dt
            prev_time, prev_pos, prev_quat = t, position, quat
            utime = int(t * 1e6)
            struct.pack_into(
                ViconShmPlugin.STRUCT_FMT,
                self._shm.buf, 0,
                utime, *position, *quat, *vel, *omega
            )
            time.sleep(1.0 / self.fs)
