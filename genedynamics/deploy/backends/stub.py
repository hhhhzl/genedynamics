"""
Stub state/control backends for testing without hardware.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class StubStateBackend:
    """Mock state backend: returns fixed pose/velocity."""

    def __init__(self, nq: int = 15, nv: int = 14) -> None:
        self.nq = nq
        self.nv = nv
        self._qpos = np.zeros(nq, dtype=np.float64)
        self._qpos[2] = 0.5
        self._qpos[7:11] = [1, 0, 0, 0]
        self._qvel = np.zeros(nv, dtype=np.float64)

    def get_qpos_qvel(self) -> tuple:
        return self._qpos.copy(), self._qvel.copy()

    def get_timestamp(self) -> float:
        return time.monotonic()

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._qpos = np.asarray(qpos, dtype=np.float64).ravel()[: self.nq]
        self._qvel = np.asarray(qvel, dtype=np.float64).ravel()[: self.nv]


class StubControlBackend:
    """Mock control backend: no-op publish."""

    def send_joint_positions(self, positions: np.ndarray) -> None:
        pass

    def set_stand_posture(self, positions: np.ndarray) -> None:
        pass

    def emergency_stop(self) -> None:
        pass
