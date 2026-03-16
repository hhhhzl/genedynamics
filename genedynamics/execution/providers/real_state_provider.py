"""
Real robot state provider (position mode quadruped).

Gets state from robot SDK / localization. Backend is pluggable
(e.g. Unitree, Boston Dynamics). Framework provides interface + stub.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from genedynamics.execution.core.contracts import HealthStatus, RobotState
from genedynamics.execution.providers.state_provider import StateProviderBase


class RobotStateBackend(Protocol):
    """Protocol for real robot state source."""

    def get_qpos_qvel(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos, qvel)."""
        ...

    def get_timestamp(self) -> float:
        """Return last update timestamp."""
        ...


class RealStateProvider(StateProviderBase):
    """
    State provider backed by real robot / localization.

    Backend: pluggable (Unitree SDK, state estimator, etc.).
    When backend is None, returns None (stub for framework testing).
    """

    def __init__(
        self,
        backend: Optional[RobotStateBackend] = None,
        nq: int = 15,
        nv: int = 14,
    ):
        self.backend = backend
        self.nq = nq
        self.nv = nv
        self._last_state: Optional[RobotState] = None
        self._last_timestamp = 0.0
        self._health = HealthStatus.OK

    def set_backend(self, backend: RobotStateBackend) -> None:
        """Set or replace robot backend."""
        self.backend = backend

    def get_state(self) -> Optional[RobotState]:
        if self.backend is None:
            return self._last_state
        try:
            qpos, qvel = self.backend.get_qpos_qvel()
            ts = self.backend.get_timestamp()
            self._last_state = RobotState(
                qpos=np.asarray(qpos, dtype=np.float32),
                qvel=np.asarray(qvel, dtype=np.float32),
                timestamp=ts,
                source="real",
            )
            self._last_timestamp = ts
            self._health = HealthStatus.OK
            return self._last_state
        except Exception:
            self._health = HealthStatus.DEGRADED
            return self._last_state

    def get_timestamp(self) -> float:
        return self._last_timestamp

    def health(self) -> HealthStatus:
        return self._health
