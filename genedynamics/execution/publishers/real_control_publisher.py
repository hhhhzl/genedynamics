"""
Real robot control publisher (position mode).

Sends joint position targets to robot. Backend is pluggable
(e.g. Unitree SDK). Framework provides interface + stub.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from genedynamics.execution.publishers.control_publisher import ControlPublisherBase


class RobotControlBackend(Protocol):
    """Protocol for real robot control interface."""

    def send_joint_positions(self, positions: np.ndarray) -> None:
        """Send joint position targets (position mode)."""
        ...

    def set_stand_posture(self, positions: np.ndarray) -> None:
        """Set safe stand posture."""
        ...

    def emergency_stop(self) -> None:
        """Emergency stop."""
        ...


class RealControlPublisher(ControlPublisherBase):
    """
    Control publisher for real robot (position mode).

    Backend: pluggable (Unitree SDK, etc.).
    When backend is None, publish is no-op (stub).
    """

    def __init__(
        self,
        backend: Optional[RobotControlBackend] = None,
        act_dim: int = 8,
        env: Any = None,
    ):
        self.backend = backend
        self.act_dim = act_dim
        self.env = env  # None for real; executor uses getattr(env, "act_dim", ...)
        self._last_action: Optional[np.ndarray] = None
        self._safe_posture: Optional[np.ndarray] = None
        self._stopped = False

    def set_backend(self, backend: RobotControlBackend) -> None:
        """Set or replace robot backend."""
        self.backend = backend

    def publish(self, action: np.ndarray, mode: str, ttl_ms: float = 0.0) -> None:
        """Send joint position targets to robot."""
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.size != self.act_dim:
            action = np.pad(action, (0, max(0, self.act_dim - action.size)), mode="edge")[: self.act_dim]
        self._last_action = action.copy()
        if self._stopped:
            if self._safe_posture is not None and self.backend is not None:
                self.backend.send_joint_positions(self._safe_posture)
            return
        if self.backend is not None:
            self.backend.send_joint_positions(action)

    def set_safe_posture(self, posture: np.ndarray) -> None:
        """Set safe stand posture for fallback."""
        self._safe_posture = np.asarray(posture, dtype=np.float32).copy()

    def emergency_stop(self) -> None:
        """Emergency stop."""
        self._stopped = True
        if self.backend is not None:
            self.backend.emergency_stop()
        if self._safe_posture is not None and self.backend is not None:
            self.backend.send_joint_positions(self._safe_posture)

    def reset_stop(self) -> None:
        """Clear emergency stop."""
        self._stopped = False

    def get_last_action(self) -> Optional[np.ndarray]:
        return self._last_action.copy() if self._last_action is not None else None
