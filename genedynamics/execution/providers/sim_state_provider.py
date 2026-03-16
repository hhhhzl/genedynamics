"""
Simulation state provider.

Wraps an env-like object with reset/step for sim state.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from genedynamics.execution.core.contracts import HealthStatus, RobotState
from genedynamics.execution.providers.state_provider import StateProviderBase


class SimStateProvider(StateProviderBase):
    """
    State provider backed by simulation environment.

    Expects env with:
    - reset(rng=None) -> (state, info)
    - step(state, action, t, info) or transition(state, action)
    - state: flat array [qpos; qvel] or dict with qpos/qvel
    """

    def __init__(
        self,
        env: Any,
        nq: int,
        nv: Optional[int] = None,
        state_to_flat: Optional[Any] = None,
    ):
        self.env = env
        self.nq = nq
        self.nv = nv if nv is not None else (self._infer_nv(env))
        self._state_to_flat = state_to_flat
        self._current_state: Optional[RobotState] = None
        self._timestamp = 0.0
        self._health = HealthStatus.OK

    def _infer_nv(self, env: Any) -> int:
        """Infer nv from env."""
        state_dim = getattr(env, "state_dim", None)
        if state_dim is not None:
            return state_dim - self.nq
        act_dim = getattr(env, "act_dim", 4)
        return max(4, act_dim * 2)

    def _to_robot_state(self, state: Any, timestamp: float) -> RobotState:
        """Convert env state to RobotState."""
        if self._state_to_flat is not None:
            flat = self._state_to_flat(state)
        elif isinstance(state, dict):
            qpos = np.asarray(state.get("qpos", state.get("qpos", [])))
            qvel = np.asarray(state.get("qvel", state.get("qvel", [])))
            return RobotState(qpos=qpos, qvel=qvel, timestamp=timestamp, source="sim")
        else:
            flat = np.asarray(state, dtype=np.float32).ravel()
        return RobotState.from_flat(flat, self.nq, timestamp=timestamp, source="sim")

    def set_state(self, state: RobotState) -> None:
        """Set current state (for sim step)."""
        self._current_state = state
        self._timestamp = state.timestamp

    def set_state_from_flat(self, flat: np.ndarray, timestamp: float = 0.0) -> None:
        """Set from flat array."""
        self._current_state = RobotState.from_flat(flat, self.nq, timestamp=timestamp, source="sim")
        self._timestamp = timestamp

    def get_state(self) -> Optional[RobotState]:
        return self._current_state

    def get_timestamp(self) -> float:
        return self._timestamp

    def health(self) -> HealthStatus:
        return self._health
