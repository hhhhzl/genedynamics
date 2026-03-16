"""
Simulation control publisher.

Applies actions to sim env via step/transition.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from genedynamics.execution.publishers.control_publisher import ControlPublisherBase


class SimControlPublisher(ControlPublisherBase):
    """
    Control publisher that applies actions to simulation env.

    Expects env with step(x_next, u, t, info) or transition(state, action).
    For drone: transition(state, action) -> next_state.
    """

    def __init__(self, env: Any):
        self.env = env
        self._last_action: Optional[np.ndarray] = None
        self._safe_posture: Optional[np.ndarray] = None
        self._stopped = False

    def publish(self, action: np.ndarray, mode: str, ttl_ms: float = 0.0) -> None:
        """Store action; actual application happens in executor step."""
        if self._stopped:
            if self._safe_posture is not None:
                self._last_action = self._safe_posture.copy()
            return
        self._last_action = np.asarray(action, dtype=np.float32).copy()

    def apply_to_env(self, state: np.ndarray, action: np.ndarray, t: int) -> np.ndarray:
        """
        Apply action to env and return next state.

        Used by executor to drive sim.
        Prefers transition() for dynamics envs (drone, etc.); uses step() for D3IL-style envs.
        """
        action = np.asarray(action, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if hasattr(self.env, "transition"):
            return np.asarray(self.env.transition(state, action), dtype=np.float32)
        if hasattr(self.env, "step"):
            next_state, _, _, _ = self.env.step(None, action, t=t, info={})
            return np.asarray(next_state, dtype=np.float32)
        raise AttributeError("env must have transition or step")

    def set_safe_posture(self, posture: np.ndarray) -> None:
        self._safe_posture = np.asarray(posture, dtype=np.float32).copy()

    def emergency_stop(self) -> None:
        self._stopped = True
        if self._safe_posture is not None:
            self._last_action = self._safe_posture.copy()

    def reset_stop(self) -> None:
        """Clear emergency stop."""
        self._stopped = False

    def get_last_action(self) -> Optional[np.ndarray]:
        return self._last_action.copy() if self._last_action is not None else None
