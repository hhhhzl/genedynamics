"""
Safety guard for action filtering and fallback.

Filters raw actions through bounds, detects state violations,
and produces fallback actions on critical events.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from genedynamics.execution.core.contracts import (
    ActionPacket,
    FallbackAction,
    RobotState,
    ViolationEvent,
)


class SafetyGuard:
    """
    Safety guard for execution layer.

    - Clips actions to configured limits
    - Checks state bounds
    - Produces fallback actions on violation
    """

    def __init__(
        self,
        action_clip_min: Optional[np.ndarray] = None,
        action_clip_max: Optional[np.ndarray] = None,
        state_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fallback_action: Optional[np.ndarray] = None,
        fallback_reason: str = "safety_trigger",
    ):
        self._action_min = np.asarray(action_clip_min, dtype=np.float32) if action_clip_min is not None else None
        self._action_max = np.asarray(action_clip_max, dtype=np.float32) if action_clip_max is not None else None
        self._state_bounds = state_bounds or {}
        self._fallback_action = np.asarray(fallback_action, dtype=np.float32) if fallback_action is not None else None
        self._fallback_reason = fallback_reason
        self._violation_history: list = []

    def filter_action(
        self,
        raw_action: np.ndarray,
        state: Optional[RobotState] = None,
        packet: Optional[ActionPacket] = None,
    ) -> Tuple[np.ndarray, Optional[FallbackAction]]:
        """
        Filter raw action through safety bounds.

        Returns:
            (safe_action, fallback_info) - fallback_info is set when action was modified
        """
        action = np.asarray(raw_action, dtype=np.float32).ravel().copy()

        if self._action_min is not None and self._action_min.size == action.size:
            low = np.maximum(action, self._action_min)
            if not np.allclose(low, action):
                self._violation_history.append(
                    ViolationEvent(
                        kind="action_limits",
                        message="Action below minimum",
                        severity="warn",
                        action=action.copy(),
                    )
                )
            action = low

        if self._action_max is not None and self._action_max.size == action.size:
            high = np.minimum(action, self._action_max)
            if not np.allclose(high, action):
                self._violation_history.append(
                    ViolationEvent(
                        kind="action_limits",
                        message="Action above maximum",
                        severity="warn",
                        action=action.copy(),
                    )
                )
            action = high

        fallback = None
        return action, fallback

    def check_state_violation(self, state: RobotState) -> Optional[ViolationEvent]:
        """Check state against bounds. Returns first violation or None."""
        flat = state.to_flat()
        for name, (lo, hi) in self._state_bounds.items():
            # Support indexed bounds like "pos_0", "pos_1", or "vel"
            if name.startswith("pos_") and name[4:].isdigit():
                idx = int(name[4:])
                if idx < flat.size and not (lo <= flat[idx] <= hi):
                    return ViolationEvent(
                        kind="state_bounds",
                        message=f"State {name} out of bounds: {flat[idx]:.4f} not in [{lo}, {hi}]",
                        severity="critical",
                        state=flat.copy(),
                        timestamp=state.timestamp,
                    )
        return None

    def fallback_policy(self, event: ViolationEvent) -> FallbackAction:
        """Produce fallback action for violation event."""
        if self._fallback_action is not None:
            return FallbackAction(
                action=self._fallback_action.copy(),
                reason=self._fallback_reason,
                original_action=event.action.copy() if event.action is not None else None,
            )
        # Default: zero action
        n = event.action.size if event.action is not None else 4
        return FallbackAction(
            action=np.zeros(n, dtype=np.float32),
            reason="zero_fallback",
            original_action=event.action.copy() if event.action is not None else None,
        )

    def set_fallback_action(self, action: np.ndarray, reason: str = "safety_trigger") -> None:
        """Set fallback action for critical events."""
        self._fallback_action = np.asarray(action, dtype=np.float32).copy()
        self._fallback_reason = reason

    def get_violation_history(self) -> list:
        """Return and optionally clear violation history."""
        out = list(self._violation_history)
        self._violation_history.clear()
        return out
