"""
Telemetry logger for execution layer.

Logs state, action, timing, and events for observability and replay.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from genedynamics.execution.core.contracts import RobotState


class TelemetryLogger:
    """
    Structured telemetry for execution.

    - log_state_action: per-step state/action
    - log_timing: plan latency, loop jitter
    - log_event: violation, fallback, etc.
    """

    def __init__(self, episode_dir: Optional[str] = None, enabled: bool = True):
        self.episode_dir = Path(episode_dir) if episode_dir else None
        self.enabled = enabled
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._plan_latencies_ms: List[float] = []
        self._events: List[Dict[str, Any]] = []
        self._meta: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

    def start_episode(self, meta: Optional[Dict[str, Any]] = None) -> None:
        """Start new episode."""
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._plan_latencies_ms.clear()
        self._events.clear()
        self._meta = dict(meta or {})
        self._start_time = time.perf_counter()

    def log_state_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        plan_latency_ms: Optional[float] = None,
    ) -> None:
        """Log one step."""
        if not self.enabled:
            return
        self._states.append(np.asarray(state, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._timestamps.append(timestamp)
        if plan_latency_ms is not None:
            self._plan_latencies_ms.append(plan_latency_ms)

    def log_timing(self, plan_latency_ms: float, loop_jitter_ms: Optional[float] = None) -> None:
        """Log timing metrics."""
        if not self.enabled:
            return
        self._plan_latencies_ms.append(plan_latency_ms)
        if loop_jitter_ms is not None:
            self._events.append({"kind": "timing", "loop_jitter_ms": loop_jitter_ms})

    def log_event(self, kind: str, **kwargs: Any) -> None:
        """Log event (violation, fallback, etc.)."""
        if not self.enabled:
            return
        ev = {"kind": kind, "ts": time.perf_counter(), **kwargs}
        self._events.append(ev)

    def flush_episode(self) -> Dict[str, Any]:
        """Flush and return episode data."""
        out = {
            "states": [s.tolist() for s in self._states],
            "actions": [a.tolist() for a in self._actions],
            "timestamps": self._timestamps,
            "plan_latencies_ms": self._plan_latencies_ms,
            "events": self._events,
            "meta": self._meta,
            "steps": len(self._states),
        }
        if self._start_time is not None:
            out["wall_time_s"] = time.perf_counter() - self._start_time
        return out
