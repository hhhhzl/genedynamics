"""JSON-Lines logger observer.

Writes one JSON object per control step to ``<out_dir>/<episode_id>.jsonl``.
The schema is intentionally flat — each line carries a small set of scalars
(time, success, reward, cost) plus the metrics from :class:`StepInfo`. The
observer never touches the heavy state arrays; use :class:`RecorderObserver`
when you need raw timeseries.

Trailing summaries from :meth:`on_episode_end` are written to a separate
``summary.json`` next to the per-step file. The whole observer keeps the
log directory open across episodes so multi-episode rollouts share one
``out_dir``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.observers.base import BaseObserver

__all__ = ["LoggerObserver"]


class LoggerObserver(BaseObserver):
    """Per-step JSONL logger.

    Args:
        out_dir: Directory where ``<episode_id>.jsonl`` and ``summary.json``
            files are written. Created if it does not exist.
        name: Display name (forwarded to :class:`Observer`'s ``name`` field).
        flush_every: Flush the file handle every N writes. Set to 1 for
            debugging at the cost of throughput.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        name: str = "logger",
        flush_every: int = 50,
    ) -> None:
        super().__init__(name=name)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = int(flush_every)
        self._fh: Optional[TextIO] = None
        self._episode_id: Optional[str] = None
        self._writes_since_flush: int = 0

    def on_episode_start(self, episode_id: str, metadata: Mapping[str, Any]) -> None:
        self._close_fh()
        self._episode_id = episode_id
        path = self.out_dir / f"{episode_id}.jsonl"
        self._fh = path.open("w")
        header = {
            "_kind": "episode_start",
            "episode_id": episode_id,
            "wall_time": time.time(),
            "metadata": _jsonable(metadata),
        }
        self._fh.write(json.dumps(header) + "\n")
        self._writes_since_flush = 1

    def on_step(
        self,
        t: float,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
        info: StepInfo,
    ) -> None:
        if self._fh is None:
            return
        record: dict[str, Any] = {
            "_kind": "step",
            "t": float(t),
            "done": bool(info.done),
            "kind": cmd.kind,
        }
        if info.success is not None:
            record["success"] = bool(info.success)
        if info.reward is not None:
            record["reward"] = float(info.reward)
        if info.cost is not None:
            record["cost"] = float(info.cost)
        if info.metrics:
            record["metrics"] = {k: float(v) for k, v in info.metrics.items()}
        if intent is not None:
            record["intent"] = {
                "base_lin_vel": _jsonable(intent.base_lin_vel),
                "base_yaw_rate": float(intent.base_yaw_rate),
                "base_yaw": float(intent.base_yaw),
                "base_height": float(intent.base_height),
            }
            if intent.contact_phase is not None:
                record["intent"]["contact_phase"] = intent.contact_phase
        self._fh.write(json.dumps(record) + "\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self._fh.flush()
            self._writes_since_flush = 0

    def on_episode_end(self, summary: Mapping[str, Any]) -> None:
        if self._fh is not None:
            tail = {
                "_kind": "episode_end",
                "wall_time": time.time(),
                "summary": _jsonable(summary),
            }
            self._fh.write(json.dumps(tail) + "\n")
            self._close_fh()
        if self._episode_id is not None:
            (self.out_dir / f"{self._episode_id}_summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2)
            )

    def _close_fh(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None
                self._writes_since_flush = 0


def _jsonable(value: Any) -> Any:
    """Best-effort JSON conversion: numpy scalars/arrays → Python natives."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
