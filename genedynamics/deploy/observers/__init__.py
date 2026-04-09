"""Telemetry, rendering, and visualization observers for the deploy pipeline.

Observers are pluggable: a pipeline accepts any number of them and forwards
per-step state / intent / command without the controller knowing.

Phase 8 surface:

* :class:`BaseObserver`     — no-op default for every hook
* :class:`LoggerObserver`   — JSONL per-step logs + summary.json
* :class:`RecorderObserver` — npz state/action timeseries

Legacy:

* :class:`WebVizService`    — html/Flask viewer used by the legacy CLI
"""

from genedynamics.deploy.observers.base import BaseObserver
from genedynamics.deploy.observers.logger import LoggerObserver
from genedynamics.deploy.observers.recorder import RecorderObserver
from genedynamics.deploy.observers.web_viz import WebVizService

__all__ = [
    "BaseObserver",
    "LoggerObserver",
    "RecorderObserver",
    "WebVizService",
]
