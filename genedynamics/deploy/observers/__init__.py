"""Telemetry, rendering, and visualization observers for the deploy pipeline.

Observers are pluggable: a pipeline accepts any number of them and forwards
per-step state / intent / command without the controller knowing.

Implementations:

* :class:`BaseObserver`          — no-op default for every hook
* :class:`LoggerObserver`        — JSONL per-step logs + summary.json
* :class:`RecorderObserver`      — npz state/action timeseries
* :class:`ROS2PublisherObserver` — publish state/cmd to ROS2 topics (Phase 13)

Legacy:

* :class:`WebVizService`    — html/Flask viewer used by the legacy CLI
"""

from genedynamics.deploy.observers.base import BaseObserver
from genedynamics.deploy.observers.logger import LoggerObserver
from genedynamics.deploy.observers.recorder import RecorderObserver
from genedynamics.deploy.observers.ros2_publisher import ROS2PublisherObserver
from genedynamics.deploy.observers.web_viz import WebVizService

__all__ = [
    "BaseObserver",
    "LoggerObserver",
    "RecorderObserver",
    "ROS2PublisherObserver",
    "WebVizService",
]
