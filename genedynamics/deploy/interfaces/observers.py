"""Observer protocol — cross-cutting telemetry.

Observers receive every step's state, intent, command and step info without
controllers / IO knowing about them. The pipeline owns a list of observers
and dispatches in order at fixed hooks: episode start, every step, episode
end. Adding an observer is a one-line config change; removing it is the
same.

Concrete observers under ``deploy/observers/``:

* ``LoggerObserver``       — structured per-step logs (parquet / jsonl)
* ``RecorderObserver``     — saves state / action timeseries to disk
* ``MujocoRendererObserver`` — renders rollouts to mp4 / gif
* ``WebVizObserver``       — wraps the existing :class:`WebVizService`
* ``Ros2PublisherObserver`` — publishes to ROS topics for monitoring
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)

__all__ = ["Observer"]


@runtime_checkable
class Observer(Protocol):
    """Per-episode telemetry sink.

    Implementations are encouraged to be side-effect-only (no return values
    other than ``None``) so that observer ordering does not affect controller
    behavior. Failure inside an observer should be caught by the pipeline and
    converted to a warning rather than aborting the episode.
    """

    name: str

    def on_episode_start(self, episode_id: str, metadata: Mapping[str, Any]) -> None:
        """Called once at the beginning of an episode."""
        ...

    def on_step(
        self,
        t: float,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
        info: StepInfo,
    ) -> None:
        """Called once per control step after the IO has advanced."""
        ...

    def on_episode_end(self, summary: Mapping[str, Any]) -> None:
        """Called once at the end of an episode (success or abort)."""
        ...
