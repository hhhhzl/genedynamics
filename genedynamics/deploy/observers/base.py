"""Base scaffolding for :class:`Observer` implementations.

Subclass :class:`BaseObserver` to inherit a no-op default for every hook so
implementations only have to override the ones they care about. The base
class also stores ``name`` so the pipeline can identify observers in logs.
"""

from __future__ import annotations

from typing import Any, Mapping

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)

__all__ = ["BaseObserver"]


class BaseObserver:
    """Optional base class — implements all hooks as no-ops."""

    def __init__(self, name: str) -> None:
        self.name = name

    def on_episode_start(self, episode_id: str, metadata: Mapping[str, Any]) -> None:
        pass

    def on_step(
        self,
        t: float,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
        info: StepInfo,
    ) -> None:
        pass

    def on_episode_end(self, summary: Mapping[str, Any]) -> None:
        pass
