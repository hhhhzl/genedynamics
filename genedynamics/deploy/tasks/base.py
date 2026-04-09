"""``ExecutionTask`` protocol and base implementation.

A deploy task is a small per-episode object the runner consults to decide
when to stop and what to record. The protocol is intentionally tiny — most
of the layout / extract_position logic lives upstream in
:mod:`genedynamics.tasks` and is consumed by followers, not tasks.

Lifecycle::

    task.reset(io)            # called once after io.reset()
    intent = follower.step(...)
    cmd    = controller.act(...)
    info = task.step(state, intent, cmd)   # returns StepInfo
    if info.done: break
    summary = task.summary()  # called once at episode end
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)

__all__ = ["ExecutionTask", "BaseExecutionTask"]


@runtime_checkable
class ExecutionTask(Protocol):
    """Per-episode task hook for the deploy runner."""

    name: str

    def reset(self, io: Any) -> None:
        """Bind the task to an IO and clear per-episode state."""
        ...

    def step(
        self,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
    ) -> StepInfo:
        """Return a :class:`StepInfo` for the current step."""
        ...

    def summary(self) -> Mapping[str, Any]:
        """Return an arbitrary summary dict at episode end."""
        ...


class BaseExecutionTask:
    """Default no-op task — never finishes, no metrics, used as a fallback."""

    def __init__(self, name: str = "noop") -> None:
        self.name = name
        self._step_count = 0

    def reset(self, io: Any) -> None:
        self._step_count = 0

    def step(
        self,
        state: RobotState,
        intent: Intent,
        cmd: ControlCommand,
    ) -> StepInfo:
        self._step_count += 1
        return StepInfo(done=False)

    def summary(self) -> Mapping[str, Any]:
        return {"steps": self._step_count}
