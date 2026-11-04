"""Interactive execution is separate from design-batch evaluation."""

from typing import Any, Protocol, runtime_checkable

from .capabilities import ExecutionCapabilities
from .results import ExecutionStep, ResetResult


@runtime_checkable
class InteractiveEnvironment(Protocol):
    capabilities: ExecutionCapabilities
    num_environments: int

    def reset(self, *, seed: int | None = None, indices: Any = None) -> ResetResult: ...
    def step(self, actions: Any) -> ExecutionStep: ...
    def close(self) -> None: ...
