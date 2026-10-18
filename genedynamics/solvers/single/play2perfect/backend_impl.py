"""Backend contract for the Play2Perfect solver frontend."""

from __future__ import annotations

from typing import Any, Protocol


class Play2PerfectBackend(Protocol):
    def reset(self, batch_size: int | None = None) -> None: ...
    def act(self, observation: Any, *, coefficient: Any = 0.0) -> Any: ...


__all__ = ["Play2PerfectBackend"]
