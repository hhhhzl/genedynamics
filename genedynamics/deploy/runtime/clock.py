"""Pluggable clock sources for the deploy loop.

Tests inject :class:`FakeClock` to drive the rate limiter deterministically.
Production code uses :class:`MonotonicClock`. The :class:`Clock` protocol
is the contract.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "MonotonicClock", "FakeClock"]


@runtime_checkable
class Clock(Protocol):
    """Minimal clock interface used by :class:`RateLimiter`."""

    def now(self) -> float:
        """Return the current time in seconds (any monotonic source)."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` (advance ``now()`` by at least that much)."""
        ...


class MonotonicClock:
    """Real wall-clock implementation. Used by default."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic clock — :meth:`sleep` advances the internal counter.

    Useful in unit tests so :class:`RateLimiter` doesn't actually block.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._t += float(seconds)

    def advance(self, seconds: float) -> None:
        self._t += float(seconds)
