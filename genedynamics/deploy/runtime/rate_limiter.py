"""Fixed-period rate limiter for the deploy loop.

Usage::

    rate = RateLimiter(hz=200)
    while running:
        rate.tick()           # blocks until next tick boundary
        do_one_control_step()

The limiter records jitter / overruns so observers can detect when the
loop falls behind. ``soft=True`` (default) sleeps to the deadline; the
timing-critical real path can use ``soft=False`` to busy-wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from genedynamics.deploy.runtime.clock import Clock, MonotonicClock

__all__ = ["RateLimiter", "RateLimiterStats"]


@dataclass
class RateLimiterStats:
    """Lightweight summary of how the rate limiter has behaved."""

    ticks: int = 0
    overruns: int = 0  # ticks where the deadline had already passed
    jitter_max_s: float = 0.0
    jitter_sum_s: float = 0.0
    overrun_max_s: float = 0.0
    overrun_sum_s: float = 0.0
    last_dt_s: float = 0.0

    @property
    def jitter_mean_s(self) -> float:
        return self.jitter_sum_s / self.ticks if self.ticks else 0.0

    @property
    def overrun_mean_s(self) -> float:
        return self.overrun_sum_s / self.overruns if self.overruns else 0.0


class RateLimiter:
    """Fixed-period rate limiter parameterized by Hz.

    Args:
        hz: Target tick frequency in Hz. Must be positive.
        clock: Clock source. Defaults to :class:`MonotonicClock`. Tests
            should pass :class:`FakeClock` for determinism.
        soft: When ``True``, the limiter sleeps. When ``False``, it
            busy-waits in 100 µs increments — use only on the real-time
            real path.
    """

    def __init__(
        self,
        hz: float,
        *,
        clock: Optional[Clock] = None,
        soft: bool = True,
    ) -> None:
        if hz <= 0.0:
            raise ValueError(f"RateLimiter hz must be positive, got {hz}")
        self.hz = float(hz)
        self.period_s = 1.0 / self.hz
        self.clock = clock or MonotonicClock()
        self.soft = bool(soft)
        self._next_deadline: Optional[float] = None
        self._last_tick: Optional[float] = None
        self.stats = RateLimiterStats()

    def reset(self) -> None:
        self._next_deadline = None
        self._last_tick = None
        self.stats = RateLimiterStats()

    def tick(self) -> float:
        """Block until the next tick boundary; return the new wall time."""
        now = self.clock.now()
        if self._next_deadline is None:
            self._next_deadline = now + self.period_s
            self._last_tick = now
            self.stats.ticks += 1
            self.stats.last_dt_s = 0.0
            return now

        wait = self._next_deadline - now
        if wait > 0:
            if self.soft:
                self.clock.sleep(wait)
            else:
                while self.clock.now() < self._next_deadline:
                    self.clock.sleep(0.0001)
            self.stats.jitter_sum_s += wait
            self.stats.jitter_max_s = max(self.stats.jitter_max_s, wait)
        else:
            overrun = -wait
            self.stats.overruns += 1
            self.stats.overrun_sum_s += overrun
            self.stats.overrun_max_s = max(self.stats.overrun_max_s, overrun)

        now = self.clock.now()
        if self._last_tick is not None:
            self.stats.last_dt_s = now - self._last_tick
        self._last_tick = now
        self.stats.ticks += 1
        self._next_deadline += self.period_s
        # If we've fallen catastrophically behind, snap forward to "now" so
        # the limiter recovers instead of sprinting through queued ticks.
        if self._next_deadline < now:
            self._next_deadline = now + self.period_s
        return now
