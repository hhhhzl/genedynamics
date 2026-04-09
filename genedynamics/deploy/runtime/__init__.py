"""Loop timing helpers — rate limiting, decimation, and clocks.

These are intentionally tiny — they exist only so the pipeline loop can
run at the right frequency on both sim (sub-real-time-factor allowed) and
real hardware (must wall-clock-pace) with the same code.

* :class:`Clock`         — pluggable clock source (monotonic by default)
* :class:`RateLimiter`   — busy-wait or sleep until the next tick
* :class:`Decimation`    — keep one sub-step per ``N`` calls (control vs sim)
"""

from genedynamics.deploy.runtime.clock import Clock, MonotonicClock, FakeClock
from genedynamics.deploy.runtime.decimation import Decimation
from genedynamics.deploy.runtime.rate_limiter import RateLimiter

__all__ = [
    "Clock",
    "Decimation",
    "FakeClock",
    "MonotonicClock",
    "RateLimiter",
]
