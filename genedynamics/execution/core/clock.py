"""
Runtime clock for control loop timing.

Supports real-time, scaled (RTF), and sync modes.
"""

from __future__ import annotations

import time
from typing import Optional


class RuntimeClock:
    """
    Runtime clock for execution control loop.

    - Real-time: sleep to maintain control rate
    - Scaled: real_time_factor for faster/slower sim
    - Sync: wait for planner timestamp before stepping
    """

    def __init__(
        self,
        control_dt: float,
        real_time_factor: float = 1.0,
        sync_mode: bool = False,
    ):
        self.control_dt = control_dt
        self.rtf = max(1e-6, real_time_factor)
        self.sync_mode = sync_mode
        self._t0: Optional[float] = None
        self._step_count = 0

    def start(self) -> None:
        """Start clock."""
        self._t0 = time.perf_counter()
        self._step_count = 0

    def elapsed(self) -> float:
        """Elapsed wall time since start."""
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def sim_time(self) -> float:
        """Simulated time (elapsed * rtf for scaled mode)."""
        return self.elapsed() * self.rtf

    def step_count(self) -> int:
        """Number of steps taken."""
        return self._step_count

    def sleep_until_next_step(self, plan_timestamp: Optional[float] = None) -> None:
        """
        Sleep until next control step.

        In sync_mode, waits until plan_timestamp + control_dt has passed.
        Otherwise sleeps for control_dt/rtf.
        """
        if self.sync_mode and plan_timestamp is not None:
            target = plan_timestamp + self.control_dt
            now = self.sim_time()
            if now < target:
                sleep_wall = (target - now) / self.rtf
                if sleep_wall > 0:
                    time.sleep(sleep_wall)
        else:
            sleep_wall = self.control_dt / self.rtf
            if sleep_wall > 0:
                time.sleep(sleep_wall)
        self._step_count += 1
