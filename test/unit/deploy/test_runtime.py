"""Unit tests for deploy/runtime/ — clock, rate limiter, decimation."""

from __future__ import annotations

import pytest

from genedynamics.deploy.runtime.clock import FakeClock, MonotonicClock
from genedynamics.deploy.runtime.decimation import Decimation
from genedynamics.deploy.runtime.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_monotonic_clock_advances():
    c = MonotonicClock()
    a = c.now()
    c.sleep(0.001)
    b = c.now()
    assert b >= a + 0.0005  # generous to absorb scheduler jitter


def test_fake_clock_is_deterministic():
    c = FakeClock()
    assert c.now() == 0.0
    c.sleep(0.5)
    assert c.now() == 0.5
    c.advance(0.25)
    assert c.now() == 0.75
    c.sleep(-1.0)  # negative sleep is a no-op
    assert c.now() == 0.75


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_first_tick_no_block():
    clock = FakeClock()
    rate = RateLimiter(hz=100, clock=clock)
    t0 = rate.tick()
    assert t0 == pytest.approx(0.0)
    assert rate.stats.ticks == 1


def test_rate_limiter_blocks_until_period_elapses():
    clock = FakeClock()
    rate = RateLimiter(hz=100, clock=clock)  # 10ms period
    rate.tick()
    rate.tick()
    assert clock.now() == pytest.approx(0.01)  # slept exactly 10ms
    rate.tick()
    assert clock.now() == pytest.approx(0.02)


def test_rate_limiter_records_overruns():
    clock = FakeClock()
    rate = RateLimiter(hz=100, clock=clock)  # 10ms period
    rate.tick()
    clock.advance(0.05)  # caller blew the budget by 40ms
    rate.tick()
    assert rate.stats.overruns == 1
    assert rate.stats.overrun_max_s == pytest.approx(0.04)


def test_rate_limiter_invalid_hz_raises():
    with pytest.raises(ValueError):
        RateLimiter(hz=0)
    with pytest.raises(ValueError):
        RateLimiter(hz=-100)


def test_rate_limiter_reset_clears_stats():
    clock = FakeClock()
    rate = RateLimiter(hz=100, clock=clock)
    rate.tick()
    rate.tick()
    assert rate.stats.ticks == 2
    rate.reset()
    assert rate.stats.ticks == 0
    rate.tick()
    assert rate.stats.ticks == 1


def test_rate_limiter_stats_means():
    clock = FakeClock()
    rate = RateLimiter(hz=100, clock=clock)
    rate.tick()
    rate.tick()  # records jitter = period
    rate.tick()
    assert rate.stats.jitter_mean_s > 0
    assert rate.stats.overrun_mean_s == 0  # no overruns


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


def test_decimation_every_one():
    dec = Decimation(every=1)
    assert all(dec.tick() for _ in range(5))


def test_decimation_every_three_fires_on_0_3_6():
    dec = Decimation(every=3)
    fires = [dec.tick() for _ in range(7)]
    assert fires == [True, False, False, True, False, False, True]
    assert dec.calls == 7


def test_decimation_offset_skips_initial_calls():
    dec = Decimation(every=2, offset=2)
    fires = [dec.tick() for _ in range(6)]
    # First 2 calls return False (offset), then every-2 starting at index 2
    assert fires == [False, False, True, False, True, False]


def test_decimation_reset():
    dec = Decimation(every=2)
    dec.tick()  # True
    dec.tick()  # False
    dec.reset()
    assert dec.tick() is True


def test_decimation_invalid_args_raise():
    with pytest.raises(ValueError):
        Decimation(every=0)
    with pytest.raises(ValueError):
        Decimation(every=2, offset=-1)
