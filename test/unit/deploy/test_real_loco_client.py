"""Unit tests for the asynchronous real-hardware locomotion client."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import List, Tuple

import numpy as np
import pytest

from genedynamics.deploy.controllers.sport_mode.real_loco_client import RealLocoClient
from genedynamics.deploy.interfaces.messages import LocoCommand


@dataclass
class _StubSdk:
    """Capture SDK calls without requiring ``unitree_sdk2py`` or hardware."""

    velocities: List[Tuple[float, float, float, float]] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    balance_calls: int = 0
    stop_calls: int = 0

    def BalanceStand(self, _flag: int) -> None:  # noqa: N802 - SDK spelling
        self.balance_calls += 1

    def SetVelocity(  # noqa: N802
        self, vx: float, vy: float, yaw: float, duration: float
    ) -> None:
        self.velocities.append((float(vx), float(vy), float(yaw), float(duration)))

    def SetStandHeight(self, height: float) -> None:  # noqa: N802
        self.heights.append(float(height))

    def StopMove(self) -> None:  # noqa: N802
        self.stop_calls += 1


def _make_client(rate_limit_hz: float = 100.0) -> Tuple[RealLocoClient, _StubSdk]:
    sdk = _StubSdk()
    client = RealLocoClient(
        sdk_client=sdk,
        rate_limit_hz=rate_limit_hz,
        move_duration_s=0.25,
    )
    return client, sdk


def _wait_until(predicate, timeout: float = 0.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("background sender did not publish before the test deadline")


def test_step_before_reset_raises():
    client, _ = _make_client()
    with pytest.raises(RuntimeError):
        client.step(LocoCommand(0.1, 0.0, 0.0), 0.02, np.zeros(3), 0.0)


def test_reset_calls_balance_stand_and_clears_state():
    client, sdk = _make_client()
    try:
        client.reset(np.zeros(3), 0.0)
        assert sdk.balance_calls == 1
        assert client.swing_foot is None
        assert client.phase == 0.0
    finally:
        client.stop()


def test_step_publishes_latest_velocity_asynchronously():
    client, sdk = _make_client()
    try:
        client.reset(np.zeros(3), 0.0)
        result = client.step(
            LocoCommand(0.5, -0.1, 0.2),
            dt=0.001,
            pelvis_world=np.zeros(3),
            pelvis_yaw=0.0,
        )

        # The control loop receives no joint overrides and never waits for RPC.
        assert dict(result) == {}
        _wait_until(lambda: len(sdk.velocities) >= 1)
        assert sdk.velocities[0] == (0.5, -0.1, 0.2, 0.25)
    finally:
        client.stop()


def test_sender_republishes_at_bounded_rate():
    client, sdk = _make_client(rate_limit_hz=20.0)
    try:
        client.reset(np.zeros(3), 0.0)
        client.step(LocoCommand(0.5, 0.0, 0.0), 0.001, np.zeros(3), 0.0)
        _wait_until(lambda: len(sdk.velocities) >= 2)
        assert len(sdk.velocities) <= 3
    finally:
        client.stop()


def test_body_height_is_sent_only_when_provided():
    client, sdk = _make_client()
    try:
        client.reset(np.zeros(3), 0.0)
        client.step(LocoCommand(0.0, 0.0, 0.0), 0.001, np.zeros(3), 0.0)
        _wait_until(lambda: len(sdk.velocities) >= 1)
        assert sdk.heights == []

        client.step(
            LocoCommand(0.0, 0.0, 0.0, body_height=0.7),
            0.001,
            np.zeros(3),
            0.0,
        )
        _wait_until(lambda: len(sdk.heights) >= 1)
        assert sdk.heights[0] == pytest.approx(0.7)
    finally:
        client.stop()


def test_stop_is_idempotent_and_requests_halt():
    client, sdk = _make_client()
    client.reset(np.zeros(3), 0.0)
    client.stop()
    client.stop()
    assert sdk.stop_calls == 2
