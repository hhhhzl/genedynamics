"""Unit tests for :class:`RealLocoClient`.

These tests inject a stub SDK client so they run on any machine — the real
``unitree_sdk2py`` is not required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pytest

from genedynamics.deploy.controllers.sport_mode.real_loco_client import (
    RealLocoClient,
    _cmd_changed,
)
from genedynamics.deploy.interfaces.messages import LocoCommand


@dataclass
class _StubSdk:
    """Captures every SDK call so the tests can assert on dispatch order."""

    moves: List[Tuple[float, float, float]] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    balance_calls: int = 0

    def Init(self) -> None:  # noqa: N802 - mimic SDK
        pass

    def SetTimeout(self, _t: float) -> None:  # noqa: N802
        pass

    def BalanceStand(self, _flag: int) -> None:  # noqa: N802
        self.balance_calls += 1

    def Move(self, vx: float, vy: float, yaw: float) -> None:  # noqa: N802
        self.moves.append((float(vx), float(vy), float(yaw)))

    def SetStandHeight(self, h: float) -> None:  # noqa: N802
        self.heights.append(float(h))


def _make_client(rate_limit_hz: float = 50.0) -> Tuple[RealLocoClient, _StubSdk]:
    sdk = _StubSdk()
    client = RealLocoClient(sdk_client=sdk, rate_limit_hz=rate_limit_hz)
    return client, sdk


def test_step_before_reset_raises():
    client, _ = _make_client()
    with pytest.raises(RuntimeError):
        client.step(LocoCommand(0.1, 0.0, 0.0), 0.02, np.zeros(3), 0.0)


def test_reset_calls_balance_stand_and_clears_state():
    client, sdk = _make_client()
    client.reset(np.zeros(3), 0.0)
    assert sdk.balance_calls == 1
    assert client.swing_foot is None
    assert client.phase == 0.0


def test_first_step_always_publishes():
    client, sdk = _make_client(rate_limit_hz=50.0)
    client.reset(np.zeros(3), 0.0)
    client.step(LocoCommand(0.5, 0.0, 0.0), dt=0.001, pelvis_world=np.zeros(3), pelvis_yaw=0.0)
    assert len(sdk.moves) == 1
    assert sdk.moves[0] == (0.5, 0.0, 0.0)


def test_rate_limit_suppresses_redundant_publishes():
    client, sdk = _make_client(rate_limit_hz=50.0)  # 20 ms min period
    client.reset(np.zeros(3), 0.0)
    cmd = LocoCommand(0.5, 0.0, 0.0)
    # Tight loop at 1 ms — only the first call should publish.
    for _ in range(10):
        client.step(cmd, dt=0.001, pelvis_world=np.zeros(3), pelvis_yaw=0.0)
    assert len(sdk.moves) == 1


def test_command_change_bypasses_rate_limit():
    client, sdk = _make_client(rate_limit_hz=50.0)
    client.reset(np.zeros(3), 0.0)
    client.step(LocoCommand(0.5, 0.0, 0.0), 0.001, np.zeros(3), 0.0)
    client.step(LocoCommand(0.0, 0.5, 0.0), 0.001, np.zeros(3), 0.0)
    assert len(sdk.moves) == 2


def test_body_height_only_published_when_provided():
    client, sdk = _make_client(rate_limit_hz=50.0)
    client.reset(np.zeros(3), 0.0)
    client.step(LocoCommand(0.0, 0.0, 0.0), 0.001, np.zeros(3), 0.0)
    assert sdk.heights == []
    client.step(LocoCommand(0.0, 0.0, 0.0, body_height=0.7), 0.001, np.zeros(3), 0.0)
    assert sdk.heights == [0.7]


def test_step_returns_empty_joint_dict():
    client, _ = _make_client()
    client.reset(np.zeros(3), 0.0)
    out = client.step(LocoCommand(0.1, 0.0, 0.0), 0.02, np.zeros(3), 0.0)
    assert dict(out) == {}


def test_cmd_changed_helpers():
    a = LocoCommand(0.1, 0.0, 0.0)
    assert not _cmd_changed(a, a)
    assert _cmd_changed(a, LocoCommand(0.2, 0.0, 0.0))
    assert _cmd_changed(a, LocoCommand(0.1, 0.05, 0.0))
    assert _cmd_changed(a, LocoCommand(0.1, 0.0, 0.1))
    assert _cmd_changed(LocoCommand(0.1, 0.0, 0.0, 0.7), LocoCommand(0.1, 0.0, 0.0))
    assert _cmd_changed(
        LocoCommand(0.1, 0.0, 0.0, 0.7),
        LocoCommand(0.1, 0.0, 0.0, 0.75),
    )
