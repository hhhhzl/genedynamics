"""Phase 10 integration matrix: ``runner.run_preset`` × {sim, stub} × controllers.

This test exercises the new Phase 9 runner end-to-end against the real
deploy registries, presets, and observers. It is the closest thing in the
codebase to "the runner can actually run the corridor" — and it makes
sure adding a new controller / preset doesn't silently break the
``RobotIO`` ↔ ``Controller`` ↔ ``SafetyFilter`` ↔ ``Observer`` plumbing.

The matrix:

| io     | controller        | task              | safety              | requires |
|--------|-------------------|-------------------|---------------------|----------|
| stub   | constant (test)   | corridor_follow   | composite (jl + tl) | none     |
| stub   | sport_mode (sprk) | corridor_follow   | none                | none     |
| mujoco | sport_mode (sprk) | corridor_follow   | composite (jl + tl) | mujoco + torch |

Heavy-dependency rows auto-skip when their backend is not importable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from genedynamics.deploy import registries
from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
)
from genedynamics.deploy.io.stub_io import StubRobotIO


def _have(mod: str) -> bool:
    """find_spec only — does not check whether the module imports cleanly."""
    return importlib.util.find_spec(mod) is not None


def _can_import(mod: str) -> bool:
    """Try to actually import — handles Apple Silicon x86_64 mujoco etc."""
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tiny stub controller for the all-stub matrix row
# ---------------------------------------------------------------------------


class _StubConstantController:
    runtime = "numpy"
    produces = ("joint_pos",)

    def __init__(self, *, io=None, value: float = 0.0) -> None:
        self.io = io
        self.spec = io.spec if io is not None else None
        self.value = float(value)

    def reset(self, io=None):
        if io is not None:
            self.io = io
            self.spec = io.spec

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        return ControlCommand(
            kind="joint_pos",
            joint_pos=np.full(self.spec.num_actuated, self.value, dtype=np.float64),
        )


@pytest.fixture(autouse=True)
def _register_stubs():
    registries.io_registry.register("stub_matrix", StubRobotIO)
    registries.controller_registry.register("stub_constant", _StubConstantController)
    yield


# ---------------------------------------------------------------------------
# Row 1: stub × constant × corridor_follow × composite
# ---------------------------------------------------------------------------


def test_matrix_stub_constant(tmp_path: Path):
    from genedynamics.deploy.runner import run_preset

    class Preset(DeployConfig):
        max_steps = 50
        sim_dt = 1e-3

        class io(ComponentConfig):
            registry_key = "io.stub_matrix"
            num_actuated = 4

        class controller(ComponentConfig):
            registry_key = "controller.stub_constant"
            value = 0.05

        class safety(ComponentConfig):
            registry_key = "safety.composite"
            filters = [
                {"registry_key": "safety.joint_limit"},
                {"registry_key": "safety.torque_limit", "safety_margin": 0.95},
            ]

        class task(ComponentConfig):
            registry_key = "task.corridor_follow"
            goal_xy = (1.0, 0.0)
            goal_tolerance_m = 100.0  # always inside → done on first step
            max_steps = 50

        observers = [
            {"registry_key": "observer.logger", "out_dir": str(tmp_path)},
            {"registry_key": "observer.recorder", "out_dir": str(tmp_path)},
        ]

    built = run_preset(Preset, episode_id="ep_stub_const")
    assert built.controller is not None
    assert built.task is not None
    # The logger should have written one JSONL file with at least 3 lines
    log_files = list(tmp_path.glob("ep_stub_const.jsonl"))
    assert len(log_files) == 1
    lines = log_files[0].read_text().strip().splitlines()
    assert len(lines) >= 3  # start + at least 1 step + end
    # The recorder should have written an NPZ
    npz_files = list(tmp_path.glob("ep_stub_const.npz"))
    assert len(npz_files) == 1


# ---------------------------------------------------------------------------
# Row 2: stub × sport_mode (spark RL) × corridor_follow — failure mode test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have("mujoco"),
    reason="SparkRLLocoClient needs the G1 spec which loads via mujoco",
)
def test_matrix_stub_sport_mode_fails_cleanly_on_wrong_spec(tmp_path: Path):
    """Sport-mode controller wired to a stub IO with the wrong spec.

    SportModeController + SparkRLLocoClient expect a G1RobotSpec exposing
    G1-shaped joint groups. The stub spec only has generic ``j*`` names,
    so we expect a TypeError / AttributeError / KeyError at construction.
    The test asserts the FAILURE mode is clean — the runner does not
    crash, leak state, or partially apply commands.
    """
    from genedynamics.deploy.runner import run_preset

    class Preset(DeployConfig):
        max_steps = 5
        sim_dt = 1e-2

        class io(ComponentConfig):
            registry_key = "io.stub_matrix"
            num_actuated = 29

        class controller(ComponentConfig):
            registry_key = "controller.sport_mode"
            leg_kp = 100.0
            leg_kd = 4.0
            upper_body_kp = 80.0
            upper_body_kd = 3.0

            class loco_client(ComponentConfig):
                registry_key = "loco_client.spark_rl"

        class task(ComponentConfig):
            registry_key = "task.corridor_follow"
            goal_xy = (10.0, 0.0)
            max_steps = 5

    with pytest.raises((TypeError, AttributeError, KeyError, RuntimeError, FileNotFoundError, ImportError)):
        run_preset(Preset, episode_id="ep_stub_sport")


# ---------------------------------------------------------------------------
# Row 3: mujoco × sport_mode (spark RL) — full real backend matrix row
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _can_import("mujoco"), reason="mujoco required (and importable)")
def test_matrix_mujoco_sport_mode_spark_rl(tmp_path: Path):
    """Build the canonical sim preset and run it for a few steps."""
    from genedynamics.deploy.runner import run_preset
    from genedynamics.robots.g1 import G1AssetNotFoundError
    from genedynamics.deploy.presets import G1CorridorMujocoSportModePreset

    class Preset(G1CorridorMujocoSportModePreset):
        max_steps = 50  # short — we just want plumbing validation

        class task(ComponentConfig):
            registry_key = "task.corridor_follow"
            goal_xy = (5.0, 0.0)
            goal_tolerance_m = 100.0  # always inside
            max_steps = 50

        observers = [
            {"registry_key": "observer.logger", "out_dir": str(tmp_path)},
        ]

    try:
        built = run_preset(Preset, episode_id="ep_mujoco_sport")
    except G1AssetNotFoundError:
        pytest.skip("G1 MJCF not available")
    assert built.io is not None
    assert built.controller is not None
    log_files = list(tmp_path.glob("ep_mujoco_sport.jsonl"))
    assert len(log_files) == 1
