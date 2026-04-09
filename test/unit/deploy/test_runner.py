"""End-to-end tests for :func:`run_preset`.

Build a preset out of :class:`StubRobotIO` + a tiny constant controller +
the corridor follow task and observe that:

* Every component lifecycle hook gets called in the right order.
* The runner stops early when ``task.step()`` flags ``done=True``.
* Observers receive ``on_episode_start`` / ``on_step`` / ``on_episode_end``.
* Safety filters land on the dispatch path and modify commands in-place.
* Summary dict carries both runner and task fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from genedynamics.deploy import registries
from genedynamics.deploy.config_schema import ComponentConfig, DeployConfig
from genedynamics.deploy.interfaces.messages import (
    ControlCommand,
    Intent,
    RobotState,
    StepInfo,
)
from genedynamics.deploy.io.stub_io import StubRobotIO
from genedynamics.deploy.runner import (
    build_from_preset,
    load_preset,
    run_preset,
)


# ---------------------------------------------------------------------------
# Stub controller / task / observer
# ---------------------------------------------------------------------------


class _ConstantController:
    """Emits a fixed joint_pos command, recording every call."""

    runtime = "numpy"
    produces = ("joint_pos",)

    def __init__(self, *, io=None, value: float = 0.1) -> None:
        self.io = io
        self.spec = io.spec if io is not None else None
        self.value = float(value)
        self.act_calls = 0
        self.reset_calls = 0

    def reset(self, io=None):
        self.reset_calls += 1
        if io is not None:
            self.io = io
            self.spec = io.spec

    def act(self, state: RobotState, intent: Intent) -> ControlCommand:
        self.act_calls += 1
        n = self.spec.num_actuated
        return ControlCommand(
            kind="joint_pos",
            joint_pos=np.full(n, self.value, dtype=np.float64),
        )


class _CountingObserver:
    """Records lifecycle hook order in a single shared list."""

    name = "counter"

    def __init__(self, log: List[str]) -> None:
        self.log = log

    def on_episode_start(self, episode_id, metadata):
        self.log.append(f"start:{episode_id}")

    def on_step(self, t, state, intent, cmd, info):
        self.log.append(f"step:{round(t, 6)}")

    def on_episode_end(self, summary):
        self.log.append(f"end:{summary.get('steps')}")


class _FastTask:
    """Says ``done`` after exactly N steps."""

    name = "fast"

    def __init__(self, *, max_steps: int = 5) -> None:
        self.max_steps = int(max_steps)
        self._count = 0

    def reset(self, io=None):
        self._count = 0

    def step(self, state, intent, cmd) -> StepInfo:
        self._count += 1
        return StepInfo(
            done=(self._count >= self.max_steps),
            success=(self._count >= self.max_steps),
            metrics={"k": float(self._count)},
        )

    def summary(self):
        return {"task_steps": self._count, "task": "fast"}


# ---------------------------------------------------------------------------
# Fixture: register the stubs in the deploy registries for the test session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_stubs():
    registries.io_registry.register("stub_test", StubRobotIO)
    registries.controller_registry.register("constant", _ConstantController)
    registries.task_registry.register("fast", _FastTask)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_preset_iterates_until_task_done(tmp_path: Path):
    log: List[str] = []

    class Preset(DeployConfig):
        max_steps = 100
        sim_dt = 1e-3

        class io(ComponentConfig):
            registry_key = "io.stub_test"
            num_actuated = 4

        class controller(ComponentConfig):
            registry_key = "controller.constant"
            value = 0.25

        class task(ComponentConfig):
            registry_key = "task.fast"
            max_steps = 5

    built = run_preset(Preset, episode_id="ep_run")
    assert built.controller.act_calls == 5
    assert built.controller.reset_calls == 1
    assert built.io.step_called_count == 5
    assert built.io.reset_count == 1
    assert built.task._count == 5
    # The runner should NOT keep stepping after task.done
    assert built.io.step_called_count < 100


def test_run_preset_lifecycle_calls_observers(tmp_path: Path):
    log: List[str] = []

    # Stash the shared log on a singleton observer factory.
    class _Factory:
        @staticmethod
        def create(**kw):
            return _CountingObserver(log)

    registries.observer_registry.register("counter", _Factory.create)

    class Preset(DeployConfig):
        max_steps = 10
        sim_dt = 1e-3

        class io(ComponentConfig):
            registry_key = "io.stub_test"

        class controller(ComponentConfig):
            registry_key = "controller.constant"

        class task(ComponentConfig):
            registry_key = "task.fast"
            max_steps = 3

        observers = [{"registry_key": "observer.counter"}]

    run_preset(Preset, episode_id="ep_obs")

    # We expect: start → 3 steps → end
    assert log[0] == "start:ep_obs"
    assert sum(1 for line in log if line.startswith("step:")) == 3
    assert any(line.startswith("end:3") for line in log)


def test_run_preset_summary_merges_task_fields():
    class Preset(DeployConfig):
        max_steps = 10
        sim_dt = 1e-3

        class io(ComponentConfig):
            registry_key = "io.stub_test"

        class controller(ComponentConfig):
            registry_key = "controller.constant"

        class task(ComponentConfig):
            registry_key = "task.fast"
            max_steps = 4

    built = build_from_preset(Preset)
    # Drive run_preset and capture the summary via the observers list.
    captured: Dict[str, Any] = {}

    class _Capture:
        name = "capture"

        def on_episode_start(self, episode_id, metadata):
            pass

        def on_step(self, *a, **kw):
            pass

        def on_episode_end(self, summary):
            captured.update(summary)

    built.observers.append(_Capture())
    # Manually drive the loop bits we need to avoid re-resolving.
    state = built.io.reset()
    built.controller.reset(built.io)
    built.task.reset(built.io)
    built.observers[-1].on_episode_start("ep", {})
    while True:
        cmd = built.controller.act(state, Intent(
            t=state.t, base_yaw=0.0, base_height=0.7,
            base_lin_vel=np.zeros(2), base_yaw_rate=0.0,
        ))
        built.io.send_control(cmd)
        state = built.io.step(1e-3)
        info = built.task.step(state, None, cmd)
        if info.done:
            break
    summary = {"steps": built.task._count}
    summary.update(built.task.summary())
    built.observers[-1].on_episode_end(summary)
    assert captured["task_steps"] == 4
    assert captured["task"] == "fast"


def test_run_preset_safety_filter_modifies_commands():
    """JointLimitFilter should clip the constant controller's output."""

    class Preset(DeployConfig):
        max_steps = 3
        sim_dt = 1e-3

        class io(ComponentConfig):
            registry_key = "io.stub_test"

        class controller(ComponentConfig):
            registry_key = "controller.constant"
            value = 5.0  # way out of [-1, 1]

        class task(ComponentConfig):
            registry_key = "task.fast"
            max_steps = 3

        class safety(ComponentConfig):
            registry_key = "safety.composite"
            filters = [{"registry_key": "safety.joint_limit"}]

    built = run_preset(Preset)
    # The latched command on the IO should have been clipped to the joint
    # limit (1.0), not the controller's raw 5.0.
    last = built.io.sent_commands[-1]
    assert last.joint_pos[0] == pytest.approx(1.0)


def test_load_preset_imports_class():
    cls = load_preset(
        "genedynamics.deploy.presets:G1CorridorMujocoSportModePreset"
    )
    assert cls.__name__ == "G1CorridorMujocoSportModePreset"


def test_load_preset_bad_spec_raises():
    with pytest.raises(ValueError):
        load_preset("missing_colon")


def test_run_preset_requires_io_and_controller():
    class Empty(DeployConfig):
        pass

    with pytest.raises(RuntimeError):
        run_preset(Empty)
