"""Nominal evidence and native reference-terminal semantics stay distinct."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from genedynamics.solvers.single.acp.learning.nominal_diagnostics import learning_summary
from genedynamics.solvers.single.acp.learning import nominal_diagnostics as acp_diagnostics
from genedynamics.solvers.single.human2sim2robot.learning import nominal_diagnostics as h2s_diagnostics
from genedynamics.solvers.single.human2sim2robot.learning.task_env import ObjectTrajectoryTaskEnv


class StaticTask:
    def __init__(self, count=1):
        self.count = count
        self.spec = SimpleNamespace(family="free")
        self.config = SimpleNamespace(control_dt=.02, timeout_steps=20)
        self.runner = SimpleNamespace(skill=SimpleNamespace(
            linear_speed=.1, angular_speed=.4, log_stiffness_limit=1.))
        self.observation_size = 3
        self.state = SimpleNamespace(steps=np.zeros(count))
        self.position = np.zeros((count, 3), np.float32)
        self.reset_masks = []

    def flat_observation(self, observation=None):
        return self.position.copy()

    def reset(self, slots, *, seed, mask=None):
        mask = np.ones(self.count, bool) if mask is None else np.asarray(mask)
        self.reset_masks.append(mask.copy())
        self.state.steps[mask] = 0
        self.position[mask] = 0

    def step(self, actions):
        self.state.steps += 1
        return SimpleNamespace(
            observation={"tcp_position": self.position.copy(),
                "tcp_rotation": np.tile(np.eye(3, dtype=np.float32), (self.count, 1, 1))},
            done=np.zeros(self.count, bool), status=np.zeros(self.count, int))

    def get_checkpoint(self):
        return deepcopy({"steps": self.state.steps, "position": self.position})

    def restore_checkpoint(self, value):
        self.state.steps = value["steps"].copy()
        self.position = value["position"].copy()


def make_env(*, reference_terminal=False, count=1, demo_dt=.02):
    return ObjectTrajectoryTaskEnv(StaticTask(count), np.tile(np.eye(4), (2, 1, 1)),
        num_envs=count, demo_dt=demo_dt, terminate_on_reference_end=reference_terminal)


def test_native_terminal_uses_pre_increment_index_and_is_not_success():
    env = make_env(reference_terminal=True)
    env.reset()
    actions = np.zeros((1, 13), np.float32)
    assert not env.step(actions, auto_reset=False)[2][0]
    assert not env.step(actions, auto_reset=False)[2][0]
    _, _, done, info = env.step(actions, auto_reset=False)
    assert done[0] and info["reference_motion_done"][0]
    assert info["task_status"][0] == 0
    assert not info["time_outs"][0]


def test_legacy_default_does_not_change_at_reference_end():
    env = make_env()
    env.reset()
    for _ in range(4):
        _, _, done, info = env.step(np.zeros((1, 13), np.float32), auto_reset=False)
        assert not done[0]
        assert "reference_motion_done" not in info
    assert "terminate_on_reference_end" not in env.get_env_state()


def test_fractional_reference_clock_preserves_native_boundary():
    env = make_env(reference_terminal=True, demo_dt=.04)
    env.reset()
    for _ in range(4):
        assert not env.step(np.zeros((1, 13), np.float32), auto_reset=False)[2][0]
    assert env.step(np.zeros((1, 13), np.float32), auto_reset=False)[2][0]


def test_stop_radius_does_not_fake_reference_completion():
    env = make_env(reference_terminal=True)
    env.reset()
    env.task.position[:] = [.3, 0, 0]
    for _ in range(4):
        assert not env.step(np.zeros((1, 13), np.float32), auto_reset=False)[2][0]
    assert env.reference_index[0] == 0


def test_reference_terminal_resets_only_finished_world():
    env = make_env(reference_terminal=True, count=2)
    env.reset()
    env.reference_index[:] = [2, 0]
    _, _, done, info = env.step(np.zeros((2, 13), np.float32))
    np.testing.assert_array_equal(done, [True, False])
    np.testing.assert_array_equal(info["task_status"], [0, 0])
    np.testing.assert_array_equal(env.task.reset_masks[-1], [True, False])
    np.testing.assert_array_equal(env.reference_index, [0, 1])


def test_continuation_binds_training_termination():
    env = make_env(reference_terminal=True)
    env.reset()
    before = env.get_env_state()
    env.step(np.zeros((1, 13), np.float32), auto_reset=False)
    env.set_env_state(before)
    np.testing.assert_array_equal(env.reference_index, before["reference_index"])
    with pytest.raises(ValueError, match="termination"):
        make_env().set_env_state(before)


def test_low_denoising_loss_does_not_prove_convergence_or_independent_windows():
    training = {"completed_updates": 8000,
        "validations": [{"update": 6000, "loss": .013}, {"update": 8000, "loss": .011}]}
    demos = {"splits": {"fit": {"episodes": 6, "windows": 3000},
                        "validation": {"episodes": 2}}}
    row = learning_summary(training, demos)
    assert row["independent_fit_episodes"] == 6
    assert not row["windows_are_independent_episodes"]
    assert not row["convergence_proven"]


@pytest.fixture(params=["acp", "human2sim2robot"])
def gpu_recipe(request, tmp_path):
    root = Path(__file__).resolve().parents[2]
    if request.param == "acp":
        relative = "arm/obstacle_transfer/training/acp_nominal_single_demo_gpu0.json"
        plan = acp_diagnostics.overfit_plan
        inputs = ("checkpoint", "training_report", "demonstration_report", "teacher_collection_config")
    else:
        relative = "arm/guided_contact_alignment/training/human2sim2robot_paired_reference_learning_gpu0.json"
        plan = h2s_diagnostics.paired_learning_plan
        inputs = ("checkpoint", "training_config", "termination_diagnostic")
    config = json.loads((root / "configs" / relative).read_text())
    prerequisite = tmp_path / "input.json"
    prerequisite.write_text(json.dumps({
        "paired_prefix_exact": True, "ppo_integration_smoke": {"finite_metrics": True}}))
    binding = {"path": prerequisite.name, "sha256": hashlib.sha256(prerequisite.read_bytes()).hexdigest()}
    for name in inputs:
        config[name] = dict(binding)
    config["output"] = "unused_output"
    config["execution"]["physical_uuid"] = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    return plan, config, tmp_path


def test_gpu_plan_uses_recipe_uuid_instead_of_a_machine_constant(gpu_recipe):
    plan, config, root = gpu_recipe
    result = plan(config, root)
    assert result["execution"] == config["execution"]
    assert not result["gpu_started"]


@pytest.mark.parametrize("uuid", [None, "", "GPU-1234", "01234567-89ab-cdef-0123-456789abcdef"])
def test_gpu_plan_rejects_invalid_uuid(gpu_recipe, uuid):
    plan, config, root = gpu_recipe
    config["execution"]["physical_uuid"] = uuid
    with pytest.raises(ValueError):
        plan(config, root)


def test_gpu_plan_requires_explicit_uuid(gpu_recipe):
    plan, config, root = gpu_recipe
    del config["execution"]["physical_uuid"]
    with pytest.raises(ValueError):
        plan(config, root)


def test_gpu_plan_accepts_explicit_gpu1_binding(gpu_recipe):
    plan, config, root = gpu_recipe
    config["execution"]["cuda_visible_devices"] = "1"
    assert plan(config, root)["execution"]["cuda_visible_devices"] == "1"


@pytest.mark.parametrize("field,value", [("device", "cuda:1"), ("cuda_visible_devices", "0,1")])
def test_gpu_plan_rejects_unbounded_device_binding(gpu_recipe, field, value):
    plan, config, root = gpu_recipe
    config["execution"][field] = value
    with pytest.raises(ValueError):
        plan(config, root)
