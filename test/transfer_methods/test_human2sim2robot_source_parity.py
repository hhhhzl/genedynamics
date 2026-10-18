"""Parity against pinned, unmodified Human2Sim2Robot actor/player source.

The reference imports only PPO utilities. Player methods are compiled directly
from upstream AST to avoid importing Isaac Gym / gym in a source-parity test.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest

torch = pytest.importorskip("torch")

from genedynamics.solvers.single.human2sim2robot import (
    Human2Sim2RobotSolver, MlpConfig, NetworkConfig, RnnConfig,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "baselines" / "Human2Sim2Robot"
PORT = ROOT / "genedynamics" / "solvers" / "single" / "human2sim2robot"
COMMIT = "894eae2ec3ae39a573b81bd1860d14cc6bdfa6df"
DEVICE = os.environ.get("GENEDYNAMICS_BASELINE_TEST_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def tensor_device():
    previous = torch.get_default_device()
    torch.set_default_device(DEVICE)
    yield
    torch.set_default_device(previous)


def rng_state():
    return torch.cuda.get_rng_state(DEVICE) if torch.device(DEVICE).type == "cuda" else torch.get_rng_state()


def set_rng_state(state):
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.set_rng_state(state, DEVICE)
    else:
        torch.set_rng_state(state)


@pytest.fixture(scope="module")
def upstream():
    if not (SOURCE / "human2sim2robot" / "ppo" / "utils" / "models.py").is_file():
        pytest.skip("Pinned Human2Sim2Robot source checkout is required for independent parity.")
    assert subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() == COMMIT
    manifest = json.loads((PORT / "source_manifest.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256((SOURCE / item["upstream_path"]).read_bytes()).hexdigest() == item["upstream_sha256"]
        if "local_sha256" in item:
            assert hashlib.sha256((PORT / item["local_path"]).read_bytes()).hexdigest() == item["local_sha256"]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(SOURCE))
        models = importlib.import_module("human2sim2robot.ppo.utils.models")
        network = importlib.import_module("human2sim2robot.ppo.utils.network")
        helpers = importlib.import_module("human2sim2robot.ppo.utils.helpers")
        yield models, network, helpers


def configs(upstream_network, rnn_kind, separate, before=False):
    options = dict(units=7, layers=2, name=rnn_kind or "lstm", layer_norm=True,
                   before_mlp=before, concat_input=not before, concat_output=True)
    local = NetworkConfig(mlp=MlpConfig([11, 9]),
                          rnn=RnnConfig(**options) if rnn_kind else None,
                          separate_value_mlp=separate)
    original = upstream_network.NetworkConfig(
        mlp=upstream_network.MlpConfig([11, 9]),
        rnn=upstream_network.RnnConfig(**options) if rnn_kind else None,
        separate_value_mlp=separate,
    )
    return local, original


@pytest.mark.parametrize("rnn_kind,separate,before", [
    (None, False, False), (None, True, False),
    ("lstm", False, False), ("gru", False, False),
    ("lstm", True, False), ("gru", True, False),
    ("lstm", False, True), ("gru", False, True),
])
def test_native_training_forward_normalizer_recurrent_reset_and_gradients(upstream, rnn_kind, separate, before):
    models, network, _ = upstream
    local_config, native_config = configs(network, rnn_kind, separate, before)
    torch.manual_seed(11)
    local = Human2Sim2RobotSolver(network_config=local_config, observation_shape=(5,),
                                 action_low=[-2, 0, -4], action_high=[1, 3, 2], device=DEVICE).model
    reference = models.ModelA2CContinuousLogStd(native_config, 3, (5,), True, True, 1, num_seqs=2)
    reference.load_state_dict(local.state_dict(), strict=True)
    local.train()
    reference.train()
    torch.manual_seed(12)
    obs = torch.randn(8, 5) * 20
    actions = torch.randn(8, 3)
    states = reference.get_default_rnn_state()
    states = tuple(torch.randn_like(s) for s in states) if states else None
    data = dict(is_train=True, obs=obs, prev_actions=actions,
                rnn_states=states, seq_length=4,
                dones=torch.tensor([0., 1., 0., 0., 0., 0., 1., 0.]))
    actual = local(copy.deepcopy(data))
    expected = reference(copy.deepcopy(data))
    for key in ("prev_neglogp", "values", "entropy", "mus", "sigmas"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    if states:
        for actual_state, expected_state in zip(actual["rnn_states"], expected["rnn_states"]):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    for result in (actual, expected):
        (result["values"].square().mean() + result["prev_neglogp"].mean()
         + result["entropy"].mean()).backward()
    for (name, actual_param), (expected_name, expected_param) in zip(local.named_parameters(), reference.named_parameters()):
        assert name == expected_name
        if expected_param.grad is None:
            assert actual_param.grad is None
        else:
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=0, atol=0)
    for name, actual_buffer in local.named_buffers():
        torch.testing.assert_close(actual_buffer, dict(reference.named_buffers())[name], rtol=0, atol=0)


def native_player_class(helpers):
    """Execute actual upstream action/preprocessing methods without gym imports."""
    tree = ast.parse((SOURCE / "human2sim2robot" / "ppo" / "ppo_player.py").read_text())
    rescale = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rescale_actions")
    player = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PpoPlayer")
    methods = [node for node in player.body if isinstance(node, ast.FunctionDef)
               and node.name in {"get_action", "_preproc_obs"}]
    player.body = methods
    module = ast.Module(body=[rescale, player], type_ignores=[])
    scope = dict(torch=torch, copy=copy, unsqueeze_obs=helpers.unsqueeze_obs)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / "human2sim2robot" / "ppo" / "ppo_player.py"), "exec"), scope)
    return scope["PpoPlayer"]


@pytest.mark.parametrize("rnn_kind", [None, "lstm", "gru"])
@pytest.mark.parametrize("clip_actions", [False, True])
def test_native_checkpoint_player_actions_rng_history_and_partial_resets(upstream, tmp_path, rnn_kind, clip_actions):
    models, network, helpers = upstream
    local_config, native_config = configs(network, rnn_kind, False)
    torch.manual_seed(41)
    native = models.ModelA2CContinuousLogStd(native_config, 3, (5,), True, True, 1, num_seqs=2).eval()
    with torch.no_grad():
        native.a2c_network.mu.bias.copy_(torch.tensor([2., -2., 0.2]))
        native.running_mean_std.running_mean.copy_(torch.arange(5).double())
        native.running_mean_std.running_var.copy_(torch.arange(1, 6).double())
        native.value_mean_std.running_mean.fill_(1.7)
        native.value_mean_std.running_var.fill_(2.3)
    separate_norm = copy.deepcopy(native.running_mean_std.state_dict())
    separate_norm["running_mean"] += 0.7
    weights = dict(model=native.state_dict(), running_mean_std=separate_norm, epoch=42, frame=108)
    path = tmp_path / "native.pth"
    torch.save(weights, path)
    solver = Human2Sim2RobotSolver(network_config=local_config, observation_shape=(5,),
                                 action_low=[-2, 0, -4], action_high=[1, 3, 2], clip_actions=clip_actions,
                                 device=DEVICE)
    solver.restore(path)
    native.load_state_dict(weights["model"], strict=True)
    native.running_mean_std.load_state_dict(weights["running_mean_std"], strict=True)
    player = native_player_class(helpers)()
    player.model = native
    player.states = native.get_default_rnn_state()
    player.has_batch_dimension = True
    player.actions_low = torch.tensor([-2., 0., -4.])
    player.actions_high = torch.tensor([1., 3., 2.])
    player.ppo_player_config = type("PlayerConfig", (), dict(clip_actions=clip_actions))()
    assert solver.checkpoint_metadata == dict(epoch=42, frame=108)
    for step in range(4):
        obs = torch.arange(10).reshape(2, 5).to(torch.uint8) if step == 0 else torch.randn(2, 5)
        if step == 2:
            # Native run() zeroes ended slots after the previous env.step.
            if player.states:
                player.states = tuple(s * torch.tensor([0., 1.])[None, :, None] for s in player.states)
            solver.reset(dones=[True, False])
        rng = rng_state()
        expected = player.get_action(obs.clone(), is_deterministic=step % 2 == 0)
        expected_rng = rng_state()
        set_rng_state(rng)
        actual = solver.act(obs.clone(), deterministic=step % 2 == 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(rng_state(), expected_rng)
        if player.states:
            for actual_state, expected_state in zip(solver.states, player.states):
                torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    solver.reset()
    assert solver.states is None
    assert not solver.model.training


def test_episode_batch_cannot_change_silently_and_no_normalizer_updates():
    solver = Human2Sim2RobotSolver(network_config=NetworkConfig(MlpConfig([8])),
                                 observation_shape=(5,), action_low=[-1.], action_high=[1.], device=DEVICE)
    before = copy.deepcopy(solver.model.running_mean_std.state_dict())
    assert solver.act(torch.ones(5)).shape == (1,)
    with pytest.raises(ValueError, match="batch changed"):
        solver.act(torch.ones(2, 5))
    solver.reset(2)
    assert solver.act(torch.ones(2, 5)).shape == (2, 1)
    for key, tensor in before.items():
        torch.testing.assert_close(tensor, solver.model.running_mean_std.state_dict()[key], rtol=0, atol=0)

