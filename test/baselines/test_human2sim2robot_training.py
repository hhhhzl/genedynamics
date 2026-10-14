"""Actual native PPO update/resume on a real short CPU source task.

The MLP32/LSTM16 network and 80 ms deadline intentionally exercise recurrent
training, auto reset and checkpoint boundaries cheaply. These are qualification
fixtures, not released model capacity or successful source demonstration evidence.
"""
from copy import deepcopy
import os
from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("gym")
pytest.importorskip("tensorboardX")

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.learning.baselines.human2sim2robot.task_env import ObjectTrajectoryTaskEnv
from genedynamics.learning.baselines.human2sim2robot.training import (
    build_learner, capture_training_state, restore_training_state, train_epoch,
)
from scripts.baselines.human2sim2robot_tasks import _state_equal, algorithm_state

ROOT = Path(__file__).resolve().parents[2]


def _stationary_measured_pose(task):
    observation = task.reset((999,), seed=33001)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = observation["tcp_rotation"][0]
    pose[:3, 3] = observation["tcp_position"][0]
    return np.repeat(pose[None], 16, axis=0)


def test_real_native_recurrent_ppo_update_full_resume_and_irreversible_source_charges(tmp_path):
    if not (ROOT / "baselines/Human2Sim2Robot/.git").exists():
        pytest.skip("native learner test requires pinned Human2Sim2Robot sources")
    device = os.environ.get("GENEDYNAMICS_BASELINE_TEST_DEVICE", "cpu")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA qualification explicitly requested but unavailable")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(33000); np.random.seed(33000); random.seed(33000)
    task = instantiate_task("T3", robot="panda", config_overrides={"timeout_seconds": .08})
    demo = _stationary_measured_pose(task)
    env = ObjectTrajectoryTaskEnv(task, demo, num_envs=2, seed=33002)
    agent, network = build_learner(env, tmp_path / "native", device=device, num_actors=2,
                             horizon=4, mlp_units=(32, 32), rnn_units=16, epochs=20)
    assert type(agent).__module__ == "human2sim2robot.ppo.ppo_agent"
    assert agent.is_rnn and network.rnn.units == 16 and network.mlp.units == [32, 32]
    before = {name: value.detach().clone() for name, value in agent.model.named_parameters()}
    try:
        first = train_epoch(agent)
        assert any(not torch.equal(before[name], value) for name, value in agent.model.named_parameters())
        assert first["frame"] == 8
        assert agent.game_lengths.current_size > 0  # at least one true auto reset
        payload = capture_training_state(agent)
        checkpoint = tmp_path / "continuation.pt"
        torch.save(payload, checkpoint)
        second_metrics = train_epoch(agent)
        expected = capture_training_state(agent)
        expected_lifetime = dict(task.lifetime_accounting)
        # Trusted, locally generated full integration/RNG state. Optimizer
        # scalar steps must remain on CPU, rollout memories move to device.
        restore_training_state(agent, torch.load(checkpoint, map_location="cpu", weights_only=False))
        assert task.lifetime_accounting == expected_lifetime
        resumed_metrics = train_epoch(agent)
        actual = capture_training_state(agent)
        assert _state_equal(algorithm_state(expected["native"]), algorithm_state(actual["native"]))
        for key in ("continuation", "scheduler", "meters", "rng"):
            assert _state_equal(expected[key], actual[key]), key
        for key in ("actor_loss", "critic_loss", "entropy", "kl", "learning_rate", "epoch", "frame"):
            assert second_metrics[key] == resumed_metrics[key], key
        assert task.accounting["control_steps"] == 16
        assert task.lifetime_accounting["control_steps"] == 24
        assert task.lifetime_accounting["reset_worlds"] > task.accounting["reset_worlds"]
        assert task.evaluate()["status_mismatches"] == 0
    finally:
        agent.writer.close()
        torch.set_num_threads(old_threads)


def test_task_action_rights_and_native_smoothing_use_normalized_units():
    task = instantiate_task("T1", robot="panda")
    env = ObjectTrajectoryTaskEnv(task, _stationary_measured_pose(task), num_envs=1, seed=33010)
    assert env.action_space.high[0] == pytest.approx(task.runner.skill.linear_speed)
    assert env.action_space.high[3] == pytest.approx(task.runner.skill.angular_speed)
    env.reset()
    normalized = np.array([[.2, -.1, 0., .1, 0., -.2, .3, 0., -.4, .1, 0., .2, 0.]], dtype=np.float32)
    half = (env.action_space.high - env.action_space.low) / 2
    middle = (env.action_space.high + env.action_space.low) / 2
    env.step(normalized * half + middle)
    np.testing.assert_allclose(env.previous_actions, normalized, rtol=1e-6, atol=1e-7)
    # T1's disabled force dimension must be zero and finite, never 0/0.
    assert env.previous_actions[0, 12] == 0 and np.isfinite(env.previous_actions).all()


def test_continuation_binds_demonstration_and_clock():
    task = instantiate_task("T3", robot="panda")
    demo = _stationary_measured_pose(task)
    env = ObjectTrajectoryTaskEnv(task, demo, num_envs=1)
    env.reset(); checkpoint = env.get_env_state()
    checkpoint["demo_dt"] *= 2
    with pytest.raises(ValueError, match="sampling clock"):
        env.set_env_state(checkpoint)
    with pytest.raises(ValueError, match="demo_dt"):
        ObjectTrajectoryTaskEnv(task, demo, num_envs=1, demo_dt=0.)
