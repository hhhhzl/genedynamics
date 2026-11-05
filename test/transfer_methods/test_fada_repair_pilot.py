"""FADA repair pilot keeps its source/target charts and frozen-planner boundary."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.fada.learning.repair_pilot import (
    _bind_target_checkpoint,
    _normalization_equal,
    _source_trajectory,
)
from genedynamics.solvers.single.fada.learning.task_training import (
    TaskPlannerIDMTrainer,
    TaskTrajectory,
    portable_observation,
    released_model_config,
    source_statistics,
)


def test_source_episode_reconstructs_public_67d_memory_and_rejects_sha_change(tmp_path):
    task = instantiate_task("T3", robot="panda", history_len=4)
    packets = [task.reset((88100,), seed=88100)]
    actions = []
    for _ in range(3):
        action = np.zeros(13, dtype=np.float32)
        action[0] = .001
        actions.append(action)
        packets.append(task.step(action).observation)
    path = tmp_path / "episode.npz"
    public = {key: np.concatenate([packet[key] for packet in packets]) for key in packets[0]}
    np.savez(path, actions=np.asarray(actions), **{"public." + key: value for key, value in public.items()})
    reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": 0}
    row = _source_trajectory(reference, task="T3")
    assert row.obs.shape == (4, 35)
    assert row.current_command.shape == (4, 67)
    np.testing.assert_array_equal(row.previous_issued, public["previous_action"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA changed"):
        _source_trajectory(reference, task="T3")


def test_target_binding_lifts_67d_source_and_preserves_planner_function():
    task = instantiate_task("T3", robot="panda", history_len=4)
    packet = task.reset((88110,), seed=88110)
    obs = portable_observation(packet)
    command = np.zeros((1, 67), dtype=np.float32)
    row = TaskTrajectory(obs, np.zeros((0, 13), dtype=np.float32), command,
                         np.zeros((1, 13), dtype=np.float32), "fixture")
    cfg = released_model_config(35, history_len=4, pred_horizon=2, cmd_dim=67)
    cfg.update(planner_d_model=16, planner_nhead=2, planner_num_layers=1, planner_dim_feedforward=24,
               idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1,
               idm_decoder_num_layers=1, idm_dim_feedforward=24)
    source = TaskPlannerIDMTrainer(cfg, source_statistics([row], task=task)).checkpoint()
    source["genedynamics_task_adapter"]["task_name"] = task.spec.name
    planner = deepcopy(source["model_state_dict"]["planner.cmd_embed.weight"])
    target = _bind_target_checkpoint(source, {"task": "T3"})
    assert target["cmd_dim"] == target["cfg"]["cmd_dim"] == 199
    torch.testing.assert_close(target["model_state_dict"]["planner.cmd_embed.weight"][:, :67], planner)
    assert torch.count_nonzero(target["model_state_dict"]["planner.cmd_embed.weight"][:, 67:]) == 0
    assert target["genedynamics_task_adapter"]["task_name"] == task.spec.name
    assert _normalization_equal({"mean": torch.zeros(2), "eps": 1e-6},
                                {"mean": torch.zeros(2), "eps": 1e-6})
    assert not _normalization_equal({"mean": torch.zeros(2), "eps": 1e-6},
                                    {"mean": torch.zeros(2), "eps": 2e-6})
