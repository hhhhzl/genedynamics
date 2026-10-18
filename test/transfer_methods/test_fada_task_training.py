"""Physical task records and actual two-pass FADA learning, with small fixtures.

Tiny widths/history/horizon are explicitly for fast qualification tests. The
task runner defaults retain the released 128-wide, history30/horizon6 model.
"""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.fada.learning.task_training import (
    PORTABLE_FIELDS, TaskPlannerIDMTrainer, TaskTrajectory, TaskWindowDataset,
    collect_target_records, oracle_shadow_dagger, portable_observation,
    released_model_config, sample_dataset, source_statistics,
    bounded_task_action, project_trajectory, portable_history, LEGACY_TO_PHYSICAL_COLUMNS,
)
from genedynamics.solvers.single.fada import FADASolver


def _measured_record(task):
    rows = [task.reset((44001,), seed=44001)]
    actions = []
    for _ in range(5):
        action = np.zeros(13, dtype=np.float32); action[0] = .003
        actions.append(action)
        rows.append(task.step(action).observation)
    obs = np.concatenate([portable_observation(row) for row in rows])
    previous = np.concatenate([row["previous_action"] for row in rows])
    command = np.concatenate([np.concatenate((row["goal_position"], row["goal_rotation"].reshape(1, 9),
                                              np.array([[1., 0.]], dtype=np.float32)), axis=-1) for row in rows])
    return TaskTrajectory(obs, np.asarray(actions), command, previous, "short_real_source_training_fixture")


def _trainer_and_data():
    task = instantiate_task("T3", robot="panda", history_len=4)
    row = _measured_record(task)
    cfg = released_model_config(row.obs.shape[-1], history_len=4, pred_horizon=2)
    cfg.update(planner_d_model=16, planner_nhead=2, planner_num_layers=1, planner_dim_feedforward=24,
               idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1, idm_decoder_num_layers=1, idm_dim_feedforward=24)
    return task, row, TaskPlannerIDMTrainer(cfg, source_statistics([row], task=task)), TaskWindowDataset([row], history_len=4, pred_horizon_k=2)


def _equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(left, right) for left, right in zip(a, b))
    return a == b


def test_portable_view_and_window_history_preserve_public_chart_and_episode_boundaries():
    task, row, _, data = _trainer_and_data()
    assert "joint_position" not in PORTABLE_FIELDS and "joint_velocity" not in PORTABLE_FIELDS
    assert row.obs.shape[-1] == 35
    assert not {"previous_request", "previous_action", "reference_position", "reference_rotation",
                "elapsed_seconds", "goal_position", "goal_rotation"}.intersection(PORTABLE_FIELDS)
    np.testing.assert_array_equal(portable_history(task)["history_obs"][0, -1], portable_observation(task.observe())[0])
    changed = deepcopy(task.observe())
    for name in ("previous_request", "previous_action", "reference_position", "reference_rotation",
                 "elapsed_seconds", "goal_position", "goal_rotation"):
        changed[name][:] = 10000.
    np.testing.assert_array_equal(portable_observation(changed), portable_observation(task.observe()))
    first = data[0]
    assert first["history_valid_mask"].tolist() == [False, False, False, True]
    np.testing.assert_array_equal(data[1]["history_act"][-1].numpy(), row.previous_issued[1])
    assert len(data) == len(row.obs) - 2


def test_both_native_passes_change_weights_and_optimizer_rng_resume_is_exact():
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    torch.manual_seed(44000)
    _, _, trainer, data = _trainer_and_data()
    planner = deepcopy(trainer.model.planner.state_dict())
    idm = deepcopy(trainer.model.idm.state_dict())
    try:
        trainer.step(sample_dataset(data, 2))
        assert any(not torch.equal(value, planner[key]) for key, value in trainer.model.planner.state_dict().items())
        assert any(not torch.equal(value, idm[key]) for key, value in trainer.model.idm.state_dict().items())
        assert all(value.requires_grad for value in trainer.model.idm.parameters())
        state = trainer.get_training_state()
        expected_metrics = trainer.step(sample_dataset(data, 2))
        expected = trainer.get_training_state()
        trainer.restore_training_state(state)
        actual_metrics = trainer.step(sample_dataset(data, 2))
        assert expected_metrics == actual_metrics
        assert _equal(expected, trainer.get_training_state())
        solver = FADASolver.from_checkpoint(trainer.checkpoint())
        assert solver.model.planner_predict_delta
    finally:
        torch.set_num_threads(old_threads)


def test_fixed_public_chart_and_amplitude_projection_do_not_amplify_constant_channels():
    task, row, trainer, _ = _trainer_and_data()
    raw = deepcopy(row)
    raw.actions[:] = 1000.
    raw.obs = np.tile(np.arange(86, dtype=np.float64), (len(raw.obs), 1))
    raw.obs[:, 24:75] = 1000.
    projected = project_trajectory(raw, task)
    assert raw.actions.max() == 1000. and raw.obs[:, 24:75].max() == 1000.
    assert projected.obs.dtype == np.float32 and projected.obs.shape[-1] == 35
    np.testing.assert_array_equal(projected.actions, bounded_task_action(task, raw.actions))
    np.testing.assert_array_equal(projected.obs, raw.obs[:, LEGACY_TO_PHYSICAL_COLUMNS].astype(np.float32))
    np.testing.assert_array_equal(projected.obs[0], np.r_[np.arange(24), np.arange(75, 86)])
    future = TaskWindowDataset([projected], history_len=4, pred_horizon_k=2)[0]["future_observations"]
    np.testing.assert_array_equal(future, projected.obs[1:3])
    changed_samples = deepcopy(row)
    changed_samples.obs[:, :24] += 7.
    changed_samples.actions[:] = .01
    assert _equal(source_statistics([changed_samples], task=task), trainer.statistics)
    stats = trainer.statistics
    assert stats["obs_norm_stats"]["obs_std"][3:12].min() == 1.
    assert stats["action_norm_stats"]["action_std"][6:12].min() == 1.
    t1 = instantiate_task("T1", robot="panda")
    clipped = bounded_task_action(t1, np.full(13, 1000.))
    np.testing.assert_allclose(clipped, [.12] * 3 + [.6] * 3 + [1.] * 6 + [0.])


def test_dagger_labels_are_actual_shadow_dynamics_and_target_records_have_no_teacher_access():
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    task, _, trainer, _ = _trainer_and_data()
    solver = FADASolver.from_checkpoint(trainer.checkpoint())
    source = instantiate_task("T3", robot="panda", history_len=4)
    try:
        labels, budget = oracle_shadow_dagger(source, solver, mode=0, chunks=1, horizon=2, seed=44010)
        assert len(labels) == 1
        assert budget["expert_shadow_control_steps"] == 2
        assert budget["student_source_calls"]["control_steps"] == 2
        assert labels[0]["future_observations"].shape == (2, 35)
        target = instantiate_task("T3", robot="xarm7", history_len=4)
        records, audit = collect_target_records(target, solver, seconds=.08, seed=44020)
        assert audit["protocol"] == "B-log" and not audit["reward_access"] and not audit["target_expert_access"]
        assert audit["accounting"]["control_steps"] == 4
        assert sum(len(row.actions) for row in records) == 4
        assert audit["episodes"]["status_mismatches"] == 0
    finally:
        torch.set_num_threads(old_threads)
