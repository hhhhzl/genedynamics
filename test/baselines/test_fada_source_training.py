"""Source support decoding, irreversible source budget and immutable data."""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from genedynamics.experiments.baselines import instantiate_task
from genedynamics.learning.baselines.fada.source_training import (
    SourceAccountingLedger, bind_source_action, load_saved_episode, source_force_support)
from genedynamics.learning.baselines.fada.source_training import (
    SourceTaskPolicy, collect_full_dagger, collect_source_teacher, sample_shared_dagger, validate_episode_split)
from genedynamics.learning.baselines.fada.task_training import (
    TaskTrajectory, TaskPlannerIDMTrainer, TaskWindowDataset, portable_history, portable_observation,
    released_model_config, sample_dataset, source_statistics)
from genedynamics.solvers.single.fada import FADASolver


def test_empirical_source_off_deadzone_preserves_unseen_positive_force_compensation():
    task = instantiate_task("T3", robot="panda")
    force = np.asarray([0., 5., 7.5, 0., 5.], dtype=np.float32)
    action = np.zeros((len(force), 13), np.float32); action[:, 12] = force
    row = TaskTrajectory(np.zeros((6, 35)), action, np.zeros((6, 14)), np.zeros((6, 13)), "teacher_fixture")
    support = source_force_support([row])
    assert support["values_N"] == [0., 5., 7.5] and support["counts"] == [2, 2, 1]
    value = np.zeros((7, 13), np.float32); value[:, 12] = [0., .01, 2.49, 2.5, 5., 7.5, 30.]
    codec = {"kind": "source_gap_off_deadzone", **support, "threshold_N": 2.5}
    np.testing.assert_allclose(bind_source_action(task, value, codec)[:, 12], [0., 0., 0., 2.5, 5., 7.5, 25.])
    np.testing.assert_allclose(bind_source_action(task, value, {"kind": "continuous13D"})[:, 12], [0., .01, 2.49, 2.5, 5., 7.5, 25.])
    assert value[-1, 12] == 30.
    with pytest.raises(ValueError, match="threshold"):
        bind_source_action(task, value, {**codec, "threshold_N": 1.})


def test_actual_source_restore_and_cpu_replay_do_not_rollback_external_charges(tmp_path):
    task = instantiate_task("T3", robot="panda", seed=28700)
    task.source_accounting_role = "collection"
    ledger = SourceAccountingLedger(tmp_path / "source_ledger.json")
    with ledger.installed():
        task.reset((28700,), seed=28700)
        state = task.get_checkpoint()
        task.step(np.zeros(13, np.float32))
        task.restore_checkpoint(state)
        task.step(np.zeros(13, np.float32))
        assert task.accounting["control_steps"] == 1
        assert task.lifetime_accounting["control_steps"] == 2
        with ledger.role("replay"):
            replay = task.replay_episode(task.records[0])
    assert replay["trace_exact"]
    assert ledger.values["roles"]["collection"]["control_steps"] == 2
    assert ledger.values["roles"]["replay"]["control_steps"] == 1
    assert SourceAccountingLedger(ledger.path).values == ledger.values
    assert ledger.values["target_calls"] == 0


def test_source_dataset_hash_is_checked_before_loading_resume_data(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, obs=np.zeros((3, 35)), actions=np.zeros((2, 13)), current_command=np.zeros((3, 14)), previous_issued=np.zeros((3, 13)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_saved_episode(path, digest).obs.dtype == np.float32
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA"):
        load_saved_episode(path, digest)


def _tiny_trainer(task):
    public = task.observe()
    goal = np.r_[public["goal_position"][0], public["goal_rotation"][0].ravel(), 1., 0.][None].astype(np.float32)
    row = TaskTrajectory(portable_observation(public), np.empty((0, 13), np.float32), goal, np.zeros((1, 13), np.float32), "test")
    cfg = released_model_config(35)
    cfg.update(planner_d_model=16, planner_nhead=2, planner_num_layers=1, planner_dim_feedforward=24,
               idm_d_model=16, idm_nhead=2, idm_encoder_num_layers=1, idm_decoder_num_layers=1, idm_dim_feedforward=24)
    return TaskPlannerIDMTrainer(cfg, source_statistics([row], task=task))


def test_unbound_decoder_checkpoint_is_rejected_and_task_loader_applies_it():
    task = instantiate_task("T3", robot="panda", history_len=30); task.reset((28710,), seed=28710)
    trainer = _tiny_trainer(task)
    with torch.no_grad():
        trainer.model.idm.action_head.weight.zero_(); trainer.model.idm.action_head.bias.zero_()
        trainer.model.idm.action_head.bias[12] = (1. - 12.5) / 12.5
    checkpoint = trainer.checkpoint()
    checkpoint["genedynamics_task_adapter"]["action_decoder"] = {
        "kind": "source_gap_off_deadzone", "values_N": [0., 5.], "threshold_N": 2.5}
    with pytest.raises(ValueError, match="decoder"):
        FADASolver.from_checkpoint(checkpoint)
    policy = SourceTaskPolicy.from_checkpoint(checkpoint, task=task)
    assert policy.act(portable_history(task))[0, 12] == 0.
    with torch.no_grad(): policy.model.idm.action_head.bias[12] = (7.5 - 12.5) / 12.5
    assert policy.act(portable_history(task))[0, 12] == pytest.approx(7.5)


def test_interrupted_current_shadow_suffix_and_zero_action_primary_are_preserved(tmp_path):
    task = instantiate_task("T3", robot="panda"); task.reset((28711,), seed=28711)
    checkpoint = _tiny_trainer(task).checkpoint()
    count = 0
    def guard():
        nonlocal count
        count += 1
        if count == 4: raise RuntimeError("partial-shadow-fixture")
    ledger = SourceAccountingLedger(tmp_path / "ledger.json")
    with ledger.installed(), pytest.raises(RuntimeError, match="partial-shadow-fixture"):
        collect_full_dagger("T3", checkpoint, seed=28712, mode=0, teacher_probability=.5,
            output=tmp_path / "dagger", device="cpu", guard=guard)
    audit = json.loads((tmp_path / "dagger/shadow_audit.json").read_text())
    assert audit["branches"][0]["suffix_controls"] == 2
    assert len(audit["branches"][0]["traces"]) == len(audit["branches"][0]["actions"]) == 2
    assert not audit["branches"][0]["complete_requested_horizon"]
    assert ledger.values["roles"]["expert_shadow"]["control_steps"] == 2
    with np.load(tmp_path / "dagger/primary/episode.npz") as data: assert data["actions"].shape == (0, 13)
    report = json.loads((tmp_path / "dagger/primary/report.json").read_text())
    assert report["evaluation"]["requested_episodes"] == 1
    assert report["evaluation"]["outcomes"] == {"incomplete": 1}
    with pytest.raises(RuntimeError, match="teacher-fixture"):
        collect_source_teacher("T3", seed=28713, mode=0, output=tmp_path / "teacher",
            guard=lambda: (_ for _ in ()).throw(RuntimeError("teacher-fixture")))
    assert json.loads((tmp_path / "teacher/report.json").read_text())["evaluation"]["requested_episodes"] == 1


def test_split_checks_copied_sha_and_canonical_data_not_only_paths():
    row = TaskTrajectory(np.zeros((8, 35)), np.zeros((7, 13)), np.zeros((8, 14)), np.zeros((8, 13)), "fixture")
    a, b = {"path": "train", "sha256": "same"}, {"path": "validation", "sha256": "same"}
    with pytest.raises(ValueError, match="overlap"): validate_episode_split([a], [b], [row], [row])
    b["sha256"] = "different-wrapper-bytes"
    with pytest.raises(ValueError, match="overlap"): validate_episode_split([a], [b], [row], [deepcopy(row)])


def test_source_mixed_sampling_respects_nondefault_configuration():
    class Pool:
        def __init__(self, value): self.value = value
        def __len__(self): return 10
        def __getitem__(self, _): return {"label": torch.tensor([self.value])}
    values = sample_shared_dagger(Pool(0), Pool(1), [{"label": torch.tensor([2])}], 10,
        offline_ratio=.4, online_real_ratio=1 / 3)["label"].ravel().tolist()
    assert values.count(0) == 4 and values.count(1) == 2 and values.count(2) == 4


@pytest.mark.parametrize("direct_checkpoint", [False, True])
def test_source_gate_rejects_paper_outputs_before_checkpoint_or_context_loading(monkeypatch, direct_checkpoint):
    from scripts.baselines import fada_source_gate as gate
    root = Path(gate.__file__).resolve().parents[2]
    args = ["fada_source_gate", "--seed", "28750", "--device", "cpu"]
    if direct_checkpoint:
        args += ["--checkpoint", "nonexistent.pt", "--task", "T3", "--output", str(root / "latex/cross_ebdm/guard_fixture")]
    else:
        args += ["--qualification", str(root / "latex/cross_ebdm/nonexistent_report.json")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="outside latex"): gate.main()


def test_source_training_rejects_paper_development_root_before_writes(tmp_path, monkeypatch):
    from scripts import fada_source_train as cli
    root = Path(cli.__file__).resolve().parents[1]
    config = tmp_path / "paper_guard.yaml"
    config.write_text(json.dumps({"name": "zero_physics_paper_guard", "output_dir": "results/arm/keyed_insertion_recovery/qualification/fada_source_train",
        "env_name": "keyed_insertion_recovery", "method": "fada_source_train", "metadata": {"source_only": True},
        "method_params": {"shared_online_offline_ratio": .2, "online_real_shadow_ratio": .5, "maximum_wall_seconds": 900, "maximum_gpu_GiB": 4}}))
    monkeypatch.setattr(sys, "argv", ["fada_source_train", "--config", str(config), "--development-root", str(root / "latex/guard_fixture"), "--device", "cpu"])
    with pytest.raises(ValueError, match="outside latex"): cli.main()


@pytest.mark.parametrize("retain_metrics", [False, True])
def test_actual_cli_resume_at_finished_training_boundary(tmp_path, monkeypatch, retain_metrics):
    import genedynamics.learning.baselines.fada.source_training as source_module
    from scripts import fada_source_train as cli
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    task = instantiate_task("T3", robot="panda")
    refs, rows = [], []
    for seed in (28720, 28721):
        public = [task.reset((seed,), seed=seed)]; actions = []
        for step in range(8):
            action = np.zeros(13, np.float32); action[12] = 5. if step % 2 else 0.
            actions.append(action); public.append(task.step(action).observation)
        obs = np.concatenate([portable_observation(value) for value in public])
        command = np.concatenate([np.c_[value["goal_position"], value["goal_rotation"].reshape(1, 9), [[1., 0.]]] for value in public]).astype(np.float32)
        row = TaskTrajectory(obs, np.asarray(actions), command, np.concatenate([value["previous_action"] for value in public]), "real_fixture")
        path = tmp_path / f"episode_{seed}.npz"
        np.savez(path, **{name: getattr(row, name) for name in ("obs", "actions", "current_command", "previous_issued")})
        refs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}); rows.append(row)
    trainer = _tiny_trainer(task)
    initial = tmp_path / "initial.pt"; torch.save(trainer.checkpoint(), initial)
    metrics = trainer.step(sample_dataset(TaskWindowDataset([rows[0]], history_len=30, pred_horizon_k=6), 4))
    recipe = {"initial_checkpoint": str(initial), "maximum_wall_seconds": 30, "maximum_gpu_GiB": 4,
        "batch_size": 4, "warmup_updates": 1, "stage_updates": 1, "teacher_probabilities": [],
        "shared_online_offline_ratio": .2, "online_real_shadow_ratio": .5, "decoder_variants": ["continuous13D", "source_gap_off_deadzone"]}
    output_relative = "results/arm/keyed_insertion_recovery/qualification/fada_source_train"
    config = tmp_path / "fixture.yaml"
    config.write_text(json.dumps({"name": "explicit_tiny_cli_resume_fixture", "output_dir": output_relative,
        "env_name": "keyed_insertion_recovery", "method": "fada_source_train", "backend": "numpy", "device": "cpu",
        "metadata": {"source_only": True}, "method_params": recipe}))
    output = tmp_path / "development" / output_relative / "level_source/seed_28730"; output.mkdir(parents=True)
    state = {"phase": "training", "stage": -1, "cursor": 1, "offline": refs[:1], "validation": refs[1:], "real": [], "shadows": [],
        "stage_reports": [], "training_state": trainer.get_training_state(), "recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        "initial_sha256": hashlib.sha256(initial.read_bytes()).hexdigest()}
    if retain_metrics: state["last_training_metrics"] = {"stage": -1, "cursor": 1, "metrics": metrics}
    continuation = output / "training_continuation.pt"; torch.save(state, continuation)
    (output / "runtime_usage.json").write_text(json.dumps({"wall_seconds_consumed": 0.}))
    root = Path(cli.__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "scripts/fada_source_train.py", "scripts/baselines/qualify_tasks.py", "scripts/baselines/fada_tasks.py",
        "genedynamics/envs/domains/manipulation/cross_embodiment/backends/contact.py")}
    (output / "provenance.json").write_text(json.dumps({"source_sha256": hashes}))
    original = source_module.instantiate_task
    monkeypatch.setattr(source_module, "instantiate_task", lambda name, **kwargs: original(name, config_overrides={"timeout_seconds": .06}, **kwargs))
    monkeypatch.setattr(sys, "argv", ["fada_source_train", "--config", str(config), "--development-root", str(tmp_path / "development"),
        "--seed", "28730", "--device", "cpu", "--resume", str(continuation)])
    try:
        cli.main()
        report = json.loads((output / "qualification.json").read_text())
        assert report["stage_reports"][0]["resume_exact"]
        assert report["target_calls"] == 0
        retained = report["stage_reports"][0]["last_training"]
        assert retained == metrics if retain_metrics else retained["availability"].startswith("not_retained")
        changed = deepcopy(json.loads((output / "provenance.json").read_text()))
        changed["source_sha256"]["scripts/fada_source_train.py"] = "bad"
        with pytest.raises(ValueError, match="source SHA changed"): cli.verify_resume_source_fingerprint(root, changed)
        changed = deepcopy(json.loads((output / "provenance.json").read_text()))
        changed["source_sha256"]["genedynamics/envs/domains/manipulation/cross_embodiment/backends/contact.py"] = "bad-physics"
        with pytest.raises(ValueError, match="source SHA changed"): cli.verify_resume_source_fingerprint(root, changed)
    finally:
        torch.set_num_threads(old_threads)
