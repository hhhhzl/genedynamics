"""Qualify a complete, explicitly adapted FADA task training/record/LoRA route.

Released Planner-IDM architecture and two-pass training are retained. The
public scripted source oracle replaces native humanoid PPO; no target expert
or reward is used. Bounded updates verify the chain, not P4 competitiveness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.experiments.baselines import instantiate_task
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.learning.baselines.fada import FADAIDMAdapter, IDMAdaptationConfig
from genedynamics.learning.baselines.fada.checkpoint import capture_rng_state, restore_rng_state, to_cpu
from genedynamics.learning.baselines.fada.task_training import (
    PORTABLE_FIELDS, LEGACY_PORTABLE_FIELDS, LEGACY_TO_PHYSICAL_COLUMNS, PUBLIC_CHART_VERSION,
    TaskPlannerIDMTrainer, TaskTrajectory, TaskWindowDataset, collect_target_records,
    load_source_trajectory, oracle_shadow_dagger, released_model_config, sample_dataset,
    sample_pools, source_statistics, portable_history, project_trajectory, bounded_task_action,
)
from genedynamics.learning.cross_embodiment.contact import ContactMechanismSpec
from genedynamics.learning.cross_embodiment.contracts import MechanismSpec
from genedynamics.solvers.single.fada import FADASolver


def _json(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-demonstrations", nargs=2, type=Path, required=True, metavar=("MODE0", "MODE1"))
    parser.add_argument("--development-root", type=Path, default=Path("results/_development/p3_20261014"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=18500)
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--dagger-chunks", type=int, default=10)
    parser.add_argument("--dagger-updates", type=int, default=100)
    parser.add_argument("--adapt-updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-seconds", type=float, default=120.)
    parser.add_argument("--target-gain", type=float, default=.65)
    parser.add_argument("--target-delay", type=int, default=3)
    parser.add_argument("--reuse-target-buffer", type=Path,
                        help="Existing committed B-log qualification.json; SHA-checked off-policy reuse without new collection")
    args = parser.parse_args()
    if any(value < 1 for value in (args.warmup_updates, args.dagger_chunks, args.dagger_updates, args.adapt_updates, args.batch_size)):
        parser.error("training/DAgger budgets must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.use_development_output_root(args.development_root)
    output = cfg.output_dir / "level_adapted" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "attempt.json").exists():
        raise RuntimeError("charged FADA task attempt exists; use a new seed, never replay unfinished target collection for free")
    atomic_json(output / "attempt.json", dict(status="running", seed=args.seed, scope="FADA task adaptation; no native paper qualification"))
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    paths = set((root / "genedynamics").rglob("*.py")) | set((root / "scripts/baselines").glob("*.py")) | {args.config.resolve()}
    source_hashes = {}
    with zipfile.ZipFile(output / "source-overlay.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes(); name = str(path.relative_to(root))
            source_hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    atomic_json(output / "source-context.json", {"sha256": source_hashes, "argv": vars(args) | {"config": str(args.config),
                "development_root": str(args.development_root), "source_demonstrations": [str(path) for path in args.source_demonstrations],
                "reuse_target_buffer": str(args.reuse_target_buffer) if args.reuse_target_buffer else None}})
    if torch.device(args.device).type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    chart_task = instantiate_task(cfg.env_name, robot="panda", seed=args.seed)
    source = [project_trajectory(load_source_trajectory(path, mode=mode), chart_task)
              for mode, path in enumerate(args.source_demonstrations)]
    config = released_model_config(source[0].obs.shape[-1])
    dataset = TaskWindowDataset(source, history_len=config["history_len"], pred_horizon_k=config["pred_horizon"])
    trainer = TaskPlannerIDMTrainer(config, source_statistics(source, task=chart_task), device=args.device)
    before = to_cpu(trainer.model.state_dict())
    warmup = [trainer.step(sample_dataset(dataset, args.batch_size)) for _ in range(args.warmup_updates)]
    atomic_json(output / "stage.json", {"stage": "source_warmup_complete", "updates": args.warmup_updates})
    shadows, source_audits = [], []
    for mode in (0, 1):
        task = instantiate_task(cfg.env_name, robot="panda", seed=args.seed + mode,
                                history_len=config["history_len"])
        solver = FADASolver.from_checkpoint(trainer.checkpoint(), device=args.device)
        labels, audit = oracle_shadow_dagger(task, solver, mode=mode, chunks=args.dagger_chunks,
                                            horizon=config["pred_horizon"], seed=args.seed + mode)
        shadows.extend(labels); source_audits.append(audit)
    if not shadows:
        raise RuntimeError("no valid oracle-shadow horizons; source qualification failed without fabricating labels")
    dagger = [trainer.step(sample_pools(dataset, shadows, args.batch_size)) for _ in range(args.dagger_updates)]
    # Exact stochastic two-pass resume uses complete optimizer and RNG state.
    state = trainer.get_training_state()
    torch.save(state, output / "source_training_continuation.pt")
    expected_metrics = trainer.step(sample_pools(dataset, shadows, args.batch_size))
    expected = trainer.get_training_state()
    trainer.restore_training_state(torch.load(output / "source_training_continuation.pt", map_location="cpu", weights_only=True))
    actual_metrics = trainer.step(sample_pools(dataset, shadows, args.batch_size))
    actual = trainer.get_training_state()
    source_resume = _equal(expected, actual) and expected_metrics == actual_metrics
    if not source_resume:
        raise RuntimeError("FADA Planner-IDM complete optimizer/RNG continuation differs")
    source_checkpoint = trainer.checkpoint()
    atomic_json(output / "stage.json", {"stage": "source_DAgger_resume_complete", "shadow_labels": len(shadows)})
    torch.save(source_checkpoint, output / "source_policy.pt")
    torch.save(shadows, output / "oracle_shadow_labels.pt")
    (output / "source_dagger_audit.json").write_text(json.dumps(source_audits, default=_json, indent=2) + "\n")
    solver = FADASolver.from_checkpoint(source_checkpoint, device=args.device)
    mechanism_type = MechanismSpec if chart_task.spec.alias == "T1" else ContactMechanismSpec
    target = instantiate_task(cfg.env_name, robot="xarm7", mechanism_spec=mechanism_type(
                              servo_gain=args.target_gain, delay_steps=args.target_delay),
                              seed=args.seed + 1000, history_len=config["history_len"])
    if args.reuse_target_buffer:
        origin = args.reuse_target_buffer.resolve()
        receipt = json.loads(origin.read_text())
        buffer_path = origin.parent / "target_B_log.npz"
        buffer_sha = hashlib.sha256(buffer_path.read_bytes()).hexdigest()
        collector_context = json.loads((origin.parent / "source-context.json").read_text())
        old_args = collector_context["argv"]
        if buffer_sha != receipt["target_buffer_sha256"]:
            raise ValueError("reused B-log SHA differs from committed qualification")
        old_dim = receipt["source_model_config"]["obs_dim"]
        known_legacy = old_dim == 86 and receipt["portable_fields"] == list(LEGACY_PORTABLE_FIELDS)
        if (args.target_gain != old_args["target_gain"] or args.target_delay != old_args["target_delay"]
                or cfg.env_name not in str(old_args["config"]) or not (known_legacy or old_dim == config["obs_dim"])):
            raise ValueError("reused B-log task, public chart dimension or mechanism differs")
        if receipt["target_B_log"]["actual_control_seconds"] < args.target_seconds:
            raise ValueError("reused B-log duration is below the requested budget")
        with np.load(buffer_path) as data:
            indices = sorted(int(key.split(".")[0][8:]) for key in data.files if key.endswith(".actions") and key.startswith("episode_"))
            records = [TaskTrajectory(*(np.array(data[f"episode_{index}.{name}"]) for name in
                        ("obs", "actions", "current_command", "previous_issued")), "off_policy_committed_B_log") for index in indices]
        budget = deepcopy(receipt["target_B_log"])
        budget["collection_in_this_run"] = False
        budget["new_collection_accounting"] = dict(target.lifetime_accounting)
        budget["reuse_provenance"] = {"qualification": str(origin), "qualification_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
            "raw_buffer": str(buffer_path), "raw_buffer_sha256": buffer_sha,
            "collector_source_context_sha256": hashlib.sha256((origin.parent / "source-context.json").read_bytes()).hexdigest(),
            "collector_source_policy_sha256": hashlib.sha256((origin.parent / "source_policy.pt").read_bytes()).hexdigest(),
            "sampling": "off-policy reuse from the original source student; not sampled by the new student",
            "transform": "public controller amplitude projection of action labels; pure physical state projection; original raw bytes unchanged",
            "raw_observation_fields": receipt["portable_fields"], "physical_observation_fields": PORTABLE_FIELDS,
            "raw_to_physical_observation_columns": LEGACY_TO_PHYSICAL_COLUMNS if known_legacy else list(range(35)),
            "raw_dtype": str(records[0].obs.dtype), "projected_dtype": "float32"}
        # Preserve the original raw records byte for byte, with a receipt that
        # makes collection ownership and irreversible historical charges clear.
        (output / "target_B_log.npz").write_bytes(buffer_path.read_bytes())
    else:
        records, budget = collect_target_records(target, solver, seconds=args.target_seconds, seed=args.seed + 1000)
        budget["collection_in_this_run"] = True
    atomic_json(output / "stage.json", {"stage": "target_B_log_complete", "actual_seconds": budget["actual_control_seconds"]})
    (output / "target_record_audit.json").write_text(json.dumps(budget, default=_json, indent=2) + "\n")
    arrays = {}
    for index, row in enumerate(records):
        for name in ("obs", "actions", "current_command", "previous_issued"):
            arrays[f"episode_{index}.{name}"] = getattr(row, name)
    for index, row in enumerate(target.records):
        if row.traces:
            for name in row.traces[0]:
                arrays[f"trace_{index}.{name}"] = np.asarray([step[name] for step in row.traces])
    if not args.reuse_target_buffer:
        np.savez_compressed(output / "target_B_log.npz", **arrays)
    records = [project_trajectory(row, target) for row in records]
    usable = [row for row in records if len(row.obs) > config["pred_horizon"]]
    if len(usable) < 2:
        raise RuntimeError("target B-log has fewer than two usable independent episodes; budget is retained, no free resampling")
    validation = usable[::5]
    train = [row for index, row in enumerate(usable) if index % 5]
    if not train:
        train, validation = usable[:-1], usable[-1:]
    target_data = TaskWindowDataset(train, history_len=config["history_len"], pred_horizon_k=config["pred_horizon"])
    validation_data = TaskWindowDataset(validation, history_len=config["history_len"], pred_horizon_k=config["pred_horizon"])
    adapter = FADAIDMAdapter(source_checkpoint, device=args.device, config=IDMAdaptationConfig())
    frozen_planner = to_cpu(adapter.model.planner.state_dict())
    adaptation = [adapter.step(sample_dataset(target_data, args.batch_size)) for _ in range(args.adapt_updates)]
    adapt_state = {"model": to_cpu(adapter.model.state_dict()), "optimizer": to_cpu(adapter.optimizer.state_dict()),
                   "steps": adapter.steps, "rng": capture_rng_state()}
    torch.save(adapt_state, output / "target_adaptation_continuation.pt")
    expected_metrics = adapter.step(sample_dataset(target_data, args.batch_size))
    expected_model, expected_optimizer, expected_rng = to_cpu(adapter.model.state_dict()), to_cpu(adapter.optimizer.state_dict()), capture_rng_state()
    adapter.model.load_state_dict(adapt_state["model"], strict=True)
    adapter.optimizer.load_state_dict(adapt_state["optimizer"])
    adapter.steps = adapt_state["steps"]; restore_rng_state(adapt_state["rng"])
    actual_metrics = adapter.step(sample_dataset(target_data, args.batch_size))
    adapt_resume = (_equal(expected_model, to_cpu(adapter.model.state_dict()))
                    and _equal(expected_optimizer, to_cpu(adapter.optimizer.state_dict()))
                    and _equal(expected_rng, capture_rng_state()) and expected_metrics == actual_metrics)
    planner_frozen = _equal(frozen_planner, to_cpu(adapter.model.planner.state_dict()))
    if not adapt_resume or not planner_frozen:
        raise RuntimeError("FADA target adaptation changed planner or optimizer/RNG continuation")
    deployed = adapter.export_checkpoint()
    deployed["genedynamics_task_adapter"] = source_checkpoint["genedynamics_task_adapter"]
    torch.save(deployed, output / "adapted_policy.pt")
    # Independent held-out target record loss never queries target physics.
    from genedynamics.learning.baselines.fada import run_idm_epoch
    validation_metrics = run_idm_epoch(model=adapter.model, loader=[sample_dataset(validation_data, args.batch_size)],
        optimizer=None, device=adapter.backend.device, io_norm_enabled=True,
        obs_stats=adapter.backend.stats["obs"], action_stats=adapter.backend.stats["act"], command_stats=adapter.backend.stats["cmd"],
        pred_horizon_k=config["pred_horizon"], grad_clip=.5, show_progress=False, epoch_desc="held-out target records",
        horizon_gamma=0.)
    # Full, independently scored first target episodes after adaptation. These
    # are charged separately from B-log and never select a model/checkpoint.
    evaluation_task = instantiate_task(cfg.env_name, robot="xarm7", mechanism_spec=mechanism_type(
        servo_gain=args.target_gain, delay_steps=args.target_delay), seed=args.seed + 2000,
        history_len=config["history_len"])
    policy = FADASolver.from_checkpoint(deployed, device=args.device)
    evaluation_task.reset((args.seed + 2000, args.seed + 2001), seed=args.seed + 2000)
    evaluation_task.set_modes([0, 1]); policy.reset()
    for _ in range(evaluation_task.config.timeout_steps):
        result = evaluation_task.step(bounded_task_action(evaluation_task, policy.act(portable_history(evaluation_task))))
        if result.done.all():
            break
    evaluation = evaluation_task.evaluate()
    replays = [evaluation_task.replay_episode(record) for record in evaluation_task.records]
    if evaluation["status_mismatches"] or not all(row["matches_status"] and row["trace_exact"] for row in replays):
        raise RuntimeError("FADA post-adaptation independent target evaluation/replay differs")
    eval_arrays = {}
    for index, record in enumerate(evaluation_task.records):
        eval_arrays[f"episode_{index}.requested_actions"] = np.asarray(record.requested_actions)
        for name in record.traces[0]:
            eval_arrays[f"trace_{index}.{name}"] = np.asarray([row[name] for row in record.traces])
    np.savez_compressed(output / "target_evaluation.npz", **eval_arrays)
    profiles = []
    observation = portable_history(evaluation_task)
    for batch in (1, 4):
        value = {key: np.repeat(array[:1], batch, axis=0) for key, array in observation.items()}
        policy.reset(); durations = []
        for _ in range(11):
            if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); action = policy.act(value)
            if torch.device(args.device).type == "cuda": torch.cuda.synchronize()
            if not torch.isfinite(action).all(): raise RuntimeError("nonfinite FADA policy resource workload")
            durations.append(time.perf_counter() - start)
        profiles.append(dict(batch=batch, history=config["history_len"], action_horizon=config["pred_horizon"],
                             cold_seconds=durations[0], warm_p50_seconds=float(np.median(durations[1:])),
                             warm_p95_seconds=float(np.percentile(durations[1:], 95))))
    report = dict(status="adapted_Planner_IDM_DAgger_B_log_LoRA_chain_qualified", native_method_qualified=False,
                  scope="bounded source/task training and exact resume; not converged strong baseline or P4 score",
                  source_model_config=config, portable_fields=PORTABLE_FIELDS, source_demonstrations=[str(path) for path in args.source_demonstrations],
                  public_chart_version=PUBLIC_CHART_VERSION,
                  parameter_count=sum(parameter.numel() for parameter in trainer.model.parameters()),
                  source_weights_changed=any(not torch.equal(before[key], value.cpu()) for key, value in trainer.model.state_dict().items()),
                  source_training_resume_exact=source_resume, adaptation_resume_exact=adapt_resume, planner_frozen=planner_frozen,
                  warmup_updates=args.warmup_updates, source_dagger_updates=args.dagger_updates,
                  adaptation_updates=args.adapt_updates, source_resume_verification_extra_updates=2,
                  adaptation_resume_verification_extra_updates=2, source_dagger=source_audits,
                  target_B_log=budget, native_record_duration_qualified=bool(args.target_seconds >= 120.),
                  target_post_adaptation_evaluation=evaluation, target_evaluation_replays=replays,
                  resource_profiles=profiles, parameter_device=str(next(trainer.model.parameters()).device),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
                  held_out_record_validation=validation_metrics,
                  last_warmup=warmup[-1], last_dagger=dagger[-1], last_adaptation=adaptation[-1],
                  target_buffer_sha256=hashlib.file_digest((output / "target_B_log.npz").open("rb"), "sha256").hexdigest(),
                  wall_seconds=time.monotonic() - started,
                  adaptations=["portable Cartesian state/history instead of humanoid joint correspondence",
                               "qualified scripted public source oracle instead of native humanoid PPO/weak-policy source",
                               "fixed public physical scales and amplitude-bounded controller Box; no target-fitted normalization",
                               "fixed-arm CPU workcell and 13D shared controller instead of humanoid locomotion"])
    atomic_json(output / "qualification.json", json.loads(json.dumps(report, default=_json)))
    atomic_json(output / "attempt.json", dict(status="complete", seed=args.seed, qualified_scope=report["scope"]))
    print(json.dumps({key: report[key] for key in ("status", "source_weights_changed", "source_training_resume_exact", "adaptation_resume_exact", "planner_frozen")}, indent=2))


if __name__ == "__main__":
    main()
