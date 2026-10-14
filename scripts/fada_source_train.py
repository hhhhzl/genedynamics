"""Bounded source-only full-episode FADA DAgger and public decoder comparison.

Both exported decoders share exactly the same weights, teacher data and online
continuous-student data. Target collection/reward and native-asset claims are
absent. Exact optimizer/RNG resume is supported at saved training boundaries;
unfinished physical collection is retained and never silently rerun for free.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import zipfile

import numpy as np
import torch

from genedynamics.experiments.artifacts import atomic_json
from genedynamics.experiments.baselines import instantiate_task
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.learning.baselines.fada.checkpoint import to_cpu
from genedynamics.learning.baselines.fada.task_training import (
    TaskPlannerIDMTrainer, TaskWindowDataset, load_source_trajectory, project_trajectory, sample_dataset)
from genedynamics.learning.baselines.fada.source_training import (
    SourceAccountingLedger, collect_source_teacher, collect_full_dagger, evaluate_source_variant,
    load_saved_episode, sample_shared_dagger, source_force_support, validate_episode_split)
from genedynamics.solvers.single.fada import FADASolver
from scripts.baselines.fada_tasks import _equal, _json
from scripts.baselines import require_non_paper_output


class SourceBudgetExceeded(RuntimeError):
    pass


def _torch_save(path, value):
    temporary = Path(str(path) + ".pending")
    torch.save(value, temporary); os.replace(temporary, path)


def verify_resume_source_fingerprint(root, provenance):
    required = ("scripts/fada_source_train.py", "scripts/baselines/qualify_tasks.py", "scripts/baselines/fada_tasks.py")
    hashes = provenance["source_sha256"]
    if any(name not in hashes for name in required): raise ValueError("resume provenance lacks required executable source fingerprints")
    for name, digest in hashes.items():
        if name.startswith("genedynamics/") or name in required:
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"resume executable source SHA changed: {name}")


def _validation(checkpoint, rows, batch_size):
    solver = FADASolver.from_checkpoint(checkpoint, device="cpu")
    data = TaskWindowDataset(rows, history_len=checkpoint["cfg"]["history_len"], pred_horizon_k=checkpoint["cfg"]["pred_horizon"])
    indices = np.unique(np.linspace(0, len(data) - 1, min(256, len(data)), dtype=int))
    errors = []
    for start in range(0, len(indices), batch_size):
        examples = [data[int(index)] for index in indices[start:start + batch_size]]
        batch = {key: torch.stack([row[key] for row in examples]) for key in examples[0]}
        predicted = solver.predict(batch["history_obs"], batch["history_act"], batch["current_command"], history_valid_mask=batch["history_valid_mask"])
        errors.extend((((predicted[:, 0] - batch["target_actions"][:, 0]) / solver.backend.stats["act"]["std"]) ** 2).mean(-1).tolist())
    return {"independent_teacher_episodes": len(rows), "windows": len(errors), "TF0_normalized_action_mse": float(np.mean(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, default=Path("results/_development/p3_20261014"))
    parser.add_argument("--seed", type=int, default=18600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config); cfg.use_development_output_root(args.development_root)
    if not cfg.metadata.get("source_only") or cfg.env_name != "keyed_insertion_recovery":
        raise ValueError("this entrypoint requires the declared source-only T3 protocol")
    recipe = cfg.method_params
    if not 0 < recipe["shared_online_offline_ratio"] < 1 or not 0 < recipe["online_real_shadow_ratio"] < 1:
        raise ValueError("configured source sampling ratios must be strictly between zero and one")
    if recipe["maximum_wall_seconds"] > 900 or recipe["maximum_gpu_GiB"] > 4:
        raise ValueError("single-run hard budgets are900seconds and4GiB")
    output = require_non_paper_output(cfg.output_dir / "level_source" / f"seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous_wall = 0.
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["phase"] != "training": raise ValueError("only saved complete-data training boundaries support exact resume; pending physics is retained")
        previous_wall = float(json.loads((output / "runtime_usage.json").read_text())["wall_seconds_consumed"])
    else:
        if (output / "attempt.json").exists(): raise RuntimeError("source attempt exists; preserve it and use explicit saved training resume")
        state = {"phase": "collection", "stage": -1, "cursor": 0, "offline": [], "validation": [], "real": [], "shadows": [], "stage_reports": []}
        atomic_json(output / "attempt.json", {"status": "running", "seed": args.seed, "maximum_wall_seconds": recipe["maximum_wall_seconds"],
            "maximum_gpu_GiB": recipe["maximum_gpu_GiB"], "target_calls": 0,
            "variants": recipe["decoder_variants"], "comparison": "identical trained weights/data; policy-owned decoder only"})
    remaining = recipe["maximum_wall_seconds"] - previous_wall
    if remaining <= 0: raise SourceBudgetExceeded("source attempt wall budget is exhausted")
    def interrupt(_signum, _frame): raise SourceBudgetExceeded("source-only900second hard wall limit")
    signal.signal(signal.SIGALRM, interrupt); signal.setitimer(signal.ITIMER_REAL, remaining)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1": raise ValueError("this run must use physical GPU1; GPU0 is externally occupied")
        torch.cuda.init()
        capacity = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction((recipe["maximum_gpu_GiB"] * 1024**3 - 512 * 1024**2) / capacity, device)
        torch.cuda.reset_peak_memory_stats(device)
    def guard():
        if previous_wall + time.monotonic() - started >= recipe["maximum_wall_seconds"]: raise SourceBudgetExceeded("source wall budget exhausted")
        if device.type == "cuda" and torch.cuda.memory_reserved(device) > (recipe["maximum_gpu_GiB"] * 1024**3 - 512 * 1024**2):
            raise SourceBudgetExceeded("source CUDA allocation budget exhausted")
    initial_path = Path(recipe["initial_checkpoint"])
    initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    if not args.resume:
        paths = set((root / "genedynamics").rglob("*.py")) | set((root / "scripts/baselines").glob("*.py")) | {Path(__file__).resolve(), args.config.resolve()}
        hashes = {}
        with zipfile.ZipFile(output / "source-overlay.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(paths):
                data = path.read_bytes(); name = str(path.relative_to(root)); hashes[name] = hashlib.sha256(data).hexdigest(); archive.writestr(name, data)
        atomic_json(output / "provenance.json", {"source_sha256": hashes, "initial_checkpoint": str(initial_path), "initial_checkpoint_sha256": initial_sha,
            "recipe_sha256": recipe_sha, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initialization": "source model weights only; new two AdamW optimizers; exact new-run continuation checkpoints include both optimizers/RNG",
            "augmentation": "unchanged previous task-adapted recipe; no inferred mapping of humanoid joint sensor augmentation"})
    else:
        if state["recipe_sha256"] != recipe_sha or state["initial_sha256"] != initial_sha: raise ValueError("resume recipe/source checkpoint changed")
        verify_resume_source_fingerprint(root, json.loads((output / "provenance.json").read_text()))
    payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    trainer = TaskPlannerIDMTrainer(payload["cfg"], {key: payload[key] for key in ("obs_norm_stats", "action_norm_stats", "command_norm_stats")}, device=args.device)
    trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
    if args.resume: trainer.restore_training_state(state["training_state"])
    ledger = SourceAccountingLedger(output / "source_accounting.json")
    def save():
        state.update(training_state=trainer.get_training_state(), recipe_sha256=recipe_sha, initial_sha256=initial_sha)
        _torch_save(output / "training_continuation.pt", state)
        atomic_json(output / "runtime_usage.json", {"wall_seconds_consumed": previous_wall + time.monotonic() - started})
    def read_rows(refs):
        for ref in refs:
            if hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]: raise ValueError("source dataset SHA changed")
        return [project_trajectory(load_source_trajectory(ref["path"], mode=ref["mode"]), instantiate_task("T3", robot="panda"))
            if ref.get("qualified_original") else load_saved_episode(ref["path"], ref["sha256"]) for ref in refs]
    try:
        with ledger.installed():
            if not args.resume:
                for mode, path in enumerate(recipe["source_demonstrations"]):
                    load_source_trajectory(Path(path), mode=mode)
                    state["offline"].append({"path": str(path), "mode": mode, "qualified_original": True, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
                for split, per_mode, offset in (("offline", recipe["teacher_train_episodes_per_mode"], 10), ("validation", recipe["teacher_validation_episodes_per_mode"], 20)):
                    for repetition in range(per_mode):
                        for mode in (0, 1):
                            guard(); seed = args.seed + offset + repetition * 2 + mode
                            folder = output / "data" / split / f"seed_{seed}"
                            atomic_json(output / "physical_phase.json", {"phase": "teacher_collection", "split": split, "seed": seed})
                            row, report = collect_source_teacher("T3", seed=seed, mode=mode, output=folder, guard=guard)
                            if report["evaluation"]["successes"] == 1 and report["observed_mode"] == mode:
                                state[split].append({"path": str(folder / "episode.npz"), "sha256": report["episode_sha256"], "mode": mode})
                if len(state["validation"]) < 2: raise RuntimeError("independent complete teacher validation episodes failed; no free replacement")
                state.update(phase="training", stage=-1, cursor=0); save()
            offline, validation = read_rows(state["offline"]), read_rows(state["validation"])
            split_audit = validate_episode_split(state["offline"], state["validation"], offline, validation)
            atomic_json(output / "episode_split_audit.json", split_audit)
            support = source_force_support(offline)
            positive = [value for value in support["values_N"] if value > 0]
            if 0. not in support["values_N"] or not positive: raise RuntimeError("source teacher bank lacks an observed off/positive force gap")
            codec = {"kind": "source_gap_off_deadzone", **support, "threshold_N": min(positive) / 2,
                     "positive_force_remains_continuous": True, "physics_predicate_unchanged": True}
            val_support = source_force_support(validation)
            atomic_json(output / "data_manifest.json", {key: state[key] for key in ("offline", "validation", "real", "shadows")})
            atomic_json(output / "force_support.json", {"training_teacher_bank": codec, "validation_teacher_bank": val_support,
                "unseen_validation_force_values": sorted(set(val_support["values_N"]) - set(support["values_N"]))})
            data = TaskWindowDataset(offline, history_len=30, pred_horizon_k=6)
            phases = [(-1, recipe["warmup_updates"])] + list(enumerate([recipe["stage_updates"]] * len(recipe["teacher_probabilities"])))
            for stage, updates in phases:
                if stage < state["stage"]: continue
                if stage >= 0 and (stage > state["stage"] or state["cursor"] == 0) and not any(ref["stage"] == stage for ref in state["shadows"]):
                    state.update(phase="collection", stage=stage, cursor=0); save()
                    checkpoint = trainer.checkpoint()
                    for mode in (0, 1):
                        folder = output / f"stage_{stage}" / f"DAgger_mode_{mode}"
                        row, labels, report = collect_full_dagger("T3", checkpoint, seed=args.seed + 100 + stage * 10 + mode, mode=mode,
                            teacher_probability=recipe["teacher_probabilities"][stage], output=folder, device=args.device, guard=guard)
                        path = folder / "shadow_labels.pt"; _torch_save(path, labels)
                        state["real"].append({"path": str(folder / "primary/episode.npz"), "sha256": report["episode_sha256"], "stage": stage})
                        state["shadows"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "stage": stage})
                    state["phase"] = "training"; save()
                real_rows = [row for row in read_rows(state["real"]) if len(row.obs) > 6]
                real_data = TaskWindowDataset(real_rows, history_len=30, pred_horizon_k=6) if real_rows else None
                shadows = []
                for ref in state["shadows"]:
                    if hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]: raise ValueError("shadow dataset SHA changed")
                    shadows.extend(torch.load(ref["path"], map_location="cpu", weights_only=True))
                state.update(stage=stage, phase="training")
                def batch(): return sample_shared_dagger(data, real_data, shadows, recipe["batch_size"],
                    offline_ratio=recipe["shared_online_offline_ratio"], online_real_ratio=recipe["online_real_shadow_ratio"])
                while state["cursor"] < updates:
                    guard(); metrics = trainer.step(batch()); state["cursor"] += 1
                    state["last_training_metrics"] = {"stage": stage, "cursor": state["cursor"], "metrics": metrics}
                    if state["cursor"] % 100 == 0: save()
                retained = state.get("last_training_metrics", {})
                metrics = retained["metrics"] if retained.get("stage") == stage and retained.get("cursor") == state["cursor"] else {
                    "availability": "not_retained_by_legacy_boundary_checkpoint"}
                save()
                # A real disk-loaded complete optimizer/dropout/RNG branch.
                before = trainer.get_training_state(); _torch_save(output / "resume_probe.pt", before)
                expected_metrics = trainer.step(batch()); expected = trainer.get_training_state()
                trainer.restore_training_state(torch.load(output / "resume_probe.pt", map_location="cpu", weights_only=True))
                actual_metrics = trainer.step(batch()); exact = expected_metrics == actual_metrics and _equal(expected, trainer.get_training_state())
                if not exact: raise RuntimeError("source two-pass optimizer/RNG resume differs")
                checkpoint = trainer.checkpoint()
                stage_report = {"stage": stage, "last_training": metrics, "resume_exact": exact,
                    "resume_verification_extra_update_executions": 2, "validation": _validation(checkpoint, validation, recipe["batch_size"]), "variants": {}}
                for variant, decoder in (("continuous13D", {"kind": "continuous13D"}), ("source_gap_off_deadzone", codec)):
                    exported = deepcopy(checkpoint); exported["genedynamics_task_adapter"]["action_decoder"] = decoder
                    exported["genedynamics_task_adapter"]["task_name"] = cfg.env_name
                    exported["genedynamics_task_adapter"]["task_action_loader"] = "genedynamics.learning.baselines.fada.source_training.SourceTaskPolicy.from_checkpoint"
                    exported["genedynamics_task_adapter"]["shared_training_weights_comparison"] = True
                    path = output / f"stage_{stage}" / f"{variant}.pt"; path.parent.mkdir(parents=True, exist_ok=True); _torch_save(path, exported)
                    stage_report["variants"][variant] = evaluate_source_variant("T3", exported,
                        seed=args.seed + 1000 + (stage + 1) * 100, output=output / f"stage_{stage}" / variant,
                        device=args.device, guard=guard, ledger=ledger)
                state["stage_reports"].append(stage_report)
                atomic_json(output / f"stage_{stage}" / "report.json", json.loads(json.dumps(stage_report, default=_json)))
                state.update(stage=stage + 1, cursor=0, phase="training"); save()
            final = state["stage_reports"][-1]
            summary = {"status": "bounded_source_training_complete", "target_calls": 0, "source_only": True,
                "recipe": recipe, "data_manifest": {key: state[key] for key in ("offline", "validation", "real", "shadows")},
                "stage_reports": state["stage_reports"], "final_continuous_gate_passed": final["variants"]["continuous13D"]["gate_passed"],
                "final_decoder_variant_gate_passed": final["variants"]["source_gap_off_deadzone"]["gate_passed"],
                "force_decoder": codec, "irreversible_source_accounting": ledger.values,
                "episode_split_audit": split_audit,
                "wall_seconds_consumed": previous_wall + time.monotonic() - started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                "scope": "shared-learner source-only decoder comparison; not native humanoid FADA reproduction or target/P4 performance"}
            atomic_json(output / "qualification.json", json.loads(json.dumps(summary, default=_json)))
            atomic_json(output / "attempt.json", {"status": "complete", "target_calls": 0, "wall_seconds": summary["wall_seconds_consumed"]})
            print(json.dumps({key: summary[key] for key in ("status", "final_continuous_gate_passed", "final_decoder_variant_gate_passed", "wall_seconds_consumed")}))
    except BaseException as error:
        atomic_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "phase": state["phase"], "stage": state["stage"],
            "cursor": state["cursor"], "source_accounting": ledger.values, "target_calls": 0})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); ledger.save()
        atomic_json(output / "runtime_usage.json", {"wall_seconds_consumed": previous_wall + time.monotonic() - started})


if __name__ == "__main__":
    main()
