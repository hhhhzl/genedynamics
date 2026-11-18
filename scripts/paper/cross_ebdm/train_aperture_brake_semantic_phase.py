#!/usr/bin/env python3
"""Train and evaluate the budgeted semantic-phase ICDT aperture actor."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.cross_ebdm.learning.ablation import treatment_metadata
from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import PublicSemanticPhaseTransport
from genedynamics.tasks.transfer.aperture import PublicApertureBrakePolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.runtime import MechanismSpec

VERSION = "aperture-brake-semantic-phase-training-v1"
TREATMENTS = ("full", "no_calibration", "no_response_anchoring", "unconditional_response_anchoring")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(value, label):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} requires an exact pointer")
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def checkpoint(task, treatment, budget, scale, diagnostic, selection):
    active = [2, 3] if treatment != "unconditional_response_anchoring" else [0, 1, 2, 3]
    if treatment in {"no_calibration", "no_response_anchoring"}:
        scale = 1.0
    return {
        "schema_version": "semantic-phase-transport-v1",
        "task": task,
        "treatment": treatment_metadata(treatment),
        "interaction_budget_seconds": budget,
        "phase_scales": [1.0, 1.0, float(scale), float(scale)],
        "active_phases": active,
        "translation_coordinates": [0, 1, 2],
        "selection": selection,
        "response_model": diagnostic,
    }


def execute(config, *, seed, mode, actor, destination, collect=False):
    mechanism = MechanismSpec(**config["mechanism"])
    task = instantiate_task(
        config["task"], robot=config["robot"], mechanism_spec=mechanism,
        seed=seed, config_overrides=config["task_overrides"],
    )
    public = task.reset((seed,), seed=seed, modes=[mode])
    samples = []
    started = time.perf_counter()
    for _ in range(task.spec.config.timeout_steps):
        if task.state.status[0] != 0:
            break
        position = np.asarray(public["tcp_position"], float)[0].copy()
        action = np.asarray(actor.act(public), np.float32)
        step = task.step(action)
        following = np.asarray(step.observation["tcp_position"], float)[0].copy()
        if collect:
            samples.append((action[0, :3].astype(float), (following - position) / task.spec.config.control_dt))
        public = step.observation
    elapsed = time.perf_counter() - started
    evaluation = task.evaluate()
    episode = evaluation["episodes"][-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(
        episode_evidence(task.records, task.spec, evaluation), sort_keys=True,
        allow_nan=False,
    ).encode(), mtime=0))
    audit = audit_episode_evidence(destination)
    event = episode.get("event_physics_steps", {})
    row = {
        "seed": seed, "mode": mode, "success": bool(episode["success"]),
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "physical_control_steps": int(episode["physical_control_steps"]),
        "physical_seconds": float(episode["physical_seconds"]),
        "capped_normalized_completion_time": float(episode["capped_normalized_completion_time"]),
        "key_events": event,
        "peak_force_N": float(episode.get("force_peak_N", 0.0)),
        "reset_physical_hash": audit["reset_physical_hashes"][0],
        "evidence": pointer(destination), "evidence_audit_passed": bool(audit["passed"]),
        "wall_seconds": elapsed,
    }
    return row, samples


def fit_response(samples, *, device):
    if not samples:
        return {"kind": "public-linear-one-step-v1", "train_samples": 0,
                "validation_samples": 0, "validation_mse": 0.0}, []
    import torch
    data = np.asarray(samples, dtype=np.float32)
    x = data[:, 0, :]
    y = data[:, 1, :]
    split = np.arange(len(x)) % 5 != 0
    train_x = torch.as_tensor(x[split], device=device)
    train_y = torch.as_tensor(y[split], device=device)
    val_x = torch.as_tensor(x[~split], device=device)
    val_y = torch.as_tensor(y[~split], device=device)
    design = torch.cat((train_x, torch.ones((len(train_x), 1), device=device)), dim=1)
    weights = torch.linalg.pinv(design) @ train_y
    validation = torch.cat((val_x, torch.ones((len(val_x), 1), device=device)), dim=1) @ weights
    mse = float(torch.mean((validation - val_y) ** 2).detach().cpu()) if len(val_x) else 0.0
    return {
        "kind": "public-linear-one-step-v1", "train_samples": int(len(train_x)),
        "validation_samples": int(len(val_x)), "validation_mse": mse,
    }, weights.detach().cpu().numpy().tolist()


def summarize(rows):
    return {
        "episodes": len(rows),
        "task_success": sum(row["success"] for row in rows) / len(rows),
        "key_event_completion": sum(bool(row["key_events"].get("terminal_hold_completed")) for row in rows) / len(rows),
        "peak_force_N": max(row["peak_force_N"] for row in rows),
        "failure_penalized_normalized_time": statistics.mean(row["capped_normalized_completion_time"] for row in rows),
    }


def select_scale(rows):
    by_scale = defaultdict(list)
    for row in rows:
        by_scale[row["scale"]].append(row)
    qualified = []
    for scale, values in by_scale.items():
        if {row["mode"] for row in values} == {0, 1} and all(row["success"] for row in values):
            qualified.append((statistics.mean(row["capped_normalized_completion_time"] for row in values), scale))
    return min(qualified)[1] if qualified else 1.0


def run(config_path):
    config = json.loads(config_path.read_text())
    required = {"version", "status", "task", "robot", "contract", "source_qualification", "mechanism", "task_overrides", "budget_seconds", "candidate_stream", "evaluation_requests", "execution", "output", "policy"}
    if set(config) != required or config["version"] != VERSION or config["status"] != "frozen_unexecuted":
        raise ValueError("aperture semantic-phase recipe changed")
    if config["budget_seconds"] != [0, 30, 60, 120] or len(config["evaluation_requests"]) != 4:
        raise ValueError("frozen budget or evaluation denominator changed")
    contract = checked(config["contract"], "task contract")
    source = checked(config["source_qualification"], "source qualification")
    if json.loads(source.read_text()).get("passed") is not True:
        raise ValueError("passed source qualification required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("physical GPU0 only")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device required")
    visible = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    if visible != config["execution"]["physical_uuid"]:
        raise RuntimeError("visible GPU UUID differs")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    plan = {"version": VERSION, "config": pointer(config_path), "contract": pointer(contract),
            "source_qualification": pointer(source), "execution": config["execution"], "policy": config["policy"]}
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    selection_rows, samples, cumulative = [], [], 0.0
    for index, request in enumerate(config["candidate_stream"]):
        base = PublicApertureBrakePolicy(
            instantiate_task(config["task"], robot=config["robot"], mechanism_spec=MechanismSpec(**config["mechanism"]),
                             seed=request["seed"], config_overrides=config["task_overrides"]).spec,
            mode=request["mode"],
        )
        provisional = checkpoint(config["task"], "full", 120, request["scale"],
                                 {"kind": "public-linear-one-step-v1", "train_samples": 0, "validation_samples": 0, "validation_mse": 0.0},
                                 {"status": "candidate"})
        actor = PublicSemanticPhaseTransport(base, provisional, task=config["task"])
        row, episode_samples = execute(config, seed=request["seed"], mode=request["mode"], actor=actor,
                                       destination=output / "selection_evidence" / f"candidate_{index:02d}.json.gz", collect=True)
        row["scale"] = float(request["scale"])
        cumulative += row["physical_seconds"]
        row["cumulative_interaction_seconds"] = cumulative
        selection_rows.append(row); samples.extend(episode_samples)
        write_json(output / "progress.json", {"phase": "selection", "rows": selection_rows})
        print(json.dumps({"phase": "selection", "scale": row["scale"], "mode": row["mode"], "success": row["success"], "cumulative_seconds": cumulative}), flush=True)
    checkpoints = {}
    curve_rows = []
    for budget in config["budget_seconds"]:
        admitted = [row for row in selection_rows if row["cumulative_interaction_seconds"] <= budget]
        admitted_samples = samples[:sum(row["physical_control_steps"] for row in admitted)]
        diagnostic, weights = fit_response(admitted_samples, device="cuda")
        scale = select_scale(admitted)
        selection = {"status": "development", "selected_scale": scale,
                     "admitted_episode_count": len(admitted), "response_weights": weights,
                     "criterion": "lowest failure-penalized normalized time among two-mode constrained-success candidates; unit scale if none"}
        value = checkpoint(config["task"], "full", budget, scale, diagnostic, selection)
        path = output / "checkpoints" / f"full_budget_{budget}.json"
        write_json(path, value); checkpoints[f"full_budget_{budget}"] = pointer(path)
        for request in config["evaluation_requests"]:
            actor = PublicSemanticPhaseTransport(PublicApertureBrakePolicy(
                instantiate_task(config["task"], robot=config["robot"], mechanism_spec=MechanismSpec(**config["mechanism"]),
                                 seed=request["seed"], config_overrides=config["task_overrides"]).spec,
                mode=request["mode"]), value, task=config["task"])
            row, _ = execute(config, seed=request["seed"], mode=request["mode"], actor=actor,
                             destination=output / "budget_evidence" / f"budget_{budget}" / f"seed_{request['seed']}.json.gz")
            row.update(budget_seconds=budget, treatment="full", selected_scale=scale)
            curve_rows.append(row)
            print(json.dumps({"phase": "budget", "budget": budget, "seed": row["seed"], "success": row["success"]}), flush=True)
    final_full = json.loads((ROOT / checkpoints["full_budget_120"]["path"]).read_text())
    treatment_rows = []
    for treatment in TREATMENTS:
        value = checkpoint(config["task"], treatment, 120,
                           final_full["selection"]["selected_scale"], final_full["response_model"],
                           {**final_full["selection"], "clean_treatment": treatment})
        path = output / "checkpoints" / f"{treatment}_budget_120.json"
        write_json(path, value); checkpoints[treatment] = pointer(path)
        for request in config["evaluation_requests"]:
            actor = PublicSemanticPhaseTransport(PublicApertureBrakePolicy(
                instantiate_task(config["task"], robot=config["robot"], mechanism_spec=MechanismSpec(**config["mechanism"]),
                                 seed=request["seed"], config_overrides=config["task_overrides"]).spec,
                mode=request["mode"]), value, task=config["task"])
            row, _ = execute(config, seed=request["seed"], mode=request["mode"], actor=actor,
                             destination=output / "treatment_evidence" / treatment / f"seed_{request['seed']}.json.gz")
            row.update(budget_seconds=120, treatment=treatment,
                       selected_scale=value["selection"]["selected_scale"] if treatment == "full" else float(value["phase_scales"][2]))
            treatment_rows.append(row)
    report = {
        "version": VERSION, "status": "complete_development_only",
        "selection_rows": selection_rows,
        "budget_curve": {str(b): summarize([r for r in curve_rows if r["budget_seconds"] == b]) for b in config["budget_seconds"]},
        "treatments": {name: summarize([r for r in treatment_rows if r["treatment"] == name]) for name in TREATMENTS},
        "budget_rows": curve_rows, "treatment_rows": treatment_rows, "checkpoints": checkpoints,
        "all_evidence_reaudited": all(r["evidence_audit_passed"] for r in selection_rows + curve_rows + treatment_rows),
        "formal_performance_evidence": False, "gpu": {"physical_uuid": visible, "purpose": "public one-step response diagnostic fit"},
        "model_mse_role": "diagnostic_only", "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "budget_curve": report["budget_curve"], "treatments": report["treatments"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/arm/obstacle_transfer/training/aperture_brake_semantic_phase.json"))
    args = parser.parse_args(); run(args.config)


if __name__ == "__main__":
    main()
