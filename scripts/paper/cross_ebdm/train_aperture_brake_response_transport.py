#!/usr/bin/env python3
"""Train and evaluate response-conditioned transport for aperture braking."""
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
from genedynamics.solvers.single.cross_ebdm.learning.ablation import (
    TREATMENTS, load_contract, treatment_metadata,
)
from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import (
    PublicSemanticPhaseTransport, RESPONSE_KIND, VERSION as CHECKPOINT_VERSION,
    select_learned_scale, validate_checkpoint,
)
from genedynamics.tasks.transfer.aperture import PublicApertureBrakePolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.runtime import MechanismSpec

VERSION = "aperture-brake-response-transport-training-v2"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def empty_fit():
    return {"weights": np.zeros((4, 3)).tolist(), "train_samples": 0,
            "validation_samples": 0, "validation_mse": 0.}


def response_model(source, target):
    return {
        "kind": RESPONSE_KIND, "source": source, "target": target,
        "input_semantics": "issued xyz velocity command plus affine bias",
        "output_semantics": "observed next-control xyz TCP velocity",
    }


def make_checkpoint(task, treatment_name, budget, selection, model):
    selected = float(selection["selected_scale"])
    value = {
        "schema_version": CHECKPOINT_VERSION, "task": task,
        "treatment": treatment_metadata(treatment_name),
        "interaction_budget_seconds": int(budget),
        "phase_scales": [1., 1., selected, selected],
        "active_phases": [2, 3],
        "translation_coordinates": [0, 1, 2],
        "selection": selection, "response_model": model,
    }
    validate_checkpoint(value, task=task)
    return value


def candidate_checkpoint(task, scale):
    selection = select_learned_scale([{
        "scale": float(scale), "task_loss": 0.,
        "calibrated_response_loss": 0., "uncalibrated_response_loss": 0.,
    }], treatment_name="full")
    return make_checkpoint(
        task, "full", 120, selection,
        response_model(empty_fit(), empty_fit()),
    )


def execute(config, *, request, mechanism, scale, destination, collect):
    task = instantiate_task(
        config["task"], robot=config["robot"],
        mechanism_spec=MechanismSpec(**mechanism), seed=request["seed"],
        config_overrides=config["task_overrides"],
    )
    base = PublicApertureBrakePolicy(task.spec, mode=request["mode"])
    actor = PublicSemanticPhaseTransport(
        base, candidate_checkpoint(config["task"], scale), task=config["task"],
    )
    public = task.reset((request["seed"],), seed=request["seed"],
                        modes=[request["mode"]])
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
            samples.append({
                "command": action[0, :3].astype(float),
                "base_command": (action[0, :3] / actor.last_scale).astype(float),
                "velocity": ((following - position)
                             / task.spec.config.control_dt).astype(float),
            })
        public = step.observation
    evaluation = task.evaluate()
    episode = evaluation["episodes"][-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(
        episode_evidence(task.records, task.spec, evaluation),
        sort_keys=True, allow_nan=False,
    ).encode(), mtime=0))
    audit = audit_episode_evidence(destination)
    row = {
        "seed": int(request["seed"]), "mode": int(request["mode"]),
        "scale": float(scale), "success": bool(episode["success"]),
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "physical_control_steps": int(episode["physical_control_steps"]),
        "physical_seconds": float(episode["physical_seconds"]),
        "capped_normalized_completion_time": float(
            episode["capped_normalized_completion_time"]),
        "key_events": episode.get("event_physics_steps", {}),
        "peak_force_N": float(episode.get("force_peak_N", 0.)),
        "reset_physical_hash": audit["reset_physical_hashes"][0],
        "evidence": pointer(destination),
        "evidence_audit_passed": bool(audit["passed"]),
        "wall_seconds": time.perf_counter() - started,
    }
    return row, samples


def fit_response(samples, *, device):
    if not samples:
        return empty_fit()
    import torch
    x = np.asarray([sample["command"] for sample in samples], np.float32)
    y = np.asarray([sample["velocity"] for sample in samples], np.float32)
    validation_mask = np.arange(len(x)) % 5 == 0
    if validation_mask.all():
        validation_mask[-1] = False
    train_x = torch.as_tensor(x[~validation_mask], device=device)
    train_y = torch.as_tensor(y[~validation_mask], device=device)
    val_x = torch.as_tensor(x[validation_mask], device=device)
    val_y = torch.as_tensor(y[validation_mask], device=device)
    design = torch.cat((train_x, torch.ones((len(train_x), 1), device=device)), 1)
    weights = torch.linalg.pinv(design) @ train_y
    validation = torch.cat(
        (val_x, torch.ones((len(val_x), 1), device=device)), 1,
    ) @ weights
    mse = float(torch.mean((validation - val_y) ** 2).detach().cpu())
    return {"weights": weights.detach().cpu().numpy().tolist(),
            "train_samples": int(len(train_x)),
            "validation_samples": int(len(val_x)),
            "validation_mse": mse}


def predicted(fit, commands):
    commands = np.asarray(commands, dtype=float)
    design = np.c_[commands, np.ones(len(commands))]
    return design @ np.asarray(fit["weights"], dtype=float)


def mismatch(samples, source_fit, target_fit):
    if not samples:
        return 0., 0.
    command = np.asarray([sample["command"] for sample in samples])
    base = np.asarray([sample["base_command"] for sample in samples])
    desired = predicted(source_fit, base)
    calibrated = predicted(target_fit, command)
    uncalibrated = predicted(source_fit, command)
    denominator = float(np.mean(desired ** 2) + 1e-8)
    return (float(np.mean((calibrated - desired) ** 2) / denominator),
            float(np.mean((uncalibrated - desired) ** 2) / denominator))


def candidate_statistics(rows, samples_by_scale, source_fit, target_fit):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scale"]].append(row)
    output = []
    for scale, values in sorted(grouped.items()):
        if {row["mode"] for row in values} != {0, 1}:
            continue
        calibrated, uncalibrated = mismatch(
            samples_by_scale[scale], source_fit, target_fit,
        )
        task_loss = statistics.mean(
            (0. if row["success"] else 1.)
            + row["capped_normalized_completion_time"] for row in values
        )
        output.append({
            "scale": float(scale), "task_loss": float(task_loss),
            "calibrated_response_loss": calibrated,
            "uncalibrated_response_loss": uncalibrated,
        })
    return output


def summarize(rows):
    if not rows:
        return {"episodes": 0, "task_success": None,
                "key_event_completion": None, "peak_force_N": None,
                "failure_penalized_normalized_time": None}
    return {
        "episodes": len(rows),
        "task_success": sum(row["success"] for row in rows) / len(rows),
        "key_event_completion": sum(
            bool(row["key_events"].get("terminal_hold_completed"))
            for row in rows) / len(rows),
        "peak_force_N": max(row["peak_force_N"] for row in rows),
        "failure_penalized_normalized_time": statistics.mean(
            row["capped_normalized_completion_time"] for row in rows),
    }


def run(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    required = {"version", "status", "task", "robot", "contract",
                "source_qualification", "ablation_contract", "source_mechanism",
                "target_mechanism", "task_overrides", "budget_seconds",
                "source_calibration_stream", "candidate_stream",
                "evaluation_requests", "execution", "output", "policy"}
    if set(config) != required or config["version"] != VERSION:
        raise ValueError("aperture response-transport recipe changed")
    if config["status"] != "frozen_unexecuted":
        raise ValueError("only a frozen unexecuted recipe may run")
    if config["budget_seconds"] != [0, 30, 60, 120]:
        raise ValueError("frozen budget grid changed")
    contract = checked(config["contract"], "task contract")
    source = checked(config["source_qualification"], "source qualification")
    ablation_path = checked(config["ablation_contract"], "ablation contract")
    load_contract(ablation_path)
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
    write_json(output / "plan.json", {
        "version": VERSION, "config": pointer(config_path),
        "contract": pointer(contract), "source_qualification": pointer(source),
        "ablation_contract": pointer(ablation_path),
        "execution": config["execution"], "policy": config["policy"],
    })
    started = time.monotonic()

    source_rows, source_samples = [], []
    for index, request in enumerate(config["source_calibration_stream"]):
        row, samples = execute(
            config, request=request, mechanism=config["source_mechanism"],
            scale=request["scale"],
            destination=output / "source_calibration_evidence"
            / f"request_{index:02d}.json.gz", collect=True,
        )
        source_rows.append(row)
        source_samples.extend(samples)
    source_fit = fit_response(source_samples, device="cuda")

    selection_rows, selection_samples, cumulative = [], [], 0.
    for index, request in enumerate(config["candidate_stream"]):
        row, samples = execute(
            config, request=request, mechanism=config["target_mechanism"],
            scale=request["scale"],
            destination=output / "target_calibration_evidence"
            / f"candidate_{index:02d}.json.gz", collect=True,
        )
        cumulative += row["physical_seconds"]
        row["cumulative_target_interaction_seconds"] = cumulative
        selection_rows.append(row)
        selection_samples.append(samples)
        write_json(output / "progress.json", {
            "phase": "target_calibration", "rows": selection_rows,
        })
        print(json.dumps({
            "phase": "target_calibration", "scale": row["scale"],
            "mode": row["mode"], "success": row["success"],
            "cumulative_seconds": cumulative,
        }), flush=True)

    checkpoints, curve_rows = {}, []
    budget_state = {}
    for budget in config["budget_seconds"]:
        admitted_indices = [
            index for index, row in enumerate(selection_rows)
            if row["cumulative_target_interaction_seconds"] <= budget
        ]
        admitted = [selection_rows[index] for index in admitted_indices]
        admitted_by_scale = defaultdict(list)
        for index in admitted_indices:
            admitted_by_scale[selection_rows[index]["scale"]].extend(
                selection_samples[index]
            )
        admitted_samples = [
            sample for values in admitted_by_scale.values() for sample in values
        ]
        target_fit = (
            fit_response(admitted_samples, device="cuda")
            if admitted_samples else source_fit
        )
        candidates = candidate_statistics(
            admitted, admitted_by_scale, source_fit, target_fit,
        )
        selection = select_learned_scale(
            candidates, treatment_name="full",
            task_band_tolerance=config["policy"]["task_band_tolerance"],
        )
        model = response_model(source_fit, target_fit)
        value = make_checkpoint(config["task"], "full", budget, selection, model)
        path = output / "checkpoints" / f"full_budget_{budget}.json"
        write_json(path, value)
        checkpoints[f"full_budget_{budget}"] = pointer(path)
        budget_state[budget] = (candidates, model)
        for request in config["evaluation_requests"]:
            actor = PublicSemanticPhaseTransport(
                PublicApertureBrakePolicy(
                    instantiate_task(
                        config["task"], robot=config["robot"],
                        mechanism_spec=MechanismSpec(**config["target_mechanism"]),
                        seed=request["seed"],
                        config_overrides=config["task_overrides"],
                    ).spec, mode=request["mode"]),
                value, task=config["task"],
            )
            row, _ = execute(
                config, request=request, mechanism=config["target_mechanism"],
                scale=selection["selected_scale"],
                destination=output / "budget_evidence" / f"budget_{budget}"
                / f"seed_{request['seed']}.json.gz", collect=False,
            )
            row.update(budget_seconds=budget, treatment="full",
                       selected_scale=selection["selected_scale"])
            curve_rows.append(row)

    candidates, model = budget_state[120]
    treatment_rows = []
    for name in TREATMENTS:
        selection = select_learned_scale(
            candidates, treatment_name=name,
            task_band_tolerance=config["policy"]["task_band_tolerance"],
        )
        value = make_checkpoint(config["task"], name, 120, selection, model)
        path = output / "checkpoints" / f"{name}_budget_120.json"
        write_json(path, value)
        checkpoints[name] = pointer(path)
        for request in config["evaluation_requests"]:
            row, _ = execute(
                config, request=request, mechanism=config["target_mechanism"],
                scale=selection["selected_scale"],
                destination=output / "treatment_evidence" / name
                / f"seed_{request['seed']}.json.gz", collect=False,
            )
            row.update(budget_seconds=120, treatment=name,
                       selected_scale=selection["selected_scale"])
            treatment_rows.append(row)

    all_rows = source_rows + selection_rows + curve_rows + treatment_rows
    report = {
        "version": VERSION, "status": "complete_development_only",
        "source_calibration": summarize(source_rows),
        "selection_rows": selection_rows,
        "budget_curve": {str(budget): summarize([
            row for row in curve_rows if row["budget_seconds"] == budget])
            for budget in config["budget_seconds"]},
        "treatments": {name: summarize([
            row for row in treatment_rows if row["treatment"] == name])
            for name in TREATMENTS},
        "treatment_selected_scales": {name: next(
            row["selected_scale"] for row in treatment_rows
            if row["treatment"] == name) for name in TREATMENTS},
        "budget_rows": curve_rows, "treatment_rows": treatment_rows,
        "checkpoints": checkpoints,
        "all_evidence_reaudited": all(
            row["evidence_audit_passed"] for row in all_rows),
        "response_model_used_for_selection": True,
        "fixed_gain_posthoc_selection": False,
        "model_mse_role": "diagnostic_only",
        "formal_performance_evidence": False,
        "gpu": {"physical_uuid": visible,
                "purpose": "public source/target response regression"},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "budget_curve": report["budget_curve"],
        "treatments": report["treatments"],
        "selected_scales": report["treatment_selected_scales"],
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/arm/obstacle_transfer/training/"
                     "aperture_brake_response_transport.json"),
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
