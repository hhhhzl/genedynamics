#!/usr/bin/env python3
"""Collect exact same-state T2 force/contact candidate-branch labels on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.specs import mechanism_spec


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "configs/arm/guided_contact_alignment"
DEFAULT_CONFIG = TASK_ROOT / "training/joint_admission_counterfactual_data_v2.json"
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
CHOICE_NAMES = ("fraction_1", "fraction_0p5", "fraction_0p25", "public_base", "emergency_brake")
PHASE_NAMES = ("pre_contact", "supported_traversal", "outlet_transition", "post_outlet")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": sha(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked(value: dict[str, str]) -> Path:
    path = ROOT / value["path"]
    if sha(path) != value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("version") != "t2-joint-force-contact-counterfactual-data-v2":
        raise ValueError("unknown joint-admission data version")
    if value.get("device") != "cpu" or value.get("status") != "frozen_cpu_design_unexecuted":
        raise ValueError("joint-admission collection requires the frozen CPU design")
    if value["candidate_generator"]["ordered_transport_fractions"] != [1.0, 0.5, 0.25, 0.0]:
        raise ValueError("deployment interpolation fractions changed")
    if not value["candidate_generator"]["append_emergency_reference_brake"]:
        raise ValueError("emergency reference brake must be declared")
    splits = value["split_by_reset_block"]
    if splits != {"fit": [0, 1, 2, 3], "calibration": [4, 5], "held_out_qualification": [6, 7]}:
        raise ValueError("reset-block split changed")
    if sorted(sum(splits.values(), [])) != list(range(value["requests"]["reset_blocks"])):
        raise ValueError("reset-block split is incomplete")
    for artifact in value["bindings"].values():
        checked(artifact)
    return value


def request_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    spec = config["requests"]
    split_for = {
        block: split for split, blocks in config["split_by_reset_block"].items()
        for block in blocks
    }
    rows = []
    sequence = 0
    for block in range(int(spec["reset_blocks"])):
        for mode in spec["requested_modes"]:
            rows.append({
                "request_id": f"T2-JOINT-ADMISSION/i{block:02d}/m{mode}",
                "reset_block": block,
                "physical_initial_slot": block,
                "requested_mode": int(mode),
                "reset_seed": int(spec["reset_seed_anchor"]) + sequence,
                "robot": spec["robot"],
                "mechanism_parameters": deepcopy(spec["mechanism_parameters"]),
                "split": split_for[block],
                "confirmation_state": False,
                "performance_evidence": False,
            })
            sequence += 1
    return rows


def build_plan(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    requests = request_rows(config)
    jobs = []
    for generator in config["state_generators"]:
        for instance in config["instances"]:
            for request in requests:
                jobs.append({
                    "record_id": f"{generator}/{instance}/{request['request_id']}",
                    "state_generator": generator,
                    "instance": instance,
                    "config": str((TASK_ROOT / "main/cross_ebdm.yaml").relative_to(ROOT)),
                    "request": request,
                })
    return {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
        "requests": requests,
        "jobs": jobs,
        "counts": {"requests": len(requests), "jobs": len(jobs)},
        "choice_names": list(CHOICE_NAMES),
        "policy": {
            "device": "cpu",
            "confirmation_states_opened": 0,
            "old_invalid_outcomes_used": False,
            "old_risk_v1_windows_reused": False,
            "performance_evidence": False,
            "failure_replacement": False,
        },
    }


def make_task(config: ExperimentConfig, request: dict[str, Any]):
    values = deepcopy(config.env_params)
    values.pop("robot", None)
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    shared_execution = values.pop("shared_execution", None)
    mechanism_values = {
        **dict(values.pop("mechanism", {}) or {}), **request["mechanism_parameters"],
    }
    body_support = dict(values.pop("body_support", {}) or {})
    task = instantiate_task(
        config.env_name,
        robot=request["robot"],
        mechanism_spec=mechanism_spec(
            "T2", mechanism_values, body_support_parameters=body_support,
        ),
        seed=int(request["reset_seed"]),
        history_len=history_len,
        config_overrides=values or None,
        shared_execution=shared_execution,
    )
    task._experiment_mode = int(request["requested_mode"])
    return task


def make_actor(task, config: ExperimentConfig, instance: str, mode: int):
    parameters = deepcopy(config.method_params)
    parameters.update(mode=mode, device="cpu")
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family="main", key="cross_ebdm", instance=instance)
    parameters["actor_binding"] = selection
    return create_transfer_actor(task, config.method, parameters)


def public_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key])) for key in left
    )


def max_false_run(values: np.ndarray) -> int:
    maximum = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def contact_state(record, task, mode: int) -> dict[str, Any]:
    rows = [row for row in record.traces if row["valid"]]
    if not rows:
        return {
            "first_contact": -1, "first_outlet": -1, "active": False,
            "outlet_without_contact": False, "coverage": 1.0,
            "maximum_loss_seconds": 0.0, "violation": False,
        }
    axial = np.concatenate([
        np.asarray(row["axial_support_substeps"], dtype=bool) for row in rows
    ])
    normal = np.concatenate([
        np.asarray(row["normal_force_substeps"], dtype=float) for row in rows
    ])
    positions = np.concatenate([
        np.asarray(row["tcp_position_substeps"], dtype=float).reshape(-1, 3)
        for row in rows
    ])
    contact = axial & (normal > task.spec.config.contact_threshold)
    outlet = positions[:, 0] > 0.455 if mode == 0 else positions[:, 1] > -0.181
    contact_indices = np.flatnonzero(contact)
    outlet_indices = np.flatnonzero(outlet)
    first_contact = int(contact_indices[0]) if len(contact_indices) else -1
    first_outlet = int(outlet_indices[0]) if len(outlet_indices) else -1
    outlet_without = first_outlet >= 0 and (first_contact < 0 or first_contact > first_outlet)
    if first_contact >= 0:
        end = first_outlet if first_outlet >= first_contact else len(contact) - 1
        segment = contact[first_contact:end + 1]
        coverage = float(segment.mean())
        maximum_loss = max_false_run(segment) * task.spec.config.physics_dt
        active = True
    else:
        coverage, maximum_loss, active = 1.0, 0.0, False
    violation = bool(
        outlet_without
        or maximum_loss > 0.15 + 1e-12
        or (first_outlet >= 0 and first_contact >= 0 and coverage < 0.5)
    )
    return {
        "first_contact": first_contact,
        "first_outlet": first_outlet,
        "active": active,
        "outlet_without_contact": outlet_without,
        "coverage": coverage,
        "maximum_loss_seconds": float(maximum_loss),
        "violation": violation,
    }


def candidate_chunks(actor, public, fractions: list[float]):
    import torch
    from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
        PublicSubstepRiskCalibration,
        risk_admission_candidates,
    )

    batch, base = actor.prepare(public)
    names = ("target_features", "source_features", "source_commands", "base_proto")
    arguments = [
        torch.as_tensor(batch[name], device=actor.device, dtype=torch.float32)
        for name in names
    ]
    if float(batch["source_query_distance"][0]) > float(actor.maximum_query_distance):
        candidate = arguments[3]
        actor.fallback_controls += 1
    else:
        with torch.inference_mode():
            candidate = actor.policy(
                *arguments,
                torch.as_tensor(batch["phase"], device=actor.device),
                torch.as_tensor(batch["own_context"], device=actor.device, dtype=torch.float32),
            )
        actor.irreversible_network_calls += 1
    calibration = PublicSubstepRiskCalibration(
        interpolation_fractions=tuple(float(value) for value in fractions),
    )
    chunks = risk_admission_candidates(candidate, arguments[3], calibration)
    normalized = chunks[0].detach().cpu().numpy().astype(np.float32)
    feature = np.asarray(batch["target_features"][0], dtype=np.float32)
    base_raw = np.asarray(base[0], dtype=np.float32)
    return feature, normalized, base_raw, int(np.asarray(batch["phase"])[0])


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def validate_npz(path: Path, *, horizon: int, choices: int) -> dict[str, Any]:
    required = {
        "features", "commands", "ticks", "controller_phase", "physical_phase",
        "force_peaks_N", "contact_coverage", "contact_loss_seconds",
        "contact_active", "outlet_without_contact", "contact_violation",
        "valid_control_steps", "terminal_status", "same_state_restored",
        "task_owned_equal", "lineage_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"unexpected joint-admission schema: {path}")
        count = len(archive["ticks"])
        if archive["commands"].shape != (count, choices, horizon, 13):
            raise ValueError("candidate command tensor shape differs")
        shapes = {
            "force_peaks_N": (count, choices, 2),
            "contact_coverage": (count, choices),
            "contact_loss_seconds": (count, choices),
            "contact_active": (count, choices),
            "outlet_without_contact": (count, choices),
            "contact_violation": (count, choices),
            "valid_control_steps": (count, choices),
            "terminal_status": (count, choices),
            "same_state_restored": (count, choices),
            "task_owned_equal": (count, choices),
            "lineage_sha256": (count,),
        }
        for name, shape in shapes.items():
            if archive[name].shape != shape:
                raise ValueError(f"{name} shape differs")
        if not np.isfinite(archive["features"]).all() or not np.isfinite(archive["commands"]).all():
            raise ValueError("nonfinite public feature or candidate command")
        if not archive["same_state_restored"].all() or not archive["task_owned_equal"].all():
            raise ValueError("same-state or task-owned identity failed")
        if np.any(archive["force_peaks_N"] < 0) or np.any(archive["contact_loss_seconds"] < 0):
            raise ValueError("negative physical label")
        phases = Counter(PHASE_NAMES[int(value)] for value in archive["physical_phase"])
        return {
            "checkpoints": count,
            "branches": count * choices,
            "branch_control_steps": int(archive["valid_control_steps"].sum()),
            "high_force_branches": int(np.sum(
                (archive["force_peaks_N"][..., 0] > 20.0)
                | (archive["force_peaks_N"][..., 1] > 8.0)
            )),
            "contact_violation_branches": int(archive["contact_violation"].sum()),
            "physical_phases": dict(sorted(phases.items())),
        }


def execute(payload: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    job, output_raw, collection = payload
    output = Path(output_raw)
    folder = output / "episodes" / hashlib.sha256(job["record_id"].encode()).hexdigest()
    result_path = folder / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    import torch
    torch.set_num_threads(1)

    experiment = ExperimentConfig.from_yaml(ROOT / job["config"])
    request = job["request"]
    task = make_task(experiment, request)
    public = task.reset(
        (int(request["physical_initial_slot"]),),
        seed=int(request["reset_seed"]), initial="clear",
        modes=[int(request["requested_mode"])],
    )
    actor = make_actor(task, experiment, job["instance"], int(request["requested_mode"]))
    fractions = collection["candidate_generator"]["ordered_transport_fractions"]
    horizon = int(collection["checkpoint_sampling"]["branch_horizon_control_steps"])
    stride = int(collection["checkpoint_sampling"]["stride_control_steps"])
    transport = collection["candidate_generator"]["transport_coordinates"]
    task_owned = collection["candidate_generator"]["task_owned_coordinates"]
    generator_choice = 0 if job["state_generator"] == "cross_ebdm" else 3

    arrays: dict[str, list[Any]] = {name: [] for name in (
        "features", "commands", "ticks", "controller_phase", "physical_phase",
        "force_peaks_N", "contact_coverage", "contact_loss_seconds",
        "contact_active", "outlet_without_contact", "contact_violation",
        "valid_control_steps", "terminal_status", "same_state_restored",
        "task_owned_equal", "lineage_sha256",
    )}
    charged_branch_controls = 0
    for _ in range(int(experiment.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        tick = int(np.rint(np.asarray(public["elapsed_seconds"]).item() / task.spec.config.control_dt))
        feature, chunks, base, phase = candidate_chunks(actor, public, fractions)
        raw_chunks = chunks * BOUNDS[None, None, :]
        selected = np.asarray(raw_chunks[generator_choice, 0], dtype=np.float32)[None]
        actor.last_action_tick = tick
        actor.cached_action = selected.copy()
        if tick % stride == 0:
            checkpoint = task.get_checkpoint()
            checkpoint_public = task.observe()
            prefix_contact = contact_state(task.records[-1], task, int(request["requested_mode"]))
            force_labels, coverage, loss, active, outlet_without = [], [], [], [], []
            violations, valid_steps, statuses, same_state, owned_equal = [], [], [], [], []
            crossed_outlet = False
            for choice in range(len(CHOICE_NAMES)):
                restored_public = task.restore_checkpoint(checkpoint)
                same_state.append(public_equal(checkpoint_public, restored_public))
                before_controls = int(task.accounting["control_steps"])
                prefix_steps = len(task.records[-1].traces)
                for command in raw_chunks[choice]:
                    if bool(np.asarray(task.state.status)[0]):
                        break
                    task.step(np.asarray(command, dtype=np.float32)[None])
                branch_rows = task.records[-1].traces[prefix_steps:]
                force_labels.append([
                    max((float(row["force_peak"]) for row in branch_rows), default=0.0),
                    max((float(row["lateral_force_peak"]) for row in branch_rows), default=0.0),
                ])
                state = contact_state(task.records[-1], task, int(request["requested_mode"]))
                coverage.append(state["coverage"])
                loss.append(state["maximum_loss_seconds"])
                active.append(state["active"])
                outlet_without.append(state["outlet_without_contact"])
                violations.append(state["violation"])
                crossed_outlet |= prefix_contact["first_outlet"] < 0 <= state["first_outlet"]
                controls = int(task.accounting["control_steps"]) - before_controls
                charged_branch_controls += controls
                valid_steps.append(controls)
                statuses.append(int(np.asarray(task.state.status)[0]))
                owned_equal.append(np.array_equal(
                    chunks[choice, :, task_owned], chunks[3, :, task_owned],
                ))
            task.restore_checkpoint(checkpoint)
            if prefix_contact["first_outlet"] >= 0:
                physical_phase = 3
            elif crossed_outlet:
                physical_phase = 2
            elif prefix_contact["first_contact"] >= 0:
                physical_phase = 1
            else:
                physical_phase = 0
            lineage = hashlib.sha256(f"{job['record_id']}@{tick}".encode()).hexdigest()
            values = {
                "features": feature,
                "commands": chunks,
                "ticks": tick,
                "controller_phase": phase,
                "physical_phase": physical_phase,
                "force_peaks_N": force_labels,
                "contact_coverage": coverage,
                "contact_loss_seconds": loss,
                "contact_active": active,
                "outlet_without_contact": outlet_without,
                "contact_violation": violations,
                "valid_control_steps": valid_steps,
                "terminal_status": statuses,
                "same_state_restored": same_state,
                "task_owned_equal": owned_equal,
                "lineage_sha256": lineage.encode(),
            }
            for name, value in values.items():
                arrays[name].append(value)
        result = task.step(selected)
        task.record_control_diagnostics(controller_diagnostics(actor))
        public = result.observation

    evaluation = task.evaluate()
    folder.mkdir(parents=True, exist_ok=True)
    evidence_path = folder / "evidence.json.gz"
    evidence = episode_evidence(task.records, task.spec, evaluation)
    temporary = evidence_path.with_suffix(".gz.tmp")
    temporary.write_bytes(gzip.compress(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6, mtime=0,
    ))
    os.replace(temporary, evidence_path)
    evidence_audit = audit_episode_evidence(evidence_path)

    typed = {
        "features": np.asarray(arrays["features"], np.float32),
        "commands": np.asarray(arrays["commands"], np.float32),
        "ticks": np.asarray(arrays["ticks"], np.int64),
        "controller_phase": np.asarray(arrays["controller_phase"], np.int64),
        "physical_phase": np.asarray(arrays["physical_phase"], np.int64),
        "force_peaks_N": np.asarray(arrays["force_peaks_N"], np.float32),
        "contact_coverage": np.asarray(arrays["contact_coverage"], np.float32),
        "contact_loss_seconds": np.asarray(arrays["contact_loss_seconds"], np.float32),
        "contact_active": np.asarray(arrays["contact_active"], bool),
        "outlet_without_contact": np.asarray(arrays["outlet_without_contact"], bool),
        "contact_violation": np.asarray(arrays["contact_violation"], bool),
        "valid_control_steps": np.asarray(arrays["valid_control_steps"], np.int64),
        "terminal_status": np.asarray(arrays["terminal_status"], np.int64),
        "same_state_restored": np.asarray(arrays["same_state_restored"], bool),
        "task_owned_equal": np.asarray(arrays["task_owned_equal"], bool),
        "lineage_sha256": np.asarray(arrays["lineage_sha256"], dtype="S64"),
    }
    data_path = folder / "branches.npz"
    save_npz(data_path, typed)
    summary = validate_npz(data_path, horizon=horizon, choices=len(CHOICE_NAMES))
    episode = evaluation["episodes"][-1]
    result = {
        **{key: job[key] for key in ("record_id", "state_generator", "instance", "request")},
        "outcome": episode["outcome"],
        "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "data_summary": summary,
        "data": pointer(data_path),
        "evidence": {**pointer(evidence_path), "audit_passed": bool(evidence_audit["passed"])},
        "accounting": dict(task.accounting),
        "charged_branch_control_steps": charged_branch_controls,
        "lifetime_accounting": dict(task.lifetime_accounting),
        "generator_choice": CHOICE_NAMES[generator_choice],
        "transport_coordinates": transport,
    }
    write_json(result_path, result)
    return result


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]):
    phases = Counter()
    for row in rows:
        phases.update(row["data_summary"]["physical_phases"])
    splits = Counter(row["request"]["split"] for row in rows)
    complete = len(rows) == len(plan["jobs"])
    checks = {
        "all_jobs_complete": complete,
        "all_episode_evidence_audited": all(row["evidence"]["audit_passed"] for row in rows),
        "all_required_phases_present": all(phases[name] > 0 for name in PHASE_NAMES),
        "all_instances_present": set(row["instance"] for row in rows) == set(config["instances"]),
        "both_modes_present": set(row["request"]["requested_mode"] for row in rows) == {0, 1},
        "both_state_generators_present": set(row["state_generator"] for row in rows) == set(config["state_generators"]),
        "all_splits_present": set(splits) == set(config["split_by_reset_block"]),
        "same_state_and_task_owned_identity": all(
            row["data_summary"]["branches"] == row["data_summary"]["checkpoints"] * len(CHOICE_NAMES)
            for row in rows
        ),
        "branch_accounting_exact": all(
            row["charged_branch_control_steps"] == row["data_summary"]["branch_control_steps"]
            for row in rows
        ),
        "old_invalid_outcomes_unused": True,
        "confirmation_states_unopened": True,
        "gpu_unused": True,
    }
    passed = all(checks.values())
    return {
        "schema_version": config["version"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "counts": {
            "planned_jobs": len(plan["jobs"]),
            "completed_jobs": len(rows),
            "checkpoints": sum(row["data_summary"]["checkpoints"] for row in rows),
            "branches": sum(row["data_summary"]["branches"] for row in rows),
            "branch_control_steps": sum(row["data_summary"]["branch_control_steps"] for row in rows),
            "actual_control_steps": sum(row["accounting"]["control_steps"] for row in rows),
            "high_force_branches": sum(row["data_summary"]["high_force_branches"] for row in rows),
            "contact_violation_branches": sum(row["data_summary"]["contact_violation_branches"] for row in rows),
            "confirmation_states_opened": 0,
        },
        "physical_phases": dict(sorted(phases.items())),
        "splits": dict(sorted(splits.items())),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "device": "cpu",
        "old_invalid_outcomes_used": False,
        "old_risk_v1_windows_reused": False,
        "performance_evidence": False,
        "failure_replacement": False,
    }


def audit(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    config = load_config(checked(plan["config"]))
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
    horizon = int(config["checkpoint_sampling"]["branch_horizon_control_steps"])
    valid = []
    summaries = []
    for row in rows:
        data = checked(row["data"])
        evidence = checked(row["evidence"])
        summary = validate_npz(data, horizon=horizon, choices=len(CHOICE_NAMES))
        summaries.append(summary)
        valid.append(summary == row["data_summary"] and audit_episode_evidence(evidence)["passed"])
    ids = [row["record_id"] for row in rows]
    expected = [job["record_id"] for job in plan["jobs"]]
    checks = {
        "unique_record_ids": len(ids) == len(set(ids)),
        "results_exactly_cover_plan": set(ids) == set(expected),
        "all_data_and_evidence_reaudited": len(valid) == len(expected) and all(valid),
        "summary_recomputed_exactly": summarize(plan, rows, config)["counts"] == report["counts"],
        "report_passed": report["passed"] is True,
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"] for item in (plan["config"], plan["runner"])
        ),
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
        "old_risk_v1_windows_unused": report["old_risk_v1_windows_reused"] is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
    }
    payload = {
        "schema_version": "t2-joint-admission-counterfactual-data-independent-audit-v2",
        "passed": all(checks.values()),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {"planned": len(expected), "results": len(rows), "jobs_reaudited": sum(valid)},
        "artifacts": {
            name: pointer(output / filename) for name, filename in {
                "plan": "plan.json", "progress": "progress.jsonl", "report": "report.json",
            }.items()
        },
    }
    write_json(output / "independent_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--probe-jobs", type=int, default=0)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("joint-admission branch data collection is CPU-only")
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    config = load_config(config_path)
    output = arguments.output or Path(config["output"])
    output = output if output.is_absolute() else ROOT / output
    if arguments.audit:
        audit(output)
    plan = build_plan(config_path, config)
    if arguments.probe_jobs:
        plan["jobs"] = plan["jobs"][:arguments.probe_jobs]
        plan["counts"]["jobs"] = len(plan["jobs"])
        plan["status"] = "probe_only_not_qualifying"
    plan_path, progress_path = output / "plan.json", output / "progress.jsonl"
    if plan_path.exists():
        if not arguments.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing collection requires an identical --resume plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    completed = {
        row["record_id"]: row for row in (
            json.loads(line) for line in progress_path.read_text().splitlines() if line
        )
    } if progress_path.is_file() else {}
    missing = [job for job in plan["jobs"] if job["record_id"] not in completed]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {
            pool.submit(execute, (job, str(output), config)): job for job in missing
        }
        for future in as_completed(futures):
            result = future.result()
            completed[result["record_id"]] = result
            append_jsonl(progress_path, result)
            write_json(output / "progress.json", {
                "completed": len(completed), "planned": len(plan["jobs"]),
                "last_record_id": result["record_id"],
            })
    rows = [completed[job["record_id"]] for job in plan["jobs"]]
    report = summarize(plan, rows, config)
    if arguments.probe_jobs:
        report["passed"] = False
        report["status"] = "probe_complete_not_qualifying"
    report["artifacts"] = {
        "plan": pointer(plan_path), "progress": pointer(progress_path),
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] or arguments.probe_jobs else 1)


if __name__ == "__main__":
    main()
