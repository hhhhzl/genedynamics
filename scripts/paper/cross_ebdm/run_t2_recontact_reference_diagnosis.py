#!/usr/bin/env python3
"""Diagnose shared T2 recontact-reference projection on accessed requests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "configs/arm/guided_contact_alignment/validation"
    / "recontact_reference_projection.json"
)
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked(value: dict[str, str]) -> Path:
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_ready(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_plan(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_plan = json.loads(checked(config["source_validation"]["plan"]).read_text())
    if "fresh_requests" in config:
        values = config["fresh_requests"]
        requests = []
        sequence = 0
        for slot in range(int(values["reset_slots"])):
            for mode in values["requested_modes"]:
                requests.append({
                    "request_id": f"T2-SHARED-EXECUTOR/i{slot:02d}/m{int(mode)}",
                    "physical_initial_slot": slot,
                    "requested_mode": int(mode),
                    "reset_seed": int(values["reset_seed_anchor"]) + sequence,
                    "robot": values["robot"],
                    "mechanism_parameters": deepcopy(values["mechanism_parameters"]),
                    "confirmation_state": False,
                    "performance_evidence": False,
                })
                sequence += 1
        source_seeds = {row["reset_seed"] for row in source_plan["requests"]}
        if source_seeds & {row["reset_seed"] for row in requests}:
            raise ValueError("fresh shared-executor requests overlap source validation")
    else:
        requests = source_plan["requests"]
        if len(requests) != 8:
            raise ValueError("accessed diagnosis requires exactly eight source requests")
    if any(request["confirmation_state"] for request in requests):
        raise ValueError("non-confirmation requests required")
    jobs = []
    for profile_key, profile_spec in config["execution_profiles"].items():
        if isinstance(profile_spec, str):
            overrides = {"execution_profile": profile_spec}
        elif isinstance(profile_spec, dict):
            overrides = dict(profile_spec.get("config_overrides", {}))
            if set(profile_spec) != {"config_overrides"}:
                raise ValueError("profile objects require only config_overrides")
        else:
            raise TypeError("execution profile must be a string or config override object")
        execution_profile = overrides.get("execution_profile")
        if not isinstance(execution_profile, str):
            raise ValueError("every diagnostic profile requires execution_profile")
        for request in requests:
            jobs.append({
                "record_id": f"{profile_key}/{request['request_id']}",
                "profile_key": profile_key,
                "execution_profile": execution_profile,
                "config_overrides": overrides,
                "request": deepcopy(request),
            })
    return {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
        "source_validation": config["source_validation"],
        "jobs": jobs,
        "counts": {"jobs": len(jobs), "requests": len(requests)},
        "policy": {
            "diagnostic_only": bool(config["diagnostic_only"]),
            "accessed_requests": bool(config["accessed_requests"]),
            "confirmation_states_opened": 0,
            "old_invalid_outcomes_used": False,
            "performance_evidence": False,
            "failure_replacement": False,
        },
    }


def make_task(experiment: ExperimentConfig, job: dict[str, Any]):
    request = job["request"]
    values = deepcopy(experiment.env_params)
    values.pop("robot", None)
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    values.pop("shared_execution", None)
    mechanism_values = {
        **dict(values.pop("mechanism", {}) or {}),
        **request["mechanism_parameters"],
    }
    values.update(job["config_overrides"])
    task = instantiate_task(
        experiment.env_name,
        robot=request["robot"],
        mechanism_spec=mechanism_spec("T2", mechanism_values),
        seed=request["reset_seed"],
        history_len=history_len,
        config_overrides=values,
        shared_execution=None,
    )
    task._experiment_mode = request["requested_mode"]
    task._provider.record_substeps = True
    return task


def executor_diagnostics(task) -> dict[str, Any]:
    values = getattr(task._provider, "last_substep_trace", {})
    if not values:
        raise ValueError("shared executor substep diagnostics were not recorded")
    phase = np.asarray(values["governor_phase"], dtype=np.int64)
    projected = np.asarray(values["reference_projected"], dtype=bool)
    return {
        "physics_substeps": int(len(phase)),
        "governor_phase_counts": np.bincount(phase, minlength=4),
        "recontact_substeps": int(np.count_nonzero(phase == 2)),
        "reference_projected_substeps": int(np.count_nonzero(projected)),
        "recontact_reference_projected_substeps": int(
            np.count_nonzero(projected & (phase == 2))
        ),
        "maximum_tangent_reference_energy_J": float(
            np.max(values["tangent_reference_energy"])
        ),
        "maximum_admitted_tangent_effort_N": float(
            np.max(values["admitted_tangent_effort"])
        ),
        "maximum_admitted_tangent_spring_N": float(
            np.max(values["admitted_tangent_spring"])
        ),
    }


def execute(payload: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    job, output_raw, sentinel_config_raw = payload
    output = Path(output_raw)
    folder = output / "episodes" / hashlib.sha256(job["record_id"].encode()).hexdigest()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    import torch
    torch.set_num_threads(1)

    experiment = ExperimentConfig.from_yaml(ROOT / sentinel_config_raw)
    task = make_task(experiment, job)
    request = job["request"]
    public = task.reset(
        (request["physical_initial_slot"],),
        seed=request["reset_seed"],
        initial="clear",
        modes=[request["requested_mode"]],
    )
    parameters = deepcopy(experiment.method_params)
    parameters.update(mode=request["requested_mode"], device="cpu")
    selection = deepcopy(parameters["actor_binding"])
    selection.update(family="main", key="public_geometry")
    parameters["actor_binding"] = selection
    actor = create_transfer_actor(task, experiment.method, parameters)
    totals = Counter()
    maxima = Counter()
    for _ in range(experiment.n_steps):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = np.asarray(actor.act(public), dtype=np.float32)
        public = task.step(action).observation
        shared = executor_diagnostics(task)
        for key in (
            "physics_substeps", "recontact_substeps",
            "reference_projected_substeps",
            "recontact_reference_projected_substeps",
        ):
            totals[key] += shared[key]
        for key in (
            "maximum_tangent_reference_energy_J",
            "maximum_admitted_tangent_effort_N",
            "maximum_admitted_tangent_spring_N",
        ):
            maxima[key] = max(maxima[key], shared[key])
        diagnostic = controller_diagnostics(actor)
        diagnostic["shared_executor"] = shared
        task.record_control_diagnostics(diagnostic)

    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
    folder.mkdir(parents=True, exist_ok=False)
    evidence_path = folder / "evidence.json.gz"
    evidence = episode_evidence(task.records, task.spec, evaluation)
    temporary = evidence_path.with_suffix(".gz.tmp")
    temporary.write_bytes(gzip.compress(
        json.dumps(json_ready(evidence), sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6,
        mtime=0,
    ))
    os.replace(temporary, evidence_path)
    audit = audit_episode_evidence(evidence_path)
    episode = evaluation["episodes"][-1]
    result = {
        **job,
        "outcome": episode["outcome"],
        "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "control_steps": task.accounting["control_steps"],
        "shared_executor_totals": dict(totals),
        "shared_executor_maxima": dict(maxima),
        "evidence": {**pointer(evidence_path), "audit_passed": bool(audit["passed"])},
        "accounting": dict(task.accounting),
    }
    write_json(folder / "result.json", result)
    return result


def frozen_source_outcomes(config: dict[str, Any]) -> dict[str, str]:
    progress_path = checked(config["source_validation"]["progress"])
    values = {}
    for line in progress_path.open():
        row = json.loads(line)
        if row["variant"] == "sentinel" and row["method_key"] == "public_geometry":
            values[row["request"]["request_id"]] = row["outcome"]
    if len(values) != 8:
        raise ValueError("source validation lacks eight public sentinel outcomes")
    return values


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]):
    cells = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["profile_key"]].append(row)
    for profile, values in sorted(grouped.items()):
        cells.append({
            "profile_key": profile,
            "execution_profile": values[0]["execution_profile"],
            "config_overrides": values[0]["config_overrides"],
            "episodes": len(values),
            "successes": sum(row["success"] for row in values),
            "constraint_violations": sum(row["constraint_violation"] for row in values),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in values).items())),
            "reference_projected_substeps": sum(
                row["shared_executor_totals"]["reference_projected_substeps"]
                for row in values
            ),
            "recontact_reference_projected_substeps": sum(
                row["shared_executor_totals"]["recontact_reference_projected_substeps"]
                for row in values
            ),
            "control_steps": sum(row["control_steps"] for row in values),
        })
    lookup = {cell["profile_key"]: cell for cell in cells}
    gates = config["gates"]
    checks = {
        "candidate_successes": (
            lookup["candidate"]["successes"] == gates["candidate_successes"]
        ),
        "candidate_constraint_violations": (
            lookup["candidate"]["constraint_violations"]
            <= gates["maximum_candidate_constraint_violations"]
        ),
        "candidate_recontact_reference_projection": (
            lookup["candidate"]["recontact_reference_projected_substeps"] > 0
        ) == gates["candidate_recontact_reference_projection_required"],
        "all_evidence_audited": all(row["evidence"]["audit_passed"] for row in rows),
        "all_cpu": all(row["accounting"]["physics_steps"] > 0 for row in rows),
        "no_confirmation_state": all(
            not row["request"]["confirmation_state"] for row in rows
        ),
    }
    if gates.get("frozen_reproduction_required", False):
        source_outcomes = frozen_source_outcomes(config)
        frozen = {
            row["request"]["request_id"]: row["outcome"]
            for row in rows if row["profile_key"] == "frozen"
        }
        checks["frozen_reproduction"] = frozen == source_outcomes
    if "fresh_requests" in config:
        source_plan = json.loads(checked(config["source_validation"]["plan"]).read_text())
        source_seeds = {row["reset_seed"] for row in source_plan["requests"]}
        checks["fresh_seed_disjoint"] = not source_seeds.intersection(
            row["request"]["reset_seed"] for row in rows
        )
    passed = all(checks.values())
    fresh = "fresh_requests" in config and not config["accessed_requests"]
    return {
        "schema_version": config["version"],
        "status": (
            "passed_fresh_qualification" if passed and fresh
            else "passed_diagnostic_retained" if passed
            else "failed_fresh_qualification" if fresh
            else "failed_diagnostic_rejected"
        ),
        "passed": passed,
        "checks": checks,
        "cells": cells,
        "counts": {
            "planned": len(plan["jobs"]),
            "completed": len(rows),
            "control_steps": sum(row["control_steps"] for row in rows),
            "physics_steps": sum(row["accounting"]["physics_steps"] for row in rows),
            "confirmation_states_opened": 0,
        },
        "diagnostic_only": bool(config["diagnostic_only"]),
        "accessed_requests": bool(config["accessed_requests"]),
        "old_invalid_outcomes_used": False,
        "performance_evidence": False,
        "failure_replacement": False,
        "next_gate": (
            "shared_protocol_rebind_and_full_predeployment"
            if passed and fresh else
            "fresh_disjoint_public_sentinel_qualification"
            if passed else "reject_shared_executor_candidate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("shared executor diagnosis is CPU-only")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("version") != "t2-recontact-reference-projection-diagnosis-v1":
        raise ValueError("unknown recontact-reference diagnosis version")
    for name in (
        "invalid_predecessor", "rejected_predecessor", "accessed_diagnostic",
        "failed_speed_qualification",
    ):
        if name in config:
            checked(config[name])
    for value in config.get("rejected_predecessors", []):
        checked(value)
    source_report = json.loads(checked(config["source_validation"]["report"]).read_text())
    source_audit = json.loads(checked(config["source_validation"]["independent_audit"]).read_text())
    if source_report.get("status") != "failed" or not source_audit.get("passed"):
        raise ValueError("faithful failed source validation and passing audit required")
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(output)
    plan = build_plan(config_path, config)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    progress_path = output / "progress.jsonl"
    completed = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                execute,
                (job, str(output), config["sentinel_config"]),
            ): job
            for job in plan["jobs"]
        }
        for future in as_completed(futures):
            row = future.result()
            completed[row["record_id"]] = row
            append_jsonl(progress_path, row)
            write_json(output / "progress.json", {
                "completed": len(completed),
                "planned": len(plan["jobs"]),
                "last_record_id": row["record_id"],
            })
    rows = [completed[job["record_id"]] for job in plan["jobs"]]
    report = summarize(plan, rows, config)
    report["artifacts"] = {
        "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
        "plan": pointer(output / "plan.json"),
        "progress": pointer(progress_path),
        "source_validation": config["source_validation"],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
