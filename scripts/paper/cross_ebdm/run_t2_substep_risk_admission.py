#!/usr/bin/env python3
"""Validate T2 substep-risk admission against unchanged paired actors."""

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
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    PublicSubstepRiskCalibration,
    PublicSubstepRiskEnsemble,
    RiskSetEmptyError,
)
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import (
    PublicCorridorGuideDenseActor,
    PublicRiskAdmittedCorridorGuideDenseActor,
)
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "configs/arm/guided_contact_alignment"
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def requests(config: dict[str, Any]):
    values = config["requests"]
    rows = []
    sequence = 0
    for slot in range(values["reset_slots"]):
        for mode in values["requested_modes"]:
            rows.append({
                "request_id": f"T2-RISK-ADMISSION/i{slot:02d}/m{mode}",
                "physical_initial_slot": slot, "requested_mode": int(mode),
                "reset_seed": int(values["reset_seed_anchor"]) + sequence,
                "robot": values["robot"],
                "mechanism_parameters": deepcopy(values["mechanism_parameters"]),
                "confirmation_state": False, "performance_evidence": False,
            })
            sequence += 1
    return rows


def build_plan(config_path: Path, config: dict[str, Any]):
    request_rows = requests(config)
    jobs = []
    for method, relative in config["methods"].items():
        for variant in config["variants"]:
            for instance in config["instances"]:
                for request in request_rows:
                    jobs.append({
                        "record_id": f"{variant}/{method}/{instance}/{request['request_id']}",
                        "method_key": method, "variant": variant, "instance": instance,
                        "config": str((TASK_ROOT / relative).relative_to(ROOT)),
                        "request": request,
                    })
    for request in request_rows:
        jobs.append({
            "record_id": f"public_geometry/fixed/{request['request_id']}",
            "method_key": "public_geometry", "variant": "sentinel", "instance": "fixed",
            "config": str((TASK_ROOT / "baseline/public_geometry.yaml").relative_to(ROOT)),
            "request": request,
        })
    return {
        "schema_version": config["version"], "status": "planned_unexecuted",
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
        "risk_qualification": config["risk_qualification"],
        "jobs": jobs, "requests": request_rows,
        "counts": {"jobs": len(jobs), "requests": len(request_rows)},
        "policy": {
            "confirmation_states_opened": 0, "old_invalid_outcomes_used": False,
            "performance_evidence": False, "failure_replacement": False,
        },
    }


def make_task(experiment: ExperimentConfig, request: dict[str, Any]):
    values = deepcopy(experiment.env_params)
    values.pop("robot", None); history_len = int(values.pop("history_len", 4))
    values.pop("mode", None); shared_execution = values.pop("shared_execution", None)
    mechanism_values = {**dict(values.pop("mechanism", {}) or {}),
                        **request["mechanism_parameters"]}
    body_support = dict(values.pop("body_support", {}) or {})
    task = instantiate_task(
        experiment.env_name, robot=request["robot"],
        mechanism_spec=mechanism_spec(
            "T2", mechanism_values, body_support_parameters=body_support,
        ),
        seed=request["reset_seed"], history_len=history_len,
        config_overrides=values or None, shared_execution=shared_execution,
    )
    task._experiment_mode = request["requested_mode"]
    return task


def json_ready(value: Any):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(child) for child in value]
    return value


def risk_actor(base_actor, instance: str, report: dict[str, Any], mode: int):
    if not isinstance(base_actor, PublicCorridorGuideDenseActor):
        raise TypeError("risk admission requires the frozen T2 dense actor")
    row = report["instances"][instance]
    model = PublicSubstepRiskEnsemble.from_checkpoint(
        __import__("torch").load(
            checked(row["checkpoint"]), map_location="cpu", weights_only=True,
        ), "cpu",
    )
    values = json.loads(checked(row["calibration_artifact"]).read_text())
    values["additive_log_correction"] = tuple(values["additive_log_correction"])
    values["soft_force_limits_N"] = tuple(values["soft_force_limits_N"])
    values["interpolation_fractions"] = tuple(values["interpolation_fractions"])
    calibration = PublicSubstepRiskCalibration(**values)
    return PublicRiskAdmittedCorridorGuideDenseActor(
        base_actor.spec, base_actor.policy, base_actor.bank, mode=mode, device="cpu",
        maximum_query_distance=base_actor.maximum_query_distance,
        risk_model=model, risk_calibration=calibration,
    )


def execute(payload):
    job, output_raw, qualification_pointer = payload
    output = Path(output_raw)
    folder = output / "episodes" / hashlib.sha256(job["record_id"].encode()).hexdigest()
    result_path = folder / "result.json"
    if result_path.is_file(): return json.loads(result_path.read_text())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if NATIVE_LIBRARY.is_file(): os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    import torch
    torch.set_num_threads(1)
    experiment = ExperimentConfig.from_yaml(ROOT / job["config"])
    task = make_task(experiment, job["request"])
    public = task.reset(
        (job["request"]["physical_initial_slot"],), seed=job["request"]["reset_seed"],
        initial="clear", modes=[job["request"]["requested_mode"]],
    )
    parameters = deepcopy(experiment.method_params)
    parameters.update(mode=job["request"]["requested_mode"], device="cpu")
    selection = deepcopy(experiment.method_params["actor_binding"])
    selection.update(family="main", key=job["method_key"])
    if job["instance"] != "fixed": selection["instance"] = job["instance"]
    parameters["actor_binding"] = selection
    actor = create_transfer_actor(task, experiment.method, parameters)
    if job["variant"] == "risk_admitted":
        qualification = json.loads(checked(qualification_pointer).read_text())
        actor = risk_actor(actor, job["instance"], qualification, job["request"]["requested_mode"])
    for _ in range(experiment.n_steps):
        if bool(np.asarray(task.state.status)[0]): break
        try:
            action = np.asarray(actor.act(public), dtype=np.float32)
        except RiskSetEmptyError:
            task.records[-1].state = "rejected"
            task.records[-1].rejection_reason = "risk_set_empty"
            break
        public = task.step(action).observation
        task.record_control_diagnostics(controller_diagnostics(actor))
    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
    folder.mkdir(parents=True, exist_ok=True)
    evidence_path = folder / "evidence.json.gz"
    evidence = episode_evidence(task.records, task.spec, evaluation)
    temporary = evidence_path.with_suffix(".gz.tmp")
    temporary.write_bytes(gzip.compress(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6, mtime=0,
    )); os.replace(temporary, evidence_path)
    audit = audit_episode_evidence(evidence_path)
    episode = evaluation["episodes"][-1]
    risk = None
    if job["variant"] == "risk_admitted":
        risk = {
            "fallback_controls": actor.risk_fallback_controls,
            "emergency_controls": actor.risk_emergency_controls,
            "empty_controls": actor.risk_empty_controls,
            "model_calls": actor.irreversible_risk_model_calls,
            "fraction_counts": actor.risk_fraction_counts.tolist(),
            "fractions": list(actor.risk_calibration.interpolation_fractions),
        }
    result = json_ready({
        **{key: job[key] for key in ("record_id", "method_key", "variant", "instance", "request")},
        "outcome": episode["outcome"], "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "control_steps": task.accounting["control_steps"], "risk": risk,
        "evidence": {**pointer(evidence_path), "audit_passed": bool(audit["passed"])},
        "accounting": dict(task.accounting),
    })
    write_json(result_path, result)
    return result


def summarize(plan, rows, config):
    cells = []
    grouped = defaultdict(list)
    for row in rows: grouped[row["variant"], row["method_key"]].append(row)
    for (variant, method), values in sorted(grouped.items()):
        counts = np.zeros(4, np.int64)
        calls = 0
        for row in values:
            if row["risk"] is not None:
                counts += np.asarray(row["risk"]["fraction_counts"], np.int64)
                calls += row["risk"]["model_calls"]
        cells.append({
            "variant": variant, "method_key": method, "episodes": len(values),
            "successes": sum(row["success"] for row in values),
            "constraint_violations": sum(row["constraint_violation"] for row in values),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in values).items())),
            "risk_model_calls": calls, "risk_fraction_counts": counts.tolist(),
        })
    lookup = {(row["variant"], row["method_key"]): row for row in cells}
    gates = config["gates"]
    checks = {
        "risk_admitted_cross_ebdm_successes": lookup[("risk_admitted", "cross_ebdm")]["successes"] == gates["risk_admitted_cross_ebdm_successes"],
        "risk_admitted_task_only_successes": lookup[("risk_admitted", "task_only")]["successes"] == gates["risk_admitted_task_only_successes"],
        "risk_admitted_constraint_violations": sum(
            row["constraint_violations"] for row in cells if row["variant"] == "risk_admitted"
        ) <= gates["maximum_risk_admitted_constraint_violations"],
        "public_geometry_sentinel_successes": lookup[("sentinel", "public_geometry")]["successes"] == gates["public_geometry_sentinel_successes"],
        "nonzero_risk_model_calls": all(
            row["risk_model_calls"] > 0 for row in cells if row["variant"] == "risk_admitted"
        ),
        "all_evidence_audited": all(row["evidence"]["audit_passed"] for row in rows),
    }
    return {
        "schema_version": config["version"], "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()), "checks": checks, "cells": cells,
        "counts": {"planned": len(plan["jobs"]), "completed": len(rows),
                   "control_steps": sum(row["control_steps"] for row in rows),
                   "confirmation_states_opened": 0},
        "old_invalid_outcomes_used": False, "performance_evidence": False,
        "failure_replacement": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("risk-admission behavior validation is CPU-only")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("version") != "t2-substep-risk-admission-validation-v1":
        raise ValueError("unknown risk-admission validation version")
    qualification = json.loads(checked(config["risk_qualification"]).read_text())
    if not qualification.get("passed"): raise ValueError("risk model qualification did not pass")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    plan = build_plan(config_path, config)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); write_json(output / "plan.json", plan)
    progress_path = output / "progress.jsonl"; completed = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(execute, (job, str(output), config["risk_qualification"])): job
            for job in plan["jobs"]
        }
        for future in as_completed(futures):
            row = future.result(); completed[row["record_id"]] = row
            append_jsonl(progress_path, row)
            write_json(output / "progress.json", {
                "completed": len(completed), "planned": len(plan["jobs"]),
                "last_record_id": row["record_id"],
            })
    rows = [completed[job["record_id"]] for job in plan["jobs"]]
    report = summarize(plan, rows, config)
    report["artifacts"] = {
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
        "plan": pointer(output / "plan.json"), "progress": pointer(progress_path),
        "risk_qualification": config["risk_qualification"],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
