#!/usr/bin/env python3
"""Run or independently audit the frozen CPU T2 execution-radius matrix."""

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
TASK_ROOT = ROOT / "configs/arm/guided_contact_alignment"
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
DEFAULT_CONFIG = TASK_ROOT / "validation/transport_execution_radius.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


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


def checked(pointer_value: dict[str, str]) -> Path:
    path = ROOT / pointer_value["path"]
    if sha(path) != pointer_value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") != "t2-transport-execution-radius-diagnosis-v1":
        raise ValueError("unknown T2 radius diagnosis version")
    if config.get("device") != "cpu":
        raise ValueError("T2 radius diagnosis must remain CPU-only")
    if config["execution_scales"] != [0.25, 0.125, 0.0625]:
        raise ValueError("predeclared execution scales changed")
    for artifact in config["qualification_evidence"].values():
        checked(artifact)
    return config


def requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    spec = config["requests"]
    result = []
    sequence = 0
    for slot in range(int(spec["reset_slots"])):
        for mode in spec["requested_modes"]:
            result.append({
                "request_id": f"T2-RADIUS/i{slot:02d}/m{mode}",
                "physical_initial_slot": slot,
                "requested_mode": int(mode),
                "reset_seed": int(spec["reset_seed_anchor"]) + sequence,
                "robot": spec["robot"],
                "mechanism_parameters": deepcopy(spec["mechanism_parameters"]),
                "confirmation_state": False,
                "performance_evidence": False,
            })
            sequence += 1
    return result


def build_plan(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    request_rows = requests(config)
    jobs = []
    for scale in config["execution_scales"]:
        scale_id = str(scale).replace(".", "p")
        for method, relative in config["methods"].items():
            for instance in config["instances"]:
                for request in request_rows:
                    record_id = (
                        f"scale_{scale_id}/{method}/{instance}/{request['request_id']}"
                    )
                    jobs.append({
                        "record_id": record_id,
                        "scale": float(scale),
                        "method_key": method,
                        "config": str((TASK_ROOT / relative).relative_to(ROOT)),
                        "instance": instance,
                        "request": request,
                    })
    for request in request_rows:
        jobs.append({
            "record_id": f"public_geometry/fixed/{request['request_id']}",
            "scale": None,
            "method_key": "public_geometry",
            "config": str((TASK_ROOT / "baseline/public_geometry.yaml").relative_to(ROOT)),
            "instance": "fixed",
            "request": request,
        })
    return {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
        "requests": request_rows,
        "jobs": jobs,
        "counts": {
            "requests": len(request_rows),
            "jobs": len(jobs),
            "learned_jobs": len(jobs) - len(request_rows),
            "sentinel_jobs": len(request_rows),
        },
        "policy": {
            "confirmation_states_opened": 0,
            "old_invalid_outcomes_used": False,
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
        **dict(values.pop("mechanism", {}) or {}),
        **request["mechanism_parameters"],
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


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    return value


def execute(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    job, output_raw = payload
    output = Path(output_raw)
    result_path = output / "episodes" / hashlib.sha256(
        job["record_id"].encode()
    ).hexdigest() / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    import torch
    torch.set_num_threads(1)
    config = ExperimentConfig.from_yaml(ROOT / job["config"])
    task = make_task(config, job["request"])
    public = task.reset(
        (int(job["request"]["physical_initial_slot"]),),
        seed=int(job["request"]["reset_seed"]),
        initial="clear",
        modes=[int(job["request"]["requested_mode"])],
    )
    parameters = deepcopy(config.method_params)
    parameters.update(mode=int(job["request"]["requested_mode"]), device="cpu")
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family="main", key=job["method_key"])
    if job["instance"] != "fixed":
        selection["instance"] = job["instance"]
    parameters["actor_binding"] = selection
    actor = create_transfer_actor(task, config.method, parameters)
    if job["scale"] is not None:
        from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
            PublicTaskResidualTransport,
        )
        if not isinstance(actor.policy, PublicTaskResidualTransport):
            raise TypeError("radius diagnosis requires the frozen T2 transport policy")
        actor.policy.t2_execution_scale = float(job["scale"])
    for _ in range(int(config.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        diagnostic = controller_diagnostics(actor)
        action = np.asarray(actor.act(public), dtype=np.float32)
        public = task.step(action).observation
        task.record_control_diagnostics(diagnostic)
    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
    evidence = episode_evidence(task.records, task.spec, evaluation)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = result_path.with_name("evidence.json.gz")
    temporary = evidence_path.with_suffix(".gz.tmp")
    temporary.write_bytes(gzip.compress(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6, mtime=0,
    ))
    os.replace(temporary, evidence_path)
    audit = audit_episode_evidence(evidence_path)
    episode = evaluation["episodes"][-1]
    passed = bool(
        episode["success"]
        and episode["mode_match"]
        and not episode["constraint_violation"]
        and episode.get("exact_physics_contact_evidence", True)
        and audit["passed"]
    )
    result = json_ready({
        **{key: job[key] for key in (
            "record_id", "scale", "method_key", "instance", "request",
        )},
        "passed": passed,
        "success": bool(episode["success"]),
        "mode_match": bool(episode["mode_match"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "outcome": episode["outcome"],
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "physical_control_steps": int(episode["physical_control_steps"]),
        "evidence": {
            "path": str(evidence_path.relative_to(ROOT)),
            "sha256": sha(evidence_path),
            "audit_passed": bool(audit["passed"]),
        },
        "accounting": dict(task.accounting),
    })
    write_json(result_path, result)
    return result


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]):
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scale"] is not None:
            grouped[float(row["scale"]), row["method_key"]].append(row)
    cells = []
    passing_scales = []
    for scale in config["execution_scales"]:
        scale_passed = True
        for method in config["methods"]:
            selected = grouped[float(scale), method]
            passed = len(selected) == len(config["instances"]) * len(plan["requests"]) and all(
                row["passed"] for row in selected
            )
            scale_passed &= passed
            cells.append({
                "scale": scale,
                "effective_normalized_radius": .025 * float(scale),
                "method": method,
                "episodes": len(selected),
                "successes": sum(row["success"] for row in selected),
                "constraint_violations": sum(row["constraint_violation"] for row in selected),
                "outcomes": dict(sorted(Counter(row["outcome"] for row in selected).items())),
                "passed": passed,
            })
        if scale_passed:
            passing_scales.append(float(scale))
    sentinel = [row for row in rows if row["method_key"] == "public_geometry"]
    sentinel_passed = len(sentinel) == len(plan["requests"]) and all(
        row["passed"] for row in sentinel
    )
    selected = max(passing_scales) if passing_scales else None
    complete = len(rows) == len(plan["jobs"])
    return {
        "schema_version": config["version"],
        "status": "passed" if complete and sentinel_passed and selected is not None else "failed",
        "passed": bool(complete and sentinel_passed and selected is not None),
        "selected_execution_scale": selected,
        "selected_effective_normalized_radius": None if selected is None else .025 * selected,
        "cells": cells,
        "public_geometry_sentinel": {
            "episodes": len(sentinel),
            "successes": sum(row["success"] for row in sentinel),
            "passed": sentinel_passed,
        },
        "counts": {
            "planned": len(plan["jobs"]),
            "completed": len(rows),
            "physical_control_steps": sum(
                int(row["accounting"].get("control_steps", 0)) for row in rows
            ),
            "confirmation_states_opened": 0,
        },
        "old_invalid_outcomes_used": False,
        "performance_evidence": False,
        "failure_replacement": False,
    }


def audit(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    ids = [row["record_id"] for row in rows]
    planned = [row["record_id"] for row in plan["jobs"]]
    evidence = []
    for row in rows:
        path = ROOT / row["evidence"]["path"]
        evidence.append(
            path.is_file() and sha(path) == row["evidence"]["sha256"]
            and audit_episode_evidence(path)["passed"]
        )
    checks = {
        "unique_record_ids": len(ids) == len(set(ids)),
        "results_exactly_cover_plan": set(ids) == set(planned),
        "all_evidence_reaudited": len(evidence) == len(rows) and all(evidence),
        "report_passed": report.get("passed") is True,
        "public_sentinel_passed": report["public_geometry_sentinel"]["passed"] is True,
        "selection_was_predeclared_largest_passing_scale": report.get(
            "selected_execution_scale"
        ) == max(
            cell["scale"] for cell in report["cells"]
            if cell["passed"] and all(
                peer["passed"] for peer in report["cells"]
                if peer["scale"] == cell["scale"]
            )
        ),
        "old_invalid_outcomes_unused": report.get("old_invalid_outcomes_used") is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"]
            for item in (plan["config"], plan["runner"])
        ),
    }
    payload = {
        "schema_version": "t2-transport-execution-radius-independent-audit-v1",
        "passed": all(checks.values()),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {"planned": len(planned), "results": len(rows), "evidence_reaudited": sum(evidence)},
        "artifacts": {
            name: pointer(output / filename)
            for name, filename in {
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("T2 radius diagnosis is CPU-only")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.audit:
        audit(output)
    config = load_config(config_path)
    plan = build_plan(config_path, config)
    plan_path = output / "plan.json"
    progress_path = output / "progress.jsonl"
    if plan_path.exists():
        if not args.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing radius diagnosis requires an identical --resume plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    completed = {
        row["record_id"]: row
        for row in (
            json.loads(line) for line in progress_path.read_text().splitlines()
        )
    } if progress_path.is_file() else {}
    missing = [job for job in plan["jobs"] if job["record_id"] not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, (job, str(output))): job for job in missing}
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
    report["artifacts"] = {
        "plan": pointer(plan_path), "progress": pointer(progress_path),
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
