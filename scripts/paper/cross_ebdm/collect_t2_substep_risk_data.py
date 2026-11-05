#!/usr/bin/env python3
"""Collect or audit T2 public histories with exact 1 ms force-peak labels."""

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
from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS
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
DEFAULT_CONFIG = TASK_ROOT / "training/substep_risk_data.json"
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"


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


def checked(value: dict[str, str]) -> Path:
    path = ROOT / value["path"]
    if sha(path) != value["sha256"]:
        raise ValueError(f"artifact changed: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") != "t2-substep-risk-data-v1":
        raise ValueError("unknown T2 substep-risk data version")
    if config.get("device") != "cpu" or config.get("execution_scales") != [0.25, 0.5, 1.0]:
        raise ValueError("substep-risk collection must use CPU and frozen scales")
    if config["split_by_physical_initial_slot"] != {
        "fit": [0, 1, 2, 3], "calibration": [4], "validation": [5],
    }:
        raise ValueError("substep-risk data split changed")
    checked(config["frozen_public_inputs"])
    for artifact in config["qualification_evidence"].values():
        checked(artifact)
    return config


def requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    spec = config["requests"]
    splits = {
        slot: split for split, slots in config["split_by_physical_initial_slot"].items()
        for slot in slots
    }
    rows = []
    sequence = 0
    for slot in range(int(spec["reset_slots"])):
        for mode in spec["requested_modes"]:
            rows.append({
                "request_id": f"T2-RISK-DATA/i{slot:02d}/m{mode}",
                "physical_initial_slot": slot,
                "requested_mode": int(mode),
                "reset_seed": int(spec["reset_seed_anchor"]) + sequence,
                "robot": spec["robot"],
                "mechanism_parameters": deepcopy(spec["mechanism_parameters"]),
                "split": splits[slot],
                "confirmation_state": False,
                "performance_evidence": False,
            })
            sequence += 1
    return rows


def build_plan(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    request_rows = requests(config)
    jobs = []
    for scale in config["execution_scales"]:
        scale_id = str(scale).replace(".", "p")
        for method, relative in config["methods"].items():
            for instance in config["instances"]:
                for request in request_rows:
                    jobs.append({
                        "record_id": f"scale_{scale_id}/{method}/{instance}/{request['request_id']}",
                        "scale": float(scale), "method_key": method,
                        "config": str((TASK_ROOT / relative).relative_to(ROOT)),
                        "instance": instance, "request": request,
                    })
    for request in request_rows:
        jobs.append({
            "record_id": f"public_geometry/fixed/{request['request_id']}",
            "scale": None, "method_key": "public_geometry",
            "config": str((TASK_ROOT / "baseline/public_geometry.yaml").relative_to(ROOT)),
            "instance": "fixed", "request": request,
        })
    return {
        "schema_version": config["version"], "status": "planned_unexecuted",
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
        "requests": request_rows, "jobs": jobs,
        "counts": {
            "requests": len(request_rows), "jobs": len(jobs),
            "learned_jobs": len(jobs) - len(request_rows),
            "sentinel_jobs": len(request_rows),
        },
        "policy": {
            "confirmation_states_opened": 0, "old_invalid_outcomes_used": False,
            "performance_evidence": False, "failure_replacement": False,
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
        config.env_name, robot=request["robot"],
        mechanism_spec=mechanism_spec("T2", mechanism_values, body_support_parameters=body_support),
        seed=int(request["reset_seed"]), history_len=history_len,
        config_overrides=values or None, shared_execution=shared_execution,
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


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def validate_npz(path: Path, horizon: int) -> dict[str, int | float]:
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        public_names = {f"public.{name}" for name in PUBLIC_FIELDS}
        required = public_names | {
            "actions", "trace.force_peak", "trace.lateral_force_peak",
        }
        if files != required:
            raise ValueError(f"unexpected risk-data schema: {path}")
        count = len(archive["actions"])
        if (archive["actions"].shape != (count, 13)
                or any(len(archive[name]) != count + 1 for name in public_names)
                or len(archive["trace.force_peak"]) != count
                or len(archive["trace.lateral_force_peak"]) != count):
            raise ValueError(f"risk-data transition lengths differ: {path}")
        total = archive["trace.force_peak"].astype(float)
        lateral = archive["trace.lateral_force_peak"].astype(float)
        starts = np.arange(max(0, count - horizon + 1))
        indices = starts[:, None] + np.arange(horizon)[None, :]
        return {
            "control_steps": count,
            "windows": len(starts),
            "maximum_total_force_N": float(total.max(initial=0.0)),
            "maximum_lateral_force_N": float(lateral.max(initial=0.0)),
            "high_risk_windows": int(np.sum(
                (total[indices].max(axis=1) >= 20.0)
                | (lateral[indices].max(axis=1) >= 8.0)
            )) if len(starts) else 0,
        }


def execute(payload: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    job, output_raw, horizon = payload
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
    from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
    torch.set_num_threads(1)
    config = ExperimentConfig.from_yaml(ROOT / job["config"])
    task = make_task(config, job["request"])
    public = task.reset(
        (int(job["request"]["physical_initial_slot"]),),
        seed=int(job["request"]["reset_seed"]), initial="clear",
        modes=[int(job["request"]["requested_mode"])],
    )
    histories = {key: [np.asarray(value[0]).copy()] for key, value in public.items()}
    parameters = deepcopy(config.method_params)
    parameters.update(mode=int(job["request"]["requested_mode"]), device="cpu")
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family="main", key=job["method_key"])
    if job["instance"] != "fixed":
        selection["instance"] = job["instance"]
    parameters["actor_binding"] = selection
    actor = create_transfer_actor(task, config.method, parameters)
    if job["scale"] is not None:
        if not isinstance(actor.policy, PublicTaskResidualTransport):
            raise TypeError("risk-data collection requires frozen T2 transport")
        actor.policy.t2_execution_scale = float(job["scale"])
    for _ in range(int(config.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = np.asarray(actor.act(public), dtype=np.float32)
        diagnostic = controller_diagnostics(actor)
        public = task.step(action).observation
        task.record_control_diagnostics(diagnostic)
        for key, value in public.items():
            histories[key].append(np.asarray(value[0]).copy())
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
    ))
    os.replace(temporary, evidence_path)
    evidence_audit = audit_episode_evidence(evidence_path)
    record = task.records[-1]
    data_path = folder / "episode.npz"
    arrays = {f"public.{key}": np.asarray(value) for key, value in histories.items()}
    arrays.update({
        "actions": np.asarray(record.requested_actions, dtype=np.float32),
        "trace.force_peak": np.asarray([row["force_peak"] for row in record.traces], dtype=np.float32),
        "trace.lateral_force_peak": np.asarray([row["lateral_force_peak"] for row in record.traces], dtype=np.float32),
    })
    save_npz(data_path, arrays)
    data_summary = validate_npz(data_path, horizon)
    episode = evaluation["episodes"][-1]
    result = json_ready({
        **{key: job[key] for key in ("record_id", "scale", "method_key", "instance", "request")},
        "outcome": episode["outcome"], "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "data_summary": data_summary,
        "data": pointer(data_path),
        "evidence": {**pointer(evidence_path), "audit_passed": bool(evidence_audit["passed"])},
        "accounting": dict(task.accounting),
    })
    write_json(result_path, result)
    return result


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]):
    learned = [row for row in rows if row["method_key"] != "public_geometry"]
    sentinel = [row for row in rows if row["method_key"] == "public_geometry"]
    fit_by_instance = defaultdict(int)
    for row in learned:
        if row["request"]["split"] == "fit":
            fit_by_instance[row["instance"]] += row["data_summary"]["high_risk_windows"]
    minimum = int(config["risk_window"]["minimum_fit_high_risk_windows_per_instance"])
    coverage = all(fit_by_instance[instance] >= minimum for instance in config["instances"])
    sentinel_passed = len(sentinel) == len(plan["requests"]) and all(row["success"] for row in sentinel)
    complete = len(rows) == len(plan["jobs"])
    passed = bool(complete and sentinel_passed and coverage)
    return {
        "schema_version": config["version"], "status": "passed" if passed else "failed",
        "passed": passed, "risk_tail_coverage_passed": coverage,
        "fit_high_risk_windows_per_instance": dict(sorted(fit_by_instance.items())),
        "required_fit_high_risk_windows_per_instance": minimum,
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["request"]["split"] for row in rows).items())),
        "public_geometry_sentinel": {
            "episodes": len(sentinel), "successes": sum(row["success"] for row in sentinel),
            "passed": sentinel_passed,
        },
        "counts": {
            "planned": len(plan["jobs"]), "completed": len(rows),
            "control_steps": sum(row["data_summary"]["control_steps"] for row in rows),
            "windows": sum(row["data_summary"]["windows"] for row in rows),
            "high_risk_windows": sum(row["data_summary"]["high_risk_windows"] for row in rows),
            "confirmation_states_opened": 0,
        },
        "old_invalid_outcomes_used": False, "performance_evidence": False,
        "failure_replacement": False,
    }


def audit(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
    horizon = 10
    valid = []
    for row in rows:
        data = ROOT / row["data"]["path"]
        evidence = ROOT / row["evidence"]["path"]
        valid.append(
            sha(data) == row["data"]["sha256"]
            and validate_npz(data, horizon) == row["data_summary"]
            and sha(evidence) == row["evidence"]["sha256"]
            and audit_episode_evidence(evidence)["passed"]
        )
    ids = [row["record_id"] for row in rows]
    planned = [row["record_id"] for row in plan["jobs"]]
    checks = {
        "unique_record_ids": len(ids) == len(set(ids)),
        "results_exactly_cover_plan": set(ids) == set(planned),
        "all_npz_and_evidence_reaudited": len(valid) == len(rows) and all(valid),
        "report_passed": report["passed"] is True,
        "public_sentinel_passed": report["public_geometry_sentinel"]["passed"] is True,
        "risk_tail_coverage_passed": report["risk_tail_coverage_passed"] is True,
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"] for item in (plan["config"], plan["runner"])
        ),
    }
    payload = {
        "schema_version": "t2-substep-risk-data-independent-audit-v1",
        "passed": all(checks.values()), "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {"planned": len(planned), "results": len(rows), "episodes_reaudited": sum(valid)},
        "artifacts": {
            name: pointer(output / filename)
            for name, filename in {"plan": "plan.json", "progress": "progress.jsonl", "report": "report.json"}.items()
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
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("T2 substep-risk data collection is CPU-only")
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    if arguments.audit:
        audit(output)
    config = load_config(config_path)
    plan = build_plan(config_path, config)
    plan_path, progress_path = output / "plan.json", output / "progress.jsonl"
    if plan_path.exists():
        if not arguments.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing risk-data collection requires an identical --resume plan")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    completed = {
        row["record_id"]: row
        for row in (json.loads(line) for line in progress_path.read_text().splitlines())
    } if progress_path.is_file() else {}
    missing = [job for job in plan["jobs"] if job["record_id"] not in completed]
    horizon = int(config["risk_window"]["horizon_control_steps"])
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {pool.submit(execute, (job, str(output), horizon)): job for job in missing}
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
        "frozen_public_inputs": config["frozen_public_inputs"],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
