#!/usr/bin/env python3
"""Measure whether frozen T2 response models foresee 1 ms overforce impulses."""

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
from genedynamics.tasks.transfer.controllers.bindings import (
    _cross_components,
    resolve_transfer_binding,
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
DEFAULT_CONFIG = TASK_ROOT / "validation/response_risk_observability.json"
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
    if config.get("version") != "t2-response-risk-observability-diagnosis-v1":
        raise ValueError("unknown T2 response-risk diagnosis version")
    if config.get("device") != "cpu" or config.get("execution_scale") != 0.25:
        raise ValueError("response-risk diagnosis must use CPU and frozen 25% radius")
    gate = config["predeclared_gate"]
    if gate != {
        "ensemble_reduction": "maximum",
        "horizon_control_steps": 10,
        "candidate_total_force_limit_N": 20.0,
        "candidate_lateral_force_limit_N": 8.0,
        "candidate_minus_base_total_limit_N": 2.0,
        "candidate_minus_base_lateral_limit_N": 1.0,
    }:
        raise ValueError("predeclared response gate changed")
    for artifact in config["qualification_evidence"].values():
        checked(artifact)
    return config


def build_plan(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    jobs = []
    for method, relative in config["methods"].items():
        for instance in config["instances"]:
            for request in config["requests"]:
                jobs.append({
                    "record_id": f"{method}/{instance}/{request['request_id']}",
                    "method_key": method,
                    "config": str((TASK_ROOT / relative).relative_to(ROOT)),
                    "instance": instance,
                    "request": deepcopy(request),
                })
    return {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "config": pointer(config_path),
        "runner": pointer(Path(__file__).resolve()),
        "jobs": jobs,
        "counts": {"jobs": len(jobs)},
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


def gate_accepts(candidate: np.ndarray, base: np.ndarray, gate: dict[str, Any]) -> bool:
    return bool(
        candidate[0] <= gate["candidate_total_force_limit_N"]
        and candidate[1] <= gate["candidate_lateral_force_limit_N"]
        and candidate[0] - base[0]
        <= gate["candidate_minus_base_total_limit_N"]
        and candidate[1] - base[1]
        <= gate["candidate_minus_base_lateral_limit_N"]
    )


def execute(payload: tuple[dict[str, Any], str, dict[str, Any], float]) -> dict[str, Any]:
    job, output_raw, gate, execution_scale = payload
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
    from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
    from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
        PublicTaskResidualTransport,
        T2_TASK_OWNED_COORDINATES,
    )
    torch.set_num_threads(1)
    config = ExperimentConfig.from_yaml(ROOT / job["config"])
    task = make_task(config, job["request"])
    public = task.reset(
        (int(job["request"]["physical_initial_slot"]),),
        seed=int(job["request"]["reset_seed"]), initial="clear",
        modes=[int(job["request"]["requested_mode"])],
    )
    parameters = deepcopy(config.method_params)
    parameters.update(mode=int(job["request"]["requested_mode"]), device="cpu")
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family="main", key=job["method_key"], instance=job["instance"])
    parameters["actor_binding"] = selection
    binding = resolve_transfer_binding(parameters, task_alias="T2")
    policy, response, bank = _cross_components(task, binding, "cpu")
    if response is None or not isinstance(policy, PublicTaskResidualTransport):
        raise TypeError("response-risk diagnosis requires paired T2 policy/response")
    policy.t2_execution_scale = float(execution_scale)
    from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import (
        PublicCorridorGuideDenseActor,
    )
    actor = PublicCorridorGuideDenseActor(
        task.spec, policy, bank, mode=int(job["request"]["requested_mode"]),
        device="cpu", maximum_query_distance=float(binding.get("maximum_query_distance", 5.0)),
    )
    response.eval()
    for parameter in response.parameters():
        parameter.requires_grad_(False)
    predictions = []
    for _ in range(int(config.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        batch, base = actor.prepare(public)
        with torch.inference_mode():
            arguments = [
                torch.as_tensor(batch[key], dtype=torch.float32)
                for key in ("target_features", "source_features", "source_commands", "base_proto")
            ]
            phase = torch.as_tensor(batch["phase"])
            context = torch.as_tensor(batch["own_context"], dtype=torch.float32)
            candidate = policy(*arguments, phase, context)
            predicted_candidate = torch.expm1(
                response(arguments[0], candidate)[..., 18:20]
            ).clamp(min=0)
            predicted_base = torch.expm1(
                response(arguments[0], arguments[3])[..., 18:20]
            ).clamp(min=0)
        candidate_peak = predicted_candidate[:, 0].amax(dim=0).cpu().numpy()
        base_peak = predicted_base[:, 0].amax(dim=0).cpu().numpy()
        if torch.equal(candidate[:, 0], arguments[3][:, 0]):
            action = np.asarray(base, dtype=np.float32)
        else:
            action = (candidate[:, 0] * candidate.new_tensor(BOUNDS)).cpu().numpy().astype(np.float32)
            action[:, list(T2_TASK_OWNED_COORDINATES)] = base[:, list(T2_TASK_OWNED_COORDINATES)]
        actor.irreversible_network_calls += 1
        tick = int(np.rint(np.asarray(public["elapsed_seconds"]).item() / task.spec.config.control_dt))
        actor.cached_action = action.copy()
        actor.last_action_tick = tick
        accepted = gate_accepts(candidate_peak, base_peak, gate)
        diagnostic = controller_diagnostics(actor)
        diagnostic["response_risk"] = {
            "candidate_peak_N": candidate_peak.copy(),
            "base_peak_N": base_peak.copy(),
            "gate_accepted": accepted,
            "prediction_semantics": "control_rate_public_wrench_peak",
        }
        predictions.append({
            "candidate_peak_N": candidate_peak.tolist(),
            "base_peak_N": base_peak.tolist(),
            "gate_accepted": accepted,
        })
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
    evidence_audit = audit_episode_evidence(evidence_path)
    episode = evaluation["episodes"][-1]
    traces = task.records[-1].traces
    horizon = int(gate["horizon_control_steps"])
    actual_horizon = []
    for index in range(len(traces)):
        window = traces[index:min(len(traces), index + horizon)]
        actual_horizon.append([
            max(float(row["force_peak"]) for row in window),
            max(float(row["lateral_force_peak"]) for row in window),
        ])
    violation_ticks = [
        index for index, row in enumerate(traces)
        if float(row["force_peak"]) > task.spec.config.force_limit
        or float(row["lateral_force_peak"]) > task.spec.config.lateral_force_limit
    ]
    first_violation = violation_ticks[0] if violation_ticks else None
    preceding = [] if first_violation is None else predictions[
        max(0, first_violation - horizon + 1):first_violation + 1
    ]
    rejected_before = bool(preceding and any(not row["gate_accepted"] for row in preceding))
    candidate_array = np.asarray([row["candidate_peak_N"] for row in predictions])
    actual_array = np.asarray(actual_horizon)
    result = json_ready({
        **{key: job[key] for key in ("record_id", "method_key", "instance", "request")},
        "outcome": episode["outcome"],
        "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "first_violation_control_tick": first_violation,
        "gate_rejected_before_violation": rejected_before,
        "gate_rejected_controls": sum(not row["gate_accepted"] for row in predictions),
        "control_steps": len(predictions),
        "candidate_prediction_max_N": candidate_array.max(axis=0),
        "actual_horizon_peak_max_N": actual_array.max(axis=0),
        "candidate_prediction_absolute_error_N": np.abs(candidate_array - actual_array).mean(axis=0),
        "evidence": {
            "path": str(evidence_path.relative_to(ROOT)),
            "sha256": sha(evidence_path),
            "audit_passed": bool(evidence_audit["passed"]),
        },
        "accounting": dict(task.accounting),
    })
    write_json(result_path, result)
    return result


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in rows if row["constraint_violation"]]
    successful = [row for row in rows if row["success"]]
    detected = sum(row["gate_rejected_before_violation"] for row in failed)
    success_with_rejection = sum(row["gate_rejected_controls"] > 0 for row in successful)
    sufficient = bool(failed and detected == len(failed) and success_with_rejection == 0)
    return {
        "schema_version": "t2-response-risk-observability-diagnosis-v1",
        "status": "passed" if sufficient else "failed",
        "passed": sufficient,
        "existing_response_sufficient_for_online_admission": sufficient,
        "counts": {
            "planned": len(plan["jobs"]), "completed": len(rows),
            "failures": len(failed), "failures_detected_before_violation": detected,
            "successful_episodes": len(successful),
            "successful_episodes_with_gate_rejection": success_with_rejection,
            "physical_control_steps": sum(row["accounting"]["control_steps"] for row in rows),
            "confirmation_states_opened": 0,
        },
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "per_episode": rows,
        "old_invalid_outcomes_used": False,
        "performance_evidence": False,
        "failure_replacement": False,
        "interpretation": (
            "The frozen response predicts 50 Hz public wrench peaks. Failure here means it cannot "
            "serve as a 1 ms impulse admission model without new substep-peak targets and calibration."
        ),
    }


def audit(output: Path) -> None:
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
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
        "reported_status_is_faithful": report["status"] in {"passed", "failed"},
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"] for item in (plan["config"], plan["runner"])
        ),
    }
    payload = {
        "schema_version": "t2-response-risk-observability-independent-audit-v1",
        "passed": all(checks.values()),
        "status": "passed" if all(checks.values()) else "failed",
        "audited_result_status": report["status"],
        "checks": checks,
        "counts": {"planned": len(planned), "results": len(rows), "evidence_reaudited": sum(evidence)},
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
        raise RuntimeError("T2 response-risk diagnosis is CPU-only")
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    if arguments.audit:
        audit(output)
    config = load_config(config_path)
    plan = build_plan(config_path, config)
    plan_path = output / "plan.json"
    progress_path = output / "progress.jsonl"
    if plan_path.exists():
        if not arguments.resume or json.loads(plan_path.read_text()) != plan:
            raise ValueError("existing response-risk diagnosis requires an identical --resume plan")
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
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {
            pool.submit(execute, (
                job, str(output), config["predeclared_gate"], config["execution_scale"],
            )): job for job in missing
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
    report = summarize(plan, rows)
    report["artifacts"] = {
        "plan": pointer(plan_path), "progress": pointer(progress_path),
        "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
