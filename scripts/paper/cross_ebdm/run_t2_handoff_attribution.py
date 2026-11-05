#!/usr/bin/env python3
"""Fork T2 from one exact public handoff state to attribute late failures."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records

from scripts.paper.cross_ebdm.qualify_method_behavior import (
    _runtime_device_proof,
    _task_from_job,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT / "configs/arm/guided_contact_alignment/validation"
    / "handoff_state_attribution.json"
)
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(child) for child in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach"):
            update(item.detach().cpu().numpy())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"[")
            for child in item:
                update(child)
            digest.update(b"]")
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"A" + str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(item, np.generic):
            # Controller restore methods deliberately normalize NumPy scalar
            # wrappers to their equivalent Python scalars.  Their values are
            # behavioral state; the wrapper class is not.
            update(item.item())
        else:
            digest.update(
                json.dumps(item, sort_keys=True, default=str).encode() + b";"
            )

    update(value)
    return digest.hexdigest()


def restorable_task_digest(checkpoint: dict[str, Any]) -> str:
    """Hash the exact rollback state while excluding monotonic resource usage.

    ``restore_checkpoint`` intentionally does not rewind ``lifetime_accounting``:
    a branch is a new charged physical execution.  Every other checkpoint field
    is part of the exact same-state fork and must round-trip byte-for-byte.
    """
    value = deepcopy(checkpoint)
    if "lifetime_accounting_at_export" not in value:
        raise ValueError("task checkpoint lacks lifetime accounting receipt")
    value.pop("lifetime_accounting_at_export")
    return stable_digest(value)


def source_inputs(config: dict[str, Any]) -> tuple[list[dict], dict[str, dict]]:
    plan_path = checked(config["source_validation"]["plan"], "source plan")
    report_path = checked(config["source_validation"]["report"], "source report")
    rejection_path = checked(
        config["source_validation"]["rejection"], "source rejection",
    )
    report = json.loads(report_path.read_text())
    rejection = json.loads(rejection_path.read_text())
    if (
        report.get("status") != "complete_recorded_development_diagnosis"
        or rejection.get("status") != "rejected_after_complete_exact_request_gate"
        or rejection.get("diagnostic_only") is not True
        or rejection.get("performance_claim") is not False
    ):
        raise ValueError("source v17 run is not the frozen rejected diagnostic")
    progress_path = checked(report["artifacts"]["progress"], "source progress")
    source_results = {
        row["record_id"]: row
        for row in map(json.loads, progress_path.read_text().splitlines())
    }
    jobs = {row["record_id"]: row for row in json.loads(plan_path.read_text())["jobs"]}
    selected = list(config["selected_record_ids"])
    if len(selected) != 6 or len(set(selected)) != len(selected):
        raise ValueError("T2 handoff attribution requires six unique paired records")
    if set(selected) - set(jobs) or set(selected) - set(source_results):
        raise ValueError("selected T2 record is absent from the frozen source run")
    outcomes = Counter(source_results[key]["outcome"] for key in selected)
    if outcomes != Counter({"success": 3, "timeout": 3}):
        raise ValueError(f"selected source outcome contract changed: {outcomes}")
    return [deepcopy(jobs[key]) for key in selected], source_results


def make_actor(job: dict[str, Any], task: Any):
    experiment = ExperimentConfig.from_yaml(ROOT / job["config"])
    parameters = deepcopy(experiment.method_params)
    parameters.update(
        mode=int(job["request"]["requested_mode"]),
        device=str(job.get("device", "cuda")),
        actor_binding=deepcopy(job["actor_binding"]),
    )
    if job.get("task_actor_interface") is not None:
        parameters["task_actor_interface"] = job["task_actor_interface"]
        parameters["task_actor_interface_parameters"] = deepcopy(
            job.get("task_actor_interface_parameters", {})
        )
    return experiment, create_transfer_actor(task, experiment.method, parameters)


def reset_task(job: dict[str, Any]):
    experiment = ExperimentConfig.from_yaml(ROOT / job["config"])
    task = _task_from_job(experiment, job["request"], job.get("task_overrides"))
    request = job["request"]
    public = task.reset(
        (int(request["physical_initial_slot"]),),
        seed=int(request["reset_seed"]), initial="clear",
        modes=[int(request["requested_mode"])],
    )
    return task, public


def step_task(task, action, diagnostic):
    public = task.step(np.asarray(action, dtype=np.float32)).observation
    task.record_control_diagnostics(diagnostic)
    return public


def handoff_summary(public, actor, task, envelope: dict[str, Any]) -> dict[str, Any]:
    position = np.asarray(public["tcp_position"], dtype=float)[0]
    reference = np.asarray(public["reference_position"], dtype=float)[0]
    twist = np.asarray(public["tcp_twist"], dtype=float)[0]
    wrench = np.asarray(public["measured_wrench"], dtype=float)[0]
    elapsed = float(np.asarray(public["elapsed_seconds"]).reshape(-1)[0])
    remaining = max(
        0, int(actor.spec.config.timeout_steps)
        - int(np.rint(elapsed / actor.spec.config.control_dt)),
    )
    lag = float(np.linalg.norm(reference - position))
    planar_speed = float(np.linalg.norm(twist[:2]))
    force_utilization = float(max(
        np.linalg.norm(wrench[:3]) / actor.spec.config.force_limit,
        np.linalg.norm(wrench[:2]) / actor.spec.config.lateral_force_limit,
    ))
    finite = bool(all(np.isfinite(value).all() for value in (
        position, reference, twist, wrench,
    )))
    pre_violation = bool(evaluate_records(task.records, actor.spec)[
        "episodes"
    ][-1]["constraint_violation"])
    checks = {
        "finite_public_state": finite,
        "remaining_controls": remaining >= int(envelope["minimum_remaining_controls"]),
        "reference_lag": lag <= float(envelope["maximum_reference_lag_m"]),
        "planar_tcp_speed": planar_speed <= float(
            envelope["maximum_planar_tcp_speed_mps"]
        ),
        "force_utilization": force_utilization <= float(
            envelope["maximum_force_utilization"]
        ),
        "no_pre_handoff_constraint_violation": not pre_violation,
    }
    return {
        "elapsed_seconds": elapsed,
        "remaining_controls": remaining,
        "reference_lag_m": lag,
        "planar_tcp_speed_mps": planar_speed,
        "force_utilization": force_utilization,
        "common_completion_stage": int(actor.common.completion.stage),
        "candidate_authority_fraction": float(actor.candidate_authority_fraction),
        "checks": checks,
        "inside_admissible_envelope": all(checks.values()),
    }

def capture_handoff(job: dict[str, Any], envelope: dict[str, Any]):
    task, public = reset_task(job)
    experiment, actor = make_actor(job, task)
    for _ in range(int(experiment.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = np.asarray(actor.act(public), dtype=np.float32)
        diagnostic = controller_diagnostics(actor)
        if bool(actor.common.switched):
            if int(job["request"]["requested_mode"]) != 1:
                raise ValueError("handoff attribution is mode-1 only")
            task_state = task.get_checkpoint()
            actor_state = actor.get_state()
            return {
                "task_state": task_state,
                "actor_state": actor_state,
                "task_checkpoint_sha256": stable_digest(task_state),
                "restorable_task_state_sha256": restorable_task_digest(task_state),
                "actor_state_sha256": stable_digest(actor_state),
                "public_sha256": stable_digest(public),
                "public": deepcopy(public),
                "normal_first_action": action.copy(),
                "common_first_action": np.asarray(
                    actor.last_common_action, dtype=np.float32,
                ).copy(),
                "normal_first_diagnostic": diagnostic,
                "prefix_control_steps": len(task.records[-1].traces),
                "handoff": handoff_summary(public, actor, task, envelope),
            }
        public = step_task(task, action, diagnostic)
    raise RuntimeError("source actor never reached the public completion handoff")


def continue_branch(job, captured, branch: str, folder: Path) -> dict[str, Any]:
    task, _ = reset_task(job)
    restored_public = task.restore_checkpoint(deepcopy(captured["task_state"]))
    experiment, actor = make_actor(job, task)
    actor.restore(deepcopy(captured["actor_state"]))
    if (
        restorable_task_digest(task.get_checkpoint())
        != captured["restorable_task_state_sha256"]
    ):
        raise ValueError("restored physical checkpoint differs")
    if stable_digest(restored_public) != captured["public_sha256"]:
        raise ValueError("restored public observation differs")
    if stable_digest(actor.get_state()) != captured["actor_state_sha256"]:
        raise ValueError("restored actor checkpoint differs")

    if branch == "normal_v17":
        action = captured["normal_first_action"]
        diagnostic = captured["normal_first_diagnostic"]
    elif branch == "exact_common":
        action = captured["common_first_action"]
        diagnostic = controller_diagnostics(actor.common)
        diagnostic["attribution_intervention"] = 1
    else:
        raise ValueError("unknown handoff continuation branch")
    public = step_task(task, action, diagnostic)
    remaining = int(experiment.n_steps) - captured["prefix_control_steps"] - 1
    for _ in range(max(0, remaining)):
        if bool(np.asarray(task.state.status)[0]):
            break
        controller = actor if branch == "normal_v17" else actor.common
        action = np.asarray(controller.act(public), dtype=np.float32)
        diagnostic = controller_diagnostics(controller)
        if branch == "exact_common":
            diagnostic["attribution_intervention"] = 1
        public = step_task(task, action, diagnostic)

    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evidence_path = folder / "evidence.json.gz"
    evidence_path.parent.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(
        json_ready(episode_evidence(task.records, task.spec, evaluation)),
        sort_keys=True, separators=(",", ":"),
    ).encode()
    temporary = evidence_path.with_suffix(".gz.tmp")
    temporary.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    os.replace(temporary, evidence_path)
    audit = audit_episode_evidence(evidence_path)
    episode = evaluation["episodes"][-1]
    return {
        "branch": branch,
        "success": bool(episode["success"]),
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "mode_match": bool(episode["mode_match"]),
        "physical_control_steps": int(episode["physical_control_steps"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "final_position_error_m": episode.get("final_position_error_m"),
        "evidence": {**pointer(evidence_path), "audit_passed": bool(audit["passed"])},
        "accounting": dict(task.accounting),
    }


def classify(source_outcome: str, normal: dict, common: dict, inside: bool) -> str:
    if (
        normal["outcome"] != source_outcome
        or not normal["evidence"]["audit_passed"]
        or not common["evidence"]["audit_passed"]
    ):
        return "inconclusive"
    if source_outcome == "success":
        return "successful_control_preserved" if common["success"] else "inconclusive"
    if common["success"]:
        return "method_or_interface_failure"
    if inside:
        return "shared_completion_defect_supported"
    return "inadmissible_method_handoff"


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    if (
        config.get("version") != "t2-handoff-state-attribution-v1"
        or config.get("diagnostic_only") is not True
        or config.get("performance_evidence") is not False
        or config.get("confirmation_states_opened") != 0
        or config.get("failure_replacement") is not False
        or config.get("execution") != {"device": "cuda"}
        or config.get("limits", {}).get("workers") != 1
    ):
        raise ValueError("T2 handoff attribution contract changed")
    os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    device = _runtime_device_proof({"execution": config["execution"]})
    jobs, source_results = source_inputs(config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "jobs": [{
            "record_id": job["record_id"],
            "source_outcome": source_results[job["record_id"]]["outcome"],
            "request": job["request"],
            "branches": ["normal_v17", "exact_common"],
        } for job in jobs],
        "handoff": config["handoff"],
        "admissible_envelope": config["admissible_envelope"],
        "decision_rule": config["decision_rule"],
        "device": device,
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "source_validation": config["source_validation"],
        },
        "policy": {
            "diagnostic_only": True,
            "performance_evidence": False,
            "confirmation_states_opened": 0,
            "failure_replacement": False,
            "same_state_fork": True,
        },
    }
    write_json(output / "plan.json", plan)
    begun = time.monotonic()
    results = []
    for job in jobs:
        if time.monotonic() - begun > float(config["limits"]["maximum_wall_seconds"]):
            raise TimeoutError("T2 handoff attribution wall ceiling")
        captured = capture_handoff(job, config["admissible_envelope"])
        key = hashlib.sha256(job["record_id"].encode()).hexdigest()
        normal = continue_branch(job, captured, "normal_v17", output / "episodes" / key / "normal_v17")
        common = continue_branch(job, captured, "exact_common", output / "episodes" / key / "exact_common")
        source_outcome = source_results[job["record_id"]]["outcome"]
        row = {
            "record_id": job["record_id"],
            "source_outcome": source_outcome,
            "prefix_control_steps": captured["prefix_control_steps"],
            "task_checkpoint_sha256": captured["task_checkpoint_sha256"],
            "restorable_task_state_sha256": captured[
                "restorable_task_state_sha256"
            ],
            "actor_state_sha256": captured["actor_state_sha256"],
            "public_sha256": captured["public_sha256"],
            "handoff": captured["handoff"],
            "normal_v17": normal,
            "exact_common": common,
        }
        row["classification"] = classify(
            source_outcome, normal, common,
            captured["handoff"]["inside_admissible_envelope"],
        )
        results.append(row)
        write_json(output / "progress.json", results)

    counts = Counter(row["classification"] for row in results)
    controls_ok = counts["successful_control_preserved"] == 3
    failures_closed = sum(
        counts[name] for name in (
            "method_or_interface_failure", "shared_completion_defect_supported",
            "inadmissible_method_handoff",
        )
    ) == 3
    inconclusive = counts["inconclusive"] > 0 or not controls_ok or not failures_closed
    continue_shared = counts["shared_completion_defect_supported"] > 0
    report = {
        "schema_version": config["version"],
        "status": "complete" if not inconclusive else "inconclusive",
        "passed": not inconclusive,
        "diagnostic_only": True,
        "performance_evidence": False,
        "confirmation_states_opened": 0,
        "failure_replacement": False,
        "device": device,
        "counts": {
            "selected_source_records": len(results),
            "physical_continuations": 2 * len(results),
            "source_failures": 3,
            "source_success_controls": 3,
            "classifications": dict(counts),
        },
        "decision": {
            "continue_shared_controller_repair": continue_shared,
            "selected_failures_closed": bool(failures_closed and not inconclusive),
            "scope": (
                "Only the three frozen v17 timeouts and their three paired successful "
                "controls; this does not certify every possible method handoff state."
            ),
        },
        "results": results,
        "artifacts": {
            "plan": pointer(output / "plan.json"),
            "progress": pointer(output / "progress.json"),
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
        },
        "wall_seconds": time.monotonic() - begun,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "decision": report["decision"],
        "counts": report["counts"],
        "report": pointer(output / "report.json"),
    }, indent=2))
    if inconclusive:
        raise AssertionError("T2 same-state handoff attribution is inconclusive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    run(config_path)


if __name__ == "__main__":
    main()
