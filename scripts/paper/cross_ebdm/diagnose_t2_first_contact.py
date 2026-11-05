#!/usr/bin/env python3
"""Replay the complementary T2 first-contact failures with public traces.

This is development diagnosis over already-accessed confirmation requests.  It
does not create confirmation evidence and its outputs must not be reused as a
qualification or final-confirmation sample.
"""

from __future__ import annotations

import argparse
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
ROSTER = (
    ROOT
    / "results/_protocol/four_task_transfer/confirmation_v13/package/request_roster.jsonl"
)
OUTCOMES = (
    ROOT
    / "results/_protocol/four_task_transfer/confirmation_v13/run/outcomes.jsonl"
)
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
DEVELOPMENT_BASE_MANIFEST = (
    TASK_ROOT / "protocol/actor_bindings_transport_only_v1.json"
)
REQUEST_IDS = (
    "CONFIRMATION-v13/T2/latency_gain_shift/xarm7_fixed_base/i00/m0",
    "CONFIRMATION-v13/T2/latency_gain_shift/xarm7_fixed_base/i06/m0",
    "CONFIRMATION-v13/T2/latency_gain_shift/xarm7_fixed_base/i00/m1",
)
METHODS = {
    "cross_ebdm": {
        "config": "main/cross_ebdm.yaml", "binding_key": "cross_ebdm",
    },
    "cross_ebdm_mode0_contact_established": {
        "config": "main/cross_ebdm.yaml", "binding_key": "cross_ebdm",
        "mode_conditioned_contact_establishment_admission": True,
    },
    "task_only": {
        "config": "baseline/task_only.yaml", "binding_key": "task_only",
    },
}
INSTANCE = "instance_08"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def frozen_requests() -> list[dict[str, Any]]:
    rows = {
        row["request_id"]: row
        for row in read_jsonl(ROSTER)
        if row.get("request_id") in REQUEST_IDS
    }
    if set(rows) != set(REQUEST_IDS):
        raise ValueError("v13 complementary first-contact requests are missing")
    for row in rows.values():
        if row.get("task") != "T2" or not row.get("confirmation_accessed") is False:
            raise ValueError("unexpected v13 request roster row")
    return [rows[request_id] for request_id in REQUEST_IDS]


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
    task = instantiate_task(
        config.env_name,
        robot=request["body_support_parameters"]["robot"],
        mechanism_spec=mechanism_spec(
            "T2",
            mechanism_values,
            body_support_parameters=request["body_support_parameters"],
        ),
        seed=int(request["reset_seed"]),
        history_len=history_len,
        config_overrides=values or None,
        shared_execution=shared_execution,
    )
    task._experiment_mode = int(request["requested_mode"])
    return task


def compact_public(public: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "elapsed_seconds",
        "tcp_position",
        "tcp_rotation",
        "tcp_twist",
        "reference_position",
        "reference_rotation",
        "measured_wrench",
        "previous_action",
    )
    return {
        key: np.asarray(public[key]).copy()
        for key in keys
        if key in public
    }


def execute(
    method: str,
    method_spec: dict[str, Any],
    request: dict[str, Any],
    *,
    output: Path,
    device: str,
) -> dict[str, Any]:
    record_id = f"{method}/{INSTANCE}/{request['request_id']}"
    episode_dir = output / "episodes" / hashlib.sha256(record_id.encode()).hexdigest()
    result_path = episode_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())

    config = ExperimentConfig.from_yaml(TASK_ROOT / method_spec["config"])
    task = make_task(config, request)
    public = task.reset(
        (int(request["physical_initial_slot"]),),
        seed=int(request["reset_seed"]),
        initial="clear",
        modes=[int(request["requested_mode"])],
    )
    parameters = deepcopy(config.method_params)
    parameters.update(mode=int(request["requested_mode"]), device=device)
    selection = deepcopy(config.method_params["actor_binding"])
    selection["manifest"] = pointer(DEVELOPMENT_BASE_MANIFEST)
    selection.update(
        family="main", key=method_spec["binding_key"], instance=INSTANCE,
    )
    parameters["actor_binding"] = selection
    actor = create_transfer_actor(task, config.method, parameters)
    if method_spec.get("mode_conditioned_contact_establishment_admission"):
        from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import (
            PublicModeConditionedContactEstablishedCorridorGuideDenseActor,
        )

        original = actor.candidate
        actor.candidate = PublicModeConditionedContactEstablishedCorridorGuideDenseActor(
            task.spec,
            original.policy,
            original.bank,
            mode=int(request["requested_mode"]),
            device=device,
            maximum_query_distance=float(original.maximum_query_distance),
        )

    action_digest = hashlib.sha256()
    public_rows: list[dict[str, Any]] = []
    for _ in range(int(config.n_steps)):
        if bool(np.asarray(task.state.status)[0]):
            break
        before = compact_public(public)
        action = np.asarray(actor.act(public), dtype=np.float32)
        action_digest.update(np.ascontiguousarray(action).tobytes())
        diagnostic = controller_diagnostics(actor)
        public = task.step(action).observation
        task.record_control_diagnostics(diagnostic)
        physical = task.records[-1].traces[-1]
        public_rows.append({
            "tick": len(public_rows),
            "public_before_act": before,
            "issued_action": action.copy(),
            "controller": diagnostic,
            "physical_after_act": physical,
        })

    evaluation = evaluate_records(task.records, task.spec)
    evaluation["accounting"] = dict(task.accounting)
    evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
    evidence = episode_evidence(task.records, task.spec, evaluation)
    episode_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = episode_dir / "evidence.json.gz"
    evidence_path.write_bytes(gzip.compress(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6,
        mtime=0,
    ))
    evidence_audit = audit_episode_evidence(evidence_path)

    trace_path = episode_dir / "public_action_trace.json.gz"
    trace_path.write_bytes(gzip.compress(
        json.dumps(json_ready(public_rows), sort_keys=True, separators=(",", ":")).encode(),
        compresslevel=6,
        mtime=0,
    ))
    episode = evaluation["episodes"][-1]
    traces = task.records[-1].traces
    contact_threshold = max(.5, float(task.spec.config.contact_threshold))
    first_contact = next((
        index for index, row in enumerate(traces)
        if float(row["normal_force"]) > contact_threshold
    ), None)
    first_violation = next((
        index for index, row in enumerate(traces)
        if (
            float(row["force_peak"]) > float(task.spec.config.force_limit)
            or float(row["lateral_force_peak"])
            > float(task.spec.config.lateral_force_limit)
        )
    ), None)
    center = first_violation if first_violation is not None else first_contact
    window = [] if center is None else public_rows[max(0, center - 12):center + 4]
    window_path = episode_dir / "event_window.json"
    write_json(window_path, window)

    result = {
        "record_id": record_id,
        "method": method,
        "instance": INSTANCE,
        "request": request,
        "outcome": episode["outcome"],
        "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "force_peak_N": episode.get("force_peak_N"),
        "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
        "physical_control_steps": int(episode["physical_control_steps"]),
        "action_stream_sha256": action_digest.hexdigest(),
        "first_contact_control_tick": first_contact,
        "first_violation_control_tick": first_violation,
        "evidence": {**pointer(evidence_path), "audit_passed": evidence_audit["passed"]},
        "public_action_trace": pointer(trace_path),
        "event_window": pointer(window_path),
        "accounting": dict(task.accounting),
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/arm/guided_contact_alignment/validation/first_contact_complementarity",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("diagnosis is authorized only with physical GPU1 visible")
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))

    import torch

    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("logical cuda:0 on physical GPU1 is required")
    rows = []
    for request in frozen_requests():
        for method, method_spec in METHODS.items():
            rows.append(execute(
                method,
                method_spec,
                request,
                output=output,
                device=args.device,
            ))
    outcomes = {
        row["record_id"].removeprefix("main/"): row
        for row in read_jsonl(OUTCOMES)
        if row.get("record_id", "").removeprefix("main/")
        in {value["record_id"] for value in rows}
    }
    indexed = {
        (row["method"], row["request"]["request_id"]): row for row in rows
    }
    mode_one_requests = [
        request for request in frozen_requests()
        if int(request["requested_mode"]) == 1
    ]
    mode_one_action_stream_parity = all(
        indexed[("cross_ebdm", request["request_id"])]["action_stream_sha256"]
        == indexed[("cross_ebdm_mode0_contact_established", request["request_id"])]["action_stream_sha256"]
        for request in mode_one_requests
    )
    mode_zero_failures_repaired = all(
        indexed[("cross_ebdm_mode0_contact_established", request["request_id"])]["success"]
        and not indexed[("cross_ebdm_mode0_contact_established", request["request_id"])]["constraint_violation"]
        for request in frozen_requests()
        if int(request["requested_mode"]) == 0
    )
    report = {
        "schema_version": "t2-mode-conditioned-contact-establishment-diagnosis-v2",
        "status": "development_diagnosis_complete",
        "performance_evidence": False,
        "confirmation_states_opened": 0,
        "accessed_confirmation_requests_reused_for_development": True,
        "may_be_reused_for_qualification_or_confirmation": False,
        "device": {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": args.device,
            "name": torch.cuda.get_device_name(0),
        },
        "inputs": {
            "request_roster": pointer(ROSTER),
            "formal_outcomes": pointer(OUTCOMES),
            "development_base_manifest": pointer(DEVELOPMENT_BASE_MANIFEST),
            "method_configs": {
                method: pointer(TASK_ROOT / method_spec["config"])
                for method, method_spec in METHODS.items()
            },
            "runner": pointer(Path(__file__).resolve()),
        },
        "formal_outcome_match": all(
            row["record_id"] in outcomes
            and row["outcome"] == outcomes[row["record_id"]]["outcome"]
            and abs(float(row["force_peak_N"]) - float(outcomes[row["record_id"]]["force_peak_N"])) < 1e-9
            for row in rows
            if row["record_id"].split("/", 1)[0]
            in {"cross_ebdm", "task_only"}
        ),
        "mode_one_action_stream_parity": mode_one_action_stream_parity,
        "mode_zero_failures_repaired": mode_zero_failures_repaired,
        "episodes": rows,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "report": pointer(output / "report.json"),
        "formal_outcome_match": report["formal_outcome_match"],
        "mode_one_action_stream_parity": report["mode_one_action_stream_parity"],
        "mode_zero_failures_repaired": report["mode_zero_failures_repaired"],
        "episodes": [
            {
                "record_id": row["record_id"],
                "outcome": row["outcome"],
                "force_peak_N": row["force_peak_N"],
                "first_contact_control_tick": row["first_contact_control_tick"],
                "first_violation_control_tick": row["first_violation_control_tick"],
            }
            for row in rows
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
