#!/usr/bin/env python3
"""Run the triggered public planar-body T4 reachability branch on frozen receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicPlanarBodyIntentSpec,
    PublicPlanarBodyMatingFeedbackPolicy,
)
from genedynamics.tasks.transfer.episode import controller_diagnostics, episode_evidence
from genedynamics.tasks.transfer.evaluation import evaluate_episode
from genedynamics.tasks.transfer.specs import mechanism_spec, task_spec


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/validation/constraint_reachability_body_intent.json"
SOURCE_PATHS = (
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/backends/floating_contact.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/evaluation.py",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    return value


def _write_json(path: Path, value) -> str:
    payload = (json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_gzip_json(path: Path, value) -> str:
    payload = (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_gzip_json(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _relative(row):
    return (Path(row["mechanism"]) / row["support_cell"]
            / f"mode_{row['requested_mode']}" / f"seed_{row['seed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("body-intent validation is CPU-only and requires CUDA_VISIBLE_DEVICES='' ")
    config_path = args.config.resolve()
    design = json.loads(config_path.read_text())
    output = (args.output or ROOT / design["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    source = design["source"]
    matrix_config_path = ROOT / source["matrix_config"]
    isolation_root = ROOT / source["isolation_results"]
    isolation_report_path = isolation_root / "report.json"
    isolation_audit_path = isolation_root / "evidence_audit.json"
    bindings_path = ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/protocol/actor_bindings.json"
    for path, expected in (
        (matrix_config_path, source["matrix_config_sha256"]),
        (isolation_report_path, source["isolation_report_sha256"]),
        (isolation_audit_path, source["isolation_audit_sha256"]),
    ):
        if _sha(path) != expected:
            raise ValueError(f"frozen source hash differs: {path}")
    matrix = json.loads(matrix_config_path.read_text())
    isolation = json.loads(isolation_report_path.read_text())
    audit = json.loads(isolation_audit_path.read_text())
    public = isolation["profiles"]["bounds_public_reference"]
    if (isolation["status"] != "complete" or not audit["passed"]
            or public["constraint_violations"] != 0
            or public["successes"] >= public["episodes"]
            or not isolation["conditional_body_intent_triggered"]):
        raise ValueError("frozen conditional body-intent trigger is not satisfied")

    feedback = json.loads(bindings_path.read_text())["artifacts"]["feedback"]
    body_spec = PublicPlanarBodyIntentSpec(**{
        key: tuple(value) if key == "velocity_world" else value
        for key, value in design["body_intent"].items() if key != "scope"
    })
    selected = [row for row in isolation["rows"]
                if row["profile"] == "bounds_public_reference"]
    if len(selected) != design["expected_receipts"]:
        raise ValueError("isolation report does not contain the expected receipt set")
    snapshot = {
        "design": design,
        "design_path": str(config_path.relative_to(ROOT)),
        "design_sha256": _sha(config_path),
        "bindings_path": str(bindings_path.relative_to(ROOT)),
        "bindings_sha256": _sha(bindings_path),
        "source_sha256": {name: _sha(ROOT / name) for name in SOURCE_PATHS},
        "source_report_sha256": _sha(isolation_report_path),
        "source_audit_sha256": _sha(isolation_audit_path),
        "device": "cpu", "cuda_visible_devices": "",
        "started_unix_seconds": time.time(),
    }
    _write_json(output / "run_manifest.json", snapshot)
    rows, failures = [], []
    for source_row in selected:
        relative = _relative(source_row)
        episode_dir = output / relative
        receipt_path = isolation_root / relative / "post_preparation_receipt.json.gz"
        try:
            if _sha(receipt_path) != source_row["receipt_sha256"]:
                raise ValueError("post-preparation receipt hash differs")
            receipt = _load_gzip_json(receipt_path)
            mechanism = mechanism_spec("T4", source_row["mechanism_parameters"])
            spec = task_spec("T4", config_overrides={
                "reset_joint_noise": matrix["reset"]["joint_noise"],
                "reset_profile": matrix["reset"]["profile"],
                "execution_profile": design["execution_profile"],
            })
            task = CpuTaskAdapter(spec, mechanism_spec=mechanism,
                                  seed=int(source_row["seed"]))
            observation = task.restore_runtime_profile_checkpoint(receipt)
            restored = task.get_checkpoint()
            if not np.array_equal(np.asarray(restored["integration"]),
                                  np.asarray(receipt["integration"])):
                raise AssertionError("body-intent restore changed integration state")
            actor = PublicPlanarBodyMatingFeedbackPolicy.load(
                spec, None,
                feedback=PublicBiasFeedbackSpec(**{
                    **feedback, "reference_governor_enabled": True,
                }),
                body_intent=body_spec,
                modes=[int(source_row["requested_mode"])],
                seed=int(source_row["seed"]),
            )
            body_requests = []
            for _ in range(spec.config.timeout_steps):
                skill, body = actor.act_with_body(observation)
                body_requests.append(np.asarray(body[0]).copy())
                step = task.step_contact(skill, body_twist=body)
                task.record_control_diagnostics(controller_diagnostics(actor))
                observation = step.observation
                if bool(step.done[0]):
                    break
            record = task.records[-1]
            evaluation = evaluate_episode(record, spec)
            traces = record.traces
            summary = {
                "mechanism": source_row["mechanism"],
                "support_cell": source_row["support_cell"],
                "mechanism_parameters": source_row["mechanism_parameters"],
                "requested_mode": source_row["requested_mode"],
                "state_index": source_row["state_index"],
                "seed": source_row["seed"],
                "source_receipt_sha256": source_row["receipt_sha256"],
                "source_receipt_integration_sha256": source_row["receipt_integration_sha256"],
                "integration_exact": True,
                "success": bool(evaluation["success"]),
                "outcome": evaluation["outcome"],
                "constraint_violation": bool(evaluation["constraint_violation"]),
                "control_steps": len(traces),
                "preparation_control_steps": record.preparation_control_steps,
                "minimum_joint_margin_rad": float(min(row["minimum_joint_margin_rad"] for row in traces)),
                "minimum_support_count": int(min(row["support_count"] for row in traces)),
                "maximum_body_tilt_rad": float(max(row["base_tilt"] for row in traces)),
                "maximum_total_force_N": float(max(row["force_peak"] for row in traces)),
                "maximum_lateral_force_N": float(max(row["lateral_force_peak"] for row in traces)),
                "final_goal_distance_m": float(np.linalg.norm(
                    np.asarray(traces[-1]["tcp_position"]) - np.asarray(spec.config.goal)
                )),
                "final_body_position": np.asarray(observation["body_position"][0]).tolist(),
                "final_body_reference": np.asarray(task.state.body_reference[0]).tolist(),
                "maximum_requested_body_speed": float(max(
                    np.linalg.norm(value) for value in body_requests
                )),
                "nonzero_body_request_steps": int(sum(
                    np.linalg.norm(value) > 0 for value in body_requests
                )),
                "accounting": dict(task.accounting),
            }
            evidence = episode_evidence(task.records, spec, task.evaluate())
            summary["episode_evidence_sha256"] = _write_gzip_json(
                episode_dir / "episode_evidence.json.gz", evidence,
            )
            _write_json(episode_dir / "summary.json", summary)
            rows.append(summary)
        except Exception as error:
            failure = {"scope": str(relative), "error": repr(error),
                       "traceback": traceback.format_exc()}
            failures.append(failure)
            _write_json(episode_dir / "failure.json", failure)
        _write_json(output / "progress.json", {
            "completed_episodes": len(rows), "failures": failures, "rows": rows,
        })
    expected = int(design["expected_episodes"])
    gate = design["gate"]
    report = {
        "version": design["version"],
        "status": "complete" if len(rows) == expected and not failures else "incomplete",
        "device": "cpu", "gpu_calls": 0,
        "episodes": len(rows), "expected_episodes": expected,
        "receipts": len({row["source_receipt_sha256"] for row in rows}),
        "expected_receipts": design["expected_receipts"],
        "successes": sum(row["success"] for row in rows),
        "constraint_violations": sum(row["constraint_violation"] for row in rows),
        "outcomes": {outcome: sum(row["outcome"] == outcome for row in rows)
                     for outcome in sorted({row["outcome"] for row in rows})},
        "minimum_joint_margin_rad": min(
            (row["minimum_joint_margin_rad"] for row in rows), default=None,
        ),
        "minimum_support_count": min(
            (row["minimum_support_count"] for row in rows), default=None,
        ),
        "all_receipt_restores_exact": all(row["integration_exact"] for row in rows),
        "gate_passed": bool(
            len(rows) == expected and not failures
            and all(row["success"] for row in rows)
            and not any(row["constraint_violation"] for row in rows)
            and min((row["minimum_support_count"] for row in rows), default=0)
                >= gate["minimum_support_count"]
        ),
        "failures": failures, "rows": rows,
        "finished_unix_seconds": time.time(),
    }
    _write_json(output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
