#!/usr/bin/env python3
"""Run the paired CPU T4 constraint/reachability isolation matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies import (
    PublicBiasFeedbackSpec,
)
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicLivenessMatingFeedbackPolicy,
)
from genedynamics.tasks.transfer.episode import (
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_episode
from genedynamics.tasks.transfer.specs import mechanism_spec, task_spec


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/validation/constraint_reachability_isolation.json"
SOURCE_PATHS = (
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/backends/floating_contact.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _seed(base: int, mechanism_index: int, support_index: int,
          mode: int, state_index: int) -> int:
    cell = 2 * mechanism_index + support_index
    return base + 10 * cell + 2 * mode + state_index


def _episode_summary(task, actor, spec, profile, receipt_sha, receipt_integration_sha):
    record = task.records[-1]
    evaluation = evaluate_episode(record, spec)
    traces = record.traces
    phases = sorted({
        int(np.asarray(row.get("memory", {}).get("phase", [-1])).reshape(-1)[0])
        for row in record.control_diagnostics
        if isinstance(row, dict) and "memory" in row
    })
    def minimum(name, default):
        values = [row.get(name) for row in traces if row.get(name) is not None]
        return default if not values else float(np.min(values))
    def maximum(name, default=0.):
        values = [row.get(name) for row in traces if row.get(name) is not None]
        return default if not values else float(np.max(values))
    final = traces[-1]
    separations = [
        np.linalg.norm(np.asarray(row["public_reference_position"])
                       - np.asarray(row["applied_reference_position"]))
        for row in traces
    ]
    return {
        "profile": profile["id"],
        "execution_profile": profile["execution_profile"],
        "public_reference_governor": profile["public_reference_governor"],
        "receipt_sha256": receipt_sha,
        "receipt_integration_sha256": receipt_integration_sha,
        "success": bool(evaluation["success"]),
        "outcome": evaluation["outcome"],
        "constraint_violation": bool(evaluation["constraint_violation"]),
        "control_steps": len(traces),
        "preparation_control_steps": record.preparation_control_steps,
        "minimum_joint_margin_rad": minimum("minimum_joint_margin_rad", None),
        "minimum_support_count": int(minimum("support_count", 0)),
        "minimum_body_height_m": minimum("body_height", None),
        "maximum_body_tilt_rad": maximum("base_tilt"),
        "maximum_wbc_residual": maximum("wbc_residual"),
        "maximum_total_force_N": maximum("force_peak"),
        "maximum_lateral_force_N": maximum("lateral_force_peak"),
        "maximum_public_applied_reference_separation_m": float(max(separations)),
        "visited_phases": phases,
        "final_tcp_position": np.asarray(final["tcp_position"]).tolist(),
        "final_goal_distance_m": float(np.linalg.norm(
            np.asarray(final["tcp_position"]) - np.asarray(spec.config.goal)
        )),
        "final_backend_status": int(final["backend_status"]),
        "body_intent": "zero; conditional branch not enabled",
        "accounting": dict(task.accounting),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("T4 isolation is CPU-only and requires CUDA_VISIBLE_DEVICES='' ")
    config_path = args.config.resolve()
    design = json.loads(config_path.read_text())
    output = (args.output or ROOT / design["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feedback_path = ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/protocol/actor_bindings.json"
    feedback = json.loads(feedback_path.read_text())["artifacts"]["feedback"]
    snapshot = {
        "design": design,
        "design_path": _display_path(config_path),
        "design_sha256": _sha(config_path),
        "feedback_path": str(feedback_path.relative_to(ROOT)),
        "feedback_sha256": _sha(feedback_path),
        "source_sha256": {name: _sha(ROOT / name) for name in SOURCE_PATHS},
        "device": "cpu",
        "cuda_visible_devices": "",
        "started_unix_seconds": time.time(),
    }
    _write_json(output / "run_manifest.json", snapshot)
    rows = []
    failures = []
    base_seed = int(design["reset"]["fresh_seed_base"])
    for mechanism_index, mechanism_row in enumerate(design["mechanisms"]):
        for support_index, support_row in enumerate(design["support_cells"]):
            parameters = {
                key: mechanism_row[key]
                for key in ("servo_gain", "delay_steps", "payload_mass")
            }
            parameters["body_servo_gain"] = support_row["body_servo_gain"]
            mechanism = mechanism_spec("T4", parameters)
            for mode in design["requested_modes"]:
                for state_index in range(design["reset"]["states_per_mode_cell"]):
                    seed = _seed(base_seed, mechanism_index, support_index,
                                 int(mode), state_index)
                    relative = Path(mechanism_row["id"]) / support_row["id"] / f"mode_{mode}" / f"seed_{seed}"
                    state_dir = output / relative
                    receipt_path = state_dir / "post_preparation_receipt.json.gz"
                    try:
                        source_spec = task_spec("T4", config_overrides={
                            "reset_joint_noise": design["reset"]["joint_noise"],
                            "reset_profile": design["reset"]["profile"],
                            "execution_profile": "legacy",
                        })
                        source = CpuTaskAdapter(
                            source_spec, mechanism_spec=mechanism, seed=seed,
                        )
                        source.reset((seed,), seed=seed, modes=(int(mode),))
                        receipt = source.get_checkpoint()
                        receipt_sha = _write_gzip_json(receipt_path, receipt)
                        receipt_integration_sha = hashlib.sha256(
                            np.ascontiguousarray(receipt["integration"]).tobytes()
                        ).hexdigest()
                    except Exception as error:
                        failure = {
                            "scope": str(relative), "stage": "preparation",
                            "error": repr(error), "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        _write_json(state_dir / "preparation_failure.json", failure)
                        continue
                    for profile in design["profiles"]:
                        profile_dir = state_dir / profile["id"]
                        try:
                            spec = task_spec("T4", config_overrides={
                                "reset_joint_noise": design["reset"]["joint_noise"],
                                "reset_profile": design["reset"]["profile"],
                                "execution_profile": profile["execution_profile"],
                            })
                            task = CpuTaskAdapter(
                                spec, mechanism_spec=mechanism, seed=seed,
                            )
                            observation = task.restore_runtime_profile_checkpoint(receipt)
                            restored = task.get_checkpoint()
                            if not np.array_equal(restored["integration"], receipt["integration"]):
                                raise AssertionError("profile restore changed the integration state")
                            actor = PublicLivenessMatingFeedbackPolicy.load(
                                spec, None,
                                feedback=PublicBiasFeedbackSpec(**{
                                    **feedback,
                                    "reference_governor_enabled": bool(
                                        profile["public_reference_governor"]
                                    ),
                                }),
                                modes=[int(mode)], seed=seed,
                            )
                            for _ in range(spec.config.timeout_steps):
                                step = task.step(actor.act(observation))
                                task.record_control_diagnostics(
                                    controller_diagnostics(actor)
                                )
                                observation = step.observation
                                if bool(step.done[0]):
                                    break
                            summary = {
                                "mechanism": mechanism_row["id"],
                                "support_cell": support_row["id"],
                                "mechanism_parameters": parameters,
                                "requested_mode": int(mode),
                                "state_index": state_index,
                                "seed": seed,
                                "integration_exact": True,
                                **_episode_summary(
                                    task, actor, spec, profile, receipt_sha,
                                    receipt_integration_sha,
                                ),
                            }
                            evaluation = task.evaluate()
                            evidence = episode_evidence(task.records, spec, evaluation)
                            evidence_path = profile_dir / "episode_evidence.json.gz"
                            summary["episode_evidence_sha256"] = _write_gzip_json(
                                evidence_path, evidence
                            )
                            _write_json(profile_dir / "summary.json", summary)
                            rows.append(summary)
                        except Exception as error:
                            failure = {
                                "scope": str(relative / profile["id"]),
                                "stage": "runtime", "error": repr(error),
                                "traceback": traceback.format_exc(),
                            }
                            failures.append(failure)
                            _write_json(profile_dir / "failure.json", failure)
                    _write_json(output / "progress.json", {
                        "completed_episodes": len(rows),
                        "failures": failures,
                        "rows": rows,
                    })
    expected = int(design["expected_episodes"])
    by_profile = {}
    for profile in design["profiles"]:
        selected = [row for row in rows if row["profile"] == profile["id"]]
        by_profile[profile["id"]] = {
            "episodes": len(selected),
            "successes": sum(row["success"] for row in selected),
            "constraint_violations": sum(
                row["constraint_violation"] for row in selected
            ),
            "outcomes": {
                outcome: sum(row["outcome"] == outcome for row in selected)
                for outcome in sorted({row["outcome"] for row in selected})
            },
            "minimum_joint_margin_rad": (
                min(row["minimum_joint_margin_rad"] for row in selected)
                if selected else None
            ),
            "minimum_support_count": (
                min(row["minimum_support_count"] for row in selected)
                if selected else None
            ),
        }
    public = by_profile["bounds_public_reference"]
    conditional_body_intent = (
        public["episodes"] > 0
        and public["constraint_violations"] == 0
        and public["successes"] < public["episodes"]
    )
    report = {
        "version": design["version"],
        "status": "complete" if len(rows) == expected and not failures else "incomplete",
        "device": "cpu", "gpu_calls": 0,
        "episodes": len(rows), "expected_episodes": expected,
        "receipts": len({row["receipt_sha256"] for row in rows}),
        "expected_receipts": design["expected_receipts"],
        "failures": failures, "profiles": by_profile,
        "all_profile_restores_exact": all(row["integration_exact"] for row in rows),
        "conditional_body_intent_triggered": conditional_body_intent,
        "rows": rows,
        "finished_unix_seconds": time.time(),
    }
    _write_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
