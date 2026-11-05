"""Task-local T3 public early-contact guard development diagnostic."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicEarlyContactMatingFeedbackPolicy,
    PublicMatchedReferenceUnloadingMatingFeedbackPolicy,
    PublicReferenceUnloadingMatingFeedbackPolicy,
)
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence, controller_diagnostics, episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec


VERSION = "t3-public-early-contact-guard-development-v1"
UNLOADING_VERSION = "t3-public-reference-unloading-development-v1"
MATCHED_UNLOADING_VERSION = "t3-public-matched-reference-unloading-development-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
ROOT = Path(__file__).resolve().parents[3]


def _checked(pointer: dict[str, str]) -> Path:
    if set(pointer) != {"path", "sha256"}:
        raise ValueError("source artifact must be SHA-bound")
    path = ROOT / pointer["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pointer["sha256"]:
        raise ValueError(f"source artifact changed: {path}")
    return path


def _key(request: dict[str, Any]) -> tuple[str, int, int]:
    return (request["scene_cell"], int(request["physical_initial_slot"]),
            int(request["requested_mode"]))


def plan(config: dict[str, Any]) -> dict[str, Any]:
    policies = {
        VERSION: PublicEarlyContactMatingFeedbackPolicy,
        UNLOADING_VERSION: PublicReferenceUnloadingMatingFeedbackPolicy,
        MATCHED_UNLOADING_VERSION: PublicMatchedReferenceUnloadingMatingFeedbackPolicy,
    }
    version = config.get("version")
    if version not in policies or config.get("task") != "T3":
        raise ValueError("T3 early-contact development config required")
    guard = policies[version]
    expected_guard = {
        "release_z_m": .5965, "clear_z_m": .604, "retreat_speed_mps": .008,
    }
    if version in (UNLOADING_VERSION, MATCHED_UNLOADING_VERSION):
        expected_guard.update(
            unload_seconds=.08,
            max_planar_speed_mps=.02 if version == MATCHED_UNLOADING_VERSION else .025,
        )
    if config.get("guard") != expected_guard:
        raise ValueError("early-contact guard parameters changed")
    implementation_guard = {
        "release_z_m": guard.RELEASE_Z_M,
        "clear_z_m": guard.CLEAR_Z_M,
        "retreat_speed_mps": guard.RETREAT_SPEED_MPS,
    }
    if version in (UNLOADING_VERSION, MATCHED_UNLOADING_VERSION):
        implementation_guard.update(
            unload_seconds=guard.UNLOAD_SECONDS,
            max_planar_speed_mps=guard.MAX_PLANAR_SPEED_MPS,
        )
        bindings = config.get("source_bindings")
        expected_paths = {
            "genedynamics/tasks/transfer/controllers/mating_feedback.py",
            "genedynamics/tasks/transfer/early_contact_diagnostic.py",
            "scripts/paper/cross_ebdm/diagnose_t3_early_contact.py",
        }
        if not isinstance(bindings, list) or {row.get("path") for row in bindings} != expected_paths:
            raise ValueError("unloading candidate requires exact source bindings")
        for binding in bindings:
            _checked(binding)
    if config["guard"] != implementation_guard:
        raise ValueError("guard implementation differs from task-local config")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 360, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("T3 guard diagnostic must retain its GPU1 resource ceiling")
    source_plan = json.loads(_checked(config["sentinel_plan"]).read_text())
    source_report = json.loads(_checked(config["sentinel_report"]).read_text())
    if (source_plan.get("task") != "T3"
            or source_report.get("status") != "complete_recorded_development_diagnosis"
            or source_report.get("counts", {}).get("completed_physical_jobs") != 16
            or source_report.get("counts", {}).get("execution_errors") != 0
            or source_report.get("artifacts", {}).get("plan", {}).get("sha256")
                != config["sentinel_plan"]["sha256"]):
        raise ValueError("source public sentinel is incomplete or differs")
    progress_path = _checked(source_report["artifacts"]["progress"])
    old_rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    requests = list(source_plan["requests"])
    if (len(requests) != 16 or len(old_rows) != 16
            or {_key(row) for row in requests} != {_key(row["request"]) for row in old_rows}
            or {row["request"]["robot"] for row in old_rows} != {"xarm7"}
            or any(row["method"] != "public_geometry" for row in old_rows)):
        raise ValueError("T3 sentinel request roster changed")
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"diagnostic output exists: {output}")
    return {
        "version": version, "status": "planned_unexecuted", "task": "T3",
        "scope": "development-only shared public early-contact guard on all 16 frozen sentinel resets",
        "sentinel_plan": config["sentinel_plan"],
        "sentinel_report": config["sentinel_report"],
        "sentinel_progress": source_report["artifacts"]["progress"],
        "guard": config["guard"], "execution": config["execution"],
        "source_bindings": config.get("source_bindings"),
        "requests": requests, "planned_physics_episodes": 16,
        "output": config["output"], "formal_performance_evidence": False,
        "failure_replacement": False,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    frozen = plan(config)
    policy = {
        VERSION: PublicEarlyContactMatingFeedbackPolicy,
        UNLOADING_VERSION: PublicReferenceUnloadingMatingFeedbackPolicy,
        MATCHED_UNLOADING_VERSION: PublicMatchedReferenceUnloadingMatingFeedbackPolicy,
    }[frozen["version"]]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("visible CUDA device differs from frozen physical GPU1")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    source_report = json.loads(_checked(config["sentinel_report"]).read_text())
    old_rows = [json.loads(line) for line in _checked(
        source_report["artifacts"]["progress"]).read_text().splitlines()]
    source = {_key(row["request"]): row for row in old_rows}
    experiment = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/keyed_insertion_recovery/baseline/public_geometry.yaml"
    )
    started = time.monotonic()
    rows = []
    report = {"version": frozen["version"], "status": "running", "plan": frozen,
              "rows": rows, "controller_version": policy.VERSION}
    try:
        for index, request in enumerate(frozen["requests"]):
            if time.monotonic() - started > config["execution"]["maximum_wall_seconds"]:
                raise TimeoutError("T3 public guard wall ceiling exceeded")
            row = {"index": index, "request": request, "status": "running"}
            task = None
            try:
                values = deepcopy(experiment.env_params)
                values.pop("robot", None)
                history_len = int(values.pop("history_len", 4))
                values.pop("mode", None)
                shared_execution = values.pop("shared_execution", None)
                mechanism_values = {
                    **dict(values.pop("mechanism", {}) or {}),
                    **dict(request["mechanism_parameters"]),
                }
                task = instantiate_task(
                    experiment.env_name, robot=request["robot"],
                    mechanism_spec=mechanism_spec(
                        "T3", mechanism_values,
                        body_support_parameters=request["body_support_parameters"],
                    ),
                    seed=int(request["reset_seed"]), history_len=history_len,
                    config_overrides=values or None, shared_execution=shared_execution,
                )
                mode = int(request["requested_mode"])
                task._experiment_mode = mode
                public = task.reset(
                    (int(request["physical_initial_slot"]),),
                    seed=int(request["reset_seed"]), initial="jammed", modes=[mode],
                )
                parameters = deepcopy(experiment.method_params)
                parameters.update(mode=mode, device="cuda")
                base = create_transfer_actor(task, experiment.method, parameters)
                actor = policy.load(
                    task.spec, None, feedback=base.pose_feedback,
                    modes=[mode], seed=base._reset_seed,
                )
                for _ in range(int(experiment.n_steps)):
                    if bool(np.asarray(task.state.status)[0]):
                        break
                    action = np.asarray(actor.act(public), dtype=np.float32)
                    diagnostic = controller_diagnostics(actor)
                    public = task.step(action).observation
                    task.record_control_diagnostics(diagnostic)
                evaluation = evaluate_records(task.records, task.spec)
                evaluation["accounting"] = dict(task.accounting)
                evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
                evidence = episode_evidence(task.records, task.spec, evaluation)
                evidence_path = output / "episodes" / f"{index:02d}" / "evidence.json.gz"
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
                evidence_path.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
                audit = audit_episode_evidence(evidence_path)
                old_evidence = json.loads(gzip.open(
                    ROOT / source[_key(request)]["evidence"]["path"], "rt"
                ).read())
                reset_sha = evidence["records"][0]["reset_evidence"]["physical_state_sha256"]
                old_reset_sha = old_evidence["records"][0]["reset_evidence"]["physical_state_sha256"]
                if reset_sha != old_reset_sha or audit.get("passed") is not True:
                    raise RuntimeError("new evidence audit or matched physical reset failed")
                episode = evaluation["episodes"][-1]
                row.update({
                    "status": "complete", "outcome": episode["outcome"],
                    "success": episode["constrained_success"],
                    "constraint_violation": episode["constraint_violation"],
                    "control_steps": episode["physical_control_steps"],
                    "force_peak_N": episode["force_peak_N"],
                    "lateral_force_peak_N": episode["lateral_force_peak_N"],
                    "source_outcome": source[_key(request)]["outcome"],
                    "source_control_steps": source[_key(request)]["physical_control_steps"],
                    "reset_sha256": reset_sha,
                    "evidence": {"path": str(evidence_path.relative_to(ROOT)),
                                 "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()},
                })
            except BaseException as exc:
                row.update({"status": "failed_retained", "error": str(exc),
                            "traceback": traceback.format_exc()})
            rows.append(row)
            atomic_json(output / "progress.json", {"rows": rows})
        report.update({
            "status": "complete" if all(row["status"] == "complete" for row in rows)
                      else "incomplete_retained",
            "counts": {"completed": sum(row["status"] == "complete" for row in rows),
                       "outcomes": dict(Counter(row.get("outcome", "execution_error") for row in rows))},
            "wall_seconds": time.monotonic() - started,
        })
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
                       "traceback": traceback.format_exc(),
                       "wall_seconds": time.monotonic() - started})
        raise
    finally:
        atomic_json(output / "report.json", report)
    return report
