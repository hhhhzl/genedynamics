"""Task-owned development evaluation for a newly trained transfer method.

The solver owns its checkpoint and inference implementation.  This module
only binds that immutable artifact to an existing task contract, executes a
fixed request list, and stores the same auditable episode evidence used by the
four-task protocol.  It is intentionally method-generic so individual
baselines do not grow their own experiment runners.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import time
import traceback
from typing import Any

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    BINDING_VERSION,
    create_transfer_actor,
    resolve_transfer_binding,
)
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec


VERSION = "transfer-method-development-evaluation-v1"
INDEPENDENT_ACTOR_VERSION = "transfer-independent-actor-development-evaluation-v1"
SOURCE_POSITIVE_CONTROL_VERSION = "transfer-source-positive-control-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
METHODS = {
    "cross_ebdm", "task_only", "fada", "acp", "human2sim2robot",
    "dial_transfer", "response_mpc", "public_geometry", "public_source_prior",
    "adaptsim", "asid",
}
SOURCE_NOMINAL_ROSTER_V2 = "method-source-nominal-actor-roster-v2"
MANIFEST_TASK_CONFIG = {
    "cross_ebdm": "main/cross_ebdm.yaml",
    "task_only": "baseline/task_only.yaml",
    "fada": "baseline/fada.yaml",
    "dial_transfer": "baseline/dial.yaml",
    "response_mpc": "baseline/response_mpc.yaml",
    "public_geometry": "baseline/public_geometry.yaml",
    "public_source_prior": "baseline/public_source_prior.yaml",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(pointer: dict[str, str], label: str) -> Path:
    if set(pointer) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(pointer["path"])
    if not path.is_absolute():
        path = _root() / path
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != pointer["sha256"]:
        raise ValueError(f"{label} SHA changed: expected {pointer['sha256']}, got {actual}")
    return path


def _request(request: dict[str, Any], *, task: str, index: int) -> dict[str, Any]:
    required = {
        "robot", "requested_mode", "reset_seed", "physical_initial_slot",
        "mechanism_parameters", "body_support_parameters",
    }
    optional = {"request_id", "scene_cell"}
    if not required <= set(request) or set(request) - required - optional:
        raise ValueError(f"request {index} must contain {sorted(required)} and only optional roster metadata")
    mode = int(request["requested_mode"])
    if mode not in (0, 1):
        raise ValueError("requested_mode must be zero or one")
    request_id = request.get("request_id", f"{task}/development/{index:03d}/m{mode}")
    if not isinstance(request_id, str) or not request_id.startswith(f"{task}/"):
        raise ValueError(f"request {index} has an invalid task-local request_id")
    result = {
        "request_id": request_id,
        "task": task,
        "robot": str(request["robot"]),
        "requested_mode": mode,
        "reset_seed": int(request["reset_seed"]),
        "physical_initial_slot": int(request["physical_initial_slot"]),
        "mechanism_parameters": deepcopy(request["mechanism_parameters"]),
        "body_support_parameters": deepcopy(request["body_support_parameters"]),
        "confirmation_state": False,
        "performance_evidence": False,
    }
    if "scene_cell" in request:
        result["scene_cell"] = str(request["scene_cell"])
    return result


def evaluation_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and freeze a zero-physics development evaluation plan."""
    independent_actor = config.get("version") == INDEPENDENT_ACTOR_VERSION
    if config.get("version") not in {VERSION, INDEPENDENT_ACTOR_VERSION}:
        raise ValueError(f"expected config version {VERSION} or {INDEPENDENT_ACTOR_VERSION}")
    method = str(config.get("method"))
    if method not in METHODS:
        raise ValueError(f"unsupported learned baseline: {method!r}")
    task = str(config.get("task"))
    task_config = Path(str(config.get("task_config", "")))
    if not task_config.is_absolute():
        task_config = _root() / task_config
    if not task_config.is_file():
        raise FileNotFoundError(f"task config does not exist: {task_config}")
    experiment = ExperimentConfig.from_yaml(task_config)
    if experiment.method != method:
        raise ValueError("task config method differs from the evaluation method")
    manifest_binding = config.get("actor_binding_source") == "task_config_manifest"
    package_binding = config.get("actor_binding_source") == "independent_actor_package"
    if independent_actor != package_binding or (independent_actor and method not in {"asid", "adaptsim"}):
        raise ValueError("independent actor evaluation requires an ASID/AdaptSim package binding")
    checkpoint = None
    if manifest_binding:
        if "checkpoint" in config:
            raise ValueError("manifest-bound evaluation cannot add a second checkpoint")
    elif method == "public_geometry":
        if "checkpoint" in config or config.get("method_identity") != "public_geometry":
            raise ValueError("public_geometry is a checkpoint-free deterministic reference")
    else:
        checkpoint = _checked(config["checkpoint"], "checkpoint")
    actor_binding = None
    instance_id = None
    if manifest_binding:
        actor_binding = deepcopy(experiment.method_params.get("actor_binding"))
        if not isinstance(actor_binding, dict) or actor_binding.get("manifest") is None:
            raise ValueError("manifest-bound evaluation requires a task-owned manifest selection")
        instance_id = config.get("instance_id")
        if instance_id is not None:
            actor_binding["instance"] = instance_id
        resolved = resolve_transfer_binding({"actor_binding": actor_binding}, task_alias=task)
        if (resolved.get("actor") != method
                or resolved.get("method_identity") != config.get("method_identity")
                or resolved.get("status") not in {
                    "instances_ready", "deterministic_single_identity",
                } or resolved.get("instance_id") != (instance_id or "fixed")):
            raise ValueError("manifest-bound actor, method identity, or true instance differs")
    elif package_binding:
        package_path = _checked(config["actor_package"], "independent actor package")
        package = json.loads(package_path.read_text())
        instance_id = config.get("instance_id")
        selected = [row for row in package.get("instances", ())
                    if row.get("instance_id") == instance_id]
        if (package.get("status") != "passed_development_only"
                or package.get("method") != method or package.get("task") != task
                or package.get("method_identity") != config.get("method_identity")
                or package.get("task_config") != {
                    "path": str(task_config.relative_to(_root())),
                    "sha256": _sha256(task_config),
                } or len(package.get("instances", ())) != 8
                or len(selected) != 1
                or selected[0].get("artifacts", {}).get("checkpoint") != config["checkpoint"]):
            raise ValueError("independent actor package identity differs")
        actor_binding = deepcopy(experiment.method_params.get("actor_binding"))
        if (not isinstance(actor_binding, dict)
                or actor_binding.get("version") != BINDING_VERSION
                or actor_binding.get("manifest") is not None
                or actor_binding.get("actor") != method
                or actor_binding.get("task") != task
                or actor_binding.get("status") != "instances_ready"):
            raise ValueError("independent actor task binding differs")
        for label, pointer in selected[0]["artifacts"].items():
            _checked(pointer, f"independent actor {label}")
            actor_binding[label] = deepcopy(pointer)
        actor_binding["instance_id"] = instance_id
    elif method in {"adaptsim", "asid"}:
        actor_binding = deepcopy(experiment.method_params.get("actor_binding"))
        if (not isinstance(actor_binding, dict)
                or actor_binding.get("version") != BINDING_VERSION
                or actor_binding.get("manifest") is not None
                or actor_binding.get("actor") != method
                or actor_binding.get("task") != task
                or actor_binding.get("checkpoint") != config["checkpoint"]
                or actor_binding.get("method_identity") != config.get("method_identity")):
            raise ValueError(f"{method} evaluation requires its exact direct task binding")
        _checked(actor_binding["training_recipe"], f"{method} training recipe")
        _checked(actor_binding["training_report"], f"{method} training report")
    elif method == "public_geometry":
        actor_binding = deepcopy(experiment.method_params.get("actor_binding"))
        if not isinstance(actor_binding, dict):
            raise ValueError("public_geometry requires a task-owned actor binding")
        resolved = resolve_transfer_binding({"actor_binding": actor_binding}, task_alias=task)
        if (resolved.get("actor") != method
                or resolved.get("status") != "deterministic_single_identity"
                or resolved.get("method_identity") != config["method_identity"]):
            raise ValueError("public_geometry task binding differs from its deterministic reference")
    execution = config.get("execution")
    if execution != {
        "device": "cuda",
        "cuda_visible_devices": "1",
        "physical_uuid": GPU1_UUID,
    }:
        raise ValueError("development evaluation is frozen to physical GPU1")
    roster_pointer = config.get("request_roster")
    if roster_pointer is not None:
        if config.get("requests"):
            raise ValueError("request_roster cannot be combined with inline requests")
        roster_path = _checked(roster_pointer, "request roster")
        roster = json.loads(roster_path.read_text())
        _checked(roster["task_config"], "roster task contract")
        if package_binding:
            source_roster = json.loads(_checked(
                roster["source_roster"], "source/nominal reference roster",
            ).read_text())
            if (roster.get("version") != "independent-baseline-source-nominal-roster-v1"
                    or roster.get("requests") != source_roster.get("requests")
                    or source_roster.get("task") != task):
                raise ValueError("independent actor source/nominal roster changed")
        if "training_recipe" in roster:
            _checked(roster["training_recipe"], "roster training recipe")
        instance_count = 8 if package_binding else 1
        if (roster.get("task") != task
                or method not in roster.get("methods", ())
                or (package_binding and (roster.get("methods") != [method]
                                         or roster.get("instance_count") != 8))
                or roster.get("formal_performance_evidence") is not False
                or roster.get("failure_replacement") is not False
                or len(roster.get("requests", ())) != roster.get("requests_per_method")
                or roster.get("maximum_total_physics_episodes") != (
                    len(roster["methods"]) * roster["requests_per_method"] * instance_count
                ) or roster.get("execution", {}).get("physical_uuid") != GPU1_UUID
                or roster.get("execution", {}).get("cuda_visible_devices") != "1"):
            raise ValueError("request roster identity or development evidence status differs")
        native = roster.get("execution", {}).get("native_library")
        if native is not None:
            if task not in {"T3", "T4"}:
                raise ValueError("native ABI binding is only valid for T3/T4")
            _checked(native, "request roster native ABI library")
        source_requests = roster["requests"]
    else:
        source_requests = config.get("requests", ())
    requests = [
        _request(row, task=task, index=index)
        for index, row in enumerate(source_requests)
    ]
    if not requests:
        raise ValueError("evaluation requires at least one request")
    identities = [(row["reset_seed"], row["physical_initial_slot"], row["requested_mode"])
                  for row in requests]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluation requests must have unique reset/slot/mode identities")
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("evaluation request IDs must be unique")
    output = Path(str(config.get("output", "")))
    if not output.is_absolute():
        output = _root() / output
    plan = {
        "version": config["version"],
        "status": "planned_unexecuted",
        "scope": "development target pilot; not formal paper performance evidence",
        "task": task,
        "method": method,
        "method_identity": str(config["method_identity"]),
        "task_config": str(task_config.relative_to(_root())),
        "task_config_sha256": _sha256(task_config),
        "execution": deepcopy(execution),
        "requests": requests,
        "output": str(output.relative_to(_root())),
        "all_failures_retained": True,
        "replacement_requests_allowed": False,
    }
    if checkpoint is not None:
        plan["checkpoint"] = {
            "path": str(checkpoint.relative_to(_root())),
            "sha256": _sha256(checkpoint),
        }
    if roster_pointer is not None:
        plan["request_roster"] = deepcopy(roster_pointer)
    if actor_binding is not None:
        plan["actor_binding"] = actor_binding
    if manifest_binding:
        plan["instance_id"] = instance_id or "fixed"
    if package_binding:
        plan["instance_id"] = instance_id
        plan["actor_package"] = deepcopy(config["actor_package"])
    return plan


def source_positive_control_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze a CPU-only Panda check of an exact FADA source policy."""
    if config.get("version") != SOURCE_POSITIVE_CONTROL_VERSION:
        raise ValueError(f"expected config version {SOURCE_POSITIVE_CONTROL_VERSION}")
    if config.get("method") != "fada":
        raise ValueError("source positive control currently supports FADA only")
    task = str(config.get("task"))
    if task not in {"T1", "T2", "T3"}:
        raise ValueError("repaired FADA source positive control covers T1--T3")
    task_config = Path(str(config.get("task_config", "")))
    if not task_config.is_absolute():
        task_config = _root() / task_config
    if not task_config.is_file():
        raise FileNotFoundError(f"task config does not exist: {task_config}")
    experiment = ExperimentConfig.from_yaml(task_config)
    if experiment.method != "fada":
        raise ValueError("source positive-control task config must bind FADA")
    checkpoint = _checked(config["checkpoint"], "checkpoint")
    if config.get("execution") != {"device": "cpu"}:
        raise ValueError("source positive control is frozen to CPU execution")

    import torch
    from genedynamics.tasks.transfer.specs import task_spec

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = payload.get("cfg", {})
    adapter = payload.get("genedynamics_task_adapter", {})
    expected_name = task_spec(task, robot="panda").name
    if (int(cfg.get("obs_dim", -1)), int(cfg.get("act_dim", -1)),
            int(cfg.get("cmd_dim", -1))) != (35, 13, 67):
        raise ValueError("repaired source checkpoint must retain the 35D/13D/67D FADA chart")
    if adapter.get("task_name") != expected_name:
        raise ValueError("source checkpoint task identity differs")

    checkpoint_role = str(config.get("checkpoint_role", "repaired_source_policy"))
    qualification = None
    if checkpoint_role == "repaired_source_policy":
        repair_phase = adapter.get("repair_pilot", {}).get("phase")
        if repair_phase != "source_oracle_and_dagger_planner_idm":
            raise ValueError(
                "checkpoint is not the repaired source artifact used before target lifting"
            )
    elif checkpoint_role == "qualified_source_reference":
        qualification_path = _checked(
            config["qualification_report"], "source qualification report",
        )
        qualification_report = json.loads(qualification_path.read_text())
        qualification_checkpoint = Path(str(qualification_report.get("checkpoint", "")))
        if not qualification_checkpoint.is_absolute():
            qualification_checkpoint = _root() / qualification_checkpoint
        stages = qualification_report.get("stages", ())
        final_stage = stages[-1] if stages else {}
        requested = int(qualification_report.get("source_final_requested", -1))
        successes = int(qualification_report.get("source_final_successes", -1))
        if (
            qualification_report.get("status") != "complete"
            or qualification_report.get("task") != task
            or qualification_report.get("source_only") is not True
            or int(qualification_report.get("new_target_calls", -1)) != 0
            or qualification_report.get("source_final_qualified") is not True
            or requested != 4
            or successes != requested
            or qualification_checkpoint.resolve() != checkpoint.resolve()
            or qualification_report.get("checkpoint_sha256") != _sha256(checkpoint)
            or final_stage.get("checkpoint_sha256") != _sha256(checkpoint)
            or int(final_stage.get("source_successes", -1))
            != int(final_stage.get("requested", -2))
            or int(final_stage.get("requested", -1)) != 4
            or int(final_stage.get("consecutive_successful_monitor_stages", 0)) < 2
        ):
            raise ValueError(
                "qualified source reference is not bound to a complete 4/4 source-only proof"
            )
        repair_phase = None
        qualification = {
            "path": str(qualification_path.relative_to(_root())),
            "sha256": _sha256(qualification_path),
            "source_final_successes": successes,
            "source_final_requested": requested,
            "consecutive_successful_monitor_stages": int(
                final_stage["consecutive_successful_monitor_stages"]
            ),
        }
    else:
        raise ValueError("unsupported FADA source checkpoint role")

    requests = [
        _request(row, task=task, index=index)
        for index, row in enumerate(config.get("requests", ()))
    ]
    if len(requests) != 8:
        raise ValueError("source positive control requires four fresh requests per mode")
    identities = [
        (row["reset_seed"], row["physical_initial_slot"], row["requested_mode"])
        for row in requests
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("source requests must have unique reset/slot/mode identities")
    nominal = {"delay_steps": 0, "payload_mass": 0.5, "servo_gain": 1.0}
    if any(
        row["robot"] != "panda"
        or row["mechanism_parameters"] != nominal
        or row["body_support_parameters"]
        for row in requests
    ):
        raise ValueError("source positive control requires nominal Panda requests")
    modes = [row["requested_mode"] for row in requests]
    slots = [row["physical_initial_slot"] for row in requests]
    if modes.count(0) != 4 or modes.count(1) != 4 or sorted(slots) != [0, 0, 1, 1, 2, 2, 3, 3]:
        raise ValueError("source positive control requires modes 0/1 over slots 0--3")

    output = Path(str(config.get("output", "")))
    if not output.is_absolute():
        output = _root() / output
    return {
        "version": SOURCE_POSITIVE_CONTROL_VERSION,
        "status": "planned_unexecuted",
        "scope": (
            "fresh exact-checkpoint source closed-loop positive control; "
            "not target performance evidence"
        ),
        "task": task,
        "method": "fada",
        "method_identity": str(config["method_identity"]),
        "actor_role": checkpoint_role,
        "task_config": str(task_config.relative_to(_root())),
        "task_config_sha256": _sha256(task_config),
        "checkpoint": {
            "path": str(checkpoint.relative_to(_root())),
            "sha256": _sha256(checkpoint),
        },
        "checkpoint_schema": {
            "obs_dim": 35, "act_dim": 13, "cmd_dim": 67,
            "history_len": int(cfg["history_len"]),
            "task_name": expected_name,
            "repair_phase": repair_phase,
        },
        "source_qualification": qualification,
        "execution": {"device": "cpu"},
        "requests": requests,
        "output": str(output.relative_to(_root())),
        "source_only": True,
        "target_physical_execution_count": 0,
        "all_failures_retained": True,
        "replacement_requests_allowed": False,
    }


def _gpu1(*, maximum_gpu_memory_gib: float | None = None) -> None:
    import torch

    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        raise RuntimeError("evaluation must expose exactly physical GPU1")
    actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if actual.removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError(f"visible GPU UUID differs: {actual} != {GPU1_UUID}")
    if maximum_gpu_memory_gib is not None:
        if maximum_gpu_memory_gib <= 0:
            raise ValueError("GPU memory ceiling must be positive")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(maximum_gpu_memory_gib * 1024**3 / total, 1.), 0,
        )
    torch.set_num_threads(1)


def _task(experiment: ExperimentConfig, request: dict[str, Any]):
    values = deepcopy(experiment.env_params)
    values.pop("robot", None)
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    shared_execution = values.pop("shared_execution", None)
    mechanism_values = {
        **dict(values.pop("mechanism", {}) or {}),
        **dict(request["mechanism_parameters"]),
    }
    mechanism = mechanism_spec(
        request["task"], mechanism_values,
        body_support_parameters=dict(request["body_support_parameters"]),
    )
    task = instantiate_task(
        experiment.env_name, robot=request["robot"], mechanism_spec=mechanism,
        seed=request["reset_seed"], history_len=history_len,
        config_overrides=values or None, shared_execution=shared_execution,
    )
    task._experiment_mode = request["requested_mode"]
    return task


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _atomic_gzip_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    os.replace(temporary, path)


def _episode(plan: dict[str, Any], request: dict[str, Any], output: Path) -> dict[str, Any]:
    record_id = request["request_id"]
    episode_dir = output / "episodes" / hashlib.sha256(record_id.encode()).hexdigest()
    result_path = episode_dir / "result.json"
    task = actor = None
    reset_succeeded = False
    try:
        experiment = ExperimentConfig.from_yaml(_root() / plan["task_config"])
        task = _task(experiment, request)
        initial = "jammed" if plan["task"] in {"T3", "T4"} else "clear"
        public = task.reset(
            (request["physical_initial_slot"],), seed=request["reset_seed"],
            initial=initial, modes=[request["requested_mode"]],
        )
        reset_succeeded = True
        if plan.get("scope") == "independent formal main table":
            reset_hash = task.records[-1].reset_evidence.get("physical_state_sha256")
            if not isinstance(reset_hash, str) or len(reset_hash) != 64:
                raise ValueError("formal reset lacks physical state evidence")
            atomic_json(episode_dir / "reset_receipt.json", {
                "request_id": record_id,
                "task": plan["task"], "method": plan["method"],
                "instance_id": plan["instance_id"],
                "reset_seed": request["reset_seed"],
                "reset_physical_hash": reset_hash,
                "written_before_first_actuation": True,
            })
        if plan.get("actor_role") in {
            "repaired_source_policy", "qualified_source_reference",
        }:
            import torch
            from genedynamics.tasks.transfer.controllers.fada.base import PublicFADATaskActor

            payload = torch.load(
                _root() / plan["checkpoint"]["path"], map_location="cpu", weights_only=True,
            )
            actor = PublicFADATaskActor(
                task.spec, payload, mode=request["requested_mode"], device="cpu",
            )
        else:
            binding = deepcopy(plan.get("actor_binding")) if plan.get("actor_binding") is not None else {
                "version": BINDING_VERSION,
                "task": plan["task"],
                "family": "main",
                "binding_key": plan["method"],
                "method_identity": plan["method_identity"],
                "actor": plan["method"],
                "status": "instances_ready",
                "instance_id": "development_pilot",
                "checkpoint": deepcopy(plan["checkpoint"]),
            }
            parameters = deepcopy(experiment.method_params)
            parameters.update({
                "mode": request["requested_mode"], "device": "cuda",
                "actor_binding": binding,
            })
            actor = create_transfer_actor(task, plan["method"], parameters)
        started = time.perf_counter()
        actor_action_seconds = []
        for _ in range(int(experiment.n_steps)):
            if bool(np.asarray(task.state.status)[0]):
                break
            actor_tick = time.perf_counter()
            action = np.asarray(actor.act(public), dtype=np.float32)
            actor_action_seconds.append(time.perf_counter() - actor_tick)
            diagnostic = controller_diagnostics(actor)
            body_intent = task.public_body_intent(public)
            public = (
                task.step(action).observation
                if body_intent is None
                else task.step_contact(action, body_twist=body_intent).observation
            )
            task.record_control_diagnostics(diagnostic)
        wall_seconds = time.perf_counter() - started
        evaluation = evaluate_records(task.records, task.spec)
        evaluation["accounting"] = dict(task.accounting)
        evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
        evidence_path = episode_dir / "evidence.json.gz"
        _atomic_gzip_json(evidence_path, episode_evidence(task.records, task.spec, evaluation))
        audit = audit_episode_evidence(evidence_path)
        episode = evaluation["episodes"][-1]
        exact_t2 = bool(episode.get("exact_physics_contact_evidence", True))
        support_ok = True
        if plan["task"] == "T4":
            valid = [row for row in task.records[-1].traces if row.get("valid")]
            support_ok = bool(valid) and min(
                int(row.get("support_count", 0)) for row in valid
            ) >= 2
        inference = list(getattr(actor, "inference_seconds", ()))
        passed = bool(
            episode["success"] and not episode["constraint_violation"]
            and episode["mode_match"] and exact_t2 and support_ok and audit["passed"]
        )
        result = {
            "record_id": record_id, "request": request, "passed": passed,
            "success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "outcome": episode["outcome"], "observed_mode": episode["observed_mode"],
            "mode_match": bool(episode["mode_match"]),
            "physical_control_steps": int(episode["physical_control_steps"]),
            "preparation_control_steps": int(episode["preparation_control_steps"]),
            "capped_normalized_completion_time": float(episode["capped_normalized_completion_time"]),
            "force_peak_N": episode.get("force_peak_N"),
            "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
            "wall_seconds": wall_seconds,
            "network_calls": int(getattr(actor, "irreversible_network_calls", 0)),
            "model_calls": int(getattr(actor, "irreversible_model_calls", 0)),
            "candidate_rollouts": int(getattr(actor, "irreversible_candidate_rollouts", 0)),
            "action_updates": int(getattr(actor, "irreversible_action_updates", 0)),
            "inference_latency_seconds": {
                "count": len(inference),
                "p50": float(np.percentile(inference, 50)) if inference else None,
                "p95": float(np.percentile(inference, 95)) if inference else None,
                "p99": float(np.percentile(inference, 99)) if inference else None,
            },
            "actor_action_latency_seconds": {
                "count": len(actor_action_seconds),
                "p50": float(np.percentile(actor_action_seconds, 50)) if actor_action_seconds else None,
                "p95": float(np.percentile(actor_action_seconds, 95)) if actor_action_seconds else None,
                "p99": float(np.percentile(actor_action_seconds, 99)) if actor_action_seconds else None,
            },
            "evidence": {
                "path": str(evidence_path.relative_to(_root())),
                "sha256": _sha256(evidence_path), "audit_passed": bool(audit["passed"]),
            },
            "accounting": dict(task.accounting),
        }
    except BaseException as error:
        result = {
            "record_id": record_id, "request": request, "passed": False,
            "success": False, "constraint_violation": True,
            "outcome": "reset_or_execution_error", "reset_succeeded": reset_succeeded,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "accounting": {} if task is None else dict(task.accounting),
        }
    result = _json_ready(result)
    atomic_json(result_path, result)
    return result


def _run_plan(plan: dict[str, Any], *, continue_claimed: bool = False) -> dict[str, Any]:
    output = _root() / plan["output"]
    if output.exists():
        if not continue_claimed:
            raise FileExistsError(f"evaluation output already exists: {output}")
        if (plan.get("scope") != "independent formal main table"
                or not (output / "plan.json").is_file()
                or json.loads((output / "plan.json").read_text()) != plan
                or (output / "report.json").exists()):
            raise ValueError("formal continuation lacks an incomplete exact claimed plan")
        missing_seen = False
        for request in plan["requests"]:
            episode_dir = output / "episodes" / hashlib.sha256(
                request["request_id"].encode()
            ).hexdigest()
            result_path = episode_dir / "result.json"
            if result_path.is_file():
                if missing_seen:
                    raise ValueError("formal continuation has noncontiguous outcomes")
                result = json.loads(result_path.read_text())
                if (result.get("record_id") != request["request_id"]
                        or result.get("request") != request):
                    raise ValueError("continued formal result differs from its request")
            else:
                missing_seen = True
                if episode_dir.exists():
                    raise ValueError(
                        "request has an incomplete or ambiguous physical attempt; "
                        "automatic replacement is forbidden"
                    )
    else:
        if continue_claimed:
            raise FileNotFoundError("formal continuation has no claimed plan")
        output.mkdir(parents=True)
        atomic_json(output / "plan.json", plan)
    results = []
    for request in plan["requests"]:
        episode_dir = output / "episodes" / hashlib.sha256(
            request["request_id"].encode()
        ).hexdigest()
        result_path = episode_dir / "result.json"
        if result_path.is_file():
            if not continue_claimed:
                raise ValueError("fresh evaluation found a previously attempted request")
            result = json.loads(result_path.read_text())
            if (result.get("record_id") != request["request_id"]
                    or result.get("request") != request):
                raise ValueError("continued formal result differs from its request")
        else:
            if episode_dir.exists():
                raise ValueError(
                    "request has an incomplete or ambiguous physical attempt; "
                    "automatic replacement is forbidden"
                )
            result = _episode(plan, request, output)
        results.append(result)
        atomic_json(output / "progress.json", {
            "status": "running", "completed": len(results),
            "requested": len(plan["requests"]), "latest": result,
        })
    latency = [
        row["inference_latency_seconds"]["p50"] for row in results
        if row.get("inference_latency_seconds", {}).get("p50") is not None
    ]
    report = {
        **plan, "status": "complete", "results": results,
        "counts": {
            "requested": len(results),
            "passed": sum(row["passed"] for row in results),
            "successes": sum(row["success"] for row in results),
            "constraint_violations": sum(row["constraint_violation"] for row in results),
            "mode_matches": sum(bool(row.get("mode_match")) for row in results),
            "execution_errors": sum(row["outcome"] == "reset_or_execution_error" for row in results),
            "outcomes": dict(Counter(row["outcome"] for row in results)),
            "physical_control_steps": sum(int(row.get("physical_control_steps", 0)) for row in results),
            "network_calls": sum(int(row.get("network_calls", 0)) for row in results),
            "model_calls": sum(int(row.get("model_calls", 0)) for row in results),
            "candidate_rollouts": sum(int(row.get("candidate_rollouts", 0)) for row in results),
            "action_updates": sum(int(row.get("action_updates", 0)) for row in results),
        },
        "evidence_audits_passed": all(
            row.get("evidence", {}).get("audit_passed", False) for row in results
        ),
        "median_episode_p50_inference_seconds": float(np.median(latency)) if latency else None,
    }
    atomic_json(output / "report.json", report)
    return report


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Execute every frozen target request once and retain every result."""
    plan = evaluation_plan(config)
    maximum_memory = None
    if "request_roster" in plan:
        roster = json.loads(_checked(plan["request_roster"], "request roster").read_text())
        maximum_memory = float(roster["execution"]["maximum_gpu_memory_gib"])
    if plan["task"] in {"T3", "T4"}:
        import ctypes

        pointer = roster.get("execution", {}).get("native_library") if "request_roster" in plan else None
        native = (_checked(pointer, "request roster native ABI library") if pointer else
                  _root() / ".runtime/cross_ebdm/native/libcross_ebdm.so")
        if not native.is_file():
            raise FileNotFoundError(f"native ABI library does not exist: {native}")
        configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
        if configured is not None and Path(configured).expanduser().resolve() != native.resolve():
            raise ValueError("CROSS_EBDM_NATIVE_LIBRARY differs from the task ABI binding")
        library = ctypes.CDLL(str(native))
        library.cross_ebdm_abi_version.restype = ctypes.c_int
        if library.cross_ebdm_abi_version() != 1:
            raise ValueError("T3/T4 actor requires native ABI v1")
        os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native.resolve())
    _gpu1(maximum_gpu_memory_gib=maximum_memory)
    return _run_plan(plan)


def audit_evaluation(config: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    """Reconcile one development run and reevaluate its saved episodes on CPU."""
    plan = evaluation_plan(config)
    output = _root() / plan["output"]
    if json.loads((output / "plan.json").read_text()) != plan:
        raise ValueError("saved development plan differs from the checked configuration")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or any(
        report.get(key) != value for key, value in plan.items() if key != "status"
    ):
        raise ValueError("development report differs from its frozen plan")
    rows = report.get("results", ())
    if len(rows) != len(plan["requests"]):
        raise ValueError("development report has missing physical requests")
    audited = []
    runner_failures = 0
    for request, row in zip(plan["requests"], rows, strict=True):
        record_id = request["request_id"]
        result_path = output / "episodes" / hashlib.sha256(record_id.encode()).hexdigest() / "result.json"
        if row.get("request") != request or row.get("record_id") != record_id:
            raise ValueError(f"development request differs: {record_id}")
        if json.loads(result_path.read_text()) != row:
            raise ValueError(f"saved result differs from report: {record_id}")
        pointer = row.get("evidence")
        if pointer is None:
            if row.get("outcome") != "reset_or_execution_error" or row.get("passed") is not False:
                raise ValueError(f"missing physical evidence without retained error: {record_id}")
            runner_failures += 1
            audited.append({"request_id": record_id, "status": "retained_runner_failure"})
            continue
        path = _checked({"path": pointer["path"], "sha256": pointer["sha256"]}, "episode evidence")
        if not path.is_relative_to(output):
            raise ValueError(f"episode evidence outside task-local run: {record_id}")
        checked = audit_episode_evidence(path)
        episode = checked["evaluation"]["episodes"][-1]
        if (not checked["passed"] or pointer.get("audit_passed") is not True
                or row["success"] != episode["success"]
                or row["outcome"] != episode["outcome"]
                or row["constraint_violation"] != episode["constraint_violation"]
                or row["mode_match"] != episode["mode_match"]
                or row["physical_control_steps"] != episode["physical_control_steps"]
                or not np.isclose(
                    row["capped_normalized_completion_time"],
                    episode["capped_normalized_completion_time"], rtol=0, atol=1e-12,
                )):
            raise ValueError(f"CPU episode reevaluation differs: {record_id}")
        audited.append({
            "request_id": record_id, "status": "physical_evidence_verified",
            "reset_physical_hash": checked["reset_physical_hashes"][-1],
            "evidence_sha256": checked["sha256"],
            "passed": row["passed"], "outcome": row["outcome"],
        })
    counts = report["counts"]
    expected_counts = {
        "requested": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "successes": sum(row["success"] for row in rows),
        "constraint_violations": sum(row["constraint_violation"] for row in rows),
        "mode_matches": sum(bool(row.get("mode_match")) for row in rows),
        "execution_errors": runner_failures,
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "physical_control_steps": sum(int(row.get("physical_control_steps", 0)) for row in rows),
        "network_calls": sum(int(row.get("network_calls", 0)) for row in rows),
        "model_calls": sum(int(row.get("model_calls", 0)) for row in rows),
        "candidate_rollouts": sum(int(row.get("candidate_rollouts", 0)) for row in rows),
        "action_updates": sum(int(row.get("action_updates", 0)) for row in rows),
    }
    if (frozenset(counts) not in {
        frozenset(expected_counts),
        frozenset(expected_counts) - {"model_calls", "candidate_rollouts", "action_updates"},
    } or any(counts[key] != expected_counts[key] for key in counts)):
        raise ValueError("development report aggregate counts differ")
    audit = {
        "version": "transfer-method-development-evidence-audit-v1",
        "status": "passed" if runner_failures == 0 else "retained_runner_failures",
        "task": plan["task"], "method": plan["method"],
        "report_sha256": _sha256(report_path),
        "request_roster": plan.get("request_roster"),
        "physical_evidence_verified": len(rows) - runner_failures,
        "runner_failures_retained": runner_failures,
        "episodes": audited,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }
    if write:
        atomic_json(output / "independent_audit.json", audit)
    return audit


def audit_paired_evaluations(
    first: dict[str, Any], second: dict[str, Any], *, write: bool = True,
) -> dict[str, Any]:
    """Compare two independently audited methods on identical physical resets."""
    first_plan, second_plan = evaluation_plan(first), evaluation_plan(second)
    roster = first_plan.get("request_roster")
    if (not roster or roster != second_plan.get("request_roster")
            or first_plan["task"] != second_plan["task"]
            or first_plan["method"] == second_plan["method"]
            or first_plan["requests"] != second_plan["requests"]):
        raise ValueError("paired methods require one exact frozen request roster")
    declared = json.loads(_checked(roster, "paired request roster").read_text())
    if not {first_plan["method"], second_plan["method"]} <= set(declared["methods"]):
        raise ValueError("paired method identities differ from the frozen roster")
    first_audit = audit_evaluation(first, write=False)
    second_audit = audit_evaluation(second, write=False)
    if first_audit["status"] != "passed" or second_audit["status"] != "passed":
        raise ValueError("paired contrast requires physical evidence for every request")
    first_report = json.loads((_root() / first_plan["output"] / "report.json").read_text())
    second_report = json.loads((_root() / second_plan["output"] / "report.json").read_text())
    groups: dict[str, list[tuple[dict, dict]]] = {}
    pairs = []
    for request, a, b, a_audit, b_audit in zip(
        first_plan["requests"], first_report["results"], second_report["results"],
        first_audit["episodes"], second_audit["episodes"], strict=True,
    ):
        reset_hash = a_audit["reset_physical_hash"]
        if not reset_hash or reset_hash != b_audit["reset_physical_hash"]:
            raise ValueError(f"physical reset mismatch: {request['request_id']}")
        group = request.get("scene_cell", "unclassified")
        groups.setdefault(group, []).append((a, b))
        pairs.append({
            "request_id": request["request_id"], "scene_cell": group,
            "reset_physical_hash": reset_hash,
            first_plan["method"]: {"passed": a["passed"], "outcome": a["outcome"]},
            second_plan["method"]: {"passed": b["passed"], "outcome": b["outcome"]},
        })

    def summarize(rows: list[tuple[dict, dict]]) -> dict[str, Any]:
        a_success = sum(a["passed"] for a, _ in rows)
        b_success = sum(b["passed"] for _, b in rows)
        return {
            "requests": len(rows),
            first_plan["method"]: {
                "constrained_successes": a_success,
                "constraint_violations": sum(a["constraint_violation"] for a, _ in rows),
                "mean_capped_completion": float(mean(a["capped_normalized_completion_time"] for a, _ in rows)),
            },
            second_plan["method"]: {
                "constrained_successes": b_success,
                "constraint_violations": sum(b["constraint_violation"] for _, b in rows),
                "mean_capped_completion": float(mean(b["capped_normalized_completion_time"] for _, b in rows)),
            },
            "first_minus_second_constrained_success_rate": (a_success - b_success) / len(rows),
        }

    contrast = {
        "version": "transfer-method-paired-development-audit-v1",
        "status": "passed_development_only",
        "task": first_plan["task"],
        "method": f"{first_plan['method']}_vs_{second_plan['method']}",
        "methods": [first_plan["method"], second_plan["method"]],
        "request_roster": roster,
        "report_sha256": {
            first_plan["method"]: first_audit["report_sha256"],
            second_plan["method"]: second_audit["report_sha256"],
        },
        "overall": summarize([(a, b) for a, b in zip(
            first_report["results"], second_report["results"], strict=True,
        )]),
        "scene_cells": {name: summarize(rows) for name, rows in sorted(groups.items())},
        "pairs": pairs,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }
    if write:
        output = _root() / first_plan["output"] / "paired_development_audit.json"
        atomic_json(output, contrast)
    return contrast


def manifest_roster_config(roster_path: Path, method: str) -> dict[str, Any]:
    """Derive one immutable manifest-bound development job from a task roster."""
    roster_path = roster_path if roster_path.is_absolute() else _root() / roster_path
    roster = json.loads(roster_path.read_text())
    if (roster.get("version") != SOURCE_NOMINAL_ROSTER_V2
            or method not in roster.get("methods", ())
            or method not in MANIFEST_TASK_CONFIG):
        raise ValueError("unsupported method or source/nominal roster version")
    base = _checked(roster["task_config"], "roster task contract")
    task_root = base.parent
    task_config = task_root / MANIFEST_TASK_CONFIG[method]
    experiment = ExperimentConfig.from_yaml(task_config)
    selection = deepcopy(experiment.method_params["actor_binding"])
    selection["instance"] = "instance_01"
    resolved = resolve_transfer_binding({"actor_binding": selection}, task_alias=roster["task"])
    if resolved["actor"] != method or resolved["status"] not in {
        "instances_ready", "deterministic_single_identity",
    }:
        raise ValueError("source/nominal roster method lacks an executable actor")
    output = experiment.output_dir / "validation" / (
        f"source_nominal_actor_screen_{roster['requests'][0]['reset_seed']}"
    )
    return {
        "version": VERSION,
        "task": roster["task"], "method": method,
        "method_identity": resolved["method_identity"],
        "task_config": str(task_config.relative_to(_root())),
        "actor_binding_source": "task_config_manifest",
        "instance_id": resolved["instance_id"],
        "request_roster": {
            "path": str(roster_path.relative_to(_root())),
            "sha256": _sha256(roster_path),
        },
        "execution": {
            "device": "cuda", "cuda_visible_devices": "1",
            "physical_uuid": GPU1_UUID,
        },
        "output": str(output.relative_to(_root())),
        "plan_output": str(output.relative_to(_root())) + "_plan.json",
    }


def _source_nominal_method_config(roster_path: Path, method: str, version: str):
    if version == SOURCE_NOMINAL_ROSTER_V2:
        config = manifest_roster_config(roster_path, method)
        provenance = {"origin": "derived_from_frozen_roster",
                      "roster_sha256": config["request_roster"]["sha256"]}
    else:
        config_path = roster_path.parent / f"{method}_source_nominal_actor_screen.json"
        config = json.loads(config_path.read_text())
        provenance = {"origin": "task_local_config",
                      "path": str(config_path.relative_to(_root())),
                      "sha256": _sha256(config_path)}
    return config, provenance


def audit_source_nominal_roster(roster_path: Path, *, write: bool = True) -> dict[str, Any]:
    """Reconcile all declared methods on one fresh task-local actor screen."""
    roster_path = roster_path if roster_path.is_absolute() else _root() / roster_path
    roster = json.loads(roster_path.read_text())
    if roster.get("version") not in {
        "method-source-nominal-actor-roster-v1", SOURCE_NOMINAL_ROSTER_V2,
    }:
        raise ValueError("source/nominal actor roster version differs")
    pointer = {"path": str(roster_path.relative_to(_root())), "sha256": _sha256(roster_path)}
    methods = roster["methods"]
    if len(methods) != len(set(methods)) or len(roster["requests"]) != roster["requests_per_method"]:
        raise ValueError("source/nominal method or request roster differs")
    method_rows = {}
    by_request: dict[str, list[dict[str, Any]]] = {
        request["request_id"]: [] for request in roster["requests"]
    }
    for method in methods:
        config, provenance = _source_nominal_method_config(
            roster_path, method, roster["version"],
        )
        if config.get("method") != method or config.get("request_roster") != pointer:
            raise ValueError(f"{method} actor screen differs from frozen roster")
        audit = audit_evaluation(config, write=False)
        report = json.loads((_root() / config["output"] / "report.json").read_text())
        cell_results = {}
        for scene in sorted({request["scene_cell"] for request in roster["requests"]}):
            rows = [row for row in report["results"] if row["request"]["scene_cell"] == scene]
            cell_results[scene] = {
                "requests": len(rows), "constrained_successes": sum(row["passed"] for row in rows),
                "constraint_violations": sum(row["constraint_violation"] for row in rows),
                "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
                "mean_capped_completion": float(mean(
                    row.get("capped_normalized_completion_time", 1.) for row in rows
                )),
            }
        method_rows[method] = {
            "instance_id": evaluation_plan(config).get("instance_id"),
            "method_identity": config["method_identity"],
            **({"config_sha256": provenance["sha256"]}
               if roster["version"] != SOURCE_NOMINAL_ROSTER_V2
               else {"config_provenance": provenance}),
            "report_sha256": audit["report_sha256"],
            "audit_status": audit["status"],
            "physical_evidence_verified": audit["physical_evidence_verified"],
            "runner_failures_retained": audit["runner_failures_retained"],
            "overall": report["counts"], "scene_cells": cell_results,
        }
        for request, row, evidence in zip(
            report["requests"], report["results"], audit["episodes"], strict=True,
        ):
            by_request[request["request_id"]].append({
                "method": method, "passed": row["passed"],
                "outcome": row["outcome"],
                "reset_physical_hash": evidence.get("reset_physical_hash"),
            })
    paired = []
    for request in roster["requests"]:
        rows = by_request[request["request_id"]]
        reset_hashes = {row["reset_physical_hash"] for row in rows}
        if (len(rows) != len(methods) or len(reset_hashes) != 1
                or None in reset_hashes):
            raise ValueError(f"source/nominal methods do not share one verified reset: {request['request_id']}")
        paired.append({"request_id": request["request_id"],
                       "scene_cell": request["scene_cell"],
                       "reset_physical_hash": next(iter(reset_hashes)),
                       "methods": rows})
    task_root = Path(roster["task_config"]["path"]).parent.relative_to("configs")
    output = _root() / "results" / task_root / "validation" / (
        f"method_source_nominal_actor_screen_{roster['requests'][0]['reset_seed']}"
    )
    result = {
        "version": "method-source-nominal-actor-screen-independent-audit-v1",
        "status": "complete_development_evidence",
        "task": roster["task"], "method": "method_roster",
        "request_roster": pointer,
        "method_order": methods, "methods": method_rows,
        "paired_requests": paired,
        "physical_evidence_verified": sum(
            row["physical_evidence_verified"] for row in method_rows.values()
        ),
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
        "source_robot_positive_control_included": False,
        "output": str(output.relative_to(_root())),
    }
    if write:
        atomic_json(output / "independent_audit.json", result)
    return result


def audit_source_nominal_resources(roster_path: Path, *, write: bool = True) -> dict[str, Any]:
    """Recover online compute counters from independently verified traces."""
    from genedynamics.tasks.transfer.episode import load_episode_evidence

    roster_audit = audit_source_nominal_roster(roster_path, write=False)
    roster_path = roster_path if roster_path.is_absolute() else _root() / roster_path
    counters = {
        "network_calls": "irreversible_network_calls",
        "model_calls": "irreversible_model_calls",
        "candidate_rollouts": "irreversible_candidate_rollouts",
        "action_updates": "irreversible_action_updates",
    }
    methods = {}
    for method in roster_audit["method_order"]:
        config, _ = _source_nominal_method_config(
            roster_path, method,
            json.loads(roster_path.read_text())["version"],
        )
        report = json.loads((_root() / config["output"] / "report.json").read_text())
        rows = []
        for episode in report["results"]:
            records, _, _ = load_episode_evidence(_root() / episode["evidence"]["path"])
            diagnostics = records[-1].control_diagnostics
            if not diagnostics:
                raise ValueError(f"{method} physical trace has no controller diagnostic")
            final = diagnostics[-1]
            candidate = final.get("candidate", {})
            if not isinstance(candidate, dict):
                raise ValueError(f"{method} candidate controller diagnostic is invalid")
            values = {}
            for name, source in counters.items():
                recorded = [int(row[source]) for row in (final, candidate) if source in row]
                if len(set(recorded)) > 1:
                    raise ValueError(f"{method} wrapper and candidate {name} counters differ")
                values[name] = recorded[0] if recorded else 0
            if values["network_calls"] != episode["network_calls"]:
                raise ValueError(f"{method} network-call report and trace differ")
            for name in counters:
                if name in episode and values[name] != episode[name]:
                    raise ValueError(f"{method} {name} report and trace differ")
            rows.append({"request_id": episode["record_id"], **values})
        totals = {name: sum(row[name] for row in rows) for name in counters}
        methods[method] = {"report_sha256": roster_audit["methods"][method]["report_sha256"],
                           "totals": totals, "episodes": rows}
    result = {
        "version": "method-source-nominal-online-resource-reconciliation-v1",
        "status": "passed",
        "task": roster_audit["task"], "method": "method_roster",
        "request_roster": roster_audit["request_roster"],
        "independent_outcome_audit_sha256": _sha256(
            _root() / roster_audit["output"] / "independent_audit.json"
        ),
        "methods": methods,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
        "output": roster_audit["output"],
    }
    if write:
        atomic_json(_root() / result["output"] / "resource_reconciliation.json", result)
    return result


def run_source_positive_control(config: dict[str, Any]) -> dict[str, Any]:
    """Execute every frozen source request once without any target access."""
    plan = source_positive_control_plan(config)
    return _run_plan(plan)


__all__ = [
    "VERSION", "SOURCE_POSITIVE_CONTROL_VERSION", "evaluation_plan",
    "source_positive_control_plan", "run_evaluation", "run_source_positive_control",
    "audit_evaluation", "audit_paired_evaluations", "audit_source_nominal_roster",
    "audit_source_nominal_resources", "manifest_roster_config",
]
