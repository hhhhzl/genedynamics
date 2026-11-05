#!/usr/bin/env python3
"""Run one shared physical predeployment gate for every four-task method row.

The task protocol owns requests and success predicates.  Method bindings own
actors and immutable artifacts.  Missing actors remain explicit admission
states; they are never replaced by a public controller or silently omitted.
"""

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
import subprocess
import time
import traceback
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION,
    PublicTargetTrajectory,
)
from genedynamics.tasks.transfer.controllers.bindings import (
    create_transfer_actor,
    resolve_transfer_binding,
    validate_transfer_actor_manifest,
)
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import mechanism_spec
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    RiskSetEmptyError,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_VERSION = 7
NATIVE_LIBRARY = ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"
TASKS = {
    "T1": ROOT / "configs/arm/obstacle_transfer",
    "T2": ROOT / "configs/arm/guided_contact_alignment",
    "T3": ROOT / "configs/arm/keyed_insertion_recovery",
    "T4": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery",
}
MAIN_CONFIG = {
    "cross_ebdm": "main/cross_ebdm.yaml",
    "task_only": "baseline/task_only.yaml",
    "fada": "baseline/fada.yaml",
    "acp": "baseline/acp.yaml",
    "human2sim2robot": "baseline/human2sim2robot.yaml",
    "dial_transfer": "baseline/dial.yaml",
    "response_mpc": "baseline/response_mpc.yaml",
    "public_geometry": "baseline/public_geometry.yaml",
    "public_source_prior": "baseline/public_source_prior.yaml",
}
ABLATION_CONFIG = {
    "frozen_warm": "ablation/frozen_warm.yaml",
    "task_only": "baseline/task_only.yaml",
    "matching_no_band": "ablation/matching_no_band.yaml",
    "band_only": "ablation/band_only.yaml",
    "ungated_full": "ablation/ungated_full.yaml",
    "gated_full": "main/cross_ebdm.yaml",
}
EXPECTED_INSTANCES = tuple(f"instance_{index:02d}" for index in range(1, 9))
T2_RETRAIN_KEYS = {
    ("main", "cross_ebdm"),
    ("main", "task_only"),
    *(('ablation', key) for key in ABLATION_CONFIG),
}
DIAGNOSTIC_SCHEMA = "four-task-paired-development-diagnosis-v1"
DIAGNOSTIC_SCHEMA_V2 = "four-task-paired-development-diagnosis-v2"
DIAGNOSTIC_SCHEMAS = {DIAGNOSTIC_SCHEMA, DIAGNOSTIC_SCHEMA_V2}
DIAGNOSTIC_QUALIFICATION_SCHEMA = "four-task-paired-development-qualification-v2"
DIAGNOSTIC_QUALIFICATION_SCHEMA_V3 = "four-task-paired-development-qualification-v3"
CONTRIBUTION_CONTRAST_SCHEMA = "four-task-paired-contribution-contrast-v1"
SAME_START_CONTRIBUTION_SCHEMA = "four-task-same-start-anchoring-contrast-v1"
RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA = (
    "four-task-same-start-relative-response-contrast-v1"
)
TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-directional-response-contrast-v1"
)
TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-directional-same-state-contrast-v1"
)
RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-runtime-dominance-contrast-v1"
)
FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-full-runtime-dominance-contrast-v2"
)
MARGIN_RUNTIME_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-margin-runtime-dominance-contrast-v1"
)
ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-adaptive-step-contrast-v1"
)
STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-stable-adaptive-step-contrast-v1"
)
EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-expanded-adaptive-step-contrast-v1"
)
TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-transient-phase-adaptive-step-contrast-v1"
)
PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-phase-scheduled-adaptive-step-contrast-v1"
)
PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA = (
    "four-task-task-first-physical-checkpoint-selector-v1"
)
ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA = (
    "four-task-task-first-ensemble-robust-checkpoint-selector-v1"
)
PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-physical-selected-adaptive-step-contrast-v1"
)
ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-ensemble-robust-adaptive-step-contrast-v1"
)
EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-expanded-ensemble-robust-adaptive-step-contrast-v1"
)
BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-bounded-endpoint-ensemble-robust-contrast-v1"
)
CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-capped-expansion-ensemble-robust-contrast-v1"
)
HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-horizon-consistent-ensemble-robust-contrast-v1"
)
VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-validated-bridge-ensemble-robust-contrast-v1"
)
COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-completion-value-ensemble-robust-contrast-v1"
)
MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA = (
    "four-task-mode-conditioned-finite-time-transport-contrast-v1"
)
MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA = (
    "four-task-mode-conditioned-finite-time-transport-confirmation-v1"
)
MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-recorded-completion-transport-contrast-v1"
)
MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-supported-phase-convex-transport-contrast-v1"
)
MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-supported-phase-capped-expansion-transport-contrast-v1"
)
MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-calibrated-authority-transport-contrast-v1"
)
MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA = (
    "four-task-mode-specific-physical-checkpoint-selector-v1"
)
MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA = (
    "four-task-mode-specific-mode-zero-physical-checkpoint-selector-v1"
)
MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-physical-selected-transport-contrast-v1"
)
MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA = (
    "four-task-mode-specific-physical-selected-transport-confirmation-v1"
)
MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-success-safe-selected-transport-contrast-v1"
)
MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA = (
    "four-task-mode-specific-success-safe-selected-transport-confirmation-v1"
)
MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA = (
    "four-task-mode-specific-dual-physical-selected-transport-contrast-v1"
)
MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA = (
    "four-task-mode-specific-dual-physical-selected-transport-confirmation-v1"
)
ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA = (
    "four-task-task-first-ensemble-robust-selected-contrast-v1"
)
PUBLIC_LOG_SCHEMA = "four-task-public-log-collection-v1"
CONSTRAINT_STRESS_SCHEMA = "four-task-paired-constraint-stress-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
GPU0_UUID = "GPU-253f0813-4432-2d51-a149-3e73b38bf77a"
T2_DIAGNOSTIC_ACTOR_INTERFACES = {
    "public-T2-terminal-completion-ownership-v1",
    "public-T2-post-outlet-completion-ownership-v2",
    "public-T2-clearance-progress-authority-v3",
    "public-T2-clearance-terminal-set-admission-v4",
    "public-T2-clearance-terminal-transport-v5",
    "public-T2-clearance-terminal-strict-progress-v6",
    "public-T2-handoff-viability-v7",
    "public-T2-stage-debt-viability-v8",
    "public-T2-completion-budget-viability-v9",
    "public-T2-completion-budget-liveness-v10",
    "public-T2-completion-budget-terminal-capture-v11",
    "public-T2-completion-budget-capture-band-v12",
    "public-T2-completion-budget-planar-braking-v13",
    "public-T2-completion-budget-pre-descent-settling-v14",
    "public-T2-completion-budget-reference-unloading-v15",
    "public-T2-completion-budget-reference-debt-unloading-v16",
    "public-T2-completion-budget-causal-reference-unloading-v17",
    "public-T2-completion-budget-causal-traverse-v18",
}
HANDOFF_VIABILITY_PARAMETER_KEYS = {
    "completion_reserve_controls",
    "reserve_taper_controls",
    "reference_lag_full_authority_m",
    "reference_lag_zero_authority_m",
    "maximum_authority_decay_per_control",
}
STAGE_DEBT_PARAMETER_KEYS = HANDOFF_VIABILITY_PARAMETER_KEYS | {
    "clearance_debt_full_controls",
    "clearance_debt_zero_controls",
}
COMPLETION_BUDGET_PARAMETER_KEYS = HANDOFF_VIABILITY_PARAMETER_KEYS | {
    "completion_progress_floor_mps",
    "post_traverse_reserve_controls",
    "completion_budget_taper_controls",
    "force_speed_full_authority_mps",
    "force_speed_zero_authority_mps",
}


def _task_actor_interface_parameters(
    interface: str | None, value: Any,
) -> dict[str, int | float]:
    parameters = dict(value or {})
    if not parameters:
        return {}
    if (
        (
            interface == "public-T2-handoff-viability-v7"
            and set(parameters) != HANDOFF_VIABILITY_PARAMETER_KEYS
        )
        or (
            interface == "public-T2-stage-debt-viability-v8"
            and set(parameters) != STAGE_DEBT_PARAMETER_KEYS
        )
        or (
            interface in {
                "public-T2-completion-budget-viability-v9",
                "public-T2-completion-budget-liveness-v10",
                "public-T2-completion-budget-terminal-capture-v11",
                "public-T2-completion-budget-capture-band-v12",
                "public-T2-completion-budget-planar-braking-v13",
                "public-T2-completion-budget-pre-descent-settling-v14",
                "public-T2-completion-budget-reference-unloading-v15",
                "public-T2-completion-budget-reference-debt-unloading-v16",
                "public-T2-completion-budget-causal-reference-unloading-v17",
                "public-T2-completion-budget-causal-traverse-v18",
            }
            and set(parameters) != COMPLETION_BUDGET_PARAMETER_KEYS
        )
        or interface not in {
            "public-T2-handoff-viability-v7",
            "public-T2-stage-debt-viability-v8",
            "public-T2-completion-budget-viability-v9",
            "public-T2-completion-budget-liveness-v10",
            "public-T2-completion-budget-terminal-capture-v11",
            "public-T2-completion-budget-capture-band-v12",
            "public-T2-completion-budget-planar-braking-v13",
            "public-T2-completion-budget-pre-descent-settling-v14",
            "public-T2-completion-budget-reference-unloading-v15",
            "public-T2-completion-budget-reference-debt-unloading-v16",
            "public-T2-completion-budget-causal-reference-unloading-v17",
            "public-T2-completion-budget-causal-traverse-v18",
        }
    ):
        raise ValueError("unsupported task actor interface parameters")
    parsed: dict[str, int | float] = {
        "completion_reserve_controls": int(
            parameters["completion_reserve_controls"]
        ),
        "reserve_taper_controls": int(parameters["reserve_taper_controls"]),
        "reference_lag_full_authority_m": float(
            parameters["reference_lag_full_authority_m"]
        ),
        "reference_lag_zero_authority_m": float(
            parameters["reference_lag_zero_authority_m"]
        ),
        "maximum_authority_decay_per_control": float(
            parameters["maximum_authority_decay_per_control"]
        ),
    }
    if interface == "public-T2-stage-debt-viability-v8":
        parsed.update({
            "clearance_debt_full_controls": int(
                parameters["clearance_debt_full_controls"]
            ),
            "clearance_debt_zero_controls": int(
                parameters["clearance_debt_zero_controls"]
            ),
        })
    if interface in {
        "public-T2-completion-budget-viability-v9",
        "public-T2-completion-budget-liveness-v10",
        "public-T2-completion-budget-terminal-capture-v11",
        "public-T2-completion-budget-capture-band-v12",
        "public-T2-completion-budget-planar-braking-v13",
        "public-T2-completion-budget-pre-descent-settling-v14",
        "public-T2-completion-budget-reference-unloading-v15",
        "public-T2-completion-budget-reference-debt-unloading-v16",
        "public-T2-completion-budget-causal-reference-unloading-v17",
        "public-T2-completion-budget-causal-traverse-v18",
    }:
        parsed.update({
            "completion_progress_floor_mps": float(
                parameters["completion_progress_floor_mps"]
            ),
            "post_traverse_reserve_controls": int(
                parameters["post_traverse_reserve_controls"]
            ),
            "completion_budget_taper_controls": int(
                parameters["completion_budget_taper_controls"]
            ),
            "force_speed_full_authority_mps": float(
                parameters["force_speed_full_authority_mps"]
            ),
            "force_speed_zero_authority_mps": float(
                parameters["force_speed_zero_authority_mps"]
            ),
        })
    if (
        parsed["completion_reserve_controls"] <= 0
        or parsed["reserve_taper_controls"] <= 0
        or parsed["reference_lag_full_authority_m"] < 0.
        or parsed["reference_lag_zero_authority_m"]
        <= parsed["reference_lag_full_authority_m"]
        or not 0. < parsed["maximum_authority_decay_per_control"] <= 1.
        or (
            interface == "public-T2-stage-debt-viability-v8"
            and (
                parsed["clearance_debt_full_controls"] < 0
                or parsed["clearance_debt_zero_controls"]
                <= parsed["clearance_debt_full_controls"]
            )
        )
        or (
            interface in {
                "public-T2-completion-budget-viability-v9",
                "public-T2-completion-budget-liveness-v10",
                "public-T2-completion-budget-terminal-capture-v11",
                "public-T2-completion-budget-capture-band-v12",
                "public-T2-completion-budget-planar-braking-v13",
                "public-T2-completion-budget-pre-descent-settling-v14",
                "public-T2-completion-budget-reference-unloading-v15",
                "public-T2-completion-budget-reference-debt-unloading-v16",
                "public-T2-completion-budget-causal-reference-unloading-v17",
                "public-T2-completion-budget-causal-traverse-v18",
            }
            and (
                parsed["completion_progress_floor_mps"] <= 0.
                or parsed["post_traverse_reserve_controls"] <= 0
                or parsed["completion_budget_taper_controls"] <= 0
                or parsed["force_speed_full_authority_mps"] < 0.
                or parsed["force_speed_zero_authority_mps"]
                <= parsed["force_speed_full_authority_mps"]
            )
        )
        or not all(np.isfinite(value) for value in parsed.values())
    ):
        raise ValueError("invalid handoff viability parameters")
    return parsed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}


def _checked_pointer(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or _sha(path) != str(value["sha256"]):
        raise ValueError(f"{label} changed or is missing")
    return path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_protocol(task: str, protocol_version: int) -> tuple[Path, dict[str, Any]]:
    expected_version = f"four-task-behavioral-predeployment-v{protocol_version}"
    path = TASKS[task] / f"protocol/predeployment_v{protocol_version}.json"
    value = json.loads(path.read_text())
    if value.get("version") != expected_version or value.get("task") != task:
        raise ValueError(f"{task} behavioral predeployment protocol differs")
    if value.get("requested_modes") != [0, 1]:
        raise ValueError(f"{task} must qualify both requested modes")
    if int(value.get("reset_slots_per_cell", 0)) < 1:
        raise ValueError(f"{task} has no qualification reset slots")
    execution = value.get("execution")
    gpu_protocol = (
        task == "T2" and protocol_version >= 14
        or task == "T3" and protocol_version >= 9
    )
    if gpu_protocol:
        exact_gpu_protocol = (
            task == "T2" and protocol_version >= 16
            or task == "T3" and protocol_version >= 10
        )
        gpu1_successor = (
            task == "T2" and protocol_version >= 17
            or task == "T3" and protocol_version >= 11
        )
        expected_execution = {"device": "cuda"}
        if exact_gpu_protocol:
            physical_index = 1 if gpu1_successor else 0
            expected_execution = {
                "device": "cuda",
                "cuda_visible_devices": str(physical_index),
                "physical_index": physical_index,
                "physical_uuid": {
                    0: GPU0_UUID,
                    1: GPU1_UUID,
                }[physical_index],
            }
        if execution != expected_execution:
            raise ValueError(
                f"{task} predeployment v{protocol_version} has the wrong frozen GPU binding"
            )
    elif execution is not None:
        raise ValueError(
            f"{task} predeployment v{protocol_version} cannot change CPU execution"
        )
    gate = value.get("gate") or {}
    if (
        gate.get("require_all_mode_correct_task_success") is not True
        or gate.get("require_zero_constraint_violations") is not True
        or gate.get("require_evidence_reaudit") is not True
        or gate.get("failure_replacement") is not False
    ):
        raise ValueError(f"{task} behavioral gate was weakened")
    return path, value


def qualification_requests(task: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sequence = 0
    anchor = int(protocol["seed_anchor"])
    protocol_version = protocol["version"].rsplit("-v", 1)[-1]
    for cell in protocol["scene_cells"]:
        for slot in range(int(protocol["reset_slots_per_cell"])):
            for mode in protocol["requested_modes"]:
                rows.append({
                    "request_id": f"PREDEPLOY-v{protocol_version}/{task}/{cell['id']}/i{slot:02d}/m{mode}",
                    "task": task,
                    "scene_cell": cell["id"],
                    "robot": cell["robot"],
                    "mechanism_parameters": deepcopy(cell["mechanism_parameters"]),
                    "body_support_parameters": deepcopy(cell["body_support_parameters"]),
                    "physical_initial_slot": slot,
                    "requested_mode": mode,
                    "reset_seed": anchor + sequence,
                    "confirmation_state": False,
                    "performance_evidence": False,
                })
                sequence += 1
    if len({row["reset_seed"] for row in rows}) != len(rows):
        raise ValueError(f"{task} qualification seeds are not unique")
    return rows


def _row_selection(
    task: str, family: str, key: str, config: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = deepcopy(config.method_params["actor_binding"])
    selection.update(family=family, key=key, instance="instance_01")
    binding = resolve_transfer_binding(
        {"actor_binding": selection}, task_alias=task,
    )
    return selection, binding


def _admission_for_rejection(
    task: str, family: str, key: str, binding: dict[str, Any],
) -> dict[str, Any]:
    if task == "T2" and (family, key) in T2_RETRAIN_KEYS:
        status = "training_artifact_required_after_shared_cell_repair"
        final = False
    elif key == "human2sim2robot":
        status = "retained_external_resource_rejection"
        final = True
    elif key == "acp":
        status = "retained_algorithm_predeployment_rejection"
        final = True
    else:
        status = "current_protocol_artifact_unavailable"
        final = False
    return {
        "admission_status": status,
        "final_for_current_protocol": final,
        "manifest_reason": binding["reason"],
        "manifest_evidence": binding["evidence"],
        "logical_instance_slots": binding["logical_instance_slots"],
        "physical_jobs": 0,
    }


def build_plan(
    tasks: tuple[str, ...], protocol_version: int = DEFAULT_PROTOCOL_VERSION,
) -> dict[str, Any]:
    requests: dict[str, list[dict[str, Any]]] = {}
    protocols = {}
    manifests = {}
    rows = []
    jobs = []
    execution = None
    for task in tasks:
        protocol_path, protocol = load_protocol(task, protocol_version)
        task_execution = protocol.get("execution")
        if task_execution is not None:
            if execution is None:
                execution = deepcopy(task_execution)
            elif execution != task_execution:
                raise ValueError("behavioral qualification mixes execution devices")
        protocols[task] = _pointer(protocol_path)
        requests[task] = qualification_requests(task, protocol)
        base = ExperimentConfig.from_yaml(TASKS[task] / "_base.yaml")
        manifest_pointer = base.method_params["actor_binding"]["manifest"]
        validate_transfer_actor_manifest(manifest_pointer)
        manifest_path = ROOT / manifest_pointer["path"]
        manifests[task] = {
            **_pointer(manifest_path),
            "declared_sha256": manifest_pointer["sha256"],
            "full_catalog_validation_passed": True,
        }
        for family, mapping in (("main", MAIN_CONFIG), ("ablation", ABLATION_CONFIG)):
            for key, relative in mapping.items():
                config_path = TASKS[task] / relative
                config = ExperimentConfig.from_yaml(config_path)
                selection, binding = _row_selection(task, family, key, config)
                row_id = f"{task}/{family}/{key}"
                common = {
                    "row_id": row_id,
                    "task": task,
                    "family": family,
                    "key": key,
                    "config": str(config_path.relative_to(ROOT)),
                    "method": config.method,
                    "method_identity": binding["method_identity"],
                    "manifest_status": binding["status"],
                }
                if binding["status"] == "predeployment_rejection":
                    rows.append({
                        **common,
                        **_admission_for_rejection(task, family, key, binding),
                    })
                    continue
                instances = (
                    ("fixed",)
                    if binding["status"] == "deterministic_single_identity"
                    else EXPECTED_INSTANCES
                )
                rows.append({
                    **common,
                    "admission_status": "behavior_execution_required",
                    "final_for_current_protocol": False,
                    "instances": list(instances),
                    "physical_jobs": len(instances) * len(requests[task]),
                })
                for instance in instances:
                    for request in requests[task]:
                        record_id = f"{row_id}/{instance}/{request['request_id']}"
                        job = {
                            "record_id": record_id,
                            "row_id": row_id,
                            "task": task,
                            "family": family,
                            "key": key,
                            "config": str(config_path.relative_to(ROOT)),
                            "method": config.method,
                            "instance": instance,
                            "selection": selection,
                            "request": request,
                        }
                        if task_execution is not None:
                            job.update(deepcopy(task_execution))
                        jobs.append(job)
    seeds = [request["reset_seed"] for task in tasks for request in requests[task]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("behavioral qualification seeds overlap across tasks")
    plan = {
        "schema_version": f"four-task-behavioral-predeployment-v{protocol_version}",
        "status": "planned_unexecuted",
        "tasks": list(tasks),
        "protocols": protocols,
        "actor_manifests": manifests,
        "requests": requests,
        "rows": rows,
        "jobs": jobs,
        "counts": {
            "logical_rows": len(rows),
            "physical_jobs": len(jobs),
            "nonphysical_admission_rows": sum(row["physical_jobs"] == 0 for row in rows),
        },
        "policy": {
            "same_requests_for_every_executable_row_within_task": True,
            "source_training_artifact_provenance_sha_validated": True,
            "target_actors_deployed_only_on_declared_target_robot": True,
            "all_independent_instances_qualified": True,
            "confirmation_states_opened": 0,
            "old_invalid_outcomes_used": False,
            "failure_replacement": False,
            "missing_actor_is_never_replaced": True,
        },
    }
    if execution is not None:
        plan["execution"] = execution
    return plan


def _mode_specific_aggregated_candidate(
    candidate: dict[str, Any], *, key: str, instances: tuple[str, ...],
) -> dict[str, Any]:
    retained_update = candidate.get("retained_update")
    expected_fields = {"method_identity", "aggregation_report"}
    if retained_update is not None:
        expected_fields.add("retained_update")
    if set(candidate) != expected_fields:
        raise ValueError(
            "mode-specific candidate requires its immutable aggregation report"
        )
    if retained_update is not None and key != "cross_ebdm":
        raise ValueError("only mode-specific transport can select a retained update")
    if instances != EXPECTED_INSTANCES or key not in {"cross_ebdm", "task_only"}:
        raise ValueError("mode-specific candidate requires all eight paired instances")
    report_path = _checked_pointer(
        candidate["aggregation_report"], "mode-specific aggregation report",
    )
    expected_root = (
        ROOT / "results/arm/keyed_insertion_recovery/training"
    ).resolve()
    if not report_path.resolve().is_relative_to(expected_root):
        raise ValueError("mode-specific aggregation is outside task-local training")
    report = json.loads(report_path.read_text())
    mode = report.get("mode")
    expected_identity = (
        f"ours_task_first_mode_specific_recorded_completion_v9_mode{mode}"
        if key == "cross_ebdm"
        else "task_only_shared_start_mode_specific_v9"
    )
    identity = report.get("training_identity", {})
    training = identity.get("training", {})
    expected_paired = {
        "linear_margin": 1e-5,
        "angular_margin": 1e-4,
        "weight": 1.0,
        "tail_fraction": 0.1,
        "tail_weight": 4.0,
        "requested_mode": mode,
    }
    if (
        candidate["method_identity"] != expected_identity
        or report.get("schema_version")
        != "t3-mode-specific-recorded-completion-aggregation-v1"
        or report.get("status") != "complete"
        or report.get("passed") is not True
        or report.get("selection_gate_passed") is not True
        or report.get("new_physics_calls") != 0
        or report.get("confirmation_states_opened") != 0
        or report.get("target_reward_or_truth_opened") is not False
        or mode not in {0, 1}
        or identity.get("version")
        != "t3-task-first-mode-specific-recorded-completion-training-v9"
        or identity.get("task") != "T3"
        or identity.get("selection", {}).get("required_modes") != [mode]
        or training.get("variant") != "task_first_ensemble_robust_anchored"
        or training.get("control_checkpoint_policy") != "frozen_selector"
        or training.get("paired_progress") != expected_paired
        or training.get("ensemble_task") != {"reduction": "worst_paired_member"}
        or training.get("target_sampling")
        != {"kind": "mode_specific", "modes": [mode]}
        or set(report.get("instances", {})) != set(EXPECTED_INSTANCES)
    ):
        raise ValueError("mode-specific aggregation did not pass its frozen contract")
    _checked_pointer(report["artifacts"]["runner"], "mode-specific aggregation runner")
    _checked_pointer(report["artifacts"]["config"], "mode-specific aggregation config")
    shared_path = _checked_pointer(
        training["shared_start_report"], "mode-specific shared task-only report",
    )
    shared = json.loads(shared_path.read_text())
    if (
        shared.get("status") != "complete"
        or shared.get("passed") is not True
        or shared.get("schema_version")
        != "t3-task-first-directional-task-only-training-v1"
        or shared.get("control_checkpoint_policy") != "frozen_selector"
        or shared.get("training_variant") != {
            "name": "task_first_directional_task_only",
            "initialization_policy": "task_only",
            "response_anchoring_weight": 0.0,
        }
        or set(shared.get("instances", {})) != set(EXPECTED_INSTANCES)
        or any(
            row.get("selection", {}).get("accepted") is not True
            for row in shared["instances"].values()
        )
    ):
        raise ValueError("mode-specific shared task-only report did not pass")
    source_rows = {}
    for source in report["artifacts"]["sources"]:
        _checked_pointer(source["plan"], "mode-specific source plan")
        source_report_path = _checked_pointer(
            source["report"], "mode-specific source report",
        )
        source_report = json.loads(source_report_path.read_text())
        overlap = set(source_rows) & set(source_report.get("instances", {}))
        if overlap:
            raise ValueError("mode-specific aggregation source instances overlap")
        source_rows.update(source_report.get("instances", {}))
    if set(source_rows) != set(EXPECTED_INSTANCES):
        raise ValueError("mode-specific aggregation sources are incomplete")

    checkpoints = {}
    task_only_checkpoints = {}
    selection_proof = {}
    pairing_proof = {}
    for instance in EXPECTED_INSTANCES:
        selected = report["instances"][instance]
        source = source_rows[instance]
        selected_pointer = selected["selected_checkpoint"]
        selected_update = selected["selected_update"]
        selected_checks = selected.get("checks", {})
        if retained_update is not None:
            retained = source.get("retained_validation_checkpoints", {})
            selected_pointer = retained.get(str(int(retained_update)))
            curve_matches = [
                point for point in source.get("curve", ())
                if point.get("update") == int(retained_update)
            ]
            if selected_pointer is None or len(curve_matches) != 1:
                raise ValueError(
                    f"mode-specific {instance} retained update is absent"
                )
            selected_update = int(retained_update)
            selected_checks = curve_matches[0].get("selection_checks", {})
        selected_path = _checked_pointer(
            selected_pointer, f"mode-specific {instance} selected checkpoint",
        )
        shared_pointer = shared["instances"][instance]["selected_checkpoint"]
        shared_checkpoint = _pointer(_checked_pointer(
            shared_pointer, f"mode-specific {instance} task-only checkpoint",
        ))
        initialization = source.get("initialization", {})
        if (
            source.get("status") != "complete"
            or source.get("selection", {}).get("accepted") is not True
            or retained_update is None and (
                source.get("selected_checkpoint") != selected["selected_checkpoint"]
                or source.get("selected_update") != selected["selected_update"]
            )
            or source.get("inputs", {}).get("task_only_checkpoint") != shared_checkpoint
            or source.get("control_checkpoint_policy") != "frozen_selector"
            or initialization.get("kind")
            != "recentered_current_checkpoint_fresh_optimizer"
            or initialization.get("source") != "task_only_checkpoint"
            or initialization.get("initial_checkpoint_matches_selected_base")
            is not True
            or initialization.get("optimizer_restored") is not False
            or initialization.get("steps_restored") != 0
        ):
            raise ValueError(
                f"mode-specific {instance} differs from its paired selected source"
            )
        checks = selected_checks
        if not checks or not all(value is True for value in checks.values()):
            raise ValueError(f"mode-specific {instance} did not pass every check")
        checkpoints[instance] = _pointer(selected_path)
        task_only_checkpoints[instance] = shared_checkpoint
        selection_proof[instance] = {
            "accepted": True,
            "checks": deepcopy(
                checks if key == "cross_ebdm"
                else shared["instances"][instance]["selection"]["checks"]
            ),
            "selected_update": (
                selected_update if key == "cross_ebdm"
                else shared["instances"][instance]["selected_update"]
            ),
            "selected": (
                checkpoints[instance] if key == "cross_ebdm"
                else shared_checkpoint
            ),
            "fallback": False,
        }
        pairing_proof[instance] = {
            "starting_checkpoint": shared_checkpoint,
            "seed": source.get("seed"),
            "updates": source.get("updates"),
            "sample_stream_sha256": source.get("sample_stream_sha256"),
            "initialization": {
                name: value for name, value in initialization.items()
                if name != "training_variant"
            },
        }
    if key == "task_only":
        checkpoints = task_only_checkpoints
    training_variant = (
        {
            "name": "task_first_ensemble_robust_anchored",
            "initialization_policy": "task_only",
            "response_anchoring_weight": 1.0,
        }
        if key == "cross_ebdm" else deepcopy(shared["training_variant"])
    )
    return {
        "method_identity": candidate["method_identity"],
        "training_report": _pointer(report_path),
        "mode": mode,
        "checkpoints": checkpoints,
        "task_only_checkpoints": task_only_checkpoints,
        "training_variant": training_variant,
        "control_checkpoint_policy": "frozen_selector",
        "resolved_shared_start_report": _pointer(shared_path),
        "data_sha256": hashlib.sha256(json.dumps(
            identity.get("data"), sort_keys=True,
        ).encode()).hexdigest(),
        "pairing_proof": pairing_proof,
        "paired_progress": expected_paired if key == "cross_ebdm" else None,
        "ensemble_task": (
            {"reduction": "worst_paired_member"}
            if key == "cross_ebdm" else None
        ),
        "checkpoint_selection": (
            {
                **deepcopy(training.get("checkpoint_selection", {})),
                **(
                    {"retained_update": int(retained_update)}
                    if retained_update is not None else {}
                ),
            }
            if key == "cross_ebdm" else None
        ),
        "selection_proof": selection_proof,
    }


def _mode_specific_physical_selected_candidate(
    candidate: dict[str, Any], *, instances: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve the mode-one checkpoints chosen by the held-out physics selector."""
    if set(candidate) != {
        "method_identity", "aggregation_report", "physical_selection_report",
    }:
        raise ValueError(
            "mode-specific physical selection requires aggregation and selector reports"
        )
    if instances != EXPECTED_INSTANCES:
        raise ValueError("mode-specific physical selection requires eight instances")
    expected_identity = "ours_task_first_mode_specific_physical_selected_v19_mode1"
    if candidate["method_identity"] != expected_identity:
        raise ValueError("mode-specific physical-selected identity changed")
    report_path = _checked_pointer(
        candidate["physical_selection_report"],
        "mode-specific physical checkpoint selection report",
    )
    report = json.loads(report_path.read_text())
    selection = report.get("physical_checkpoint_selection", {})
    if (
        report.get("schema_version")
        != MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA
        or report.get("status")
        != "complete_recorded_physical_checkpoint_selection"
        or report.get("passed") is not True
        or report.get("diagnostic_only") is not True
        or report.get("performance_evidence") is not False
        or report.get("confirmation_states_opened") != 0
        or report.get("counts", {}).get("execution_errors") != 0
        or selection.get("selection_complete") is not True
        or set(selection.get("instances", {})) != set(EXPECTED_INSTANCES)
    ):
        raise ValueError("mode-specific physical selector report is not admitted")
    selection_device = report.get("device", {})
    if (
        selection_device.get("cuda_visible_devices") not in {"0", "1"}
        or selection_device.get("physical_index") not in {0, 1}
        or str(selection_device["physical_index"])
        != selection_device["cuda_visible_devices"]
        or selection_device.get("physical_uuid")
        != {0: GPU0_UUID, 1: GPU1_UUID}[selection_device["physical_index"]]
    ):
        raise ValueError("mode-specific physical selector device changed")
    selector_rows = {
        row["key"]: row for row in report.get("rows", ())
        if row["key"].startswith("mode_specific_selector_update_")
    }
    task_rows = [
        row for row in report.get("rows", ())
        if row["key"] == "mode_specific_selector_task_only"
    ]
    if set(selector_rows) != {
        f"mode_specific_selector_update_{update:04d}"
        for update in (1200, 1300, 1400, 1500)
    } or len(task_rows) != 1:
        raise ValueError("mode-specific physical selector source rows changed")
    aggregation_path = _checked_pointer(
        candidate["aggregation_report"], "mode-specific mode-one aggregation report",
    )
    base = _mode_specific_aggregated_candidate(
        {
            "method_identity": "ours_task_first_mode_specific_recorded_completion_v9_mode1",
            "aggregation_report": _pointer(aggregation_path),
        },
        key="cross_ebdm", instances=instances,
    )
    task_source = task_rows[0]["candidate_policy_set"]
    for source in [
        task_source,
        *(row["candidate_policy_set"] for row in selector_rows.values()),
    ]:
        if (
            source.get("training_report") != base["training_report"]
            or source.get("mode") != 1
            or source.get("resolved_shared_start_report")
            != base["resolved_shared_start_report"]
            or source.get("data_sha256") != base["data_sha256"]
        ):
            raise ValueError("mode-specific selector provenance changed")
    checkpoints = {}
    proof = {}
    for instance in EXPECTED_INSTANCES:
        selected = selection["instances"][instance]
        update = int(selected["selected_update"])
        pointer_value = selected["selected_checkpoint"]
        if update == 0:
            source = task_source
        else:
            source = selector_rows[
                f"mode_specific_selector_update_{update:04d}"
            ]["candidate_policy_set"]
        if (
            source["checkpoints"][instance] != pointer_value
            or not selected.get("selected_metrics", {}).get("eligible", False)
        ):
            raise ValueError(
                f"mode-specific physical-selected {instance} provenance changed"
            )
        checkpoints[instance] = _pointer(_checked_pointer(
            pointer_value, f"mode-specific physical-selected {instance} checkpoint",
        ))
        proof[instance] = {
            "selected_update": update,
            "selected_checkpoint": checkpoints[instance],
            "selected_metrics": deepcopy(selected["selected_metrics"]),
        }
    base.update({
        "method_identity": expected_identity,
        "checkpoints": checkpoints,
        "physical_selection_report": _pointer(report_path),
        "physical_selection_device": deepcopy(selection_device),
        "physical_selection_proof": proof,
    })
    return base


def _mode_specific_mode_zero_physical_selected_candidate(
    candidate: dict[str, Any], *, instances: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve mode-zero checkpoints chosen on full held-out episodes."""
    if set(candidate) != {
        "method_identity", "aggregation_report", "physical_selection_report",
    }:
        raise ValueError("mode-zero physical selection requires immutable reports")
    expected_identity = (
        "ours_task_first_mode_specific_dual_physical_selected_v21_mode0"
    )
    if candidate["method_identity"] != expected_identity:
        raise ValueError("mode-zero physical-selected identity changed")
    report_path = _checked_pointer(
        candidate["physical_selection_report"],
        "mode-zero physical checkpoint selection report",
    )
    report = json.loads(report_path.read_text())
    selection = report.get("physical_checkpoint_selection", {})
    if (
        report.get("schema_version")
        != MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA
        or report.get("status")
        != "complete_recorded_physical_checkpoint_selection"
        or report.get("passed") is not True
        or report.get("diagnostic_only") is not True
        or report.get("performance_evidence") is not False
        or report.get("confirmation_states_opened") != 0
        or report.get("counts", {}).get("execution_errors") != 0
        or selection.get("selection_complete") is not True
        or set(selection.get("instances", {})) != set(EXPECTED_INSTANCES)
    ):
        raise ValueError("mode-zero physical selector report is not admitted")
    selection_device = report.get("device", {})
    if (
        selection_device.get("cuda_visible_devices") not in {"0", "1"}
        or selection_device.get("physical_index") not in {0, 1}
        or str(selection_device["physical_index"])
        != selection_device["cuda_visible_devices"]
        or selection_device.get("physical_uuid")
        != {0: GPU0_UUID, 1: GPU1_UUID}[selection_device["physical_index"]]
    ):
        raise ValueError("mode-zero physical selector device changed")
    selector_rows = {
        row["key"]: row for row in report.get("rows", ())
        if row["key"].startswith("mode_zero_selector_update_")
    }
    task_rows = [
        row for row in report.get("rows", ())
        if row["key"] == "mode_zero_selector_task_only"
    ]
    if set(selector_rows) != {
        f"mode_zero_selector_update_{update:04d}"
        for update in (1100, 1400, 1500)
    } or len(task_rows) != 1:
        raise ValueError("mode-zero physical selector source rows changed")
    aggregation_path = _checked_pointer(
        candidate["aggregation_report"], "mode-zero aggregation report",
    )
    base = _mode_specific_aggregated_candidate({
        "method_identity": "ours_task_first_mode_specific_recorded_completion_v9_mode0",
        "aggregation_report": _pointer(aggregation_path),
    }, key="cross_ebdm", instances=instances)
    task_source = task_rows[0]["candidate_policy_set"]
    for source in [
        task_source,
        *(row["candidate_policy_set"] for row in selector_rows.values()),
    ]:
        if (
            source.get("training_report") != base["training_report"]
            or source.get("mode") != 0
            or source.get("resolved_shared_start_report")
            != base["resolved_shared_start_report"]
            or source.get("data_sha256") != base["data_sha256"]
        ):
            raise ValueError("mode-zero selector provenance changed")
    checkpoints, proof = {}, {}
    for instance in EXPECTED_INSTANCES:
        selected = selection["instances"][instance]
        update = int(selected["selected_update"])
        pointer_value = selected["selected_checkpoint"]
        source = (
            task_source if update == 0 else selector_rows[
                f"mode_zero_selector_update_{update:04d}"
            ]["candidate_policy_set"]
        )
        if (
            source["checkpoints"][instance] != pointer_value
            or not selected.get("selected_metrics", {}).get("eligible", False)
        ):
            raise ValueError(f"mode-zero selected {instance} provenance changed")
        checkpoints[instance] = _pointer(_checked_pointer(
            pointer_value, f"mode-zero physical-selected {instance} checkpoint",
        ))
        proof[instance] = {
            "selected_update": update,
            "selected_checkpoint": checkpoints[instance],
            "selected_metrics": deepcopy(selected["selected_metrics"]),
        }
    base.update({
        "method_identity": expected_identity,
        "checkpoints": checkpoints,
        "mode_zero_physical_selection_report": _pointer(report_path),
        "mode_zero_physical_selection_device": deepcopy(selection_device),
        "mode_zero_physical_selection_proof": proof,
    })
    return base


def _mode_specific_success_safe_selected_candidate(
    candidate: dict[str, Any], *, instances: tuple[str, ...],
) -> dict[str, Any]:
    """Apply a frozen success/time safety fallback on a second dev cohort."""
    if set(candidate) != {
        "method_identity", "aggregation_report", "physical_selection_report",
        "safety_calibration_report",
    }:
        raise ValueError("success-safe selection requires all immutable reports")
    expected_identity = (
        "ours_task_first_mode_specific_success_safe_selected_v20_mode1"
    )
    if candidate["method_identity"] != expected_identity:
        raise ValueError("success-safe selected identity changed")
    base = _mode_specific_physical_selected_candidate({
        "method_identity": "ours_task_first_mode_specific_physical_selected_v19_mode1",
        "aggregation_report": candidate["aggregation_report"],
        "physical_selection_report": candidate["physical_selection_report"],
    }, instances=instances)
    report_path = _checked_pointer(
        candidate["safety_calibration_report"],
        "mode-specific success safety calibration report",
    )
    report = json.loads(report_path.read_text())
    if (
        report.get("schema_version")
        != MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA
        or report.get("status")
        != "complete_recorded_development_contribution_contrast"
        or not isinstance(report.get("passed"), bool)
        or report.get("diagnostic_only") is not True
        or report.get("performance_evidence") is not False
        or report.get("confirmation_states_opened") != 0
        or report.get("counts", {}).get("execution_errors") != 0
    ):
        raise ValueError("success safety calibration report is not admitted")
    safety_device = report.get("device", {})
    if safety_device != base["physical_selection_device"]:
        raise ValueError("success safety calibration changed physical device")
    progress_path = _checked_pointer(
        report["artifacts"]["progress"], "success safety calibration progress",
    )
    plan_path = _checked_pointer(
        report["artifacts"]["plan"], "success safety calibration plan",
    )
    calibration_plan = json.loads(plan_path.read_text())
    if calibration_plan.get("schema_version") != report["schema_version"]:
        raise ValueError("success safety calibration plan changed")
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    expected_keys = {
        "mode_specific_physical_selected_transport",
        "mode_specific_physical_selected_task_only",
    }
    mode_one = [row for row in progress if row["request"]["requested_mode"] == 1]
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in mode_one:
        pair = (row["instance"], row["request"]["request_id"])
        if row["key"] in paired[pair]:
            raise ValueError("duplicate success safety calibration result")
        paired[pair][row["key"]] = row
    task_checkpoints = base["task_only_checkpoints"]
    calibrated = {}
    proof = {}
    for instance in EXPECTED_INSTANCES:
        rows = [value for (owner, _), value in paired.items() if owner == instance]
        complete = len(rows) == 4 and all(set(value) == expected_keys for value in rows)
        transport = [
            value["mode_specific_physical_selected_transport"] for value in rows
        ]
        task_only = [
            value["mode_specific_physical_selected_task_only"] for value in rows
        ]
        integrity = complete and all(
            row["outcome"] != "reset_or_execution_error"
            and row.get("evidence", {}).get("audit_passed", False)
            for row in [*transport, *task_only]
        )
        preserves = all(
            not task["passed"] or candidate_row["passed"]
            for candidate_row, task in zip(transport, task_only, strict=True)
        )
        extra_constraints = sum(
            bool(candidate_row["constraint_violation"])
            and not bool(task["constraint_violation"])
            for candidate_row, task in zip(transport, task_only, strict=True)
        )
        deltas = [
            task["physical_control_steps"] - candidate_row["physical_control_steps"]
            for candidate_row, task in zip(transport, task_only, strict=True)
            if candidate_row["passed"] and task["passed"]
        ]
        nonnegative_time = bool(deltas) and float(np.mean(deltas)) >= 0.
        retained = bool(
            integrity and preserves and extra_constraints == 0 and nonnegative_time
        )
        calibrated[instance] = (
            base["checkpoints"][instance]
            if retained else task_checkpoints[instance]
        )
        proof[instance] = {
            "complete": complete,
            "integrity_passed": integrity,
            "task_only_successes_preserved": preserves,
            "extra_constraint_violations": int(extra_constraints),
            "paired_successes_for_time": len(deltas),
            "mean_control_step_reduction": (
                float(np.mean(deltas)) if deltas else None
            ),
            "retained_transport_checkpoint": retained,
            "selected_checkpoint": calibrated[instance],
        }
    if set(paired) != {
        (instance, request["request_id"])
        for instance in EXPECTED_INSTANCES
        for request in calibration_plan["requests"]
        if request["requested_mode"] == 1
    }:
        raise ValueError("success safety calibration denominator changed")
    base.update({
        "method_identity": expected_identity,
        "checkpoints": calibrated,
        "safety_calibration_report": _pointer(report_path),
        "safety_calibration_plan": _pointer(plan_path),
        "safety_calibration_device": deepcopy(safety_device),
        "success_safety_proof": proof,
    })
    return base


def _mode_one_authority_calibration(value: dict[str, Any]) -> dict[str, Any]:
    """Select mode-one authority from immutable paired development cohorts."""
    expected_rule = {
        "requested_mode": 1,
        "reset_slots_per_source": 4,
        "require_both_methods_to_succeed_on_every_pair": True,
        "require_zero_constraint_violations": True,
        "require_nonnegative_control_step_delta_on_every_pair": True,
        "selection": "largest_mean_control_step_reduction_then_smallest_blend",
        "fallback_blend": 0,
    }
    if not isinstance(value, dict) or set(value) != {
        "version", "rule", "sources", "selected_mode_one_max_blend_by_instance",
    }:
        raise ValueError("mode-one authority calibration fields changed")
    if (
        value["version"] != "t3-mode-one-paired-authority-calibration-v1"
        or value["rule"] != expected_rule
        or set(value["sources"]) != {"blend_1", "blend_2", "blend_8"}
    ):
        raise ValueError("mode-one authority calibration contract changed")
    expected_sources = {
        "blend_1": (
            1., MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
            "public_mode_specific_supported_phase_convex_dominance_v16",
        ),
        "blend_2": (
            2., MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
            "public_mode_specific_supported_phase_capped_expansion_dominance_v17",
        ),
        "blend_8": (
            8., MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
            "public_mode_specific_recorded_completion_dominance_v15",
        ),
    }
    per_instance: dict[str, list[dict[str, Any]]] = {
        instance: [{
            "maximum_blend": 0., "eligible": True,
            "paired_control_step_deltas": [0, 0, 0, 0],
            "mean_control_step_reduction": 0.,
            "source": "task_only_fallback",
        }]
        for instance in EXPECTED_INSTANCES
    }
    source_proof = {}
    seed_sets = []
    for source_name, (blend, expected_schema, expected_version) in expected_sources.items():
        report_path = _checked_pointer(
            value["sources"][source_name], f"{source_name} authority report",
        )
        report = json.loads(report_path.read_text())
        if (
            report.get("schema_version") != expected_schema
            or report.get("status")
            != "complete_recorded_development_contribution_contrast"
            or report.get("diagnostic_only") is not True
            or report.get("performance_evidence") is not False
            or report.get("confirmation_states_opened") != 0
            or report.get("failure_replacement") is not False
            or report.get("counts", {}).get("planned_physical_jobs") != 128
            or report.get("counts", {}).get("completed_physical_jobs") != 128
            or report.get("counts", {}).get("execution_errors") != 0
            or report.get("contribution_contrast", {}).get("execution_complete")
            is not True
            or report.get("contribution_contrast", {}).get("integrity_passed")
            is not True
        ):
            raise ValueError(f"{source_name} is not a complete development cohort")
        rows = {row["key"]: row for row in report.get("rows", ())}
        transport_labels = [
            key for key in rows if key.endswith("_transport")
        ]
        task_labels = [
            key for key in rows if key.endswith("_task_only")
        ]
        if len(transport_labels) != 1 or len(task_labels) != 1:
            raise ValueError(f"{source_name} paired labels changed")
        transport_label, task_label = transport_labels[0], task_labels[0]
        admission = rows[transport_label].get("execution_admission", {})
        if admission.get("version") != expected_version:
            raise ValueError(f"{source_name} execution authority changed")
        plan_path = _checked_pointer(
            report["artifacts"]["plan"], f"{source_name} authority plan",
        )
        progress_path = _checked_pointer(
            report["artifacts"]["progress"], f"{source_name} authority ledger",
        )
        plan = json.loads(plan_path.read_text())
        seeds = {int(request["reset_seed"]) for request in plan.get("requests", ())}
        if (
            plan.get("schema_version") != expected_schema
            or len(seeds) != 8
            or plan.get("acceptance", {}).get("transport") != transport_label
            or plan.get("acceptance", {}).get("task_only") != task_label
        ):
            raise ValueError(f"{source_name} plan differs from its completed report")
        seed_sets.append(seeds)
        records = [
            json.loads(line) for line in progress_path.read_text().splitlines()
            if line.strip()
        ]
        if len(records) != 128 or len({row["record_id"] for row in records}) != 128:
            raise ValueError(f"{source_name} authority ledger is incomplete")
        indexed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
            transport_label: {}, task_label: {},
        }
        for row in records:
            if row.get("key") not in indexed:
                raise ValueError(f"{source_name} authority ledger has an unknown row")
            key = (row["instance"], row["request"]["request_id"])
            if key in indexed[row["key"]]:
                raise ValueError(f"{source_name} authority ledger duplicated a pair")
            if (
                row.get("evidence", {}).get("audit_passed") is not True
                or row.get("outcome") == "reset_or_execution_error"
            ):
                raise ValueError(f"{source_name} authority ledger failed integrity")
            indexed[row["key"]][key] = row
        for instance in EXPECTED_INSTANCES:
            transport = {
                key: row for key, row in indexed[transport_label].items()
                if key[0] == instance and row["request"]["requested_mode"] == 1
            }
            task_only = {
                key: row for key, row in indexed[task_label].items()
                if key[0] == instance and row["request"]["requested_mode"] == 1
            }
            paired = sorted(set(transport) & set(task_only))
            if set(transport) != set(task_only) or len(paired) != 4:
                raise ValueError(f"{source_name} {instance} mode-one pairs changed")
            deltas = [
                int(task_only[key]["physical_control_steps"])
                - int(transport[key]["physical_control_steps"])
                for key in paired
            ]
            eligible = bool(
                all(transport[key]["passed"] and task_only[key]["passed"] for key in paired)
                and not any(
                    transport[key]["constraint_violation"]
                    or task_only[key]["constraint_violation"]
                    for key in paired
                )
                and all(delta >= 0 for delta in deltas)
            )
            per_instance[instance].append({
                "maximum_blend": blend,
                "eligible": eligible,
                "paired_control_step_deltas": deltas,
                "mean_control_step_reduction": float(np.mean(deltas)),
                "source": source_name,
            })
        source_proof[source_name] = {
            "report": _pointer(report_path),
            "plan": _pointer(plan_path),
            "progress": _pointer(progress_path),
            "maximum_blend": blend,
        }
    if any(left & right for index, left in enumerate(seed_sets)
           for right in seed_sets[index + 1:]):
        raise ValueError("authority calibration development seeds overlap")
    selected = {}
    instance_proof = {}
    for instance, candidates in per_instance.items():
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        choice = max(
            eligible,
            key=lambda candidate: (
                candidate["mean_control_step_reduction"],
                -candidate["maximum_blend"],
            ),
        )
        selected[instance] = choice["maximum_blend"]
        instance_proof[instance] = {
            "candidates": candidates,
            "selected": deepcopy(choice),
        }
    declared = value["selected_mode_one_max_blend_by_instance"]
    if set(declared) != set(EXPECTED_INSTANCES) or any(
        float(declared[instance]) != selected[instance]
        for instance in EXPECTED_INSTANCES
    ):
        raise ValueError("declared mode-one authority differs from frozen rule")
    return {
        "version": value["version"],
        "rule": deepcopy(expected_rule),
        "sources": source_proof,
        "selected_mode_one_max_blend_by_instance": selected,
        "instances": instance_proof,
    }


def _candidate_policy_set(
    method_spec: dict[str, Any], *, schema: str, task: str,
    family: str, key: str, instances: tuple[str, ...],
) -> dict[str, Any] | None:
    candidate = method_spec.get("candidate_policy_set")
    if candidate is None:
        return None
    if schema in {
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
    }:
        if task != "T3" or family != "main":
            raise ValueError("mode-specific candidate is outside T3 main methods")
        if (
            schema in {
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }
            and key == "cross_ebdm"
            and "physical_selection_report" in candidate
            and "safety_calibration_report" not in candidate
        ):
            return _mode_specific_mode_zero_physical_selected_candidate(
                candidate, instances=instances,
            )
        if (
            schema in {
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }
            and key == "cross_ebdm"
            and "safety_calibration_report" in candidate
        ):
            return _mode_specific_success_safe_selected_candidate(
                candidate, instances=instances,
            )
        if (
            schema in {
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }
            and key == "cross_ebdm"
            and "physical_selection_report" in candidate
        ):
            return _mode_specific_physical_selected_candidate(
                candidate, instances=instances,
            )
        return _mode_specific_aggregated_candidate(
            candidate, key=key, instances=instances,
        )
    if (
        schema in {
            PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
            ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        }
        and key == "cross_ebdm"
    ):
        robust_trained = schema == ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA
        if set(candidate) != {"method_identity", "physical_selection_report"}:
            raise ValueError(
                "physical-selected candidate requires its immutable selector report"
            )
        expected_identity = (
            "ours_task_first_ensemble_robust_selected_v1"
            if robust_trained else "ours_task_first_physical_selected_v1"
        )
        if candidate["method_identity"] != expected_identity:
            raise ValueError("physical-selected candidate identity changed")
        report_path = _checked_pointer(
            candidate["physical_selection_report"],
            "physical checkpoint selection report",
        )
        expected_root = (
            ROOT / "results/arm/keyed_insertion_recovery/validation"
        ).resolve()
        if not report_path.resolve().is_relative_to(expected_root):
            raise ValueError("physical selector report is outside task-local validation")
        report = json.loads(report_path.read_text())
        selection = report.get("physical_checkpoint_selection", {})
        if (
            report.get("schema_version") != (
                ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA
                if robust_trained else PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA
            )
            or report.get("status")
            != "complete_recorded_physical_checkpoint_selection"
            or report.get("passed") is not True
            or report.get("diagnostic_only") is not True
            or report.get("performance_evidence") is not False
            or report.get("confirmation_states_opened") != 0
            or report.get("counts", {}).get("execution_errors") != 0
            or selection.get("selection_complete") is not True
            or set(selection.get("instances", {})) != set(EXPECTED_INSTANCES)
        ):
            raise ValueError("physical selector report is not an admitted development selector")
        selector_prefix = (
            "ensemble_robust_selector_update_"
            if robust_trained else "physical_selector_update_"
        )
        source_rows = [
            row for row in report.get("rows", ())
            if row["key"].startswith(selector_prefix)
        ]
        if len(source_rows) != 6:
            raise ValueError("physical selector source rows changed")
        source = source_rows[0]["candidate_policy_set"]
        immutable_fields = {
            name: source.get(name) for name in (
                "training_variant", "control_checkpoint_policy",
                "resolved_base_training_config", "resolved_shared_start_report",
                "data_sha256", "pairing_proof", "paired_progress",
                "checkpoint_selection", "ensemble_task",
            )
        }
        if any(
            any(row["candidate_policy_set"].get(name) != value
                for name, value in immutable_fields.items())
            for row in source_rows
        ):
            raise ValueError("physical selector candidate provenance differs by update")
        checkpoints = {}
        proof = {}
        for instance, row in selection["instances"].items():
            pointer_value = row.get("selected_checkpoint")
            path = _checked_pointer(
                pointer_value, f"physical-selected {instance} checkpoint",
            )
            update = int(row["selected_update"])
            label = f"{selector_prefix}{update:04d}"
            source_row = next(
                (item for item in source_rows if item["key"] == label), None,
            )
            if (
                source_row is None
                or source_row["candidate_policy_set"]["checkpoints"][instance]
                != pointer_value
                or not row.get("selected_metrics", {}).get("eligible", False)
            ):
                raise ValueError(f"physical-selected {instance} provenance changed")
            checkpoints[instance] = _pointer(path)
            proof[instance] = {
                "selected_update": update,
                "selected_metrics": deepcopy(row["selected_metrics"]),
                "selected_checkpoint": checkpoints[instance],
            }
        return {
            "method_identity": candidate["method_identity"],
            "physical_selection_report": _pointer(report_path),
            "checkpoints": checkpoints,
            "physical_selection_proof": proof,
            **deepcopy(immutable_fields),
        }
    if schema not in {
        DIAGNOSTIC_QUALIFICATION_SCHEMA, DIAGNOSTIC_QUALIFICATION_SCHEMA_V3,
        CONTRIBUTION_CONTRAST_SCHEMA, SAME_START_CONTRIBUTION_SCHEMA,
        RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        PUBLIC_LOG_SCHEMA,
    }:
        raise ValueError("candidate policy sets require paired qualification v2")
    allowed_key = (
        family == "main"
        and key == "cross_ebdm"
        or schema in {
            CONTRIBUTION_CONTRAST_SCHEMA, SAME_START_CONTRIBUTION_SCHEMA,
            RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
            TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
            TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
            RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
            FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
            MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
            ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
            ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
            PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
            ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        }
        and family == "main"
        and key == "task_only"
    )
    if task != "T3" or not allowed_key:
        raise ValueError("candidate policy set is outside the T3 contribution scope")
    if instances != EXPECTED_INSTANCES:
        raise ValueError("candidate policy qualification requires all eight instances")
    retained_selector = (
        schema in {
            PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
            ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
        }
        and key == "cross_ebdm"
    )
    expected_candidate_keys = {"method_identity", "training_report"}
    if retained_selector:
        expected_candidate_keys.add("retained_update")
    if set(candidate) != expected_candidate_keys:
        raise ValueError(
            "candidate_policy_set fields do not match the frozen schema"
        )
    identity = str(candidate["method_identity"])
    report_path = _checked_pointer(
        candidate["training_report"], "candidate training report",
    )
    expected_root = (
        ROOT / "results/arm/keyed_insertion_recovery/training"
    ).resolve()
    if not report_path.resolve().is_relative_to(expected_root):
        raise ValueError("candidate training report is outside task-local training results")
    report = json.loads(report_path.read_text())
    training_schema = report.get("schema_version")
    expected_identity = {
        "t3-finite-time-terminal-progress-training-v3": "ours_gated_v3_finite_time",
        "t3-finite-time-terminal-progress-training-v4": "ours_gated_v4_terminal_stable",
        "t3-finite-time-terminal-progress-training-v5": "ours_gated_v5_late_phase",
        "t3-finite-time-terminal-progress-matched-task-only-training-v1": (
            "task_only_v1_late_phase_matched"
        ),
        "t3-finite-time-same-start-anchored-training-v1": (
            "ours_same_start_anchored_v1"
        ),
        "t3-finite-time-same-start-task-only-training-v1": (
            "same_start_task_only_v1"
        ),
        "t3-finite-time-same-start-relative-anchored-training-v1": (
            "ours_relative_response_v1"
        ),
        "t3-finite-time-same-start-relative-task-only-training-v1": (
            "relative_response_task_only_v1"
        ),
        "t3-task-first-directional-anchored-training-v1": (
            "ours_task_first_directional_v1"
        ),
        "t3-task-first-directional-same-state-anchored-training-v1": (
            "ours_task_first_directional_same_state_v1"
        ),
        "t3-task-first-directional-task-only-training-v1": (
            "task_first_directional_task_only_v1"
        ),
        "t3-task-first-directional-same-state-task-only-training-v1": (
            "task_first_directional_same_state_task_only_v1"
        ),
        "t3-task-first-margin-anchored-training-v1": (
            "ours_task_first_margin_v1"
        ),
        "t3-task-first-margin-task-only-training-v1": (
            "task_first_margin_task_only_v1"
        ),
        "t3-task-first-stable-margin-anchored-training-v2": (
            "ours_task_first_stable_margin_v2"
        ),
        "t3-task-first-physical-selector-anchored-training-v3": (
            "ours_task_first_physical_selector_v3"
        ),
        "t3-task-first-ensemble-robust-anchored-training-v4": (
            "ours_task_first_ensemble_robust_v4"
        ),
    }.get(training_schema)
    if identity != expected_identity:
        raise ValueError("candidate identity does not match its training report")
    if (
        expected_identity is None
        or report.get("status") != "complete"
        or report.get("passed") is not True
        or report.get("confirmation_states_opened") != 0
        or report.get("target_reward_or_truth_opened") is not False
        or report.get("new_physics_calls") != 0
    ):
        raise ValueError("candidate training report is not an admitted zero-physics run")
    variant_name = report.get("training_variant", {}).get("name")
    same_start_schema = schema in {
        SAME_START_CONTRIBUTION_SCHEMA, RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
    }
    if same_start_schema:
        expected_variant = {
            SAME_START_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "same_start_anchored",
                "task_only": "same_start_task_only",
            },
            RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "same_start_relative_anchored",
                "task_only": "same_start_relative_task_only",
            },
            TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_directional_anchored",
                "task_only": "task_first_directional_task_only",
            },
            TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_directional_same_state_anchored",
                "task_only": "task_first_directional_task_only",
            },
            RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_directional_anchored",
                "task_only": "task_first_directional_task_only",
            },
            FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_directional_anchored",
                "task_only": "task_first_directional_task_only",
            },
            MARGIN_RUNTIME_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_stable_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_margin_anchored",
                "task_only": "task_first_margin_task_only",
            },
            PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA: {
                "cross_ebdm": "task_first_ensemble_robust_anchored",
                "task_only": "task_first_margin_task_only",
            },
            PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA: {
                "cross_ebdm": "task_first_physical_selector_anchored",
                "task_only": "task_first_margin_task_only",
            },
            ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA: {
                "cross_ebdm": "task_first_ensemble_robust_anchored",
                "task_only": "task_first_margin_task_only",
            },
        }[schema][key]
        expected_control_policy = (
            "frozen_selector"
            if schema in {
                TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
                PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
            }
            else "terminal_update"
        )
        if (
            variant_name != expected_variant
            or report.get("control_checkpoint_policy") != expected_control_policy
        ):
            raise ValueError("same-start candidate changed its isolated training factor")
        if (
            schema == RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA
            and report.get("response_normalization") != {"reference_floor": 0.0001}
        ):
            raise ValueError("relative-response candidate changed its normalization")
        if (
            schema in {
                TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
                PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
            }
            and report.get("response_normalization") != {
                "reference_floor": 0.0001,
                "minimum_source_task_advantage": 0.0,
                **(
                    {"source_task_state": "target"}
                    if schema == TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA
                    else {}
                ),
            }
        ):
            raise ValueError("task-first candidate changed its directional response gate")
    elif key == "task_only" and variant_name != "matched_task_only":
        raise ValueError("matched task-only candidate retained response anchoring")
    trained_device = report.get("device", {})
    if (
        trained_device.get("process_visible_devices") != "1"
        or str(trained_device.get("physical_uuid", "")).lower()
        != GPU1_UUID.removeprefix("GPU-").lower()
    ):
        raise ValueError("candidate was not trained on the frozen physical GPU1")
    trained_instances = report.get("instances", {})
    if set(trained_instances) != set(EXPECTED_INSTANCES):
        raise ValueError("candidate training report does not contain eight instances")
    checkpoints = {}
    pairing_proof = {}
    selection_proof = {}
    for instance in EXPECTED_INSTANCES:
        row = trained_instances[instance]
        same_start = same_start_schema
        if row.get("status") != "complete" or (
            not same_start and not row.get("selection", {}).get("accepted", False)
        ):
            raise ValueError(f"candidate {instance} was not selected by the frozen gate")
        if same_start:
            initialization = row.get("initialization", {})
            expected_control_policy = (
                "frozen_selector"
                if schema in {
                    TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                    TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                    RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                    FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                    MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                    ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                    ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
                    PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                    MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                    ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
                    ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
                }
                else "terminal_update"
            )
            if (
                row.get("control_checkpoint_policy") != expected_control_policy
                or initialization.get("kind")
                != "recentered_current_checkpoint_fresh_optimizer"
                or initialization.get("source") != "task_only_checkpoint"
                or initialization.get("optimizer_restored") is not False
                or initialization.get("steps_restored") != 0
                or initialization.get("initial_checkpoint_matches_selected_base")
                is not True
            ):
                raise ValueError(f"candidate {instance} violated same-start initialization")
            if schema in {
                RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
                TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
            }:
                if retained_selector:
                    update = int(candidate["retained_update"])
                    retained = row.get(
                        "retained_validation_checkpoints", {}
                    )
                    matches = [
                        point for point in row.get("curve", ())
                        if point.get("update") == update
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"candidate {instance} retained update is absent"
                        )
                    checks = matches[0].get("selection_checks", {})
                    accepted = bool(checks) and all(checks.values())
                    pointer_value = retained.get(str(update))
                    if pointer_value is None:
                        raise ValueError(
                            f"candidate {instance} retained checkpoint is absent"
                        )
                else:
                    pointer_value = row.get("selected_checkpoint")
                    accepted = row.get("selection", {}).get(
                        "accepted", False,
                    )
                    checks = row.get("selection", {}).get("checks")
                if (
                    schema in {
                        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
                        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
                    }
                    and not accepted
                ):
                    raise ValueError(
                        f"task-first candidate {instance} failed its frozen selector"
                    )
                if not accepted and pointer_value != row.get("inputs", {}).get(
                    "task_only_checkpoint"
                ):
                    raise ValueError(
                        f"candidate {instance} did not use its frozen task-only fallback"
                    )
                selection_proof[instance] = {
                    "accepted": bool(accepted),
                    "checks": checks,
                    "selected": pointer_value,
                    "fallback": not bool(accepted),
                }
            else:
                pointer_value = row.get("control_checkpoint")
            pairing_proof[instance] = {
                "starting_checkpoint": row.get("inputs", {}).get(
                    "task_only_checkpoint"
                ),
                "seed": row.get("seed"),
                "updates": row.get("updates"),
                "sample_stream_sha256": row.get("sample_stream_sha256"),
                "initialization": {
                    name: value for name, value in initialization.items()
                    if name != "training_variant"
                },
            }
        else:
            pointer_value = row.get("selected_checkpoint")
        path = _checked_pointer(
            pointer_value, f"candidate {instance} checkpoint",
        )
        checkpoints[instance] = _pointer(path)
    return {
        "method_identity": identity,
        "training_report": _pointer(report_path),
        "checkpoints": checkpoints,
        **(
            {
                "training_variant": report["training_variant"],
                "control_checkpoint_policy": report["control_checkpoint_policy"],
                "resolved_base_training_config": report.get(
                    "resolved_base_training_config"
                ),
                "resolved_shared_start_report": report.get(
                    "resolved_shared_start_report"
                ),
                "data_sha256": hashlib.sha256(json.dumps(
                    report.get("data"), sort_keys=True,
                ).encode()).hexdigest(),
                "pairing_proof": pairing_proof,
                "paired_progress": report.get("paired_progress"),
                "ensemble_task": report.get("ensemble_task"),
                "checkpoint_selection": report.get("instances", {}).get(
                    "instance_01", {}
                ).get("checkpoint_selection"),
                **(
                    {"retained_update": int(candidate["retained_update"])}
                    if retained_selector else {}
                ),
                **(
                    {"selection_proof": selection_proof}
                    if schema in {
                        RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
                        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
                        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
                        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
                        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
                    } else {}
                ),
            }
            if same_start_schema else {}
        ),
    }


def build_diagnostic_plan(spec_path: Path) -> dict[str, Any]:
    """Expand one task-local bounded diagnosis into the existing job schema."""
    spec_path = spec_path if spec_path.is_absolute() else ROOT / spec_path
    spec = json.loads(spec_path.read_text())
    schema = spec.get("version")
    if schema not in {
        *DIAGNOSTIC_SCHEMAS, DIAGNOSTIC_QUALIFICATION_SCHEMA,
        DIAGNOSTIC_QUALIFICATION_SCHEMA_V3, CONTRIBUTION_CONTRAST_SCHEMA,
        SAME_START_CONTRIBUTION_SCHEMA,
        RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        PUBLIC_LOG_SCHEMA,
        CONSTRAINT_STRESS_SCHEMA,
    }:
        raise ValueError("unsupported paired diagnostic schema")
    task = spec.get("task")
    if task not in {"T2", "T3"}:
        raise ValueError("paired diagnosis is currently bounded to T2/T3")
    confirmation = schema in {
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
    }
    if confirmation:
        if (
            spec.get("diagnostic_only") is not False
            or spec.get("performance_evidence") is not True
            or spec.get("confirmation_states_opened") != 1
            or spec.get("failure_replacement") is not False
        ):
            raise ValueError("transport confirmation must be one-shot formal evidence")
    else:
        if not spec.get("diagnostic_only") or spec.get("performance_evidence") is not False:
            raise ValueError("paired diagnosis must be development-only evidence")
        if spec.get("confirmation_states_opened") != 0 or spec.get("failure_replacement") is not False:
            raise ValueError("paired diagnosis cannot open or replace confirmation states")
    modes = list(spec.get("requested_modes", ()))
    if not modes or any(mode not in (0, 1) for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("paired diagnosis requires unique public requested modes")
    slots = int(spec.get("reset_slots_per_cell", 0))
    if slots < 1:
        raise ValueError("paired diagnosis requires fresh reset slots")
    cells = list(spec.get("scene_cells", ()))
    if not cells or len({cell["id"] for cell in cells}) != len(cells):
        raise ValueError("paired diagnosis requires unique scene cells")

    requests = []
    anchor = int(spec["seed_anchor"])
    for cell in cells:
        if cell.get("robot") not in {"panda", "xarm7"}:
            raise ValueError("paired diagnosis requires an explicit supported robot")
        for slot in range(slots):
            for mode in modes:
                sequence = len(requests)
                request_prefix = "CONFIRMATION" if confirmation else "DEVELOPMENT"
                requests.append({
                    "request_id": (
                        f"{request_prefix}/{spec['name']}/{task}/{cell['id']}/"
                        f"i{slot:02d}/m{mode}"
                    ),
                    "task": task, "scene_cell": cell["id"],
                    "robot": cell["robot"],
                    "mechanism_parameters": deepcopy(cell["mechanism_parameters"]),
                    "body_support_parameters": deepcopy(cell["body_support_parameters"]),
                    "physical_initial_slot": slot, "requested_mode": mode,
                    "reset_seed": anchor + sequence,
                    "confirmation_state": confirmation,
                    "performance_evidence": confirmation,
                })
    if len({request["reset_seed"] for request in requests}) != len(requests):
        raise ValueError("paired diagnostic reset seeds are not unique")

    development_gate = None
    if confirmation:
        declared_gate = spec.get("development_gate")
        if not isinstance(declared_gate, dict) or set(declared_gate) != {
            "plan", "report",
        }:
            raise ValueError("transport confirmation requires its frozen development plan and report")
        development_plan_path = _checked_pointer(
            declared_gate["plan"], "transport development plan",
        )
        development_report_path = _checked_pointer(
            declared_gate["report"], "transport development report",
        )
        development_plan = json.loads(development_plan_path.read_text())
        development_report = json.loads(development_report_path.read_text())
        development_spec_path = _checked_pointer(
            development_plan["diagnostic_spec"], "transport development spec",
        )
        development_spec = json.loads(development_spec_path.read_text())
        frozen_fields = (
            "task", "execution", "methods", "requested_modes", "scene_cells",
            "reset_slots_per_cell", "acceptance", "completion_value_admission",
            "mode_one_candidate_policy_set",
        )
        expected_development_schema = (
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA
            if schema == MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA
            else
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA
            if schema == MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA
            else
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA
            if schema == MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA
            else MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA
        )
        if (
            development_plan.get("schema_version")
            != expected_development_schema
            or development_report.get("schema_version")
            != expected_development_schema
            or development_report.get("passed") is not True
            or development_report.get("contribution_contrast", {}).get(
                "transport_claim_supported_development"
            ) is not True
            or development_report.get("confirmation_states_opened") != 0
            or any(spec.get(name) != development_spec.get(name) for name in frozen_fields)
            or {request["reset_seed"] for request in requests}
            & {request["reset_seed"] for request in development_plan["requests"]}
        ):
            raise ValueError("transport confirmation differs from or overlaps its passed development gate")
        development_gate = {
            "plan": _pointer(development_plan_path),
            "report": _pointer(development_report_path),
            "seed_disjoint": True,
            "method_and_gate_unchanged": True,
        }
    elif spec.get("development_gate") is not None:
        raise ValueError("development gate pointer is reserved for formal confirmation")

    base = ExperimentConfig.from_yaml(TASKS[task] / "_base.yaml")
    manifest_pointer = base.method_params["actor_binding"]["manifest"]
    validate_transfer_actor_manifest(manifest_pointer)
    manifest_path = ROOT / manifest_pointer["path"]
    execution = dict(spec.get("execution", {"device": "cpu"}))
    explicit_cuda_identity = set(execution) == {
        "device", "cuda_visible_devices", "physical_index", "physical_uuid",
    }
    if (
        (set(execution) != {"device"} and not explicit_cuda_identity)
        or execution["device"] not in {"cpu", "cuda"}
        or explicit_cuda_identity and (
            schema not in {
                MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }
            or execution["device"] != "cuda"
            or execution["cuda_visible_devices"] not in {"0", "1"}
            or execution["physical_index"] not in {0, 1}
            or str(execution["physical_index"])
            != execution["cuda_visible_devices"]
            or execution["physical_uuid"]
            != ({0: GPU0_UUID, 1: GPU1_UUID}[execution["physical_index"]])
        )
    ):
        raise ValueError("paired execution must select exactly cpu or cuda")
    if schema == DIAGNOSTIC_SCHEMA and execution["device"] != "cpu":
        raise ValueError("paired diagnosis v1 is CPU-only")
    if schema == DIAGNOSTIC_SCHEMA_V2 and execution["device"] != "cuda":
        raise ValueError("paired diagnosis v2 is GPU1-only")
    if schema in {
        CONTRIBUTION_CONTRAST_SCHEMA, SAME_START_CONTRIBUTION_SCHEMA,
        RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
    } and execution["device"] != "cuda":
        raise ValueError("paired contribution contrast is GPU1-only")
    if schema == CONSTRAINT_STRESS_SCHEMA and execution["device"] != "cpu":
        raise ValueError("paired constraint stress is CPU-only")
    public_log = deepcopy(spec.get("public_log"))
    if schema == PUBLIC_LOG_SCHEMA:
        if public_log != {
            "chart": "shared13", "permission": "B_log", "purpose": "fit",
        }:
            raise ValueError("public-log collection requires frozen B_log metadata")
    elif public_log is not None:
        raise ValueError("public_log requires the public-log collection schema")
    rows, jobs = [], []
    for method_spec in spec.get("methods", ()):
        family, key = method_spec["family"], method_spec["key"]
        mapping = MAIN_CONFIG if family == "main" else ABLATION_CONFIG if family == "ablation" else None
        if mapping is None or key not in mapping:
            raise ValueError(f"unknown paired diagnostic row: {family}/{key}")
        config_path = TASKS[task] / mapping[key]
        config = ExperimentConfig.from_yaml(config_path)
        selection, binding = _row_selection(task, family, key, config)
        declared_instances = method_spec.get("instances")
        if declared_instances == "fixed":
            if binding["status"] != "deterministic_single_identity":
                raise ValueError(f"{family}/{key} is not a fixed identity")
            instances = ("fixed",)
        elif declared_instances == "all":
            if binding["status"] != "instances_ready":
                raise ValueError(f"{family}/{key} has no complete instance set")
            instances = EXPECTED_INSTANCES
        elif isinstance(declared_instances, list):
            instances = tuple(declared_instances)
            if binding["status"] != "instances_ready" or not instances:
                raise ValueError(f"{family}/{key} cannot use declared instances")
            if any(instance not in EXPECTED_INSTANCES for instance in instances):
                raise ValueError(f"{family}/{key} names an unknown instance")
        else:
            raise ValueError("diagnostic method instances must be fixed, all, or an explicit list")
        label = str(method_spec.get("label", key))
        if any(row["key"] == label for row in rows):
            raise ValueError(f"duplicate paired diagnostic label: {label}")
        task_actor_interface = method_spec.get("task_actor_interface")
        if task_actor_interface is not None and (
            schema not in DIAGNOSTIC_SCHEMAS
            or task != "T2"
            or key == "public_geometry"
            or task_actor_interface not in T2_DIAGNOSTIC_ACTOR_INTERFACES
        ):
            raise ValueError("unsupported paired diagnostic task actor interface")
        task_actor_interface_parameters = _task_actor_interface_parameters(
            task_actor_interface,
            method_spec.get("task_actor_interface_parameters"),
        )
        overrides = dict(method_spec.get("binding_overrides", {}))
        if overrides and not (
            schema == CONSTRAINT_STRESS_SCHEMA
            and family == "ablation"
            and key in {"gated_full", "ungated_full"}
            and str(method_spec.get("label", key)) == "ungated_full"
            and overrides == {"contact_guard_version": None}
        ):
            raise ValueError("paired diagnostic binding overrides are unsupported")
        candidate = _candidate_policy_set(
            method_spec, schema=schema, task=task, family=family, key=key,
            instances=instances,
        )
        task_overrides = dict(method_spec.get("task_overrides", {}))
        if set(task_overrides) - {
            "guide_reference_admission", "log_stiffness_limit",
            "tangential_wrench_limit",
        }:
            raise ValueError("unsupported paired diagnostic task override")
        if task_overrides and task != "T2":
            raise ValueError("guide reference admission is a T2-only task override")
        admission = task_overrides.get("guide_reference_admission")
        if admission == "deadline_aware_v1":
            if task_overrides != {
                "guide_reference_admission": "deadline_aware_v1",
            }:
                raise ValueError("deadline admission cannot alter task support")
        elif admission == "response_normalized_completion":
            if task_overrides != {
                "guide_reference_admission": "response_normalized_completion",
                "log_stiffness_limit": 1.5,
                "tangential_wrench_limit": 16.0,
            }:
                raise ValueError("response normalization requires fixed declared support")
        elif task_overrides:
            raise ValueError("unknown T2 guide reference admission")
        row_id = f"{task}/{family}/{label}"
        rows.append({
            "row_id": row_id, "task": task, "family": family, "key": label,
            "binding_key": key, "binding_overrides": overrides,
            "task_overrides": task_overrides,
            **(
                {"task_actor_interface": task_actor_interface}
                if task_actor_interface is not None else {}
            ),
            **(
                {"task_actor_interface_parameters": task_actor_interface_parameters}
                if task_actor_interface_parameters else {}
            ),
            "config": str(config_path.relative_to(ROOT)), "method": config.method,
            "method_identity": (
                candidate["method_identity"] if candidate else binding["method_identity"]
            ),
            "manifest_status": binding["status"], "instances": list(instances),
            "physical_jobs": len(instances) * len(requests),
            **({"candidate_policy_set": candidate} if candidate else {}),
        })
        for instance in instances:
            instance_selection = deepcopy(selection)
            if instance != "fixed":
                instance_selection["instance"] = instance
            instance_binding = resolve_transfer_binding(
                {"actor_binding": instance_selection}, task_alias=task,
                instance_id=None if instance == "fixed" else instance,
            )
            for name, value in overrides.items():
                if value is None:
                    instance_binding.pop(name, None)
                else:
                    instance_binding[name] = value
            if candidate:
                instance_binding = deepcopy(instance_binding)
                instance_binding["policy_checkpoint"] = candidate["checkpoints"][instance]
                instance_binding["method_identity"] = candidate["method_identity"]
            for request in requests:
                record_id = f"{row_id}/{instance}/{request['request_id']}"
                jobs.append({
                    "record_id": record_id, "row_id": row_id, "task": task,
                    "family": family, "key": label, "binding_key": key,
                    "config": str(config_path.relative_to(ROOT)),
                    "method": config.method, "instance": instance,
                    "selection": instance_selection,
                    "actor_binding": instance_binding,
                    "device": execution["device"],
                    **(
                        {
                            "cuda_visible_devices": execution.get(
                                "cuda_visible_devices", "1"
                            ),
                            "physical_index": execution.get("physical_index", 1),
                            "physical_uuid": execution.get(
                                "physical_uuid", GPU1_UUID
                            ),
                        }
                        if execution["device"] == "cuda" else {}
                    ),
                    **(
                        {"task_actor_interface": task_actor_interface}
                        if task_actor_interface is not None else {}
                    ),
                    **(
                        {
                            "task_actor_interface_parameters": (
                                task_actor_interface_parameters
                            )
                        }
                        if task_actor_interface_parameters else {}
                    ),
                    **({"public_log": public_log} if public_log else {}),
                    "task_overrides": task_overrides,
                    "request": request,
                })
    if not rows:
        raise ValueError("paired diagnosis has no method rows")
    acceptance = deepcopy(spec.get("acceptance"))
    if schema in {
        DIAGNOSTIC_QUALIFICATION_SCHEMA, DIAGNOSTIC_QUALIFICATION_SCHEMA_V3,
    }:
        expected_acceptance = {
            "candidate": "cross_ebdm_finite_time",
            "current": "cross_ebdm_current",
            "band_only": "band_only",
            "strong_reference": "response_mpc",
            "required_candidate_successes": len(EXPECTED_INSTANCES) * len(requests),
            "preserve_every_paired_current_success": True,
            "require_zero_candidate_constraint_violations": True,
            "require_candidate_to_beat_band_only": True,
        }
        if schema == DIAGNOSTIC_QUALIFICATION_SCHEMA:
            expected_acceptance["require_candidate_to_match_strong_reference"] = True
        else:
            expected_acceptance[
                "require_candidate_to_meet_or_exceed_strong_reference"
            ] = True
        if acceptance != expected_acceptance:
            raise ValueError("paired qualification acceptance gate was changed")
        labels = {row["key"] for row in rows}
        if labels != {
            expected_acceptance[name]
            for name in ("candidate", "current", "band_only", "strong_reference")
        }:
            raise ValueError("paired qualification must contain exactly four gate rows")
    elif schema == CONTRIBUTION_CONTRAST_SCHEMA:
        expected_acceptance = {
            "transport": "cross_ebdm_matched",
            "task_only": "task_only_matched",
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.05,
            "minimum_nonnegative_instances": 6,
            "require_zero_extra_transport_constraint_violations": True,
        }
        if (
            task != "T3" or acceptance != expected_acceptance
            or {row["key"] for row in rows}
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("paired contribution contrast contract was changed")
    elif schema == SAME_START_CONTRIBUTION_SCHEMA:
        expected_acceptance = {
            "transport": "same_start_anchored",
            "task_only": "same_start_task_only",
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.05,
            "minimum_nonnegative_instances": 6,
            "maximum_two_sided_exact_mcnemar_p_value": 0.05,
            "require_zero_extra_transport_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows)
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("same-start anchoring contrast contract was changed")
        anchored = indexed_rows[expected_acceptance["transport"]][
            "candidate_policy_set"
        ]
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if (
            anchored["training_variant"] != {
                "name": "same_start_anchored",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 1.0,
            }
            or task_only["training_variant"] != {
                "name": "same_start_task_only",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
            or anchored["resolved_base_training_config"]
            != task_only["resolved_base_training_config"]
            or anchored["data_sha256"] != task_only["data_sha256"]
            or anchored["pairing_proof"] != task_only["pairing_proof"]
        ):
            raise ValueError("same-start pair differs beyond response anchoring")
    elif schema == RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA:
        expected_acceptance = {
            "transport": "relative_response_transport",
            "task_only": "relative_response_task_only",
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.0,
            "minimum_nonnegative_instances": 8,
            "minimum_mean_control_step_reduction_fraction": 0.02,
            "minimum_nonnegative_time_instances": 6,
            "maximum_one_sided_exact_sign_p_value": 0.05,
            "require_transport_to_preserve_every_task_only_success": True,
            "require_zero_extra_transport_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows)
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("relative-response contribution contract was changed")
        anchored = indexed_rows[expected_acceptance["transport"]][
            "candidate_policy_set"
        ]
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if (
            anchored["training_variant"] != {
                "name": "same_start_relative_anchored",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 1.0,
            }
            or task_only["training_variant"] != {
                "name": "same_start_relative_task_only",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
            or anchored["resolved_base_training_config"]
            != task_only["resolved_base_training_config"]
            or anchored["data_sha256"] != task_only["data_sha256"]
            or anchored["pairing_proof"] != task_only["pairing_proof"]
            or not all(
                row["accepted"] for row in task_only["selection_proof"].values()
            )
        ):
            raise ValueError(
                "relative-response pair differs beyond its response objective"
            )
    elif schema in {
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
    }:
        expected_acceptance = {
            "transport": "directional_transport",
            "task_only": "directional_task_only",
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.0,
            "minimum_nonnegative_instances": 8,
            "minimum_mean_control_step_reduction_fraction": 0.02,
            "minimum_nonnegative_time_instances": 6,
            "maximum_one_sided_exact_sign_p_value": 0.05,
            "require_transport_to_preserve_every_task_only_success": True,
            "require_zero_extra_transport_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows)
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("task-first directional contrast contract was changed")
        anchored = indexed_rows[expected_acceptance["transport"]][
            "candidate_policy_set"
        ]
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if (
            anchored["training_variant"] != {
                "name": (
                    "task_first_directional_same_state_anchored"
                    if schema == TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA
                    else "task_first_directional_anchored"
                ),
                "initialization_policy": "task_only",
                "response_anchoring_weight": 1.0,
            }
            or task_only["training_variant"] != {
                "name": "task_first_directional_task_only",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
            or anchored["resolved_base_training_config"]
            != task_only["resolved_base_training_config"]
            or anchored["resolved_shared_start_report"]
            != task_only["resolved_shared_start_report"]
            or anchored["data_sha256"] != task_only["data_sha256"]
            or anchored["pairing_proof"] != task_only["pairing_proof"]
            or not all(
                row["accepted"]
                for candidate in (anchored, task_only)
                for row in candidate["selection_proof"].values()
            )
        ):
            raise ValueError(
                "task-first pair differs beyond directional response anchoring"
            )
    elif schema in {
        PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
    }:
        robust_selector = schema == ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA
        label_prefix = (
            "ensemble_robust_selector" if robust_selector
            else "physical_selector"
        )
        selector_updates = (
            (900, 1100, 1200, 1300, 1400, 1500)
            if robust_selector
            else (900, 1000, 1200, 1300, 1400, 1500)
        )
        expected_acceptance = {
            "task_only": f"{label_prefix}_task_only",
            "candidate_updates": {
                f"{label_prefix}_update_{update:04d}": update
                for update in selector_updates
            },
            "required_requests_per_instance": len(requests),
            "selection_rule": (
                "maximize_successes_then_minimize_mean_capped_control_steps_"
                "then_lower_update"
            ),
            "require_candidate_to_preserve_every_task_only_success": True,
            "require_zero_extra_candidate_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        expected_labels = {
            expected_acceptance["task_only"],
            *expected_acceptance["candidate_updates"],
        }
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows) != expected_labels
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("physical checkpoint selector contract was changed")
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if task_only["training_variant"] != {
            "name": "task_first_margin_task_only",
            "initialization_policy": "task_only",
            "response_anchoring_weight": 0.0,
        }:
            raise ValueError("physical selector changed its task-only control")
        for label, update in expected_acceptance["candidate_updates"].items():
            candidate = indexed_rows[label]["candidate_policy_set"]
            if (
                candidate.get("retained_update") != update
                or candidate["training_variant"] != {
                    "name": (
                        "task_first_ensemble_robust_anchored"
                        if robust_selector
                        else "task_first_physical_selector_anchored"
                    ),
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 1.0,
                }
                or candidate["ensemble_task"] != (
                    {"reduction": "worst_paired_member"}
                    if robust_selector else None
                )
                or candidate["paired_progress"] != {
                    "linear_margin": 1e-5,
                    "angular_margin": 1e-4,
                    "weight": 1.0,
                }
                or candidate["resolved_base_training_config"]
                != task_only["resolved_base_training_config"]
                or candidate["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or candidate["data_sha256"] != task_only["data_sha256"]
                or candidate["pairing_proof"] != task_only["pairing_proof"]
                or not all(
                    row["accepted"]
                    for row in candidate["selection_proof"].values()
                )
            ):
                raise ValueError(
                    f"physical selector candidate {update} changed the paired contract"
                )
            indexed_rows[label]["execution_admission"] = {
                "version": (
                    "public_expanded_ensemble_robust_adaptive_step_dominance_v6"
                    if robust_selector
                    else "public_adaptive_step_transport_dominance_v3"
                ),
                "paired_task_only_checkpoints": deepcopy(
                    task_only["checkpoints"]
                ),
                "public_response_model_required": True,
            }
        for job in jobs:
            if job["key"] not in expected_acceptance["candidate_updates"]:
                continue
            job["actor_binding"] = deepcopy(job["actor_binding"])
            job["actor_binding"]["task_dominance_version"] = (
                "public_expanded_ensemble_robust_adaptive_step_dominance_v6"
                if robust_selector
                else "public_adaptive_step_transport_dominance_v3"
            )
            job["actor_binding"]["admission_task_only_checkpoint"] = deepcopy(
                task_only["checkpoints"][job["instance"]]
            )
    elif schema == MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA:
        selector_updates = (1100, 1400, 1500)
        expected_acceptance = {
            "task_only": "mode_zero_selector_task_only",
            "candidate_updates": {
                f"mode_zero_selector_update_{update:04d}": update
                for update in selector_updates
            },
            "required_requests_per_instance": len(requests),
            "selection_rule": (
                "maximize_successes_then_minimize_mean_capped_control_steps_"
                "then_task_only_fallback_then_lower_update"
            ),
            "require_candidate_to_preserve_every_task_only_success": True,
            "require_zero_extra_candidate_constraint_violations": True,
            "allow_task_only_fallback": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        expected_labels = {
            expected_acceptance["task_only"],
            *expected_acceptance["candidate_updates"],
        }
        if (
            task != "T3" or modes != [0] or slots != 4
            or acceptance != expected_acceptance
            or set(indexed_rows) != expected_labels
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("mode-zero physical selector contract changed")
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if task_only["mode"] != 0:
            raise ValueError("mode-zero selector changed task-only mode")
        for label, update in expected_acceptance["candidate_updates"].items():
            candidate = indexed_rows[label]["candidate_policy_set"]
            if (
                candidate["mode"] != 0
                or candidate.get("checkpoint_selection", {}).get(
                    "retained_update"
                ) != update
                or candidate["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or candidate["data_sha256"] != task_only["data_sha256"]
                or candidate["task_only_checkpoints"]
                != task_only["checkpoints"]
                or not all(
                    row["accepted"]
                    for row in candidate["selection_proof"].values()
                )
            ):
                raise ValueError(
                    f"mode-zero selector candidate {update} changed paired contract"
                )
            indexed_rows[label]["execution_admission"] = {
                "version": "public_mode_specific_recorded_completion_dominance_v15",
                "paired_task_only_checkpoints": deepcopy(task_only["checkpoints"]),
                "public_response_model_required": True,
            }
        for job in jobs:
            if job["key"] not in expected_acceptance["candidate_updates"]:
                continue
            job["actor_binding"] = deepcopy(job["actor_binding"])
            job["actor_binding"]["task_dominance_version"] = (
                "public_mode_specific_recorded_completion_dominance_v15"
            )
            job["actor_binding"]["admission_task_only_checkpoint"] = deepcopy(
                task_only["checkpoints"][job["instance"]]
            )
    elif schema == MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA:
        selector_updates = (1200, 1300, 1400, 1500)
        expected_acceptance = {
            "task_only": "mode_specific_selector_task_only",
            "candidate_updates": {
                f"mode_specific_selector_update_{update:04d}": update
                for update in selector_updates
            },
            "required_requests_per_instance": len(requests),
            "selection_rule": (
                "maximize_successes_then_minimize_mean_capped_control_steps_"
                "then_task_only_fallback_then_lower_update"
            ),
            "require_candidate_to_preserve_every_task_only_success": True,
            "require_zero_extra_candidate_constraint_violations": True,
            "allow_task_only_fallback": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        expected_labels = {
            expected_acceptance["task_only"],
            *expected_acceptance["candidate_updates"],
        }
        if (
            task != "T3" or modes != [1] or slots != 4
            or acceptance != expected_acceptance
            or set(indexed_rows) != expected_labels
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("mode-specific physical selector contract changed")
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if (
            task_only["mode"] != 1
            or task_only["training_variant"] != {
                "name": "task_first_directional_task_only",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
        ):
            raise ValueError("mode-specific selector changed task-only control")
        for label, update in expected_acceptance["candidate_updates"].items():
            candidate = indexed_rows[label]["candidate_policy_set"]
            if (
                candidate["mode"] != 1
                or candidate.get("checkpoint_selection", {}).get(
                    "retained_update"
                ) != update
                or candidate["training_variant"] != {
                    "name": "task_first_ensemble_robust_anchored",
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 1.0,
                }
                or candidate["ensemble_task"]
                != {"reduction": "worst_paired_member"}
                or candidate["paired_progress"] != {
                    "linear_margin": 1e-5,
                    "angular_margin": 1e-4,
                    "weight": 1.0,
                    "tail_fraction": 0.1,
                    "tail_weight": 4.0,
                    "requested_mode": 1,
                }
                or candidate["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or candidate["data_sha256"] != task_only["data_sha256"]
                or candidate["task_only_checkpoints"]
                != task_only["checkpoints"]
                or not all(
                    row["accepted"]
                    for row in candidate["selection_proof"].values()
                )
            ):
                raise ValueError(
                    f"mode-specific selector candidate {update} changed paired contract"
                )
            indexed_rows[label]["execution_admission"] = {
                "version": "public_mode_specific_recorded_completion_dominance_v15",
                "paired_task_only_checkpoints": deepcopy(task_only["checkpoints"]),
                "public_response_model_required": True,
            }
        for job in jobs:
            if job["key"] not in expected_acceptance["candidate_updates"]:
                continue
            job["actor_binding"] = deepcopy(job["actor_binding"])
            job["actor_binding"]["task_dominance_version"] = (
                "public_mode_specific_recorded_completion_dominance_v15"
            )
            job["actor_binding"]["admission_task_only_checkpoint"] = deepcopy(
                task_only["checkpoints"][job["instance"]]
            )
    elif schema in {
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
    }:
        mode_specific_recorded = (
            schema in {
                MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }
        )
        mode_specific_physical_selected = schema in {
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        }
        supported_phase_convex = (
            schema == MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA
        )
        supported_phase_capped_expansion = (
            schema
            == MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA
        )
        calibrated_authority = (
            schema == MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA
        )
        robust_trained = schema == ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA
        bounded_endpoint = (
            schema == BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
        )
        capped_expansion = (
            schema == CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
        )
        horizon_consistent = (
            schema == HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
        )
        validated_bridge = (
            schema == VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
        )
        mode_conditioned = schema in {
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        }
        completion_value = schema in {
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        }
        ensemble_robust = schema in {
            ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        }
        expanded_ensemble_robust = schema in {
            EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
            MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
            MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
        }
        authority_calibration = (
            _mode_one_authority_calibration(spec.get("mode_one_authority_calibration"))
            if calibrated_authority else None
        )
        if not calibrated_authority and spec.get("mode_one_authority_calibration") is not None:
            raise ValueError("mode-one authority calibration requires its dedicated schema")
        expected_acceptance = {
            "transport": (
                "mode_specific_physical_selected_transport"
                if mode_specific_physical_selected
                else "mode_specific_calibrated_authority_transport"
                if calibrated_authority
                else "mode_specific_supported_phase_capped_expansion_transport"
                if supported_phase_capped_expansion
                else "mode_specific_supported_phase_convex_transport"
                if supported_phase_convex
                else "mode_specific_recorded_completion_transport"
                if mode_specific_recorded
                else "robust_trained_selected_transport"
                if robust_trained
                else "mode_conditioned_transport" if mode_conditioned
                else "bounded_endpoint_transport" if bounded_endpoint
                else "capped_expansion_transport" if capped_expansion
                else "horizon_consistent_transport" if horizon_consistent
                else "validated_bridge_transport" if validated_bridge
                else "completion_value_transport" if completion_value
                else
                "expanded_ensemble_robust_transport"
                if expanded_ensemble_robust
                else "ensemble_robust_transport" if ensemble_robust
                else "physical_selected_transport"
            ),
            "task_only": (
                "mode_specific_physical_selected_task_only"
                if mode_specific_physical_selected
                else "mode_specific_calibrated_authority_task_only"
                if calibrated_authority
                else "mode_specific_supported_phase_capped_expansion_task_only"
                if supported_phase_capped_expansion
                else "mode_specific_supported_phase_convex_task_only"
                if supported_phase_convex
                else "mode_specific_recorded_completion_task_only"
                if mode_specific_recorded
                else "robust_trained_selected_task_only"
                if robust_trained
                else "mode_conditioned_task_only" if mode_conditioned
                else "bounded_endpoint_task_only" if bounded_endpoint
                else "capped_expansion_task_only" if capped_expansion
                else "horizon_consistent_task_only" if horizon_consistent
                else "validated_bridge_task_only" if validated_bridge
                else "completion_value_task_only" if completion_value
                else
                "expanded_ensemble_robust_task_only"
                if expanded_ensemble_robust
                else "ensemble_robust_task_only" if ensemble_robust
                else "physical_selected_task_only"
            ),
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.0,
            "minimum_nonnegative_instances": 8,
            "minimum_mean_control_step_reduction_fraction": 0.02,
            "minimum_nonnegative_time_instances": 6,
            "maximum_one_sided_exact_sign_p_value": 0.05,
            "require_transport_to_preserve_every_task_only_success": True,
            "require_zero_extra_transport_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows)
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError(
                "physical-selected contribution contract was changed"
            )
        anchored = indexed_rows[expected_acceptance["transport"]][
            "candidate_policy_set"
        ]
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if mode_specific_recorded:
            if (
                anchored["mode"] != 0
                or task_only["mode"] != 0
                or anchored["training_variant"] != {
                    "name": "task_first_ensemble_robust_anchored",
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 1.0,
                }
                or anchored["ensemble_task"]
                != {"reduction": "worst_paired_member"}
                or task_only["training_variant"] != {
                    "name": "task_first_directional_task_only",
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 0.0,
                }
                or anchored["paired_progress"] != {
                    "linear_margin": 1e-5,
                    "angular_margin": 1e-4,
                    "weight": 1.0,
                    "tail_fraction": 0.1,
                    "tail_weight": 4.0,
                    "requested_mode": 0,
                }
                or task_only["paired_progress"] is not None
                or anchored["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or anchored["data_sha256"] != task_only["data_sha256"]
                or anchored["pairing_proof"] != task_only["pairing_proof"]
                or anchored["task_only_checkpoints"]
                != task_only["checkpoints"]
                or not all(
                    row["accepted"]
                    for candidate in (anchored, task_only)
                    for row in candidate["selection_proof"].values()
                )
            ):
                raise ValueError(
                    "mode-specific pair differs before recorded-completion admission"
                )
            if schema in {
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
                MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
            }:
                zero_proof = anchored.get(
                    "mode_zero_physical_selection_proof", {}
                )
                zero_device = anchored.get(
                    "mode_zero_physical_selection_device", {}
                )
                zero_report_path = _checked_pointer(
                    anchored["mode_zero_physical_selection_report"],
                    "mode-zero physical selection report",
                )
                zero_report = json.loads(zero_report_path.read_text())
                zero_plan_path = _checked_pointer(
                    zero_report["artifacts"]["plan"],
                    "mode-zero physical selection plan",
                )
                zero_plan = json.loads(zero_plan_path.read_text())
                if (
                    set(zero_proof) != set(EXPECTED_INSTANCES)
                    or not all(
                        row["selected_metrics"]["eligible"]
                        for row in zero_proof.values()
                    )
                    or zero_device.get("cuda_visible_devices")
                    != execution.get("cuda_visible_devices")
                    or zero_device.get("physical_index")
                    != execution.get("physical_index")
                    or zero_device.get("physical_uuid")
                    != execution.get("physical_uuid")
                    or {request["reset_seed"] for request in requests}
                    & {
                        request["reset_seed"]
                        for request in zero_plan["requests"]
                    }
                ):
                    raise ValueError(
                        "mode-zero physical selection is incomplete, mismatched, or seed-overlapping"
                    )
        elif (
            anchored["training_variant"] != {
                "name": (
                    "task_first_ensemble_robust_anchored"
                    if robust_trained
                    else "task_first_physical_selector_anchored"
                ),
                "initialization_policy": "task_only",
                "response_anchoring_weight": 1.0,
            }
            or anchored["ensemble_task"] != (
                {"reduction": "worst_paired_member"}
                if robust_trained else None
            )
            or task_only["training_variant"] != {
                "name": "task_first_margin_task_only",
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
            or anchored["paired_progress"] != {
                "linear_margin": 1e-5,
                "angular_margin": 1e-4,
                "weight": 1.0,
            }
            or task_only["paired_progress"] is not None
            or anchored["resolved_base_training_config"]
            != task_only["resolved_base_training_config"]
            or anchored["resolved_shared_start_report"]
            != task_only["resolved_shared_start_report"]
            or anchored["data_sha256"] != task_only["data_sha256"]
            or anchored["pairing_proof"] != task_only["pairing_proof"]
            or set(anchored["physical_selection_proof"])
            != set(EXPECTED_INSTANCES)
            or not all(
                row["selected_metrics"]["eligible"]
                for row in anchored["physical_selection_proof"].values()
            )
            or not all(
                row["accepted"]
                for row in task_only["selection_proof"].values()
            )
        ):
            raise ValueError(
                "physical-selected pair differs before adaptive admission"
            )
        indexed_rows[expected_acceptance["transport"]]["execution_admission"] = {
            "version": (
                "public_mode_specific_calibrated_authority_dominance_v18"
                if calibrated_authority
                else "public_mode_specific_supported_phase_capped_expansion_dominance_v17"
                if supported_phase_capped_expansion
                else "public_mode_specific_supported_phase_convex_dominance_v16"
                if supported_phase_convex
                else "public_mode_specific_recorded_completion_dominance_v15"
                if mode_specific_recorded
                else "public_bounded_endpoint_ensemble_robust_dominance_v7"
                if bounded_endpoint
                else
                "public_capped_expansion_ensemble_robust_dominance_v8"
                if capped_expansion
                else
                "public_horizon_consistent_ensemble_robust_dominance_v9"
                if horizon_consistent
                else
                "public_validated_bridge_ensemble_robust_dominance_v10"
                if validated_bridge
                else
                "public_completion_value_ensemble_robust_dominance_v11"
                if completion_value and not mode_conditioned
                else
                "public_mode_conditioned_finite_time_dominance_v14"
                if mode_conditioned
                else
                "public_expanded_ensemble_robust_adaptive_step_dominance_v6"
                if expanded_ensemble_robust
                else
                "public_ensemble_robust_adaptive_step_dominance_v5"
                if ensemble_robust
                else "public_adaptive_step_transport_dominance_v3"
            ),
            "paired_task_only_checkpoints": deepcopy(task_only["checkpoints"]),
            "public_response_model_required": True,
            **(
                {"mode_one_authority_calibration": deepcopy(authority_calibration)}
                if authority_calibration is not None else {}
            ),
        }
        completion_inputs = None
        if completion_value:
            admission = spec.get("completion_value_admission")
            if not isinstance(admission, dict) or set(admission) != {
                "response_training_report", "completion_value_checkpoint",
                "completion_value_calibration",
            }:
                raise ValueError("completion-value admission proof is incomplete")
            response_report_path = _checked_pointer(
                admission["response_training_report"],
                "mode-complete response training report",
            )
            value_path = _checked_pointer(
                admission["completion_value_checkpoint"],
                "completion-value checkpoint",
            )
            calibration_path = _checked_pointer(
                admission["completion_value_calibration"],
                "completion-value calibration",
            )
            response_report = json.loads(response_report_path.read_text())
            calibration_report = json.loads(calibration_path.read_text())
            if (
                response_report.get("schema_version")
                != "t3-mode-complete-public-response-training-v1"
                or response_report.get("passed") is not True
                or response_report.get("new_physics_calls") != 0
                or set(response_report.get("instances", {}))
                != set(EXPECTED_INSTANCES)
                or calibration_report.get("schema_version")
                != "t3-public-completion-response-calibration-v1"
                or calibration_report.get("passed") is not True
                or calibration_report.get("new_physics_calls") != 0
                or calibration_report.get("artifacts", {}).get(
                    "completion_value_checkpoint"
                ) != _pointer(value_path)
            ):
                raise ValueError("completion-value admission proof did not pass")
            response_checkpoints = {}
            for instance in EXPECTED_INSTANCES:
                checkpoint = _pointer(_checked_pointer(
                    response_report["instances"][instance]["checkpoint"],
                    f"mode-complete {instance} response checkpoint",
                ))
                if calibration_report["artifacts"]["response_checkpoints"].get(
                    instance
                ) != checkpoint:
                    raise ValueError(
                        f"{instance} response differs from completion calibration"
                    )
                response_checkpoints[instance] = checkpoint
            completion_inputs = {
                "response_training_report": _pointer(response_report_path),
                "response_checkpoints": response_checkpoints,
                "completion_value_checkpoint": _pointer(value_path),
                "completion_value_calibration": _pointer(calibration_path),
                "minimum_admission_gain_controls": calibration_report[
                    "calibration"
                ]["minimum_admission_gain_controls"],
            }
            indexed_rows[expected_acceptance["transport"]][
                "completion_value_admission"
            ] = deepcopy(completion_inputs)
        elif spec.get("completion_value_admission") is not None:
            raise ValueError("completion-value proof requires its dedicated schema")
        mode_one_candidate = None
        if mode_conditioned:
            raw_mode_one = spec.get("mode_one_candidate_policy_set")
            mode_one_candidate = _candidate_policy_set(
                {"candidate_policy_set": raw_mode_one},
                schema=(
                    schema
                    if mode_specific_recorded
                    else ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
                ),
                task=task, family="main", key="cross_ebdm",
                instances=EXPECTED_INSTANCES,
            )
            if mode_specific_recorded and (
                mode_one_candidate["mode"] != 1
                or mode_one_candidate["training_variant"] != {
                    "name": "task_first_ensemble_robust_anchored",
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 1.0,
                }
                or mode_one_candidate["ensemble_task"]
                != {"reduction": "worst_paired_member"}
                or mode_one_candidate["paired_progress"] != {
                    "linear_margin": 1e-5,
                    "angular_margin": 1e-4,
                    "weight": 1.0,
                    "tail_fraction": 0.1,
                    "tail_weight": 4.0,
                    "requested_mode": 1,
                }
                or mode_one_candidate["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or mode_one_candidate["data_sha256"] != task_only["data_sha256"]
                or mode_one_candidate["task_only_checkpoints"]
                != task_only["checkpoints"]
                or not all(
                    row["accepted"]
                    for row in mode_one_candidate["selection_proof"].values()
                )
            ):
                raise ValueError(
                    "mode-one recorded-completion branch differs from its paired task-only contract"
                )
            if mode_specific_physical_selected:
                proof = mode_one_candidate.get("physical_selection_proof", {})
                selected_device = mode_one_candidate.get(
                    "physical_selection_device", {}
                )
                if (
                    selected_device.get("cuda_visible_devices")
                    != execution.get("cuda_visible_devices")
                    or selected_device.get("physical_index")
                    != execution.get("physical_index")
                    or selected_device.get("physical_uuid")
                    != execution.get("physical_uuid")
                ):
                    raise ValueError(
                        "mode-specific selector and contribution devices differ"
                    )
                selector_report_path = _checked_pointer(
                    mode_one_candidate["physical_selection_report"],
                    "mode-specific selected checkpoint report",
                )
                selector_report = json.loads(selector_report_path.read_text())
                selector_plan_path = _checked_pointer(
                    selector_report["artifacts"]["plan"],
                    "mode-specific selected checkpoint plan",
                )
                selector_plan = json.loads(selector_plan_path.read_text())
                excluded_seeds = {
                    request["reset_seed"] for request in selector_plan["requests"]
                }
                if "safety_calibration_plan" in mode_one_candidate:
                    safety_plan_path = _checked_pointer(
                        mode_one_candidate["safety_calibration_plan"],
                        "mode-specific success safety calibration plan",
                    )
                    safety_plan = json.loads(safety_plan_path.read_text())
                    excluded_seeds.update(
                        request["reset_seed"] for request in safety_plan["requests"]
                    )
                    safety_proof = mode_one_candidate.get(
                        "success_safety_proof", {}
                    )
                    if (
                        set(safety_proof) != set(EXPECTED_INSTANCES)
                        or not all(
                            row["complete"] and row["integrity_passed"]
                            for row in safety_proof.values()
                        )
                    ):
                        raise ValueError("success safety calibration is incomplete")
                if (
                    set(proof) != set(EXPECTED_INSTANCES)
                    or not all(
                        row["selected_metrics"]["eligible"]
                        for row in proof.values()
                    )
                    or {request["reset_seed"] for request in requests}
                    & excluded_seeds
                ):
                    raise ValueError(
                        "mode-specific physical selection is incomplete or seed-overlapping"
                    )
            if not mode_specific_recorded and (
                mode_one_candidate["training_variant"] != {
                    "name": "task_first_margin_anchored",
                    "initialization_policy": "task_only",
                    "response_anchoring_weight": 1.0,
                }
                or mode_one_candidate["paired_progress"] != {
                    "linear_margin": 1e-5,
                    "angular_margin": 1e-4,
                    "weight": 1.0,
                }
                or mode_one_candidate["resolved_base_training_config"]
                != task_only["resolved_base_training_config"]
                or mode_one_candidate["resolved_shared_start_report"]
                != task_only["resolved_shared_start_report"]
                or mode_one_candidate["data_sha256"] != task_only["data_sha256"]
                or mode_one_candidate["pairing_proof"] != task_only["pairing_proof"]
                or not all(
                    row["accepted"]
                    for row in mode_one_candidate["selection_proof"].values()
                )
            ):
                raise ValueError("mode-one transport branch differs from its paired task-only contract")
            indexed_rows[expected_acceptance["transport"]][
                "mode_conditioned_branches"
            ] = {
                "public_mode_field": "requested_mode",
                "mode_0": {
                    "candidate_policy_set": deepcopy(anchored),
                    "execution_admission": (
                        "public_mode_specific_supported_phase_capped_expansion_dominance_v17"
                        if supported_phase_capped_expansion
                        else "public_mode_specific_supported_phase_convex_dominance_v16"
                        if supported_phase_convex
                        else "public_mode_specific_recorded_completion_dominance_v15"
                        if mode_specific_recorded
                        else "public_completion_value_ensemble_robust_dominance_v11"
                    ),
                },
                "mode_1": {
                    "candidate_policy_set": deepcopy(mode_one_candidate),
                    "execution_admission": (
                        "public_mode_specific_supported_phase_capped_expansion_dominance_v17"
                        if supported_phase_capped_expansion
                        else "public_mode_specific_supported_phase_convex_dominance_v16"
                        if supported_phase_convex
                        else "public_mode_specific_recorded_completion_dominance_v15"
                        if mode_specific_recorded
                        else "public_adaptive_step_transport_dominance_v3"
                    ),
                },
            }
        elif spec.get("mode_one_candidate_policy_set") is not None:
            raise ValueError("mode-one policy source requires its dedicated schema")
        for job in jobs:
            if job["key"] != expected_acceptance["transport"]:
                continue
            job["actor_binding"] = deepcopy(job["actor_binding"])
            job["actor_binding"]["task_dominance_version"] = (
                "public_mode_specific_calibrated_authority_dominance_v18"
                if calibrated_authority
                else "public_mode_specific_supported_phase_capped_expansion_dominance_v17"
                if supported_phase_capped_expansion
                else "public_mode_specific_supported_phase_convex_dominance_v16"
                if supported_phase_convex
                else "public_mode_specific_recorded_completion_dominance_v15"
                if mode_specific_recorded
                else "public_bounded_endpoint_ensemble_robust_dominance_v7"
                if bounded_endpoint
                else
                "public_capped_expansion_ensemble_robust_dominance_v8"
                if capped_expansion
                else
                "public_horizon_consistent_ensemble_robust_dominance_v9"
                if horizon_consistent
                else
                "public_validated_bridge_ensemble_robust_dominance_v10"
                if validated_bridge
                else
                "public_completion_value_ensemble_robust_dominance_v11"
                if completion_value and not mode_conditioned
                else
                "public_mode_conditioned_finite_time_dominance_v14"
                if mode_conditioned
                else
                "public_expanded_ensemble_robust_adaptive_step_dominance_v6"
                if expanded_ensemble_robust
                else
                "public_ensemble_robust_adaptive_step_dominance_v5"
                if ensemble_robust
                else "public_adaptive_step_transport_dominance_v3"
            )
            job["actor_binding"]["admission_task_only_checkpoint"] = deepcopy(
                task_only["checkpoints"][job["instance"]]
            )
            if mode_conditioned and job["request"]["requested_mode"] == 1:
                job["actor_binding"]["policy_checkpoint"] = deepcopy(
                    mode_one_candidate["checkpoints"][job["instance"]]
                )
            if authority_calibration is not None:
                job["actor_binding"]["mode_one_max_blend"] = (
                    authority_calibration[
                        "selected_mode_one_max_blend_by_instance"
                    ][job["instance"]]
                )
            if completion_inputs is not None and (
                not mode_conditioned
                or job["request"]["requested_mode"] == 0
            ):
                job["actor_binding"]["response_checkpoint"] = deepcopy(
                    completion_inputs["response_checkpoints"][job["instance"]]
                )
                job["actor_binding"]["completion_value_checkpoint"] = deepcopy(
                    completion_inputs["completion_value_checkpoint"]
                )
                job["actor_binding"]["completion_value_calibration"] = deepcopy(
                    completion_inputs["completion_value_calibration"]
                )
    elif schema in {
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
    }:
        margin_training = schema in {
            MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
            ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        }
        adaptive_step = schema in {
            ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        }
        stable_selector = schema == STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
        expanded_step = schema == EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
        transient_phase = (
            schema == TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
        )
        phase_scheduled = (
            schema == PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
        )
        full_only = schema in {
            FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
            MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
            ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
            PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        }
        expected_acceptance = {
            "transport": (
                "expanded_adaptive_step_transport" if expanded_step
                else "phase_scheduled_adaptive_step_transport" if phase_scheduled
                else "transient_phase_adaptive_step_transport" if transient_phase
                else "stable_adaptive_step_transport" if stable_selector
                else "adaptive_step_transport" if adaptive_step
                else "margin_runtime_transport" if margin_training
                else "full_runtime_transport" if full_only
                else "runtime_transport"
            ),
            "task_only": (
                "expanded_adaptive_step_task_only" if expanded_step
                else "phase_scheduled_adaptive_step_task_only" if phase_scheduled
                else "transient_phase_adaptive_step_task_only" if transient_phase
                else "stable_adaptive_step_task_only" if stable_selector
                else "adaptive_step_task_only" if adaptive_step
                else "margin_runtime_task_only" if margin_training
                else "full_runtime_task_only" if full_only
                else "runtime_task_only"
            ),
            "required_pairs": len(EXPECTED_INSTANCES) * len(requests),
            "minimum_aggregate_success_rate_gain": 0.0,
            "minimum_nonnegative_instances": 8,
            "minimum_mean_control_step_reduction_fraction": 0.02,
            "minimum_nonnegative_time_instances": 6,
            "maximum_one_sided_exact_sign_p_value": 0.05,
            "require_transport_to_preserve_every_task_only_success": True,
            "require_zero_extra_transport_constraint_violations": True,
        }
        indexed_rows = {row["key"]: row for row in rows}
        if (
            task != "T3" or acceptance != expected_acceptance
            or set(indexed_rows)
            != {expected_acceptance["transport"], expected_acceptance["task_only"]}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
            or any("candidate_policy_set" not in row for row in rows)
        ):
            raise ValueError("runtime task-dominance contrast contract was changed")
        anchored = indexed_rows[expected_acceptance["transport"]][
            "candidate_policy_set"
        ]
        task_only = indexed_rows[expected_acceptance["task_only"]][
            "candidate_policy_set"
        ]
        if (
            anchored["training_variant"] != {
                "name": (
                    "task_first_stable_margin_anchored" if stable_selector
                    else "task_first_margin_anchored" if margin_training
                    else "task_first_directional_anchored"
                ),
                "initialization_policy": "task_only",
                "response_anchoring_weight": 1.0,
            }
            or task_only["training_variant"] != {
                "name": (
                    "task_first_margin_task_only" if margin_training
                    else "task_first_directional_task_only"
                ),
                "initialization_policy": "task_only",
                "response_anchoring_weight": 0.0,
            }
            or margin_training and anchored["paired_progress"] != {
                "linear_margin": 1e-5,
                "angular_margin": 1e-4,
                "weight": 1.0,
            }
            or margin_training and task_only["paired_progress"] is not None
            or stable_selector and anchored.get("checkpoint_selection") != {
                "rule": "best_validation_paired_progress",
                "interval_updates": 100,
                "minimum_update": 100,
            }
            or anchored["resolved_base_training_config"]
            != task_only["resolved_base_training_config"]
            or anchored["resolved_shared_start_report"]
            != task_only["resolved_shared_start_report"]
            or anchored["data_sha256"] != task_only["data_sha256"]
            or anchored["pairing_proof"] != task_only["pairing_proof"]
            or not all(
                row["accepted"]
                for candidate in (anchored, task_only)
                for row in candidate["selection_proof"].values()
            )
        ):
            raise ValueError(
                "runtime pair differs before task-dominance admission"
            )
        indexed_rows[expected_acceptance["transport"]]["execution_admission"] = {
            "version": (
                "public_expanded_adaptive_step_transport_dominance_v4"
                if expanded_step else
                "public_phase_scheduled_adaptive_step_dominance_v13"
                if phase_scheduled else
                "public_transient_phase_adaptive_step_dominance_v12"
                if transient_phase else
                "public_adaptive_step_transport_dominance_v3"
                if adaptive_step else
                "public_full_transport_over_task_only_dominance_v2"
                if full_only else "public_transport_over_task_only_dominance_v1"
            ),
            "paired_task_only_checkpoints": deepcopy(task_only["checkpoints"]),
            "public_response_model_required": True,
        }
        for job in jobs:
            if job["key"] != expected_acceptance["transport"]:
                continue
            job["actor_binding"] = deepcopy(job["actor_binding"])
            job["actor_binding"]["task_dominance_version"] = (
                "public_expanded_adaptive_step_transport_dominance_v4"
                if expanded_step else
                "public_phase_scheduled_adaptive_step_dominance_v13"
                if phase_scheduled else
                "public_transient_phase_adaptive_step_dominance_v12"
                if transient_phase else
                "public_adaptive_step_transport_dominance_v3"
                if adaptive_step else
                "public_full_transport_over_task_only_dominance_v2"
                if full_only else "public_transport_over_task_only_dominance_v1"
            )
            job["actor_binding"]["admission_task_only_checkpoint"] = deepcopy(
                task_only["checkpoints"][job["instance"]]
            )
    elif acceptance is not None:
        raise ValueError("paired diagnosis v1 cannot define a qualification gate")
    stress_contract = deepcopy(spec.get("stress_contract"))
    source_decision = deepcopy(spec.get("source_decision"))
    if schema == CONSTRAINT_STRESS_SCHEMA:
        expected_contract = {
            "comparators": ["gated_full", "ungated_full"],
            "event": "measured_force_or_joint_speed_limit",
            "force_limit_N": 25.0,
            "lateral_force_limit_N": 10.0,
            "joint_speed_limit_radps": 3.0,
            "require_exact_1ms_substeps": True,
            "claim_rule": "exact_paired_two_sided_p_below_0.05_and_ungated_only_exceeds_gated_only",
            "predicted_margin_policy": "unavailable_without_a_calibrated_public_predictor",
        }
        if (
            task != "T2" or modes != [1] or slots != 8 or len(cells) != 1
            or cells[0].get("id") != "payload_shift"
            or cells[0].get("robot") != "xarm7"
            or stress_contract != expected_contract
            or {(row["family"], row["key"]) for row in rows}
            != {("ablation", "gated_full"), ("ablation", "ungated_full")}
            or any(row["instances"] != list(EXPECTED_INSTANCES) for row in rows)
        ):
            raise ValueError("paired constraint stress contract was changed")
        _checked_pointer(source_decision, "constraint stress source decision")
    elif stress_contract is not None or source_decision is not None:
        raise ValueError("stress metadata requires the paired constraint stress schema")
    if schema == PUBLIC_LOG_SCHEMA:
        if acceptance is not None or len(rows) != 1:
            raise ValueError("public-log collection requires one row and no behavior gate")
        row = rows[0]
        if (
            row["key"] != "cross_ebdm_late_phase"
            or row["method_identity"] != "ours_gated_v4_terminal_stable"
            or row["instances"] != list(EXPECTED_INSTANCES)
        ):
            raise ValueError("public-log collection must use all terminal-stable instances")
    return {
        "schema_version": schema, "status": "planned_unexecuted",
        "task": task, "name": spec["name"], "diagnostic_spec": _pointer(spec_path),
        "execution": execution,
        "public_log": public_log,
        "acceptance": acceptance,
        "development_gate": development_gate,
        "stress_contract": stress_contract,
        "source_decision": source_decision,
        "actor_manifest": {
            **_pointer(manifest_path), "declared_sha256": manifest_pointer["sha256"],
            "full_catalog_validation_passed": True,
        },
        "requests": requests, "rows": rows, "jobs": jobs,
        "counts": {"logical_rows": len(rows), "requests": len(requests),
                   "physical_jobs": len(jobs)},
        "policy": {
            "diagnostic_only": not confirmation,
            "performance_evidence": confirmation,
            "same_requests_for_every_selected_row": True,
            "all_declared_instances_included": all(
                row["instances"] == list(EXPECTED_INSTANCES) or row["instances"] == ["fixed"]
                for row in rows
            ),
            "confirmation_states_opened": int(confirmation),
            "confirmation_seed_disjoint": bool(confirmation),
            "old_invalid_outcomes_used": False,
            "failure_replacement": False,
            "same_start_anchoring_is_only_training_factor": (
                schema == SAME_START_CONTRIBUTION_SCHEMA
            ),
            "same_start_relative_response_is_only_training_factor": (
                schema == RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA
            ),
            "task_first_directional_response_is_only_training_factor": (
                schema in {
                    TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
                    TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
                }
            ),
            "directional_source_task_uses_target_state": (
                schema == TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA
            ),
            "runtime_task_dominance_is_only_execution_factor": (
                schema == RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA
            ),
            "full_runtime_task_dominance_is_only_execution_factor": (
                schema == FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA
            ),
            "margin_training_and_runtime_dominance_define_transport": (
                schema == MARGIN_RUNTIME_CONTRIBUTION_SCHEMA
            ),
            "adaptive_step_transport_is_only_execution_change": (
                schema == ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "stable_paired_selector_and_adaptive_step_define_transport": (
                schema == STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "expanded_adaptive_step_is_only_execution_change": (
                schema == EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "terminal_task_only_ownership_is_only_execution_change": (
                schema == TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "phase_scheduled_authority_is_only_execution_change": (
                schema == PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "physical_checkpoint_selection_is_frozen_before_contribution_test": (
                schema in {
                    PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
                    MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                    MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                    MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
                    ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
                }
            ),
            "ensemble_consensus_is_only_execution_change": (
                schema in {
                    ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                    EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
                }
            ),
            "expanded_grid_is_only_change_from_ensemble_consensus": (
                schema == EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA
            ),
            "bounded_endpoint_is_only_change_from_expanded_ensemble_consensus": (
                schema == BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
            ),
            "capped_expansion_is_only_change_from_expanded_ensemble_consensus": (
                schema == CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
            ),
            "horizon_consistency_is_only_change_from_expanded_ensemble_consensus": (
                schema == HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
            ),
            "validated_bridge_is_only_change_from_expanded_ensemble_consensus": (
                schema == VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
            ),
            "completion_value_is_only_execution_change_from_expanded_ensemble_consensus": (
                schema == COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA
            ),
            "public_mode_conditioned_branch_is_frozen_before_execution": (
                schema in {
                    MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
                    MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
                    MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
                    MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
                }
            ),
            "mode_one_supported_phase_convex_authority_is_only_execution_change": (
                schema
                == MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA
            ),
            "mode_one_supported_phase_capped_expansion_is_only_execution_change": (
                schema
                == MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA
            ),
            "mode_one_authority_is_selected_from_disjoint_development_cohorts": (
                schema == MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA
            ),
            "ensemble_robust_training_and_runtime_define_transport": (
                schema == ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA
            ),
        },
    }


def _task_from_job(
    config: ExperimentConfig,
    request: dict[str, Any],
    task_overrides: dict[str, Any] | None = None,
):
    values = deepcopy(config.env_params)
    values.pop("robot", None)
    history_len = int(values.pop("history_len", 4))
    values.pop("mode", None)
    shared_execution = values.pop("shared_execution", None)
    values.update(task_overrides or {})
    mechanism_values = {
        **dict(values.pop("mechanism", {}) or {}),
        **dict(request["mechanism_parameters"]),
    }
    body_support = dict(request["body_support_parameters"])
    mechanism = mechanism_spec(
        request["task"], mechanism_values,
        body_support_parameters=body_support,
    )
    task = instantiate_task(
        config.env_name,
        robot=request["robot"],
        mechanism_spec=mechanism,
        seed=int(request["reset_seed"]),
        history_len=history_len,
        config_overrides=values or None,
        shared_execution=shared_execution,
    )
    task._experiment_mode = int(request["requested_mode"])
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


def _public_snapshot(public: dict[str, Any]) -> dict[str, np.ndarray]:
    snapshot = {}
    for key, value in public.items():
        array = np.asarray(value)
        if array.shape[0] != 1:
            raise ValueError("public-log collection requires batch-one observations")
        snapshot[key] = np.array(array[0], copy=True)
    return snapshot


def _execute_job(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
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
    device = str(job.get("device", "cpu"))
    try:
        import torch

        torch.set_num_threads(1)
        if device == "cuda":
            expected_visible = str(job.get("cuda_visible_devices", "1"))
            expected_uuid = str(job.get("physical_uuid", GPU1_UUID))
            if (
                os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible
                or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1
            ):
                raise RuntimeError("paired CUDA execution exposed the wrong device")
            actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
            if actual.lower() != expected_uuid.removeprefix("GPU-").lower():
                raise RuntimeError(f"wrong physical CUDA device: {actual}")
    except ImportError:
        if device == "cuda":
            raise
    request = job["request"]
    task = actor = None
    reset_succeeded = False
    actor_action_latencies: list[float] = []
    evidence_path = result_path.with_name("evidence.json.gz")
    try:
        config = ExperimentConfig.from_yaml(ROOT / job["config"])
        task = _task_from_job(config, request, job.get("task_overrides"))
        initial = "jammed" if job["task"] in {"T3", "T4"} else "clear"
        public = task.reset(
            (int(request["physical_initial_slot"]),),
            seed=int(request["reset_seed"]),
            initial=initial,
            modes=[int(request["requested_mode"])],
        )
        reset_succeeded = True
        public_log = job.get("public_log")
        public_rows = [_public_snapshot(public)] if public_log else []
        issued_requests = []
        parameters = deepcopy(config.method_params)
        parameters.update(mode=int(request["requested_mode"]), device=device)
        if job.get("actor_binding") is not None:
            selection = deepcopy(job["actor_binding"])
        else:
            selection = deepcopy(config.method_params["actor_binding"])
            selection.update(family=job["family"], key=job["key"])
            if job["instance"] != "fixed":
                selection["instance"] = job["instance"]
        parameters["actor_binding"] = selection
        task_actor_interface = job.get("task_actor_interface")
        if task_actor_interface is not None:
            if task_actor_interface not in T2_DIAGNOSTIC_ACTOR_INTERFACES:
                raise ValueError("unknown task actor interface")
            parameters["task_actor_interface"] = task_actor_interface
            interface_parameters = _task_actor_interface_parameters(
                task_actor_interface,
                job.get("task_actor_interface_parameters"),
            )
            if interface_parameters:
                parameters["task_actor_interface_parameters"] = (
                    interface_parameters
                )
        actor = create_transfer_actor(task, config.method, parameters)
        for _ in range(int(config.n_steps)):
            if bool(np.asarray(task.state.status)[0]):
                break
            action_started = time.perf_counter()
            try:
                action = np.asarray(actor.act(public), dtype=np.float32)
            except RiskSetEmptyError:
                task.records[-1].state = "rejected"
                task.records[-1].rejection_reason = "risk_set_empty"
                break
            finally:
                actor_action_latencies.append(time.perf_counter() - action_started)
            diagnostic = controller_diagnostics(actor)
            body_intent = task.public_body_intent(public)
            issued = np.asarray(action, dtype=np.float32).reshape(-1, 13)[0].copy()
            public = (
                task.step(action).observation
                if body_intent is None
                else task.step_contact(action, body_twist=body_intent).observation
            )
            if public_log:
                issued_requests.append(issued)
                public_rows.append(_public_snapshot(public))
            task.record_control_diagnostics(diagnostic)
        evaluation = evaluate_records(task.records, task.spec)
        evaluation["accounting"] = dict(task.accounting)
        evaluation["lifetime_accounting"] = dict(task.lifetime_accounting)
        evidence = episode_evidence(task.records, task.spec, evaluation)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(
            evidence, sort_keys=True, separators=(",", ":")
        ).encode()
        temporary = evidence_path.with_suffix(".gz.tmp")
        temporary.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        os.replace(temporary, evidence_path)
        audit = audit_episode_evidence(evidence_path)
        episode = evaluation["episodes"][-1]
        public_trajectory = None
        if public_log:
            if not issued_requests or len(public_rows) != len(issued_requests) + 1:
                raise ValueError("public-log transition accounting differs")
            trajectory = PublicTargetTrajectory(
                {
                    "version": FOUR_TASK_VERSION,
                    "permission": public_log["permission"],
                    "robot": request["robot"],
                    "mode": int(request["requested_mode"]),
                    "seed": int(request["reset_seed"]),
                    "purpose": public_log["purpose"],
                    "chart": public_log["chart"],
                    "controls": len(issued_requests),
                    "recording_end": (
                        "terminal" if bool(np.asarray(task.state.status)[0])
                        else "budget_stop"
                    ),
                    "task": job["task"],
                },
                {
                    key: np.stack([row[key] for row in public_rows])
                    for key in public_rows[0]
                },
                np.stack(issued_requests),
            )
            trajectory_path = result_path.parent / "public_trajectory"
            trajectory.save(trajectory_path)
            public_trajectory = {
                "path": str(trajectory_path.relative_to(ROOT)),
                "digest": trajectory.digest,
                "files": {
                    child.name: _sha(child)
                    for child in sorted(trajectory_path.iterdir()) if child.is_file()
                },
            }
        exact_t2 = bool(episode.get("exact_physics_contact_evidence", True))
        support_ok = True
        if job["task"] == "T4":
            valid = [row for row in task.records[-1].traces if row.get("valid")]
            support_ok = bool(valid) and min(
                int(row.get("support_count", 0)) for row in valid
            ) >= 2
        passed = bool(
            episode["success"]
            and not episode["constraint_violation"]
            and episode["mode_match"]
            and audit["passed"]
            and exact_t2
            and support_ok
        )
        result = {
            "record_id": job["record_id"],
            "row_id": job["row_id"],
            "task": job["task"],
            "family": job["family"],
            "key": job["key"],
            "method": job["method"],
            "instance": job["instance"],
            "request": request,
            "reset_succeeded": True,
            "passed": passed,
            "success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "outcome": episode["outcome"],
            "observed_mode": episode["observed_mode"],
            "mode_match": bool(episode["mode_match"]),
            "exact_physics_contact_evidence": exact_t2,
            "support_gate_passed": support_ok,
            "physical_control_steps": int(episode["physical_control_steps"]),
            "preparation_control_steps": int(episode["preparation_control_steps"]),
            "capped_normalized_completion_time": (
                min(
                    1.0,
                    float(episode["physical_control_steps"]) / float(config.n_steps),
                )
                if passed else 1.0
            ),
            "actor_action_latency_seconds": {
                "count": len(actor_action_latencies),
                "p50": (
                    float(np.quantile(actor_action_latencies, 0.50))
                    if actor_action_latencies else None
                ),
                "p95": (
                    float(np.quantile(actor_action_latencies, 0.95))
                    if actor_action_latencies else None
                ),
                "maximum": (
                    max(actor_action_latencies) if actor_action_latencies else None
                ),
            },
            "force_peak_N": episode.get("force_peak_N"),
            "lateral_force_peak_N": episode.get("lateral_force_peak_N"),
            "episode_metrics": {
                name: episode.get(name) for name in (
                    "final_position_error_m", "final_orientation_error_rad",
                    "first_contact_step", "first_outlet_step",
                    "contact_before_outlet", "strict_contact_success",
                    "continuous_guide_contact_ratio", "maximum_contact_loss_seconds",
                )
            },
            "evidence": {
                "path": str(
                    evidence_path.relative_to(ROOT)
                    if evidence_path.is_relative_to(ROOT)
                    else evidence_path
                ),
                "sha256": _sha(evidence_path),
                "audit_passed": bool(audit["passed"]),
            },
            **({"public_trajectory": public_trajectory} if public_trajectory else {}),
            "accounting": dict(task.accounting),
        }
    except BaseException as error:
        result = {
            "record_id": job["record_id"],
            "row_id": job["row_id"],
            "task": job["task"],
            "family": job["family"],
            "key": job["key"],
            "method": job["method"],
            "instance": job["instance"],
            "request": request,
            "reset_succeeded": reset_succeeded,
            "passed": False,
            "success": False,
            "constraint_violation": True,
            "outcome": "reset_or_execution_error",
            "capped_normalized_completion_time": 1.0,
            "actor_action_latency_seconds": {
                "count": len(actor_action_latencies),
                "p50": (
                    float(np.quantile(actor_action_latencies, 0.50))
                    if actor_action_latencies else None
                ),
                "p95": (
                    float(np.quantile(actor_action_latencies, 0.95))
                    if actor_action_latencies else None
                ),
                "maximum": (
                    max(actor_action_latencies) if actor_action_latencies else None
                ),
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "accounting": {} if task is None else dict(task.accounting),
        }
    result = _json_ready(result)
    _write_json(result_path, result)
    return result


def summarize(plan: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_row[result["row_id"]].append(result)
    rows = []
    for planned in plan["rows"]:
        row = deepcopy(planned)
        executed = by_row.get(row["row_id"], [])
        if row["physical_jobs"]:
            complete = len(executed) == row["physical_jobs"]
            passed = complete and all(result["passed"] for result in executed)
            row.update({
                "completed_physical_jobs": len(executed),
                "behavior_successes": sum(result["success"] for result in executed),
                "constraint_violations": sum(result["constraint_violation"] for result in executed),
                "outcomes": dict(Counter(result["outcome"] for result in executed)),
                "admission_status": (
                    "behavior_qualified" if passed else
                    "behavior_rejected" if complete else
                    "behavior_execution_incomplete"
                ),
                "final_for_current_protocol": complete,
            })
        rows.append(row)
    counts = Counter(row["admission_status"] for row in rows)
    all_jobs_complete = len(results) == len(plan["jobs"])
    public_sentinels = [
        row for row in rows
        if row["family"] == "main" and row["key"] == "public_geometry"
    ]
    common_task_gate = bool(public_sentinels) and all(
        row["admission_status"] == "behavior_qualified"
        for row in public_sentinels
    )
    # A method-owned load/runtime error is a retained behavior rejection for
    # that method.  Only the public task sentinel can diagnose a shared runner
    # or task-cell failure that blocks every later training job.
    infrastructure_errors = [
        result for result in results
        if result["outcome"] == "reset_or_execution_error"
        and result["family"] == "main"
        and result["key"] == "public_geometry"
    ]
    training_allowed = bool(
        all_jobs_complete and common_task_gate and not infrastructure_errors
    )
    formal_allowed = bool(
        training_allowed
        and all(row["final_for_current_protocol"] for row in rows)
        and not any(
            row["admission_status"] in {
                "training_artifact_required_after_shared_cell_repair",
                "current_protocol_artifact_unavailable",
                "behavior_execution_incomplete",
            }
            for row in rows
        )
    )
    return {
        "schema_version": plan["schema_version"],
        "status": (
            "formal_evaluation_admitted" if formal_allowed else
            "gpu_training_admitted" if training_allowed else
            "qualification_failed_or_incomplete"
        ),
        "passed": training_allowed,
        "gpu_training_allowed": training_allowed,
        "formal_evaluation_allowed": formal_allowed,
        "performance_evidence": False,
        "confirmation_states_opened": 0,
        "old_invalid_outcomes_used": False,
        "failure_replacement": False,
        "counts": {
            "logical_rows": len(rows),
            "planned_physical_jobs": len(plan["jobs"]),
            "completed_physical_jobs": len(results),
            "physical_control_steps": sum(
                int(result.get("accounting", {}).get("control_steps", 0))
                for result in results
            ),
            "preparation_control_steps": sum(
                int(result.get("accounting", {}).get("preparation_control_steps", 0))
                for result in results
            ),
            "admission_status": dict(counts),
        },
        "common_task_public_sentinel_passed": common_task_gate,
        "infrastructure_error_count": len(infrastructure_errors),
        "rows": rows,
    }


def _paired_qualification_checks(
    plan: dict[str, Any], results: list[dict[str, Any]],
) -> dict[str, bool]:
    gate = plan["acceptance"]
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_key[result["key"]].append(result)

    def passed_pairs(key: str) -> set[tuple[str, str]]:
        return {
            (result["instance"], result["request"]["request_id"])
            for result in by_key[key] if result["passed"]
        }

    candidate = by_key[gate["candidate"]]
    candidate_pairs = passed_pairs(gate["candidate"])
    current_pairs = passed_pairs(gate["current"])
    band_pairs = passed_pairs(gate["band_only"])
    reference_pairs = passed_pairs(gate["strong_reference"])
    required = int(gate["required_candidate_successes"])
    checks = {
        "complete_four_way_denominator": all(
            len(by_key[key]) == required
            for key in (
                gate["candidate"], gate["current"], gate["band_only"],
                gate["strong_reference"],
            )
        ),
        "zero_execution_errors": not any(
            result["outcome"] == "reset_or_execution_error" for result in results
        ),
        "candidate_required_successes": len(candidate_pairs) == required,
        "candidate_zero_constraint_violations": not any(
            result["constraint_violation"] for result in candidate
        ),
        "candidate_preserves_every_current_success": current_pairs <= candidate_pairs,
        "candidate_beats_band_only": len(candidate_pairs) > len(band_pairs),
    }
    if "require_candidate_to_match_strong_reference" in gate:
        checks["candidate_matches_strong_reference"] = (
            candidate_pairs == reference_pairs
        )
    else:
        checks["candidate_meets_or_exceeds_strong_reference"] = (
            len(candidate_pairs) >= len(reference_pairs)
        )
    return checks


def _paired_contribution_contrast(
    plan: dict[str, Any], results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the predeclared T3 transport-minus-matched-task-only effect."""
    gate = plan["acceptance"]
    confirmation = plan.get("schema_version") in {
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
    }
    rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for result in results:
        pair = (result["instance"], result["request"]["request_id"])
        if pair in rows[result["key"]]:
            raise ValueError("duplicate paired contribution result")
        rows[result["key"]][pair] = result
    transport = rows[gate["transport"]]
    task_only = rows[gate["task_only"]]
    required = int(gate["required_pairs"])
    paired = set(transport) == set(task_only) and len(transport) == required
    if not paired:
        return {
            "execution_complete": False,
            (
                "transport_claim_supported_confirmation"
                if confirmation else "transport_claim_supported_development"
            ): False,
            "reason": "paired denominator incomplete",
        }
    ordered = sorted(transport)
    transport_success = np.asarray([
        int(transport[key]["passed"]) for key in ordered
    ], dtype=float)
    task_success = np.asarray([
        int(task_only[key]["passed"]) for key in ordered
    ], dtype=float)
    deltas = transport_success - task_success
    by_instance = {}
    for instance in EXPECTED_INSTANCES:
        mask = np.asarray([key[0] == instance for key in ordered])
        by_instance[instance] = {
            "transport_successes": int(transport_success[mask].sum()),
            "task_only_successes": int(task_success[mask].sum()),
            "success_rate_gain": float(deltas[mask].mean()),
        }
    nonnegative = sum(
        row["success_rate_gain"] >= 0. for row in by_instance.values()
    )
    extra_constraints = sum(
        bool(transport[key]["constraint_violation"])
        and not bool(task_only[key]["constraint_violation"])
        for key in ordered
    )
    integrity = not any(
        row["outcome"] == "reset_or_execution_error"
        or not row.get("evidence", {}).get("audit_passed", False)
        for row in (*transport.values(), *task_only.values())
    )
    gain = float(deltas.mean())
    favorable = int((deltas > 0).sum())
    unfavorable = int((deltas < 0).sum())
    discordant = favorable + unfavorable
    exact_mcnemar_p = float(
        binomtest(
            favorable, discordant, p=0.5, alternative="two-sided",
        ).pvalue
    ) if discordant else 1.0
    task_success_preserved = all(
        not task_only[key]["passed"] or transport[key]["passed"]
        for key in ordered
    )
    paired_success_keys = [
        key for key in ordered
        if transport[key]["passed"] and task_only[key]["passed"]
    ]
    step_reductions = np.asarray([
        float(task_only[key]["physical_control_steps"])
        - float(transport[key]["physical_control_steps"])
        for key in paired_success_keys
    ])
    task_steps = np.asarray([
        float(task_only[key]["physical_control_steps"])
        for key in paired_success_keys
    ])
    mean_step_reduction_fraction = float(
        step_reductions.mean() / task_steps.mean()
    ) if len(step_reductions) and float(task_steps.mean()) > 0 else float("-inf")
    faster = int((step_reductions > 0).sum())
    slower = int((step_reductions < 0).sum())
    step_discordant = faster + slower
    exact_sign_p = float(
        binomtest(
            faster, step_discordant, p=0.5, alternative="greater",
        ).pvalue
    ) if step_discordant else 1.0
    nonnegative_time_instances = 0
    for instance in EXPECTED_INSTANCES:
        values = np.asarray([
            float(task_only[key]["physical_control_steps"])
            - float(transport[key]["physical_control_steps"])
            for key in paired_success_keys if key[0] == instance
        ])
        by_instance[instance].update({
            "paired_successes_for_time": int(len(values)),
            "mean_control_step_reduction": (
                float(values.mean()) if len(values) else None
            ),
        })
        nonnegative_time_instances += bool(len(values) and values.mean() >= 0.)
    time_gate_present = "minimum_mean_control_step_reduction_fraction" in gate
    time_supported = bool(
        not time_gate_present
        or mean_step_reduction_fraction
        >= float(gate["minimum_mean_control_step_reduction_fraction"])
        and nonnegative_time_instances
        >= int(gate["minimum_nonnegative_time_instances"])
        and exact_sign_p
        <= float(gate["maximum_one_sided_exact_sign_p_value"])
    )
    supported = bool(
        integrity
        and gain >= float(gate["minimum_aggregate_success_rate_gain"])
        and nonnegative >= int(gate["minimum_nonnegative_instances"])
        and exact_mcnemar_p <= float(
            gate.get("maximum_two_sided_exact_mcnemar_p_value", 1.0)
        )
        and (
            not gate["require_zero_extra_transport_constraint_violations"]
            or extra_constraints == 0
        )
        and (
            not gate.get(
                "require_transport_to_preserve_every_task_only_success", False,
            )
            or task_success_preserved
        )
        and time_supported
    )
    return {
        "execution_complete": True,
        "integrity_passed": integrity,
        "paired_requests": required,
        "transport_successes": int(transport_success.sum()),
        "task_only_successes": int(task_success.sum()),
        "transport_minus_task_only_success_rate": gain,
        "transport_favorable_pairs": favorable,
        "task_only_favorable_pairs": unfavorable,
        "tied_pairs": int((deltas == 0).sum()),
        "two_sided_exact_mcnemar_p_value": exact_mcnemar_p,
        "task_only_successes_preserved": task_success_preserved,
        "paired_successes_for_time": int(len(paired_success_keys)),
        "mean_control_step_reduction": (
            float(step_reductions.mean()) if len(step_reductions) else None
        ),
        "mean_control_step_reduction_fraction": (
            mean_step_reduction_fraction if np.isfinite(mean_step_reduction_fraction)
            else None
        ),
        "transport_faster_pairs": faster,
        "task_only_faster_pairs": slower,
        "control_step_ties": int((step_reductions == 0).sum()),
        "one_sided_exact_sign_p_value": exact_sign_p,
        "nonnegative_time_instances": nonnegative_time_instances,
        "nonnegative_instances": nonnegative,
        "extra_transport_constraint_violations": int(extra_constraints),
        "instances": by_instance,
        "decision_thresholds": {
            key: value for key, value in gate.items()
            if key.startswith("minimum_") or key.startswith("maximum_")
            or key.startswith("require_")
        },
        "transport_claim_supported_development": (
            supported if not confirmation else False
        ),
        "transport_claim_supported_confirmation": (
            supported if confirmation else False
        ),
        "interpretation": (
            "Seed-disjoint one-shot confirmation of the frozen transport effect."
            if confirmation else
            "Development direction gate only. A positive decision admits a fresh "
            "seed-disjoint confirmation contrast; it is not paper performance evidence."
        ),
    }


def _physical_checkpoint_selection(
    plan: dict[str, Any], results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select one offline-admitted checkpoint per instance on held-out physics."""
    gate = plan["acceptance"]
    by_key: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for result in results:
        pair = (result["instance"], result["request"]["request_id"])
        if pair in by_key[result["key"]]:
            raise ValueError("duplicate physical checkpoint selector result")
        by_key[result["key"]][pair] = result
    planned_rows = {row["key"]: row for row in plan["rows"]}
    task_key = gate["task_only"]
    required = int(gate["required_requests_per_instance"])
    selections = {}
    for instance in EXPECTED_INSTANCES:
        task_rows = {
            request: row for (owner, request), row in by_key[task_key].items()
            if owner == instance
        }
        task_successes = {
            request for request, row in task_rows.items() if row["passed"]
        }
        candidates = []
        if gate.get("allow_task_only_fallback", False):
            candidates.append({
                "label": task_key, "update": 0,
                "complete": len(task_rows) == required,
                "integrity_passed": all(
                    row["outcome"] != "reset_or_execution_error"
                    and row.get("evidence", {}).get("audit_passed", False)
                    for row in task_rows.values()
                ),
                "successes": len(task_successes),
                "task_only_successes_preserved": True,
                "extra_constraint_violations": 0,
                "mean_capped_control_steps": (
                    float(np.mean([
                        row["physical_control_steps"] for row in task_rows.values()
                    ])) if task_rows else None
                ),
                "eligible": len(task_rows) == required and all(
                    row["outcome"] != "reset_or_execution_error"
                    and row.get("evidence", {}).get("audit_passed", False)
                    for row in task_rows.values()
                ),
                "task_only_fallback": True,
            })
        for label, update in gate["candidate_updates"].items():
            rows = {
                request: row for (owner, request), row in by_key[label].items()
                if owner == instance
            }
            same_requests = set(rows) == set(task_rows) and len(rows) == required
            integrity = bool(
                same_requests
                and all(
                    row["outcome"] != "reset_or_execution_error"
                    and row.get("evidence", {}).get("audit_passed", False)
                    for row in rows.values()
                )
            )
            successes = {
                request for request, row in rows.items() if row["passed"]
            }
            extra_constraints = sum(
                bool(rows[request]["constraint_violation"])
                and not bool(task_rows[request]["constraint_violation"])
                for request in set(rows) & set(task_rows)
            )
            preserves = task_successes <= successes
            eligible = bool(
                integrity and preserves and extra_constraints == 0
            )
            mean_steps = (
                float(np.mean([
                    row["physical_control_steps"] for row in rows.values()
                ])) if rows else None
            )
            candidates.append({
                "label": label, "update": int(update),
                "complete": same_requests, "integrity_passed": integrity,
                "successes": len(successes),
                "task_only_successes_preserved": preserves,
                "extra_constraint_violations": int(extra_constraints),
                "mean_capped_control_steps": mean_steps,
                "eligible": eligible,
                "task_only_fallback": False,
            })
        eligible = [row for row in candidates if row["eligible"]]
        selected = min(
            eligible,
            key=lambda row: (
                -row["successes"], row["mean_capped_control_steps"],
                not row["task_only_fallback"],
                row["update"],
            ),
        ) if eligible else None
        selections[instance] = {
            "task_only_successes": len(task_successes),
            "task_only_mean_capped_control_steps": (
                float(np.mean([
                    row["physical_control_steps"] for row in task_rows.values()
                ])) if task_rows else None
            ),
            "candidates": candidates,
            "selected_update": None if selected is None else selected["update"],
            "selected_checkpoint": (
                None if selected is None else deepcopy(
                    planned_rows[selected["label"]]["candidate_policy_set"]
                    ["checkpoints"][instance]
                )
            ),
            "selected_metrics": selected,
        }
    complete = all(
        row["selected_checkpoint"] is not None for row in selections.values()
    )
    return {
        "selection_complete": complete,
        "selection_rule": gate["selection_rule"],
        "development_only": True,
        "performance_evidence": False,
        "instances": selections,
        "interpretation": (
            "Held-out development selector only. Selected checkpoints require a "
            "fresh seed-disjoint contribution contrast before confirmation."
        ),
    }


def _write_public_log_manifest(
    output: Path, plan: dict[str, Any], results: list[dict[str, Any]],
) -> Path:
    entries = []
    for result in results:
        value = result.get("public_trajectory")
        if value is None:
            raise ValueError("public-log result has no public trajectory")
        path = ROOT / value["path"]
        trajectory = PublicTargetTrajectory.load(path)
        if trajectory.digest != value["digest"]:
            raise ValueError("public trajectory digest changed")
        actual_files = {
            child.name: _sha(child)
            for child in sorted(path.iterdir()) if child.is_file()
        }
        if actual_files != value["files"]:
            raise ValueError("public trajectory files changed")
        entries.append({
            "path": str(path), "parent_sha256": trajectory.digest,
            "controls": len(trajectory.requests),
            "prefix_sha256": trajectory.digest,
            "mode": int(trajectory.metadata["mode"]),
        })
    controls = sum(entry["controls"] for entry in entries)
    manifest = {
        "permission": "B_log", "action_seconds": controls * .02,
        "controls": controls, "entries": entries,
        "available_to": ["FADA", "ours", "ordinary_model_reference"],
        "reward": False, "success_filter": False,
        "physical_episodes": len(entries),
        "physical_control_steps": controls,
        "confirmation_states_opened": 0,
        "performance_evidence": False,
        "source_plan": _pointer(output / "plan.json"),
    }
    path = output / "public_log_manifest.json"
    _write_json(path, manifest)
    return path


def _clopper_pearson(successes: int, trials: int, alpha: float = .05) -> list[float] | None:
    if trials == 0:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def _constraint_stress_episode(
    result: dict[str, Any], contract: dict[str, Any],
) -> dict[str, Any]:
    """Reaudit one episode at the physics rate without exposing evidence to actors."""
    if result["outcome"] == "reset_or_execution_error":
        return {"exact_substep_evidence": False, "event": False, "alignment": None}
    evidence_path = Path(result["evidence"]["path"])
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    payload = json.loads(gzip.decompress(evidence_path.read_bytes()))
    records = payload.get("records", [])
    if len(records) != 1:
        raise ValueError("constraint stress evidence must contain exactly one episode")
    record = records[0]
    traces = record.get("traces", [])
    diagnostics = record.get("control_diagnostics", [])
    if len(traces) != len(diagnostics):
        raise ValueError("constraint stress command/evidence lengths differ")
    selector_guard_controls = sum(
        int(row.get("last_selector_code", 0)) == 1 for row in diagnostics
    )
    selector_changed_controls = sum(
        row.get("last_candidate_action") is not None
        and row.get("last_selected_action") is not None
        and not np.allclose(
            np.asarray(row["last_candidate_action"], dtype=float),
            np.asarray(row["last_selected_action"], dtype=float),
        )
        for row in diagnostics
    )
    required = {
        "joint_speed_substeps", "force_peak_substeps", "lateral_force_substeps",
        "issued_action", "applied_action",
    }
    exact = bool(traces) and all(required <= set(row) for row in traces)
    if not exact:
        return {
            "exact_substep_evidence": False, "event": False, "alignment": None,
            "selector_guard_controls": selector_guard_controls,
            "selector_changed_controls": selector_changed_controls,
        }
    speed_limit = float(contract["joint_speed_limit_radps"])
    force_limit = float(contract["force_limit_N"])
    lateral_limit = float(contract["lateral_force_limit_N"])
    samples = []
    for control_index, row in enumerate(traces):
        speed = np.asarray(row["joint_speed_substeps"], dtype=float)
        force = np.asarray(row["force_peak_substeps"], dtype=float)
        lateral = np.asarray(row["lateral_force_substeps"], dtype=float)
        if speed.shape != force.shape or speed.shape != lateral.shape or speed.ndim != 1:
            raise ValueError("constraint stress substep evidence shapes differ")
        for substep_index in range(len(speed)):
            samples.append({
                "control_index": control_index,
                "substep_index": substep_index,
                "global_substep_index": len(samples),
                "joint_speed_radps": float(speed[substep_index]),
                "force_peak_N": float(force[substep_index]),
                "lateral_force_N": float(lateral[substep_index]),
                "speed_margin_radps": float(speed_limit - speed[substep_index]),
                "force_margin_N": float(force_limit - force[substep_index]),
                "lateral_force_margin_N": float(lateral_limit - lateral[substep_index]),
            })
    violating = [
        index for index, sample in enumerate(samples)
        if sample["speed_margin_radps"] < 0
        or sample["force_margin_N"] < 0
        or sample["lateral_force_margin_N"] < 0
    ]
    if not violating:
        return {
            "exact_substep_evidence": True, "event": False, "alignment": None,
            "selector_guard_controls": selector_guard_controls,
            "selector_changed_controls": selector_changed_controls,
            "maximum_joint_speed_radps": max(sample["joint_speed_radps"] for sample in samples),
            "maximum_force_peak_N": max(sample["force_peak_N"] for sample in samples),
            "maximum_lateral_force_N": max(sample["lateral_force_N"] for sample in samples),
        }
    first_index = violating[0]
    first = samples[first_index]
    control_index = int(first["control_index"])
    diagnostic = diagnostics[control_index]
    trace = traces[control_index]
    actions = {
        "common": diagnostic.get("last_common_action"),
        "candidate": diagnostic.get("last_candidate_action"),
        "selected": diagnostic.get("last_selected_action"),
        "task_requested": record.get("requested_actions", [])[control_index],
        "issued": trace["issued_action"],
        "applied": trace["applied_action"],
        "selector_code": diagnostic.get("last_selector_code"),
        "selector_semantics": {"0": "candidate", "1": "public_contact_guard"},
    }
    return {
        "exact_substep_evidence": True,
        "event": True,
        "selector_guard_controls": selector_guard_controls,
        "selector_changed_controls": selector_changed_controls,
        "alignment": {
            "last_safe_substep": samples[first_index - 1] if first_index else None,
            "first_violating_substep": first,
            "actions_at_first_violation_control": actions,
            "predicted_margins": {
                "status": "unavailable",
                "reason": contract["predicted_margin_policy"],
            },
        },
        "maximum_joint_speed_radps": max(sample["joint_speed_radps"] for sample in samples),
        "maximum_force_peak_N": max(sample["force_peak_N"] for sample in samples),
        "maximum_lateral_force_N": max(sample["lateral_force_N"] for sample in samples),
    }


def _constraint_stress_checks(
    plan: dict[str, Any], results: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = plan["stress_contract"]
    audited = {}
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    alignments = []
    for result in results:
        detail = _constraint_stress_episode(result, contract)
        audited[result["record_id"]] = detail
        key = (str(result["instance"]), int(result["request"]["reset_seed"]))
        pairs[key][str(result["key"])] = detail
        if detail.get("alignment") is not None:
            alignments.append({
                "record_id": result["record_id"], "comparator": result["key"],
                "instance": result["instance"],
                "reset_seed": result["request"]["reset_seed"],
                **detail["alignment"],
            })
    expected = set(contract["comparators"])
    complete_pairs = len(pairs) == 64 and all(set(pair) == expected for pair in pairs.values())
    exact = complete_pairs and all(
        detail["exact_substep_evidence"] for detail in audited.values()
    )
    cells = {"both": 0, "gated_only": 0, "ungated_only": 0, "neither": 0}
    if complete_pairs:
        for pair in pairs.values():
            gated = bool(pair["gated_full"]["event"])
            ungated = bool(pair["ungated_full"]["event"])
            cells[
                "both" if gated and ungated else
                "gated_only" if gated else
                "ungated_only" if ungated else "neither"
            ] += 1
    discordant = cells["gated_only"] + cells["ungated_only"]
    p_value = (
        float(binomtest(cells["ungated_only"], discordant, .5).pvalue)
        if discordant else None
    )
    direction_interval = _clopper_pearson(cells["ungated_only"], discordant)
    selector_controls = {
        comparator: sum(
            int(detail.get("selector_guard_controls", 0))
            for record_id, detail in audited.items()
            if f"/ablation/{comparator}/" in record_id
        )
        for comparator in contract["comparators"]
    }
    changed_controls = {
        comparator: sum(
            int(detail.get("selector_changed_controls", 0))
            for record_id, detail in audited.items()
            if f"/ablation/{comparator}/" in record_id
        )
        for comparator in contract["comparators"]
    }
    comparator_semantics_valid = bool(
        selector_controls["gated_full"] > 0
        and selector_controls["ungated_full"] == 0
        and changed_controls["ungated_full"] == 0
    )
    claim_supported = bool(
        exact and comparator_semantics_valid and discordant
        and cells["ungated_only"] > cells["gated_only"]
        and p_value is not None and p_value < .05
    )
    episode_summary = {}
    for comparator in contract["comparators"]:
        details = [
            detail for record_id, detail in audited.items()
            if f"/ablation/{comparator}/" in record_id
        ]
        episode_summary[comparator] = {}
        for name in (
            "maximum_joint_speed_radps", "maximum_force_peak_N",
            "maximum_lateral_force_N",
        ):
            values = np.asarray([detail[name] for detail in details], dtype=float)
            episode_summary[comparator][name] = {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95)), "maximum": float(values.max()),
            }
    result_by_pair = defaultdict(dict)
    for result in results:
        key = (str(result["instance"]), int(result["request"]["reset_seed"]))
        result_by_pair[key][str(result["key"])] = result
    step_delta = np.asarray([
        pair["gated_full"]["physical_control_steps"]
        - pair["ungated_full"]["physical_control_steps"]
        for pair in result_by_pair.values() if set(pair) == expected
    ], dtype=float)
    return {
        "execution_complete": complete_pairs,
        "all_episodes_have_exact_1ms_evidence": exact,
        "comparator_semantics_valid": comparator_semantics_valid,
        "public_contact_guard_selector_controls": selector_controls,
        "candidate_to_selected_changed_controls": changed_controls,
        "observed_episode_maxima": episode_summary,
        "paired_completion_step_delta_gated_minus_ungated": {
            "mean": float(step_delta.mean()), "median": float(np.median(step_delta)),
            "minimum": float(step_delta.min()), "maximum": float(step_delta.max()),
            "gated_faster_pairs": int((step_delta < 0).sum()),
            "equal_pairs": int((step_delta == 0).sum()),
            "gated_slower_pairs": int((step_delta > 0).sum()),
        },
        "paired_force_or_speed_events": cells,
        "event_counts": {
            "gated_full": cells["both"] + cells["gated_only"],
            "ungated_full": cells["both"] + cells["ungated_only"],
        },
        "discordant_pairs": discordant,
        "exact_two_sided_mcnemar_p": p_value,
        "ungated_event_direction_probability_exact_95pct_interval": direction_interval,
        "guard_reduction_claim_supported": claim_supported,
        "claim_rule": contract["claim_rule"],
        "reproduced_measured_event": bool(cells["both"] + cells["gated_only"] + cells["ungated_only"]),
        "predicted_margin_status": contract["predicted_margin_policy"],
        "event_alignments": alignments,
    }


def summarize_diagnostic(
    plan: dict[str, Any], results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_row[result["row_id"]].append(result)
    rows = []
    for planned in plan["rows"]:
        executed = by_row.get(planned["row_id"], [])
        rows.append({
            **deepcopy(planned), "completed_physical_jobs": len(executed),
            "successes": sum(result["success"] for result in executed),
            "constraint_violations": sum(result["constraint_violation"] for result in executed),
            "outcomes": dict(Counter(result["outcome"] for result in executed)),
            "execution_errors": sum(
                result["outcome"] == "reset_or_execution_error" for result in executed
            ),
        })
    complete = len(results) == len(plan["jobs"])
    integrity = complete and all(
        result.get("evidence", {}).get("audit_passed", False)
        for result in results if result["outcome"] != "reset_or_execution_error"
    )
    qualification = plan["schema_version"] in {
        DIAGNOSTIC_QUALIFICATION_SCHEMA, DIAGNOSTIC_QUALIFICATION_SCHEMA_V3,
    }
    contribution = plan["schema_version"] in {
        CONTRIBUTION_CONTRAST_SCHEMA, SAME_START_CONTRIBUTION_SCHEMA,
        RELATIVE_RESPONSE_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_CONTRIBUTION_SCHEMA,
        TASK_FIRST_DIRECTIONAL_SAME_STATE_CONTRIBUTION_SCHEMA,
        RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        FULL_RUNTIME_DOMINANCE_CONTRIBUTION_SCHEMA,
        MARGIN_RUNTIME_CONTRIBUTION_SCHEMA,
        ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        STABLE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        TRANSIENT_PHASE_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHASE_SCHEDULED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        PHYSICAL_SELECTED_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        EXPANDED_ENSEMBLE_ROBUST_ADAPTIVE_STEP_CONTRIBUTION_SCHEMA,
        BOUNDED_ENDPOINT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        CAPPED_EXPANSION_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        HORIZON_CONSISTENT_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        VALIDATED_BRIDGE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        COMPLETION_VALUE_ENSEMBLE_ROBUST_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONTRIBUTION_SCHEMA,
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_RECORDED_COMPLETION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CONVEX_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUPPORTED_PHASE_CAPPED_EXPANSION_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_CALIBRATED_AUTHORITY_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONTRIBUTION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        ENSEMBLE_ROBUST_SELECTED_CONTRIBUTION_SCHEMA,
    }
    confirmation = plan["schema_version"] in {
        MODE_CONDITIONED_FINITE_TIME_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_SUCCESS_SAFE_SELECTED_CONFIRMATION_SCHEMA,
        MODE_SPECIFIC_DUAL_PHYSICAL_SELECTED_CONFIRMATION_SCHEMA,
    }
    checkpoint_selector = (
        plan["schema_version"] in {
            PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
            ENSEMBLE_ROBUST_CHECKPOINT_SELECTOR_SCHEMA,
            MODE_SPECIFIC_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
            MODE_SPECIFIC_MODE_ZERO_PHYSICAL_CHECKPOINT_SELECTOR_SCHEMA,
        }
    )
    collection = plan["schema_version"] == PUBLIC_LOG_SCHEMA
    stress = plan["schema_version"] == CONSTRAINT_STRESS_SCHEMA
    qualification_checks = (
        _paired_qualification_checks(plan, results) if qualification else {}
    )
    contribution_contrast = (
        _paired_contribution_contrast(plan, results)
        if contribution and complete else {}
    )
    checkpoint_selection = (
        _physical_checkpoint_selection(plan, results)
        if checkpoint_selector and complete else {}
    )
    stress_checks = _constraint_stress_checks(plan, results) if stress and complete else {}
    passed = bool(
        complete and integrity
        and (not qualification or all(qualification_checks.values()))
        and (
            not contribution
            or contribution_contrast.get("execution_complete", False)
            and contribution_contrast.get("integrity_passed", False)
            and contribution_contrast.get(
                (
                    "transport_claim_supported_confirmation"
                    if confirmation
                    else "transport_claim_supported_development"
                ),
                False,
            )
        )
        and (
            not checkpoint_selector
            or checkpoint_selection.get("selection_complete", False)
        )
        and (
            not stress
            or stress_checks.get("execution_complete", False)
            and stress_checks.get("all_episodes_have_exact_1ms_evidence", False)
            and stress_checks.get("comparator_semantics_valid", False)
        )
        and (
            not collection
            or all(result.get("public_trajectory") is not None for result in results)
        )
    )
    return {
        "schema_version": plan["schema_version"],
        "status": (
            "complete_recorded_transport_confirmation"
            if confirmation and complete and integrity else
            "complete_recorded_public_log_collection"
            if collection and complete and integrity else
            "complete_recorded_development_contribution_contrast"
            if contribution and complete and integrity else
            "complete_recorded_physical_checkpoint_selection"
            if checkpoint_selector and complete and integrity else
            "complete_recorded_development_qualification"
            if qualification and complete and integrity else
            "complete_recorded_development_diagnosis" if complete and integrity
            else "incomplete_or_integrity_failed"
        ),
        "passed": passed, "diagnostic_only": not confirmation,
        "performance_evidence": confirmation,
        "confirmation_states_opened": int(confirmation),
        "old_invalid_outcomes_used": False, "failure_replacement": False,
        **({"qualification_checks": qualification_checks} if qualification else {}),
        **({"contribution_contrast": contribution_contrast} if contribution else {}),
        **(
            {"physical_checkpoint_selection": checkpoint_selection}
            if checkpoint_selector else {}
        ),
        **({"constraint_stress_checks": stress_checks} if stress else {}),
        **({
            "public_log_checks": {
                "complete_denominator": complete,
                "all_public_trajectories_written": all(
                    result.get("public_trajectory") is not None for result in results
                ),
                "no_success_filter": plan["public_log"]["permission"] == "B_log",
            }
        } if collection else {}),
        "counts": {
            "logical_rows": len(rows), "planned_physical_jobs": len(plan["jobs"]),
            "completed_physical_jobs": len(results),
            "physical_control_steps": sum(
                int(result.get("accounting", {}).get("control_steps", 0)) for result in results
            ),
            "execution_errors": sum(
                result["outcome"] == "reset_or_execution_error" for result in results
            ),
        },
        "rows": rows,
    }


def _runtime_device_proof(plan: dict[str, Any]) -> dict[str, Any]:
    device = str(plan.get("execution", {}).get("device", "cpu"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device == "cpu":
        if visible not in {"", "-1"}:
            raise RuntimeError("CPU behavioral execution cannot expose a CUDA device")
        return {"device": "cpu", "cuda_visible_devices": visible}
    expected_visible = str(plan.get("execution", {}).get(
        "cuda_visible_devices", "1"
    ))
    expected_index = int(plan.get("execution", {}).get("physical_index", 1))
    expected_uuid = str(plan.get("execution", {}).get(
        "physical_uuid", GPU1_UUID
    ))
    if device != "cuda" or visible != expected_visible:
        raise RuntimeError("paired CUDA execution exposed the wrong physical index")
    rows = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(expected_index), "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().split(", ", 1)
    if len(rows) != 2 or rows[0] != expected_uuid:
        raise RuntimeError(f"physical GPU identity changed: {rows}")
    return {
        "device": "cuda:0", "cuda_visible_devices": visible,
        "physical_index": expected_index, "physical_uuid": rows[0], "name": rows[1],
        "required_visible_device_count": 1,
    }


def compose_task_qualifications(
    declarations: list[str], output: Path,
) -> dict[str, Any]:
    """Compose independently audited task qualifications with unchanged manifests."""
    sources = {}
    for declaration in declarations:
        if "=" not in declaration:
            raise ValueError("compose sources must use TASK=RUN_DIRECTORY")
        task, raw = declaration.split("=", 1)
        if task not in TASKS or task in sources:
            raise ValueError(f"invalid or duplicate composed task: {task}")
        run = Path(raw)
        run = run if run.is_absolute() else ROOT / run
        sources[task] = run
    if set(sources) != set(TASKS):
        raise ValueError("qualification composition requires exactly T1,T2,T3,T4")

    rows = []
    source_pointers = {}
    checks = {}
    for task, run in sources.items():
        raw_report_path = run / "report.json"
        report_path = raw_report_path
        audit_path = run / "independent_audit.json"
        execution_admission_used = False
        for candidate in (
            run / "independent_execution_admission_from_prior_audit_v2.json",
            run / "independent_execution_admission_from_prior_audit.json",
            run / "independent_execution_admission_audit.json",
            run / "independent_audit.json",
        ):
            if not candidate.is_file():
                continue
            candidate_audit = json.loads(candidate.read_text())
            admission_pointer = candidate_audit.get("artifacts", {}).get(
                "execution_admission"
            )
            if (
                candidate_audit.get("passed") is True
                and candidate_audit.get("integrity_passed") is True
                and isinstance(admission_pointer, dict)
            ):
                admission_path = _checked_pointer(
                    admission_pointer, f"{task} execution admission",
                )
                report_path = admission_path
                audit_path = candidate
                execution_admission_used = True
                break
        plan_path = run / "plan.json"
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        plan = json.loads(plan_path.read_text())
        base = ExperimentConfig.from_yaml(TASKS[task] / "_base.yaml")
        current = base.method_params["actor_binding"]["manifest"]
        planned = plan["actor_manifests"][task]
        manifest_match = bool(
            current["path"] == planned["path"]
            and current["sha256"] == planned["sha256"]
            and _sha(ROOT / current["path"]) == current["sha256"]
        )
        selected = [deepcopy(row) for row in report["rows"] if row["task"] == task]
        expected_ids = {
            f"{task}/{family}/{key}"
            for family, mapping in (("main", MAIN_CONFIG), ("ablation", ABLATION_CONFIG))
            for key in mapping
        }
        source_ok = bool(
            report.get("formal_evaluation_allowed") is True
            and report.get("confirmation_states_opened") == 0
            and report.get("old_invalid_outcomes_used") is False
            and audit.get("passed") is True
            and audit.get("integrity_passed") is True
            and audit.get("artifacts", {}).get(
                "execution_admission", {}
            ).get("sha256") == _sha(report_path)
            and execution_admission_used
            and {row["row_id"] for row in selected} == expected_ids
            and all(row.get("final_for_current_protocol") is True for row in selected)
            and manifest_match
        )
        if not source_ok:
            raise ValueError(f"{task} qualification source is not composition-safe")
        rows.extend(selected)
        source_pointers[task] = {
            "report": _pointer(report_path), "independent_audit": _pointer(audit_path),
            "plan": _pointer(plan_path), "actor_manifest": deepcopy(current),
            **({"raw_behavior_report": _pointer(raw_report_path)}
               if report_path != raw_report_path else {}),
        }
        checks[task] = {
            "source_formal_gate_passed": True,
            "source_integrity_audit_passed": True,
            "exact_fifteen_row_coverage": True,
            "all_rows_final_for_source_protocol": True,
            "current_actor_manifest_matches_source_plan": True,
            "execution_admission_used": execution_admission_used,
        }
    if len(rows) != 60 or len({row["row_id"] for row in rows}) != 60:
        raise ValueError("composed qualification does not contain 60 unique rows")
    public_rows = [
        row for row in rows
        if row["family"] == "main" and row["key"] == "public_geometry"
    ]
    public_gate = len(public_rows) == 4 and all(
        row["admission_status"] == "behavior_qualified" for row in public_rows
    )
    counts = Counter(row["admission_status"] for row in rows)
    report = {
        "schema_version": "four-task-behavioral-predeployment-composition-v1",
        "status": "formal_evaluation_admitted" if public_gate else "qualification_failed",
        "passed": public_gate,
        "gpu_training_allowed": public_gate,
        "formal_evaluation_allowed": public_gate,
        "confirmation_states_opened": 0,
        "old_invalid_outcomes_used": False,
        "failure_replacement": False,
        "composition_scope": (
            "Task-wise composition of independently audited behavior gates; only a task "
            "whose current actor manifest exactly matches its source plan may be reused."
        ),
        "counts": {
            "logical_rows": len(rows), "admission_status": dict(counts),
            "source_task_reports": len(source_pointers),
        },
        "rows": rows,
        "sources": source_pointers,
        "composition_checks": checks,
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    _write_json(report_path, report)
    audit = {
        "schema_version": "behavioral-predeployment-composition-audit-v1",
        "status": "passed" if public_gate else "failed", "passed": public_gate,
        "integrity_passed": public_gate,
        "checks": {
            "four_task_sources_present": set(checks) == set(TASKS),
            "all_source_checks_passed": all(all(row.values()) for row in checks.values()),
            "sixty_unique_rows": len(rows) == len({row["row_id"] for row in rows}) == 60,
            "public_sentinel_qualified_for_every_task": public_gate,
            "confirmation_states_unopened": True,
            "old_invalid_outcomes_unused": True,
        },
        "artifacts": {"report": _pointer(report_path), **source_pointers},
    }
    audit_path = output / "independent_audit.json"
    _write_json(audit_path, audit)
    print(json.dumps({
        "status": report["status"], "output": str(report_path),
        "sha256": _sha(report_path), "counts": report["counts"],
        "audit_sha256": _sha(audit_path),
    }, indent=2, sort_keys=True))
    if not public_gate:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/_qualification/four_task_transfer/behavioral_predeployment_20261022_v13",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task", action="append", choices=tuple(TASKS))
    parser.add_argument("--protocol-version", type=int, default=DEFAULT_PROTOCOL_VERSION)
    parser.add_argument("--diagnostic-plan", type=Path)
    parser.add_argument(
        "--compose-source", action="append", default=[],
        help="Compose audited task results as TASK=RUN_DIRECTORY (repeat for T1-T4).",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--only-record-id", action="append", default=[],
        help=(
            "Execute only the named records from a diagnostic plan. Repeat for "
            "a bounded preflight; omit on --resume to complete the frozen plan."
        ),
    )
    args = parser.parse_args()
    if NATIVE_LIBRARY.is_file():
        os.environ.setdefault("CROSS_EBDM_NATIVE_LIBRARY", str(NATIVE_LIBRARY))
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.protocol_version < 1:
        parser.error("--protocol-version must be positive")
    if args.diagnostic_plan and args.task:
        parser.error("--diagnostic-plan cannot be mixed with --task")
    if args.diagnostic_plan and args.protocol_version != DEFAULT_PROTOCOL_VERSION:
        parser.error("--diagnostic-plan cannot be mixed with --protocol-version")
    if args.only_record_id and not args.diagnostic_plan:
        parser.error("--only-record-id requires --diagnostic-plan")
    if args.only_record_id and args.plan_only:
        parser.error("--only-record-id cannot be mixed with --plan-only")
    if args.compose_source:
        if args.diagnostic_plan or args.task or args.resume or args.plan_only:
            parser.error("--compose-source cannot be mixed with execution modes")
        if args.protocol_version != DEFAULT_PROTOCOL_VERSION or args.workers != 1:
            parser.error("qualification composition has no protocol-version or workers")
        output = args.output if args.output.is_absolute() else ROOT / args.output
        compose_task_qualifications(args.compose_source, output)
        return
    tasks = tuple(args.task or TASKS)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    plan_path = output / "plan.json"
    report_path = output / "report.json"
    progress_path = output / "progress.jsonl"
    diagnostic = args.diagnostic_plan is not None
    fresh_plan = (
        build_diagnostic_plan(args.diagnostic_plan)
        if diagnostic else build_plan(tasks, args.protocol_version)
    )
    device_proof = _runtime_device_proof(fresh_plan)
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"qualification output exists; use --resume without changing the plan: {output}"
            )
        plan = json.loads(plan_path.read_text())
        if plan != fresh_plan:
            raise ValueError("behavioral qualification plan changed during resume")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing nonempty qualification output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        plan = fresh_plan
        _write_json(plan_path, plan)
    if args.plan_only:
        print(json.dumps({
            "status": "planned_unexecuted", "output": str(plan_path),
            "sha256": _sha(plan_path), "counts": plan["counts"],
        }, indent=2, sort_keys=True))
        return
    completed = {
        row["record_id"]: row for row in _read_jsonl(progress_path)
    }
    missing = [job for job in plan["jobs"] if job["record_id"] not in completed]
    if args.only_record_id:
        selected_ids = set(args.only_record_id)
        if len(selected_ids) != len(args.only_record_id):
            parser.error("--only-record-id values must be unique")
        known_ids = {job["record_id"] for job in plan["jobs"]}
        unknown_ids = sorted(selected_ids - known_ids)
        if unknown_ids:
            parser.error(f"unknown diagnostic record ids: {unknown_ids}")
        missing = [job for job in missing if job["record_id"] in selected_ids]
    if args.workers == 1:
        for job in missing:
            result = _execute_job((job, str(output)))
            completed[result["record_id"]] = result
            _append_jsonl(progress_path, result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_execute_job, (job, str(output))): job["record_id"]
                for job in missing
            }
            for future in as_completed(futures):
                result = future.result()
                completed[result["record_id"]] = result
                _append_jsonl(progress_path, result)
                _write_json(output / "progress.json", {
                    "completed": len(completed),
                    "planned": len(plan["jobs"]),
                    "last_record_id": result["record_id"],
                })
    results = [completed[job["record_id"]] for job in plan["jobs"] if job["record_id"] in completed]
    report = summarize_diagnostic(plan, results) if diagnostic else summarize(plan, results)
    if plan.get("schema_version") == PUBLIC_LOG_SCHEMA and report["passed"]:
        report["public_log_manifest"] = _pointer(
            _write_public_log_manifest(output, plan, results)
        )
    report["device"] = device_proof
    if args.only_record_id:
        report["execution_selection"] = {
            "scope": "bounded diagnostic preflight",
            "record_ids": list(args.only_record_id),
            "resume_without_selection_completes_frozen_plan": True,
        }
    report["artifacts"] = {
        "plan": _pointer(plan_path),
        "progress": _pointer(progress_path),
        "runner": _pointer(Path(__file__).resolve()),
    }
    _write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(report_path),
        "sha256": _sha(report_path),
        "counts": report["counts"],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
