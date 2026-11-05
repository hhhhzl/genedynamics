#!/usr/bin/env python3
"""Run a bounded T4 development gate for projected completion execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import resolve_transfer_binding
from scripts.paper.cross_ebdm.qualify_method_behavior import (
    _execute_job,
    _runtime_device_proof,
)


ROOT = Path(__file__).resolve().parents[3]
VERSION = "t4-completion-projected-execution-development-v1"
FACTORIAL_VERSION = "t4-transport-completion-factorial-v1"
SAME_START_FACTORIAL_VERSION = "t4-transport-completion-same-start-factorial-v1"
EXPECTED_INSTANCES = tuple(f"instance_{index:02d}" for index in range(1, 9))


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_factorial_config(config_path: Path) -> dict[str, Any]:
    """Load the original full factorial or a narrow SHA-bound policy variant."""
    raw = json.loads(config_path.read_text())
    parent = raw.get("base_factorial_config")
    if parent is None:
        return raw
    if set(raw) != {"base_factorial_config", "overrides"}:
        raise ValueError("compact factorial requires a base and overrides only")
    parent_path = checked(parent, "base factorial config")
    base = json.loads(parent_path.read_text())
    if base.get("version") != FACTORIAL_VERSION:
        raise ValueError("compact factorial base version changed")
    overrides = raw["overrides"]
    if set(overrides) != {"version", "policy_sets", "output"}:
        raise ValueError("compact factorial may only replace policy sets and output")
    if overrides["version"] != SAME_START_FACTORIAL_VERSION:
        raise ValueError("unknown compact factorial version")
    if set(overrides["policy_sets"]) != {"transport", "task_only"}:
        raise ValueError("same-start factorial requires two policy sets")
    config = deepcopy(base)
    config.update(deepcopy(overrides))
    config["resolved_base_factorial_config"] = pointer(parent_path)
    return config


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def final_completion_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    evidence_path = Path(result["evidence"]["path"])
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    evidence = json.loads(gzip.decompress(evidence_path.read_bytes()))
    rows = evidence["records"][0]["control_diagnostics"]
    if not rows:
        raise ValueError("episode has no public controller diagnostics")
    final = rows[-1]
    expected = (
        "genedynamics.tasks.transfer.controllers.recovery_execution."
        "PublicCompletionDominantRecoveryExecution"
    )
    if final.get("implementation") != expected:
        raise ValueError("episode did not execute the projected completion wrapper")
    learned = int(final["learned_controls"])
    terminal = int(final["completion_terminal_controls"])
    admission = int(final["completion_admission_controls"])
    eligible = learned - terminal
    blends = [int(value) for value in final["completion_blend_counts"]]
    if eligible < 0 or admission < 0 or admission > eligible:
        raise ValueError("invalid completion-admission accounting")
    if sum(blends) != int(result["accounting"]["control_steps"]):
        raise ValueError("completion blend denominator differs from control accounting")
    return {
        "common_prefix_controls": int(final["common_controls"]),
        "learned_enabled_controls": learned,
        "terminal_controls": terminal,
        "eligible_nonterminal_controls": eligible,
        "positive_admission_controls": admission,
        "common_fallback_controls": int(final["completion_common_fallback_controls"]),
        "blend_counts": {
            "1.0": blends[0], "0.5": blends[1],
            "0.25": blends[2], "0.0": blends[3],
        },
    }


def run(config: dict[str, Any], config_path: Path) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("T4 completion development is CPU-only")
    preflight_path = checked(config["projection_preflight"], "projection preflight")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("new_physics_calls") != 0:
        raise ValueError("passed zero-physics projection preflight required")
    base_path = checked(config["base_config"], "T4 base method config")
    base = ExperimentConfig.from_yaml(base_path)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)

    jobs = []
    bindings = {}
    for instance, row in sorted(config["instances"].items()):
        checkpoint_path = checked(row["candidate_checkpoint"], f"{instance} candidate")
        selection = deepcopy(base.method_params["actor_binding"])
        selection.update(family="main", key="cross_ebdm", instance=instance)
        binding = resolve_transfer_binding(
            {"actor_binding": selection}, task_alias="T4", instance_id=instance,
        )
        binding["policy_checkpoint"] = pointer(checkpoint_path)
        binding["completion_admission_version"] = "public_completion_dominance_v1"
        binding["method_identity"] = "ours_gated_v2_completion_projected"
        bindings[instance] = binding
        for request in config["requests"]:
            if request.get("confirmation_state") is not False:
                raise ValueError("development request cannot open confirmation state")
            if request.get("performance_evidence") is not False:
                raise ValueError("development request cannot claim performance evidence")
            record_id = f"T4/completion_projected/{instance}/{request['request_id']}"
            jobs.append({
                "record_id": record_id,
                "row_id": f"T4/validation/cross_ebdm/{instance}",
                "task": "T4", "family": "main", "key": "cross_ebdm",
                "method": "cross_ebdm", "instance": instance,
                "config": str(base_path.relative_to(ROOT)),
                "actor_binding": binding,
                "request": deepcopy(request),
            })
    expected = int(config["gate"]["expected_episodes_per_instance"])
    if len(config["requests"]) != expected:
        raise ValueError("development request denominator changed")
    plan = {
        "schema_version": VERSION,
        "status": "planned_unexecuted",
        "jobs": jobs,
        "bindings": bindings,
        "gate": config["gate"],
        "confirmation_states_opened": 0,
        "performance_evidence": False,
        "artifacts": {
            "config": pointer(config_path),
            "projection_preflight": pointer(preflight_path),
            "runner": pointer(Path(__file__).resolve()),
        },
    }
    write_json(output / "plan.json", plan)
    begun = time.monotonic()
    workers = int(config["limits"]["workers"])
    if workers < 1:
        raise ValueError("positive CPU worker count required")
    completed = {}
    if workers == 1:
        for job in jobs:
            result = _execute_job((job, str(output)))
            completed[result["record_id"]] = result
            write_json(output / "progress.json", list(completed.values()))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_job, (job, str(output))): job["record_id"]
                for job in jobs
            }
            for future in as_completed(futures):
                result = future.result()
                completed[result["record_id"]] = result
                write_json(output / "progress.json", list(completed.values()))
                if time.monotonic() - begun > float(
                    config["limits"]["maximum_wall_seconds"]
                ):
                    raise TimeoutError("T4 completion development wall ceiling")
    results = [completed[job["record_id"]] for job in jobs]

    diagnostics = {}
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"admission": 0, "eligible": 0}
    )
    instances: dict[str, dict[str, int]] = defaultdict(
        lambda: {"admission": 0, "eligible": 0, "episodes": 0, "successes": 0}
    )
    total_admission = total_eligible = 0
    diagnostic_errors = []
    for result in results:
        if result.get("evidence") is None:
            diagnostic_errors.append(result["record_id"])
            continue
        try:
            value = final_completion_diagnostics(result)
        except BaseException as error:
            diagnostic_errors.append(f"{result['record_id']}: {error}")
            continue
        diagnostics[result["record_id"]] = value
        admission = value["positive_admission_controls"]
        eligible = value["eligible_nonterminal_controls"]
        total_admission += admission
        total_eligible += eligible
        instance = result["instance"]
        instances[instance]["admission"] += admission
        instances[instance]["eligible"] += eligible
        instances[instance]["episodes"] += 1
        instances[instance]["successes"] += int(result["passed"])
        group = f"{instance}/{result['request']['scene_cell']}"
        groups[group]["admission"] += admission
        groups[group]["eligible"] += eligible

    fraction = total_admission / max(total_eligible, 1)
    instance_summary = {
        name: {
            **value,
            "candidate_admission_fraction": value["admission"] / max(value["eligible"], 1),
        }
        for name, value in sorted(instances.items())
    }
    checks = {
        "complete_denominator": len(results) == len(jobs),
        "all_task_success": len(results) == len(jobs) and all(
            result["passed"] for result in results
        ),
        "zero_constraint_violations": not any(
            result["constraint_violation"] for result in results
        ),
        "diagnostics_complete": not diagnostic_errors and len(diagnostics) == len(jobs),
        "minimum_candidate_admission": len(instance_summary) == len(config["instances"])
        and all(
            value["candidate_admission_fraction"]
            >= float(config["gate"]["minimum_candidate_admission_fraction"])
            for value in instance_summary.values()
        ),
        "every_mechanism_support_group_uses_candidate": bool(groups) and all(
            value["admission"] > 0 for value in groups.values()
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": VERSION,
        "status": "complete" if passed else "rejected",
        "passed": passed,
        "checks": checks,
        "counts": {
            "planned_episodes": len(jobs),
            "completed_episodes": len(results),
            "task_successes": sum(bool(result["success"]) for result in results),
            "constraint_violations": sum(
                bool(result["constraint_violation"]) for result in results
            ),
            "eligible_nonterminal_controls": total_eligible,
            "positive_admission_controls": total_admission,
            "candidate_admission_fraction": fraction,
        },
        "groups": dict(groups),
        "instances": instance_summary,
        "diagnostic_errors": diagnostic_errors,
        "diagnostics": diagnostics,
        "results": results,
        "confirmation_states_opened": 0,
        "performance_evidence": False,
        "new_physics_calls": len(results),
        "wall_seconds": time.monotonic() - begun,
        "artifacts": {
            "plan": pointer(output / "plan.json"),
            "config": pointer(config_path),
            "projection_preflight": pointer(preflight_path),
            "runner": pointer(Path(__file__).resolve()),
        },
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"), "counts": report["counts"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("T4 completion projected development gate failed")


def _factorial_policy_set(
    name: str, value: dict[str, Any], base_path: Path,
) -> tuple[ExperimentConfig, dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = checked(value["training_report"], f"{name} training report")
    report = json.loads(report_path.read_text())
    expected = {
        "transport": "t4-completion-tail-aligned-transport-training-v2",
        "task_only": "t4-completion-tail-aligned-matched-task-only-training-v1",
    }[name]
    if (
        report.get("schema_version") != expected
        or set(report.get("instances", {})) != set(EXPECTED_INSTANCES)
        or report.get("new_physics_calls") != 0
        or report.get("target_reward_or_truth_opened") is not False
    ):
        raise ValueError(f"{name} training report is not admitted")
    variant = report.get("training_variant", {"name": "transport"})
    if variant.get("name") != name.replace("task_only", "matched_task_only"):
        raise ValueError(f"{name} training factor differs")
    matched_training_admitted = True
    composite_qualification = None
    if name == "transport":
        if (
            report.get("status") != "selection_failed"
            or report.get("passed") is not False
            or value.get("control_policy")
            != "qualified_completion_composite_endpoints"
        ):
            raise ValueError("transport composite control policy changed")
        matched_training_admitted = False

        qualification_value = value.get("composite_qualification", {})
        qualification_path = checked(
            qualification_value["report"], "transport composite qualification report",
        )
        qualification_config_path = checked(
            qualification_value["config"], "transport composite qualification config",
        )
        qualification = json.loads(qualification_path.read_text())
        qualification_config = json.loads(qualification_config_path.read_text())
        counts = qualification.get("counts", {})
        if (
            qualification.get("schema_version")
            != "t4-completion-projected-execution-development-v1"
            or qualification.get("status") != "complete"
            or qualification.get("passed") is not True
            or qualification.get("performance_evidence") is not False
            or qualification.get("confirmation_states_opened") != 0
            or qualification.get("diagnostic_errors") != []
            or not all(qualification.get("checks", {}).values())
            or counts.get("planned_episodes") != 64
            or counts.get("completed_episodes") != 64
            or counts.get("task_successes") != 64
            or counts.get("constraint_violations") != 0
            or qualification.get("artifacts", {}).get("config")
            != pointer(qualification_config_path)
        ):
            raise ValueError("transport composite qualification did not pass")
        if (
            qualification_config.get("version")
            != "t4-completion-projected-execution-development-v1"
            or qualification_config.get("base_config") != pointer(base_path)
            or set(qualification_config.get("instances", {}))
            != set(EXPECTED_INSTANCES)
            or qualification_config.get("gate", {}).get(
                "require_all_mode_correct_task_success"
            ) is not True
            or qualification_config.get("gate", {}).get(
                "require_zero_constraint_violations"
            ) is not True
        ):
            raise ValueError("transport composite qualification config changed")
        qualification_results = qualification.get("results", [])
        if len(qualification_results) != 64:
            raise ValueError("transport composite evidence denominator changed")
        for result in qualification_results:
            if (
                result.get("passed") is not True
                or result.get("constraint_violation") is not False
                or result.get("evidence", {}).get("audit_passed") is not True
            ):
                raise ValueError("transport composite episode did not qualify")
            checked(
                result["evidence"],
                f"transport composite evidence {result.get('record_id')}",
            )
        composite_qualification = {
            "report": pointer(qualification_path),
            "config": pointer(qualification_config_path),
            "qualified_policy_identity": "checkpoint_plus_public_completion_dominance_v1",
            "qualified_episodes": 64,
        }
    else:
        if (
            report.get("status") != "selection_failed"
            or report.get("passed") is not False
            or value.get("control_policy")
            != "predeclared_frozen_task_only_fallback"
        ):
            raise ValueError("task-only factorial control policy changed")
        matched_training_admitted = False
    checkpoints = {}
    for instance in EXPECTED_INSTANCES:
        row = report["instances"][instance]
        accepted = row.get("selection", {}).get("accepted", False)
        if name == "transport":
            if accepted:
                raise ValueError(
                    f"{name} {instance} unexpectedly passed the raw selector"
                )
            endpoint = report_path.parent / instance / "checkpoint.pt"
            qualified = qualification_config["instances"][instance][
                "candidate_checkpoint"
            ]
            if pointer(endpoint) != qualified:
                raise ValueError(
                    f"{name} {instance} qualified endpoint lineage differs"
                )
            path = checked(qualified, f"{name} {instance} qualified checkpoint")
        if name == "task_only":
            if (
                accepted
                or row.get("checkpoint_selection") is not None
                or row.get("selected_update") != row.get("updates")
                or row.get("selected_checkpoint")
                != row.get("inputs", {}).get("task_only_checkpoint")
            ):
                raise ValueError(
                    f"task-only {instance} did not use its predeclared fallback"
                )
            path = checked(
                row["selected_checkpoint"], f"{name} {instance} checkpoint"
            )
        checkpoints[instance] = pointer(path)
    return ExperimentConfig.from_yaml(base_path), checkpoints, {
        "training_report": pointer(report_path),
        "training_variant": variant,
        "method_identity": value["method_identity"],
        "binding_key": value["binding_key"],
        "matched_training_admitted": matched_training_admitted,
        "raw_training_selection_admitted": bool(
            report.get("status") == "complete" and report.get("passed") is True
        ),
        "composite_qualification": composite_qualification,
        "control_policy": value.get("control_policy", "selected_trained_checkpoint"),
    }


def _same_start_factorial_policy_set(
    name: str, value: dict[str, Any], base_path: Path,
) -> tuple[ExperimentConfig, dict[str, dict[str, Any]], dict[str, Any]]:
    expected_schema = {
        "transport": "t4-completion-same-start-anchored-training-v1",
        "task_only": "t4-completion-same-start-task-only-training-v1",
    }[name]
    expected_variant = {
        "transport": "same_start_anchored",
        "task_only": "same_start_task_only",
    }[name]
    expected_weight = 1.0 if name == "transport" else 0.0
    report_path = checked(value["training_report"], f"{name} training report")
    report = json.loads(report_path.read_text())
    instances = report.get("instances", {})
    variant = report.get("training_variant", {})
    if (
        report.get("schema_version") != expected_schema
        or report.get("status") != "complete"
        or report.get("passed") is not True
        or set(instances) != set(EXPECTED_INSTANCES)
        or report.get("new_physics_calls") != 0
        or report.get("target_reward_or_truth_opened") is not False
        or report.get("training_updates") != 12000
        or variant.get("name") != expected_variant
        or variant.get("initialization_policy") != "task_only"
        or float(variant.get("response_anchoring_weight", -1.0)) != expected_weight
        or value.get("control_policy")
        != "same_start_terminal_checkpoint_plus_shared_projection"
    ):
        raise ValueError(f"{name} same-start training report is not admitted")

    qualification_value = value["projection_qualification"]
    qualification_config_path = checked(
        qualification_value["config"], f"{name} projection config",
    )
    qualification_path = checked(
        qualification_value["report"], f"{name} projection report",
    )
    qualification = json.loads(qualification_path.read_text())
    qualified_instances = qualification.get("instances", {})
    if (
        qualification.get("schema_version")
        != "t4-completion-projected-execution-preflight-v1"
        or qualification.get("status") != "complete"
        or qualification.get("passed") is not True
        or qualification.get("new_physics_calls") != 0
        or qualification.get("target_reward_or_truth_opened") is not False
        or qualification.get("projection", {}).get("version")
        != "public_completion_dominance_v1"
        or qualification.get("artifacts", {}).get("config")
        != pointer(qualification_config_path)
        or set(qualified_instances) != set(EXPECTED_INSTANCES)
    ):
        raise ValueError(f"{name} projected checkpoint qualification changed")

    checkpoints = {}
    sample_streams = {}
    input_digests = {}
    initialization_digests = {}
    for instance in EXPECTED_INSTANCES:
        row = instances[instance]
        checkpoint = row.get("control_checkpoint")
        endpoint = report_path.parent / instance / "checkpoint.pt"
        initialization = dict(row.get("initialization", {}))
        initialization.pop("training_variant", None)
        if (
            row.get("control_checkpoint_policy") != "terminal_update"
            or checkpoint != pointer(endpoint)
            or qualified_instances[instance].get("accepted") is not True
            or qualified_instances[instance].get("candidate_checkpoint") != checkpoint
        ):
            raise ValueError(f"{name} {instance} terminal checkpoint lineage changed")
        checked(checkpoint, f"{name} {instance} terminal checkpoint")
        checkpoints[instance] = checkpoint
        sample_streams[instance] = row["sample_stream_sha256"]
        input_digests[instance] = _json_digest(row["inputs"])
        initialization_digests[instance] = _json_digest(initialization)

    return ExperimentConfig.from_yaml(base_path), checkpoints, {
        "training_report": pointer(report_path),
        "training_variant": variant,
        "method_identity": value["method_identity"],
        "binding_key": value["binding_key"],
        "matched_training_admitted": True,
        "raw_training_selection_admitted": all(
            row.get("selection", {}).get("accepted", False)
            for row in instances.values()
        ),
        "composite_qualification": {
            "report": pointer(qualification_path),
            "config": pointer(qualification_config_path),
            "qualified_policy_identity": (
                "same_start_terminal_checkpoint_plus_"
                "public_completion_dominance_v1"
            ),
            "qualified_instances": 8,
        },
        "control_policy": value["control_policy"],
        "matching_contract": {
            "training_updates": report["training_updates"],
            "sample_streams": sample_streams,
            "input_digests": input_digests,
            "initialization_digests": initialization_digests,
        },
    }


def _factorial_effects(
    results: list[dict[str, Any]], decision: dict[str, Any], *,
    matched_control_admitted: bool = True,
    transport_composite_admitted: bool = True,
) -> dict[str, Any]:
    indexed = {
        (row["key"], row["instance"], row["request"]["request_id"]): row
        for row in results
    }
    labels = (
        "task_only_projection_off", "task_only_projection_on",
        "transport_projection_off", "transport_projection_on",
    )
    pairs = sorted({(row["instance"], row["request"]["request_id"]) for row in results})
    if len(indexed) != len(results) or any(
        (label, *pair) not in indexed for label in labels for pair in pairs
    ):
        raise ValueError("T4 factorial paired denominator differs")

    success = {
        label: np.asarray([int(indexed[(label, *pair)]["passed"]) for pair in pairs])
        for label in labels
    }
    task_projection = success["task_only_projection_on"] - success["task_only_projection_off"]
    transport_projection = success["transport_projection_on"] - success["transport_projection_off"]
    primary = success["transport_projection_on"] - success["task_only_projection_on"]
    interaction = transport_projection - task_projection
    instances = {}
    for instance in EXPECTED_INSTANCES:
        mask = np.asarray([pair[0] == instance for pair in pairs])
        instances[instance] = {
            "transport_on_minus_task_only_on": float(primary[mask].mean()),
            "task_only_projection_effect": float(task_projection[mask].mean()),
            "transport_projection_effect": float(transport_projection[mask].mean()),
            "interaction": float(interaction[mask].mean()),
        }
    extra_constraints = sum(
        bool(indexed[("transport_projection_on", *pair)]["constraint_violation"])
        and not bool(indexed[("task_only_projection_on", *pair)]["constraint_violation"])
        for pair in pairs
    )
    nonnegative = sum(
        value["transport_on_minus_task_only_on"] >= 0.
        for value in instances.values()
    )
    primary_gain = float(primary.mean())
    favorable = int((primary > 0).sum())
    unfavorable = int((primary < 0).sum())
    discordant = favorable + unfavorable
    exact_mcnemar_p = 1.0 if not discordant else min(
        1.0,
        2.0 * sum(
            math.comb(discordant, k) for k in range(min(favorable, unfavorable) + 1)
        ) / (2.0 ** discordant),
    )
    supported = bool(
        transport_composite_admitted
        and primary_gain >= float(decision["minimum_primary_success_rate_gain"])
        and nonnegative >= int(decision["minimum_nonnegative_instances"])
        and extra_constraints == 0
        and exact_mcnemar_p <= float(
            decision.get("maximum_two_sided_exact_mcnemar_p_value", 1.0)
        )
    )
    return {
        "cell_successes": {
            label: int(values.sum()) for label, values in success.items()
        },
        "paired_requests_per_cell": len(pairs),
        "primary_transport_on_minus_task_only_on": primary_gain,
        "task_only_projection_main_effect": float(task_projection.mean()),
        "transport_projection_main_effect": float(transport_projection.mean()),
        "transport_by_projection_interaction": float(interaction.mean()),
        "primary_favorable_pairs": favorable,
        "task_only_favorable_pairs": unfavorable,
        "primary_tied_pairs": int((primary == 0).sum()),
        "primary_two_sided_exact_mcnemar_p_value": exact_mcnemar_p,
        "nonnegative_instances": nonnegative,
        "extra_transport_constraint_violations": int(extra_constraints),
        "instances": instances,
        "decision_thresholds": decision,
        "transport_vs_strong_task_only_direction_supported": supported,
        "qualified_transport_composite_admitted": transport_composite_admitted,
        "matched_task_only_training_admitted": matched_control_admitted,
        "transport_policy_increment_supported_development": supported,
        "response_anchoring_causal_claim_supported": bool(
            supported and matched_control_admitted
        ),
        "transport_claim_supported_development": bool(
            supported and matched_control_admitted
        ),
        "interpretation": (
            "Development direction gate only. The primary effect holds the public "
            "completion projection on in both arms. When matched_control_admitted is "
            "true, the arms also share initialization, public data, sample streams, "
            "optimizer reset, update budget, and terminal-checkpoint policy, isolating "
            "the response-anchoring training term. Fresh frozen confirmation remains "
            "required for performance evidence."
        ),
    }


def run_factorial(
    config: dict[str, Any], config_path: Path, *, plan_only: bool = False,
) -> None:
    if (
        config.get("execution") != {"device": "cuda"}
        or config.get("diagnostic_only") is not True
        or config.get("performance_evidence") is not False
        or config.get("confirmation_states_opened") != 0
        or config.get("failure_replacement") is not False
    ):
        raise ValueError("T4 contribution contrast contract changed")
    device = _runtime_device_proof({"execution": config["execution"]})
    base_paths = {
        name: checked(value, f"{name} base config")
        for name, value in config["base_configs"].items()
    }
    if set(base_paths) != {"transport", "task_only"}:
        raise ValueError("T4 factorial requires transport and task-only bases")
    policies = {}
    for name in ("transport", "task_only"):
        loader = (
            _same_start_factorial_policy_set
            if config.get("version") == SAME_START_FACTORIAL_VERSION
            else _factorial_policy_set
        )
        policies[name] = loader(
            name, config["policy_sets"][name], base_paths[name],
        )
    same_start_matching = None
    if config.get("version") == SAME_START_FACTORIAL_VERSION:
        left = policies["transport"][2]["matching_contract"]
        right = policies["task_only"][2]["matching_contract"]
        same_start_matching = {
            key: left[key] == right[key]
            for key in (
                "training_updates", "sample_streams", "input_digests",
                "initialization_digests",
            )
        }
        if not all(same_start_matching.values()):
            raise ValueError("same-start factorial training arms are not matched")
    requests = list(config["requests"])
    expected = int(config["design"]["requests_per_instance"])
    if len(requests) != expected or any(
        request.get("confirmation_state") is not False
        or request.get("performance_evidence") is not False
        for request in requests
    ):
        raise ValueError("T4 factorial request denominator changed")
    profiles = {"projection_off": None, "projection_on": "public_completion_dominance_v1"}
    jobs = []
    bindings = {}
    for policy_name, (base, checkpoints, metadata) in policies.items():
        key = metadata["binding_key"]
        base_selection = deepcopy(base.method_params["actor_binding"])
        for instance in EXPECTED_INSTANCES:
            for profile, projection in profiles.items():
                selection = deepcopy(base_selection)
                selection.update(family="main", key=key, instance=instance)
                binding = resolve_transfer_binding(
                    {"actor_binding": selection}, task_alias="T4", instance_id=instance,
                )
                binding["policy_checkpoint"] = checkpoints[instance]
                binding["method_identity"] = metadata["method_identity"]
                if projection is None:
                    binding.pop("completion_admission_version", None)
                else:
                    binding["completion_admission_version"] = projection
                label = f"{policy_name}_{profile}"
                bindings[f"{label}/{instance}"] = binding
                for request in requests:
                    jobs.append({
                        "record_id": f"T4/{label}/{instance}/{request['request_id']}",
                        "row_id": f"T4/contribution/{label}",
                        "task": "T4", "family": "main", "key": label,
                        "method": base.method, "instance": instance,
                        "config": str(base_paths[policy_name].relative_to(ROOT)),
                        "actor_binding": binding, "device": "cuda",
                        "request": deepcopy(request),
                    })
    plan = {
        "schema_version": config["version"],
        "status": "planned_unexecuted",
        "jobs": jobs, "bindings": bindings,
        "design": config["design"], "decision": config["decision"],
        "policy_sets": {
            name: value[2] for name, value in policies.items()
        },
        "device": device,
        "policy": {
            "diagnostic_only": True, "performance_evidence": False,
            "confirmation_states_opened": 0, "failure_replacement": False,
            "same_checkpoint_within_projection_pair": True,
            "same_requests_for_all_four_cells": True,
            "same_start_training_checks": same_start_matching,
        },
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
        },
    }
    if plan_only:
        print(json.dumps({
            "status": "planned_unexecuted",
            "planned_episodes": len(jobs),
            "bindings": len(bindings),
            "policy_sets": plan["policy_sets"],
            "device": device,
            "artifacts": plan["artifacts"],
        }, indent=2, sort_keys=True))
        return
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    begun = time.monotonic()
    workers = int(config["limits"]["workers"])
    completed = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute_job, (job, str(output))): job["record_id"]
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            completed[result["record_id"]] = result
            write_json(output / "progress.json", list(completed.values()))
            if time.monotonic() - begun > float(config["limits"]["maximum_wall_seconds"]):
                raise TimeoutError("T4 factorial wall ceiling")
    results = [completed[job["record_id"]] for job in jobs]
    diagnostic_errors = []
    projection_diagnostics = {}
    for result in results:
        if result["key"].endswith("projection_on") and result.get("evidence"):
            try:
                projection_diagnostics[result["record_id"]] = (
                    final_completion_diagnostics(result)
                )
            except BaseException as error:
                diagnostic_errors.append(f"{result['record_id']}: {error}")
    counts = Counter(result["outcome"] for result in results)
    integrity = bool(
        len(results) == len(jobs)
        and not diagnostic_errors
        and not counts["reset_or_execution_error"]
        and all(result.get("evidence", {}).get("audit_passed", False) for result in results)
        and len(projection_diagnostics) == len(jobs) // 2
    )
    effects = _factorial_effects(
        results, config["decision"],
        matched_control_admitted=bool(
            policies["task_only"][2]["matched_training_admitted"]
        ),
        transport_composite_admitted=bool(
            policies["transport"][2]["composite_qualification"]
        ),
    ) if integrity else {}
    report = {
        "schema_version": config["version"],
        "status": "complete_recorded_development_contribution_contrast" if integrity else "incomplete_or_integrity_failed",
        "passed": integrity,
        "diagnostic_only": True, "performance_evidence": False,
        "confirmation_states_opened": 0, "failure_replacement": False,
        "device": device,
        "counts": {
            "planned_episodes": len(jobs), "completed_episodes": len(results),
            "outcomes": dict(counts),
            "constraint_violations": sum(result["constraint_violation"] for result in results),
            "physical_control_steps": sum(
                int(result.get("accounting", {}).get("control_steps", 0))
                for result in results
            ),
        },
        "effects": effects,
        "diagnostic_errors": diagnostic_errors,
        "projection_diagnostics": projection_diagnostics,
        "results": results,
        "wall_seconds": time.monotonic() - begun,
        "artifacts": {
            "plan": pointer(output / "plan.json"),
            "progress": pointer(output / "progress.json"),
            "config": pointer(config_path), "runner": pointer(Path(__file__).resolve()),
        },
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "counts": report["counts"], "effects": report["effects"],
        "report": pointer(output / "report.json"),
    }, indent=2, sort_keys=True))
    if not integrity:
        raise AssertionError("T4 contribution contrast failed execution integrity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_factorial_config(config_path)
    if config.get("version") in {FACTORIAL_VERSION, SAME_START_FACTORIAL_VERSION}:
        run_factorial(config, config_path, plan_only=args.plan_only)
        return
    if args.plan_only:
        parser.error("--plan-only is only available for the factorial contract")
    if config.get("version") != VERSION:
        raise ValueError("unknown T4 completion development version")
    run(config, config_path)


if __name__ == "__main__":
    main()
