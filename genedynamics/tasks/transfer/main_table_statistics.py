"""Independent four-task main-table arithmetic with explicit missing cells.

This module consumes audited physical outcome rows. It does not launch physics,
certify evidence, or infer an algorithm score from predeployment rejection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

import numpy as np

from .task_identity import LEGACY_TASKS

VERSION = "four-task-independent-main-table-statistics-v5"
# This v5 schema is an immutable reader for the completed formal table.  New
# runtime dispatch uses task_identity.CANONICAL_TASKS; the legacy keys remain
# here so old protocols/checkpoints are read without rewriting their evidence.
TASKS = LEGACY_TASKS
PRIMARY_TASKS = TASKS[:3]
METHODS = (
    "cross_ebdm", "task_only", "fada",
    "dial_transfer", "response_mpc", "public_geometry", "public_source_prior",
    "adaptsim", "asid",
)
FIXED_METHODS = frozenset(("public_geometry", "public_source_prior"))
TRUE_TRAINING_INSTANCES = 8
METRICS = (
    "constrained_success_rate", "constraint_violation_rate",
    "capped_normalized_completion_time", "worst_group_constrained_success",
)


def _unique(values: list[str], label: str) -> None:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be nonempty and unique")


def _validate_protocol(protocol: dict[str, Any]) -> tuple[dict, dict]:
    if protocol.get("version") != VERSION:
        raise ValueError("independent-table statistics version differs")
    if protocol.get("task_order") != list(TASKS) or protocol.get("method_order") != list(METHODS):
        raise ValueError("independent-table task/method roster differs")
    requests = protocol.get("requests")
    cells = protocol.get("cells")
    if not isinstance(requests, dict) or set(requests) != set(TASKS):
        raise ValueError("every task requires a declared request roster")
    if not isinstance(cells, dict) or set(cells) != set(METHODS):
        raise ValueError("every main-table method requires a cell declaration")
    request_index = {}
    for task in TASKS:
        rows = requests[task]
        if not isinstance(rows, list):
            raise ValueError("request roster must be a list")
        ids = [row.get("request_id") for row in rows]
        _unique(ids, f"{task} requests")
        if any(set(row) != {"request_id", "mechanism_group", "requested_mode", "body_support_group"}
               or row["requested_mode"] not in (0, 1)
               or not isinstance(row["mechanism_group"], str)
               or not row["mechanism_group"]
               or not isinstance(row["body_support_group"], str)
               or not row["body_support_group"] for row in rows):
            raise ValueError(f"{task} request contract differs")
        group_sizes = Counter(
            (row["mechanism_group"], row["requested_mode"], row["body_support_group"])
            for row in rows
        )
        if len(set(group_sizes.values())) != 1:
            raise ValueError(f"{task} mechanism/mode/body-support request groups are unbalanced")
        request_index[task] = {row["request_id"]: row for row in rows}
    for method in METHODS:
        method_cells = cells[method]
        if not isinstance(method_cells, dict) or set(method_cells) != set(TASKS):
            raise ValueError(f"{method} requires all four declared cells")
        for task in TASKS:
            cell = method_cells[task]
            if cell.get("status") == "executable":
                if set(cell) != {"status", "instance_ids", "method_identity"}:
                    raise ValueError(f"{method}/{task} executable cell differs")
                ids = cell["instance_ids"]
                if not isinstance(ids, list):
                    raise ValueError("true instance IDs must be a list")
                _unique(ids, f"{method}/{task} true training instances")
                if not all(isinstance(value, str) and value for value in ids):
                    raise ValueError("training instance IDs must be nonempty strings")
                expected = 1 if method in FIXED_METHODS else TRUE_TRAINING_INSTANCES
                if len(ids) != expected or (method in FIXED_METHODS and ids != ["fixed"]):
                    raise ValueError(
                        f"{method}/{task} requires {expected} true independent actor identity"
                    )
            elif cell.get("status") == "unavailable":
                if set(cell) != {"status", "reason", "method_identity"} or not cell["reason"]:
                    raise ValueError(f"{method}/{task} unavailable cell needs a reason")
            else:
                raise ValueError(f"{method}/{task} has unknown admission status")
            if not isinstance(cell["method_identity"], str) or not cell["method_identity"]:
                raise ValueError("method identity is required for every cell")
    return request_index, cells


def _validate_outcome(row: dict[str, Any], request: dict[str, Any]) -> None:
    required = {
        "task", "method", "instance_id", "request_id", "dispatch_status",
        "target_physical_execution", "constrained_success",
        "constraint_violation", "capped_normalized_completion_time",
        "outcome", "evidence_sha256", "reset_physical_hash",
        "evidence_audit_status",
    }
    if not required <= set(row):
        raise ValueError("physical outcome lacks required provenance or metrics")
    if (row["dispatch_status"] != "executed"
            or row["target_physical_execution"] != 1
            or row["constrained_success"] not in (0, 1)
            or not isinstance(row["constraint_violation"], bool)
            or not isinstance(row["outcome"], str) or not row["outcome"]
            or not isinstance(row["evidence_sha256"], str)
            or len(row["evidence_sha256"]) != 64
            or not isinstance(row["reset_physical_hash"], str)
            or len(row["reset_physical_hash"]) != 64
            or row["evidence_audit_status"] != "physical_evidence_verified"):
        raise ValueError("physical outcome status, provenance, or success differs")
    completion = row["capped_normalized_completion_time"]
    if not isinstance(completion, (int, float)) or not 0 <= completion <= 1:
        raise ValueError("capped completion must lie in [0,1]")
    if row["constrained_success"] == 0 and completion != 1:
        raise ValueError("non-success must receive capped completion time one")
    if row["constrained_success"] == 1 and (
        row["constraint_violation"] or row["outcome"] != "success"
    ):
        raise ValueError("success cannot coincide with violation or runner failure")
    if row.get("mechanism_group", request["mechanism_group"]) != request["mechanism_group"]:
        raise ValueError("outcome mechanism group differs from frozen request")
    if row.get("requested_mode", request["requested_mode"]) != request["requested_mode"]:
        raise ValueError("outcome mode differs from frozen request")
    if row.get("body_support_group", request["body_support_group"]) != request["body_support_group"]:
        raise ValueError("outcome body/support group differs from frozen request")


def aggregate_independent_main_table(protocol: dict[str, Any], outcomes: list[dict]) -> dict:
    """Reject missing executable outcomes and leave unavailable metrics null."""
    requests, cells = _validate_protocol(protocol)
    if not isinstance(outcomes, list):
        raise ValueError("outcomes must be a list")
    indexed = {}
    reset_hashes = defaultdict(set)
    for row in outcomes:
        task, method = row.get("task"), row.get("method")
        if task not in TASKS or method not in METHODS:
            raise ValueError("outcome references an undeclared task or method")
        cell = cells[method][task]
        if cell["status"] != "executable":
            raise ValueError("unavailable method cannot have a scored outcome")
        if row.get("instance_id") not in cell["instance_ids"]:
            raise ValueError("outcome has no true declared training instance")
        request = requests[task].get(row.get("request_id"))
        if request is None:
            raise ValueError("outcome uses an undeclared request")
        _validate_outcome(row, request)
        reset_hashes[task, row["request_id"]].add(row["reset_physical_hash"])
        key = (task, method, row["instance_id"], row["request_id"])
        if key in indexed:
            raise ValueError("duplicate physical outcome")
        indexed[key] = row
    if any(len(hashes) != 1 for hashes in reset_hashes.values()):
        raise ValueError("paired outcomes have inconsistent physical reset states")
    task_rows = []
    by_method = defaultdict(dict)
    missing = []
    for method in METHODS:
        for task in TASKS:
            cell = cells[method][task]
            if cell["status"] == "unavailable":
                result = {
                    "method": method, "task": task, "status": "unavailable",
                    "method_identity": cell["method_identity"],
                    "reason": cell["reason"], "true_instance_count": 0,
                    "physical_outcomes": 0,
                    **{metric: None for metric in METRICS},
                }
                missing.append({"method": method, "task": task, "reason": cell["reason"]})
            else:
                selected = []
                for instance in cell["instance_ids"]:
                    for request_id in requests[task]:
                        key = (task, method, instance, request_id)
                        if key not in indexed:
                            raise ValueError(f"missing executable physical outcome: {key}")
                        selected.append(indexed[key])
                groups = defaultdict(list)
                for row in selected:
                    request = requests[task][row["request_id"]]
                    group = (
                        request["mechanism_group"], request["requested_mode"],
                        request["body_support_group"],
                    )
                    groups[group].append(row["constrained_success"])
                result = {
                    "method": method, "task": task, "status": "executed",
                    "method_identity": cell["method_identity"],
                    "true_instance_count": len(cell["instance_ids"]),
                    "physical_outcomes": len(selected),
                    "constrained_success_rate": mean(row["constrained_success"] for row in selected),
                    "constraint_violation_rate": mean(row["constraint_violation"] for row in selected),
                    "capped_normalized_completion_time": mean(
                        row["capped_normalized_completion_time"] for row in selected
                    ),
                    "worst_group_constrained_success": min(mean(values) for values in groups.values()),
                    "group_constrained_success": [
                        {"mechanism_group": group[0], "requested_mode": group[1],
                         "body_support_group": group[2],
                         "constrained_success_rate": mean(values)}
                        for group, values in sorted(groups.items())
                    ],
                    "outcomes": dict(sorted(Counter(row["outcome"] for row in selected).items())),
                }
            task_rows.append(result)
            by_method[method][task] = result
    macro = []
    for method in METHODS:
        primary = [by_method[method][task] for task in PRIMARY_TASKS]
        complete = all(row["status"] == "executed" for row in primary)
        aggregate = (
            {
                "constrained_success_rate": mean(row["constrained_success_rate"] for row in primary),
                "constraint_violation_rate": mean(row["constraint_violation_rate"] for row in primary),
                "capped_normalized_completion_time": mean(
                    row["capped_normalized_completion_time"] for row in primary
                ),
                "worst_group_constrained_success": min(
                    row["worst_group_constrained_success"] for row in primary
                ),
            } if complete else {metric: None for metric in METRICS}
        )
        macro.append({"method": method, "primary_tasks_complete": complete, **aggregate})
    return {
        "version": VERSION,
        "status": "complete_outcome_matrix" if not missing else "incomplete_method_coverage",
        "formal_performance_evidence": False,
        "evidence_audit_required_before_publication": True,
        "main_rows": task_rows,
        "primary_macro_T1_T3": macro,
        "shape_extension_T4": [by_method[method]["T4"] for method in METHODS],
        "unavailable_cells": missing,
        "physical_outcomes": len(indexed),
        "missing_execution_score": None,
        "missing_execution_in_macro_denominator": False,
    }


def bootstrap_independent_main_table(
    protocol: dict[str, Any], outcomes: list[dict],
    full_requests: dict[str, list[dict]], *, resamples: int = 10000,
    seed: int = 267100,
) -> dict:
    """Crossed instance/request bootstrap; common request draws remain paired.

    A deterministic reference has one actor, not eight pseudo-instances.
    Distinct learned methods resample their training instances independently;
    the same physical-initial blocks are drawn for every method.
    """
    table = aggregate_independent_main_table(protocol, outcomes)
    if table["status"] != "complete_outcome_matrix" or resamples < 1:
        raise ValueError("bootstrap requires a complete executed matrix and positive draws")
    if set(full_requests) != set(TASKS):
        raise ValueError("bootstrap requires every frozen full request roster")
    by_outcome = {
        (row["task"], row["method"], row["instance_id"], row["request_id"]): row
        for row in outcomes
    }
    draws: dict[tuple[str, str], np.ndarray] = {}
    points = {(row["task"], row["method"]): row for row in table["main_rows"]}
    intervals = []
    for task_index, task in enumerate(TASKS):
        declared = protocol["requests"][task]
        frozen = full_requests[task]
        if ({row["request_id"] for row in frozen} !=
                {row["request_id"] for row in declared} or
                len(frozen) != len(declared)):
            raise ValueError("bootstrap full request roster differs")
        strata = sorted({
            (row["mechanism_group"], row["body_support_group"])
            for row in frozen
        })
        strata_index = {value: index for index, value in enumerate(strata)}
        blocks = sorted({row["physical_initial_slot"] for row in frozen})
        if blocks != list(range(5)):
            raise ValueError("bootstrap requires five frozen physical-initial blocks")
        block_index = {value: index for index, value in enumerate(blocks)}
        by_request = {}
        for row in frozen:
            key = row["request_id"]
            expected = next(item for item in declared if item["request_id"] == key)
            if any(row[field] != expected[field] for field in (
                "mechanism_group", "requested_mode", "body_support_group",
            )):
                raise ValueError("bootstrap request labels differ from scoring protocol")
            by_request[key] = (
                strata_index[row["mechanism_group"], row["body_support_group"]],
                block_index[row["physical_initial_slot"]], row["requested_mode"],
            )
        if len(set(by_request.values())) != len(frozen):
            raise ValueError("bootstrap has duplicated physical-initial/mode blocks")
        expected_slots = len(strata) * len(blocks) * 2
        if len(frozen) != expected_slots:
            raise ValueError("bootstrap mechanism/support/mode grid is incomplete")
        request_rng = np.random.default_rng(seed + 1009 * (task_index + 1))
        shared_block_draws = request_rng.integers(
            len(blocks), size=(resamples, len(strata), len(blocks)),
        )
        for method_index, method in enumerate(METHODS):
            ids = protocol["cells"][method][task]["instance_ids"]
            values = np.empty((len(ids), len(strata), len(blocks), 2, 3), dtype=float)
            filled = np.zeros(values.shape[:-1], dtype=bool)
            for actor_index, instance_id in enumerate(ids):
                for request_id, (s, b, mode) in by_request.items():
                    row = by_outcome[task, method, instance_id, request_id]
                    values[actor_index, s, b, mode] = (
                        row["constrained_success"], row["constraint_violation"],
                        row["capped_normalized_completion_time"],
                    )
                    filled[actor_index, s, b, mode] = True
            if not filled.all():
                raise ValueError("bootstrap actor/request grid is incomplete")
            actor_rng = np.random.default_rng(
                seed + 1000003 * (task_index + 1) + 7919 * (method_index + 1)
            )
            parts = []
            for start in range(0, resamples, 200):
                count = min(200, resamples - start)
                actor_draw = actor_rng.integers(len(ids), size=(count, len(ids)))
                block_draw = shared_block_draws[start:start + count]
                selected = values[
                    actor_draw[:, :, None, None],
                    np.arange(len(strata))[None, None, :, None],
                    block_draw[:, None, :, :],
                ]
                strata_modes = selected.mean(axis=(1, 3))
                linear = strata_modes.mean(axis=(1, 2))
                worst = strata_modes[..., 0].min(axis=(1, 2))
                parts.append(np.column_stack((linear, worst)))
            draw = np.concatenate(parts)
            draws[task, method] = draw
            intervals.append({
                "task": task, "method": method,
                "true_training_instances": len(ids),
                "ci95": {
                    metric: np.quantile(draw[:, index], [.025, .975]).tolist()
                    for index, metric in enumerate(METRICS)
                },
            })

    comparisons = []
    for comparator in METHODS:
        if comparator == "cross_ebdm":
            continue
        for task in (*TASKS, "primary_macro_T1_T3"):
            if task in TASKS:
                left = draws[task, "cross_ebdm"]
                right = draws[task, comparator]
                left_point, right_point = points[task, "cross_ebdm"], points[task, comparator]
                estimates = [left_point[metric] - right_point[metric] for metric in METRICS]
            else:
                left_stack = np.stack([draws[t, "cross_ebdm"] for t in PRIMARY_TASKS])
                right_stack = np.stack([draws[t, comparator] for t in PRIMARY_TASKS])
                left = np.column_stack((
                    left_stack[..., :3].mean(axis=0), left_stack[..., 3].min(axis=0)
                ))
                right = np.column_stack((
                    right_stack[..., :3].mean(axis=0), right_stack[..., 3].min(axis=0)
                ))
                macro = {row["method"]: row for row in table["primary_macro_T1_T3"]}
                estimates = [
                    macro["cross_ebdm"][metric] - macro[comparator][metric]
                    for metric in METRICS
                ]
            delta = left - right
            comparisons.append({
                "task": task, "reference": "cross_ebdm", "comparator": comparator,
                "metrics": {
                    metric: {"estimate": estimate,
                             "ci95": np.quantile(delta[:, index], [.025, .975]).tolist()}
                    for index, (metric, estimate) in enumerate(zip(METRICS, estimates))
                },
            })
    return {
        "version": "four-task-independent-main-table-inference-v1",
        "status": "complete_descriptive_inference",
        "resamples": resamples, "seed": seed,
        "confidence_interval": "pointwise 95% empirical percentile",
        "sampling": "independent trained actors and shared physical-initial blocks within frozen mechanism/support strata; modes remain paired; fixed actors are not replicated",
        "inference_scope": "conditional on frozen mechanisms, supports, and target simulator",
        "cell_intervals": intervals,
        "cross_ebdm_comparisons": comparisons,
        "formal_performance_evidence": False,
        "physical_calls": 0, "gpu_calls": 0,
    }
