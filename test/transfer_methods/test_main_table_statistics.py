"""Independent main-table arithmetic keeps absent methods outside performance."""

import pytest

from genedynamics.tasks.transfer.main_table_statistics import (
    METHODS, TASKS, VERSION, aggregate_independent_main_table,
    bootstrap_independent_main_table,
)

INSTANCES = [f"true_{index:02d}" for index in range(1, 9)]


def _protocol():
    return {
        "version": VERSION, "task_order": list(TASKS),
        "method_order": list(METHODS),
        "requests": {
            task: [
                {"request_id": f"{task}/a", "mechanism_group": "a", "requested_mode": 0,
                 "body_support_group": "fixed"},
                {"request_id": f"{task}/b", "mechanism_group": "b", "requested_mode": 1,
                 "body_support_group": "fixed"},
            ] for task in TASKS
        },
        "cells": {
            method: {
                task: {"status": "unavailable", "reason": "no executable actor",
                       "method_identity": method}
                for task in TASKS
            } for method in METHODS
        },
    }


def _outcome(task, method, instance, group, success, violation=False):
    return {
        "task": task, "method": method, "instance_id": instance,
        "request_id": f"{task}/{group}", "dispatch_status": "executed",
        "target_physical_execution": 1,
        "constrained_success": int(success), "constraint_violation": violation,
        "capped_normalized_completion_time": .5 if success else 1.,
        "outcome": "success" if success else "overforce" if violation else "timeout",
        "evidence_sha256": "a" * 64,
        "reset_physical_hash": "b" * 64,
        "evidence_audit_status": "physical_evidence_verified",
    }


def test_missing_method_has_null_metrics_and_eight_true_instances_are_scored():
    protocol = _protocol()
    protocol["cells"]["cross_ebdm"]["T1"] = {
        "status": "executable", "instance_ids": INSTANCES,
        "method_identity": "cross_ebdm",
    }
    outcomes = [row for instance in INSTANCES for row in (
        _outcome("T1", "cross_ebdm", instance, "a", True),
        _outcome("T1", "cross_ebdm", instance, "b", False, True),
    )]
    result = aggregate_independent_main_table(protocol, outcomes)
    executed = next(row for row in result["main_rows"]
                    if row["method"] == "cross_ebdm" and row["task"] == "T1")
    absent = next(row for row in result["main_rows"]
                  if row["method"] == "asid" and row["task"] == "T1")
    assert executed["true_instance_count"] == 8
    assert executed["constrained_success_rate"] == .5
    assert executed["constraint_violation_rate"] == .5
    assert executed["worst_group_constrained_success"] == 0.
    assert absent["constrained_success_rate"] is None
    assert result["missing_execution_score"] is None
    assert result["primary_macro_T1_T3"][0]["constrained_success_rate"] is None


def test_missing_or_dummy_executable_outcome_is_rejected():
    protocol = _protocol()
    protocol["cells"]["task_only"]["T2"] = {
        "status": "executable", "instance_ids": INSTANCES,
        "method_identity": "task_only",
    }
    rows = [_outcome("T2", "task_only", "true_01", "a", True)]
    with pytest.raises(ValueError, match="missing executable physical outcome"):
        aggregate_independent_main_table(protocol, rows)
    rows.append(_outcome("T2", "task_only", "true_01", "b", False))
    rows[1]["target_physical_execution"] = 0
    with pytest.raises(ValueError, match="physical outcome status"):
        aggregate_independent_main_table(protocol, rows)
    rows[1]["target_physical_execution"] = 1
    rows[1]["dispatch_status"] = "retained_runner_failure"
    with pytest.raises(ValueError, match="physical outcome status"):
        aggregate_independent_main_table(protocol, rows)
    unavailable = _outcome("T2", "asid", "fake_01", "a", False)
    with pytest.raises(ValueError, match="unavailable method"):
        aggregate_independent_main_table(protocol, [unavailable])
    for archived_method in ("acp", "human2sim2robot"):
        with pytest.raises(ValueError, match="undeclared task or method"):
            aggregate_independent_main_table(
                protocol, [_outcome("T2", archived_method, "fake_01", "a", False)]
            )


def test_primary_macro_requires_three_tasks_but_not_shape_extension():
    protocol = _protocol()
    for task in TASKS[:3]:
        protocol["cells"]["cross_ebdm"][task] = {
            "status": "executable", "instance_ids": INSTANCES,
            "method_identity": "cross_ebdm",
        }
    outcomes = []
    for task, successes in zip(TASKS[:3], ((1, 1), (1, 0), (0, 0))):
        for instance in INSTANCES:
            for group, success in zip(("a", "b"), successes):
                outcomes.append(_outcome(task, "cross_ebdm", instance, group, success))
    result = aggregate_independent_main_table(protocol, outcomes)
    macro = result["primary_macro_T1_T3"][0]
    assert macro["primary_tasks_complete"] is True
    assert macro["constrained_success_rate"] == .5
    assert macro["worst_group_constrained_success"] == 0.
    assert result["shape_extension_T4"][0]["constrained_success_rate"] is None


def test_worst_group_stratifies_mode_and_body_support():
    protocol = _protocol()
    protocol["requests"]["T4"] = [
        {"request_id": "T4/a", "mechanism_group": "same", "requested_mode": 0,
         "body_support_group": "supported"},
        {"request_id": "T4/b", "mechanism_group": "same", "requested_mode": 1,
         "body_support_group": "derated"},
        {"request_id": "T4/c", "mechanism_group": "same", "requested_mode": 0,
         "body_support_group": "derated"},
        {"request_id": "T4/d", "mechanism_group": "same", "requested_mode": 1,
         "body_support_group": "supported"},
    ]
    protocol["cells"]["asid"]["T4"] = {
        "status": "executable", "instance_ids": INSTANCES,
        "method_identity": "asid",
    }
    outcomes = [row for instance in INSTANCES for row in (
        _outcome("T4", "asid", instance, "a", True),
        _outcome("T4", "asid", instance, "b", False),
        _outcome("T4", "asid", instance, "c", False),
        _outcome("T4", "asid", instance, "d", True),
    )]
    result = aggregate_independent_main_table(protocol, outcomes)
    row = next(row for row in result["main_rows"]
               if row["method"] == "asid" and row["task"] == "T4")
    assert row["constrained_success_rate"] == .5
    assert row["worst_group_constrained_success"] == 0.
    assert row["group_constrained_success"] == [
        {"mechanism_group": "same", "requested_mode": 0,
         "body_support_group": "derated", "constrained_success_rate": 0.},
        {"mechanism_group": "same", "requested_mode": 0,
         "body_support_group": "supported", "constrained_success_rate": 1.},
        {"mechanism_group": "same", "requested_mode": 1,
         "body_support_group": "derated", "constrained_success_rate": 0.},
        {"mechanism_group": "same", "requested_mode": 1,
         "body_support_group": "supported", "constrained_success_rate": 1.},
    ]
    outcomes[1]["body_support_group"] = "supported"
    with pytest.raises(ValueError, match="body/support group differs"):
        aggregate_independent_main_table(protocol, outcomes)


def test_paired_physical_resets_must_match_across_methods():
    protocol = _protocol()
    for method in ("public_geometry", "public_source_prior"):
        protocol["cells"][method]["T1"] = {
            "status": "executable", "instance_ids": ["fixed"],
            "method_identity": method,
        }
    rows = [
        _outcome("T1", method, "fixed", group, True)
        for method in ("public_geometry", "public_source_prior")
        for group in ("a", "b")
    ]
    rows[2]["reset_physical_hash"] = "c" * 64
    with pytest.raises(ValueError, match="inconsistent physical reset states"):
        aggregate_independent_main_table(protocol, rows)


def test_unbalanced_request_groups_are_rejected():
    protocol = _protocol()
    protocol["requests"]["T1"].append({
        "request_id": "T1/c", "mechanism_group": "a",
        "requested_mode": 0, "body_support_group": "fixed",
    })
    with pytest.raises(ValueError, match="groups are unbalanced"):
        aggregate_independent_main_table(protocol, [])


def test_single_development_instance_cannot_be_scored_as_formal_method():
    protocol = _protocol()
    protocol["cells"]["asid"]["T1"] = {
        "status": "executable", "instance_ids": ["development_01"],
        "method_identity": "asid",
    }
    with pytest.raises(ValueError, match="requires 8 true independent actor identity"):
        aggregate_independent_main_table(protocol, [])
    protocol["cells"]["asid"]["T1"] = {
        "status": "unavailable", "reason": "qualification pending", "method_identity": "asid",
    }
    protocol["cells"]["public_geometry"]["T1"] = {
        "status": "executable", "instance_ids": ["true_01"],
        "method_identity": "public_geometry",
    }
    with pytest.raises(ValueError, match="requires 1 true independent actor identity"):
        aggregate_independent_main_table(protocol, [])


def test_crossed_bootstrap_keeps_fixed_actor_single_and_identical_pair_zero():
    protocol = _protocol()
    full = {}
    rows = []
    for task in TASKS:
        supports = ("nominal", "derated") if task == "T4" else ("nominal",)
        full[task] = []
        for mechanism in ("source", "shift"):
            for support in supports:
                for block in range(5):
                    for mode in (0, 1):
                        request_id = f"{task}/{mechanism}/{support}/{block}/{mode}"
                        full[task].append({
                            "request_id": request_id, "mechanism_group": mechanism,
                            "body_support_group": support, "physical_initial_slot": block,
                            "requested_mode": mode,
                        })
        protocol["requests"][task] = [
            {key: row[key] for key in (
                "request_id", "mechanism_group", "requested_mode", "body_support_group",
            )} for row in full[task]
        ]
        for method in METHODS:
            ids = ["fixed"] if method.startswith("public_") else INSTANCES
            protocol["cells"][method][task] = {
                "status": "executable", "instance_ids": ids,
                "method_identity": method,
            }
            for instance in ids:
                for request in full[task]:
                    success = (
                        request["physical_initial_slot"] < 2
                        if method in {"cross_ebdm", "task_only"} else True
                    )
                    outcome = _outcome(task, method, instance, "a", success)
                    outcome["request_id"] = request["request_id"]
                    rows.append(outcome)
    inference = bootstrap_independent_main_table(
        protocol, rows, full, resamples=200, seed=123,
    )
    assert inference["status"] == "complete_descriptive_inference"
    fixed = next(row for row in inference["cell_intervals"]
                 if row["task"] == "T1" and row["method"] == "public_geometry")
    assert fixed["true_training_instances"] == 1
    pair = next(row for row in inference["cross_ebdm_comparisons"]
                if row["task"] == "primary_macro_T1_T3"
                and row["comparator"] == "task_only")
    assert all(value["estimate"] == 0 and value["ci95"] == [0, 0]
               for value in pair["metrics"].values())
    ours = next(row for row in inference["cell_intervals"]
                if row["task"] == "T1" and row["method"] == "cross_ebdm")
    ci = ours["ci95"]["constrained_success_rate"]
    assert ci[0] < ci[1]
    full["T1"][0]["mechanism_group"] = "changed"
    with pytest.raises(ValueError, match="request labels differ"):
        bootstrap_independent_main_table(protocol, rows, full, resamples=2)
