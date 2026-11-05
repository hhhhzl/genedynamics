"""CPU-only statistical checks using synthetic saved outcomes, without physics."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from genedynamics.experiments.utils import metrics


def _rows(values, role="independent_training_or_calibration_instance"):
    """Create binary results for instance/stratum/initial/mode axes."""
    rows = []
    for instance, stratum, initial, mode in np.ndindex(values.shape):
        rejected = role == "planned_evaluation_slot_not_an_independent_trained_instance"
        success = float(values[instance, stratum, initial, mode])
        rows.append({
            "training_or_calibration_instance": f"instance_{instance:02d}",
            "instance_role": role,
            "mechanism_group": "mechanism",
            "body_support_cell": f"support_{stratum}",
            "physical_initial_slot": initial,
            "requested_mode": mode,
            "constrained_success": success,
            "constraint_violation": float(rejected),
            "capped_normalized_completion_time": 1.0 - success / 2,
            "predeployment_rejection": float(rejected),
        })
    return rows


def test_bootstrap_keeps_modes_and_crossed_request_uncertainty():
    # Identical training instances must not turn one request cohort into eight.
    # Within each initial block, the two mode scores cancel exactly.
    values = np.zeros((8, 1, 2, 2, 1))
    values[:, 0, :, 0, 0] = [0, 1]
    values[:, 0, :, 1, 0] = [1, 0]
    draws = metrics.paired_block_bootstrap(values, resamples=3000, seed=7)
    assert draws.shape == (3000, 1, 2, 1)
    np.testing.assert_allclose(draws.mean(axis=2), 0.5)
    assert set(np.unique(draws[:, 0, 0, 0])) == {0, 0.5, 1}
    # Var(mean of two draws from Bernoulli(.5)) = .125, regardless of eight
    # identical training instances. Independent reset resampling gives .015625.
    assert np.var(draws[:, 0, 0, 0]) == pytest.approx(0.125, abs=0.01)


def test_bootstrap_preserves_fixed_strata_and_metric_columns():
    values = np.zeros((3, 2, 4, 2, 2))
    values[:, 0, :, :, :] = [1, 10]
    values[:, 1, :, :, :] = [2, 20]
    draws = metrics.paired_block_bootstrap(values, resamples=203, seed=6)
    np.testing.assert_allclose(draws[:, 0], np.broadcast_to([1, 10], (203, 2, 2)))
    np.testing.assert_allclose(draws[:, 1], np.broadcast_to([2, 20], (203, 2, 2)))


def test_deterministic_reference_has_one_identity():
    rows = _rows(np.ones((1, 2, 3, 2)), "deterministic_single_identity")
    tensor, metadata = metrics._outcome_tensor(rows)
    assert tensor.shape == (1, 2, 3, 2, 3)
    assert metadata["independent_training_instances"] == 0
    assert metadata["deterministic_identities"] == 1
    assert metadata["planned_instance_slots"] == 1


def test_rejection_slots_are_denominators_not_training_repeats():
    rows = _rows(
        np.zeros((8, 2, 3, 2)),
        "planned_evaluation_slot_not_an_independent_trained_instance",
    )
    tensor, metadata = metrics._outcome_tensor(rows)
    assert len(rows) == 96
    assert tensor.shape == (1, 2, 3, 2, 3)
    assert metadata["planned_instance_slots"] == 8
    assert metadata["independent_training_instances"] == 0
    assert metadata["deterministic_identities"] == 0
    assert metadata["all_predeployment_rejected"]
    rows[-1]["capped_normalized_completion_time"] = 0.5
    with pytest.raises(ValueError, match="identical scoring aliases"):
        metrics._outcome_tensor(rows)


@pytest.mark.parametrize("damage", ["duplicate", "missing"])
def test_tensor_rejects_duplicate_or_incomplete_pairs(damage):
    rows = _rows(np.ones((2, 2, 3, 2)))
    rows = rows + [rows[0]] if damage == "duplicate" else rows[:-1]
    with pytest.raises(ValueError, match="duplicate|unbalanced"):
        metrics._outcome_tensor(rows)


def test_joint_worst_group_includes_mode():
    group_means = np.array([[[0.8, 0, 0.6], [0.2, 0, 0.9]],
                            [[0.5, 0, 0.7], [0.5, 0, 0.7]]])
    result = metrics._outcome_statistics(group_means)
    assert result[0] == pytest.approx(0.5)
    assert result[3] == pytest.approx(0.2)


def test_exact_sign_flip_and_holm_include_ties():
    assert metrics.exact_instance_sign_flip(np.zeros(8)) == 1.0
    assert metrics.exact_instance_sign_flip(np.ones(3)) == 0.25
    assert metrics.exact_instance_sign_flip(np.array([1.0, -1.0, 0])) == 1.0
    assert metrics.exact_instance_sign_flip(np.array([1.0])) is None
    assert metrics.holm_adjust([0.04, 0.01, 0.01, 0.4]) == pytest.approx(
        [0.08, 0.04, 0.04, 0.4]
    )
    assert metrics.holm_adjust([]) == []


def test_strongest_baseline_selection_uses_frozen_lexicographic_order():
    points = {
        "zeta": np.array([.8, .1, .5, .2]),
        "alpha": np.array([.8, .1, .5, .1]),
        "safer": np.array([.8, .05, .9, .1]),
        "faster": np.array([.8, .05, .4, .1]),
        "lower_success": np.array([.7, 0., 0., .7]),
    }
    assert metrics._select_strongest_baseline(points) == "faster"
    points.pop("faster")
    assert metrics._select_strongest_baseline(points) == "safer"
    points.pop("safer")
    assert metrics._select_strongest_baseline(points) == "alpha"


def test_alias_specs_support_task_scope_and_retained_runner_failures():
    specs = metrics._alias_specs({
        "candidate": {
            "method": "gated", "tasks": ["T1", "T3"],
            "tolerated_failure_categories": ["runner_interruption_after_reset"],
        },
    }, ["T1", "T2", "T3"])
    assert specs == [{
        "main": "candidate", "ablation": "gated", "tasks": ["T1", "T3"],
        "tolerated_failure_categories": ["runner_interruption_after_reset"],
    }]
    with pytest.raises(ValueError, match="unknown tasks"):
        metrics._alias_specs({"candidate": {"method": "gated", "tasks": ["T4"]}}, ["T1"])


def _saved_confirmation(
    root: Path, monkeypatch, *, tasks=("T1",), method_rows=None, package=None,
):
    """Small complete manifest with trained, fixed, and rejected methods."""
    source = root / "confirmation"
    source.mkdir()
    code = root / "analysis_source.py"
    code.write_text("# synthetic provenance only\n")
    monkeypatch.setattr(metrics, "__file__", str(code))

    def pointer(path):
        return {"path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(name, data, *, lines=False):
        path = source / name
        path.write_text("".join(json.dumps(row) + "\n" for row in data)
                        if lines else json.dumps(data))
        return pointer(path)

    values = np.zeros((2, 2, 2, 2))
    values[:, :, 0] = 1  # Both group success means are .5.
    fixed = np.zeros((1, 2, 2, 2))
    fixed[:, 1] = 1  # Same overall mean; worst group is zero.
    methods = method_rows or {
        "candidate": _rows(values),
        "fixed": _rows(fixed, "deterministic_single_identity"),
        "rejected": _rows(np.zeros((8, 2, 2, 2)),
                          "planned_evaluation_slot_not_an_independent_trained_instance"),
    }
    rows, schedule, receipts, roster, aggregates = [], [], [], {}, []
    for task_index, task in enumerate(tasks):
        for method, templates in methods.items():
          members = [dict(row) for row in templates]
          for row in members:
              request_id = (f"{task}/request/{row['body_support_cell']}/"
                            f"{row['physical_initial_slot']}/{row['requested_mode']}")
              reset_seed = (1000 * task_index + 100
                            + row["physical_initial_slot"] * 2 + row["requested_mode"])
              roster[request_id] = {
                  "request_id": request_id, "task": task, "reset_seed": reset_seed,
                  **{key: row[key] for key in ("mechanism_group", "body_support_cell",
                                              "physical_initial_slot", "requested_mode")},
              }
              record_id = f"{method}/{row['training_or_calibration_instance']}/{request_id}"
              rejection = bool(row["predeployment_rejection"])
              row.update({
                  "family": "main", "method": method, "task": task,
                  "base_request_id": request_id, "record_id": record_id,
                  "confirmation_seed": reset_seed, "ordinal": len(rows) + 1,
                  "target_physical_execution": int(not rejection),
                  "failure_category": "rejected" if rejection else
                      ("success" if row["constrained_success"] else "timeout"),
                  "rejection_reason": "synthetic qualification" if rejection else None,
              })
              rows.append(row)
              schedule.append({"schedule_id": record_id, "request_id": request_id,
                               **{key: row[key] for key in (
                                   "family", "method", "task", "base_request_id",
                                   "training_or_calibration_instance", "instance_role")}})
              if not rejection:
                  receipts.append({"record_id": record_id, "base_request_id": request_id,
                                   **{key: row[key] for key in (
                                       "task", "family", "method", "instance_role",
                                       "training_or_calibration_instance", "requested_mode",
                                       "body_support_cell", "mechanism_group",
                                       "physical_initial_slot", "confirmation_seed")},
                                   "written_before_first_actuation": True,
                                   "post_reset_state_sha256": request_id})
          aggregates.append({
              "family": "main", "method": method, "task": task,
              "requested_episodes": len(members),
              "target_physical_execution_count": sum(x["target_physical_execution"] for x in members),
              "predeployment_rejection_rate": float(np.mean([x["predeployment_rejection"] for x in members])),
              "outcomes": dict(Counter(x["failure_category"] for x in members)),
              "constrained_success_rate": float(np.mean([x["constrained_success"] for x in members])),
              "constraint_violation_rate": float(np.mean([x["constraint_violation"] for x in members])),
              "capped_normalized_completion_time": float(np.mean([x["capped_normalized_completion_time"] for x in members])),
              # Old execution aggregator omitted support and requested mode.
              "worst_group_constrained_success": float(np.mean([x["constrained_success"] for x in members])),
          })
    launch = {
        "status": "complete", "completed_records": len(rows),
        "physical_executions": len(receipts),
        "outcomes": write("outcomes.jsonl", rows, lines=True),
        "aggregates": write("aggregates.json", aggregates),
        "state_receipts": write("state_receipts.jsonl", receipts, lines=True),
        "package_report": write("package.json", package or {}),
        "sealed_artifacts": {
            "evaluation_schedule": write("schedule.jsonl", schedule, lines=True),
            "request_roster": write("roster.jsonl", list(roster.values()), lines=True),
        },
    }
    write("launch.json", launch)
    write("audit.json", {"passed": True, "artifacts": {}})
    return source


def test_saved_summary_worst_contrast_is_difference_of_minima(tmp_path, monkeypatch):
    source = _saved_confirmation(tmp_path, monkeypatch)
    output = tmp_path / "derived"
    result = metrics.summarize_confirmation(
        tmp_path, source, output, references={"main": "candidate"},
        ablation_terms={}, aliases={}, resamples=1000, seed=8,
        permission_classes={"candidate": "B_log", "fixed": "A_fixed", "rejected": "B_log"},
        contract_limitations=["synthetic recorded scores"],
    )
    assert result["counts"]["corrected_scalar_cells"] == 1
    summaries = json.loads((output / "summary.json").read_text())
    rejected = next(row for row in summaries if row["method"] == "rejected")
    assert rejected["requested_episodes"] == 64
    assert rejected["independent_training_instances"] == 0
    assert rejected["constrained_success_rate_ci95"] is None
    fixed = next(row for row in summaries if row["method"] == "fixed")
    assert fixed["deterministic_identities"] == 1
    assert fixed["worst_group_constrained_success"] == 0
    contrasts = json.loads((output / "contrasts.json").read_text())
    contrast = next(row for row in contrasts if row["contrast"] == "candidate_minus_fixed"
                    and row["metric"] == "worst_group_constrained_success")
    assert contrast["estimate"] == pytest.approx(0.5)
    assert contrast["ci95"][0] >= 0  # min(A-B) would instead have a negative tail.
    assert contrast["independent_training_instances"] == 2
    assert contrast["fixed_reference_broadcast_is_not_replication"]
    assert contrast["supplemental_instance_sign_flip_p"] is None
    assert all(row["supplemental_instance_sign_flip_p"] is None
               for row in contrasts if row["contrast"] == "candidate_minus_fixed")


def test_v13_four_task_claim_requires_stable_transport_and_strong_superiority(
    tmp_path, monkeypatch,
):
    cross = np.ones((2, 1, 2, 2))
    task_only = np.ones_like(cross)
    task_only[:, 0, 0, 0] = 0
    strong = np.zeros_like(cross)
    methods = {
        "cross_ebdm": _rows(cross),
        "task_only": _rows(task_only),
        "fada": _rows(strong),
        "dial_transfer": _rows(strong),
        "response_mpc": _rows(strong),
    }
    contract = {
        "version": "four-task-stable-transport-and-strong-baseline-v1",
        "task_weighting": "equal_macro_average_over_T1_T2_T3_T4",
        "transport_control": "task_only",
        "strong_executable_baselines": ["fada", "dial_transfer", "response_mpc"],
        "stable_transport_gate": {
            "minimum_nonnegative_task_success_deltas": 4,
            "minimum_strictly_positive_task_success_deltas": 1,
            "minimum_macro_success_delta": 0.,
            "maximum_macro_constraint_violation_delta": 0.,
            "maximum_macro_capped_completion_time_delta": 0.,
            "require_transport_to_preserve_every_task_only_success": True,
        },
        "coverage_only_rows": ["acp", "human2sim2robot"],
    }
    source = _saved_confirmation(
        tmp_path, monkeypatch, tasks=("T1", "T2", "T3", "T4"),
        method_rows=methods,
        package={
            "version": "four-task-transfer-confirmation-package-v13",
            "transport_claim_contract": contract,
        },
    )
    output = tmp_path / "derived"

    result = metrics.summarize_confirmation(
        tmp_path, source, output, references={"main": "cross_ebdm"},
        ablation_terms={}, aliases={}, resamples=200, seed=17,
        permission_classes={method: "B-log" for method in methods},
    )

    claim = result["transport_claim"]
    assert claim["stable_transport_supported"] is True
    assert claim["strong_baseline_superiority_supported"] is True
    assert claim["core_claim_supported"] is True
    assert claim["selected_strongest_baseline"] == "dial_transfer"
    assert all(
        row["constrained_success_rate"] == pytest.approx(.25)
        for row in claim["transport_pair"]["task_deltas"].values()
    )
    assert json.loads((output / "transport_claim.json").read_text())[
        "core_claim_supported"
    ] is True


def test_four_task_claim_rejects_any_lost_task_only_success(
    tmp_path, monkeypatch,
):
    cross = np.ones((2, 1, 2, 2))
    task_only = np.zeros_like(cross)
    # The candidate has a positive net success delta but loses one paired
    # success.  The frozen preservation rule must therefore fail.
    cross[0, 0, 0, 0] = 0
    cross[1, 0, 1, 1] = 0
    task_only[0, 0, 0, 0] = 1
    task_only[0, 0, 0, 1] = 1
    strong = np.zeros_like(cross)
    methods = {
        "cross_ebdm": _rows(cross),
        "task_only": _rows(task_only),
        "fada": _rows(strong),
        "dial_transfer": _rows(strong),
        "response_mpc": _rows(strong),
    }
    contract = {
        "version": "four-task-stable-transport-and-strong-baseline-v1",
        "task_weighting": "equal_macro_average_over_T1_T2_T3_T4",
        "transport_control": "task_only",
        "strong_executable_baselines": ["fada", "dial_transfer", "response_mpc"],
        "stable_transport_gate": {
            "minimum_nonnegative_task_success_deltas": 4,
            "minimum_strictly_positive_task_success_deltas": 1,
            "minimum_macro_success_delta": 0.,
            "maximum_macro_constraint_violation_delta": 0.,
            "maximum_macro_capped_completion_time_delta": 0.,
            "require_transport_to_preserve_every_task_only_success": True,
        },
        "coverage_only_rows": ["acp", "human2sim2robot"],
    }
    source = _saved_confirmation(
        tmp_path, monkeypatch, tasks=("T1", "T2", "T3", "T4"),
        method_rows=methods,
        package={
            "version": "four-task-transfer-confirmation-package-v13",
            "transport_claim_contract": contract,
        },
    )
    result = metrics.summarize_confirmation(
        tmp_path, source, tmp_path / "derived",
        references={"main": "cross_ebdm"}, ablation_terms={}, aliases={},
        resamples=200, seed=17,
        permission_classes={method: "B-log" for method in methods},
    )
    claim = result["transport_claim"]
    assert claim["transport_pair"]["paired_success_outcomes"]["T1"] == {
        "favorable_pairs": 5,
        "adverse_pairs": 1,
        "tied_success_pairs": 1,
        "tied_failure_pairs": 1,
        "total_pairs": 8,
    }
    assert claim["stable_transport_checks"][
        "transport_preserves_every_task_only_success"
    ] is False
    assert claim["stable_transport_supported"] is False
    assert claim["core_claim_supported"] is False


def test_four_task_superiority_rejects_cross_permission_comparisons(
    tmp_path, monkeypatch,
):
    cross = np.ones((2, 1, 2, 2))
    task_only = cross.copy()
    task_only[:, 0, 0, 0] = 0
    strong = np.zeros_like(cross)
    methods = {
        "cross_ebdm": _rows(cross),
        "task_only": _rows(task_only),
        "fada": _rows(strong),
        "dial_transfer": _rows(strong),
        "response_mpc": _rows(strong),
    }
    contract = {
        "version": "four-task-stable-transport-and-strong-baseline-v1",
        "task_weighting": "equal_macro_average_over_T1_T2_T3_T4",
        "transport_control": "task_only",
        "strong_executable_baselines": ["fada", "dial_transfer", "response_mpc"],
        "stable_transport_gate": {
            "minimum_nonnegative_task_success_deltas": 4,
            "minimum_strictly_positive_task_success_deltas": 1,
            "minimum_macro_success_delta": 0.,
            "maximum_macro_constraint_violation_delta": 0.,
            "maximum_macro_capped_completion_time_delta": 0.,
            "require_transport_to_preserve_every_task_only_success": True,
        },
        "coverage_only_rows": ["acp", "human2sim2robot"],
    }
    source = _saved_confirmation(
        tmp_path, monkeypatch, tasks=("T1", "T2", "T3", "T4"),
        method_rows=methods,
        package={
            "version": "four-task-transfer-confirmation-package-v13",
            "transport_claim_contract": contract,
        },
    )
    permissions = {method: "B-log" for method in methods}
    permissions["response_mpc"] = "B-log-model"
    result = metrics.summarize_confirmation(
        tmp_path, source, tmp_path / "derived",
        references={"main": "cross_ebdm"}, ablation_terms={}, aliases={},
        resamples=200, seed=17, permission_classes=permissions,
    )
    claim = result["transport_claim"]
    assert claim["strong_baseline_checks"][
        "all_superiority_comparisons_same_permission"
    ] is False
    assert claim["strong_baseline_superiority_supported"] is False
    assert claim["core_claim_supported"] is False


def test_saved_summary_refuses_changed_bound_input(tmp_path, monkeypatch):
    source = _saved_confirmation(tmp_path, monkeypatch)
    with (source / "outcomes.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        metrics.summarize_confirmation(
            tmp_path, source, tmp_path / "derived", references={"main": "candidate"},
            ablation_terms={}, aliases={}, resamples=20,
            permission_classes={"candidate": "B_log", "fixed": "A_fixed", "rejected": "B_log"},
        )
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("damage, message", [
    ("instance", "scheduled instance identity"),
    ("receipt", "receipt identity"),
    ("roster", "duplicate roster"),
])
def test_semantic_audit_rejects_corruption_even_with_updated_hashes(
    tmp_path, monkeypatch, damage, message,
):
    source = _saved_confirmation(tmp_path, monkeypatch)
    launch = json.loads((source / "launch.json").read_text())
    if damage == "instance":
        pointer = launch["outcomes"]
    elif damage == "receipt":
        pointer = launch["state_receipts"]
    else:
        pointer = launch["sealed_artifacts"]["request_roster"]
    path = tmp_path / pointer["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "instance":
        rows[0]["training_or_calibration_instance"] = "wrong_instance"
    elif damage == "receipt":
        rows[0]["base_request_id"] = "wrong_request"
    else:
        rows.append(rows[0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pointer["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "launch.json").write_text(json.dumps(launch))
    with pytest.raises(ValueError, match=message):
        metrics.summarize_confirmation(
            tmp_path, source, tmp_path / "derived", references={"main": "candidate"},
            ablation_terms={}, aliases={}, resamples=20,
            permission_classes={"candidate": "B_log", "fixed": "A_fixed", "rejected": "B_log"},
        )
    assert not (tmp_path / "derived").exists()
