import json
from pathlib import Path

import pytest

from genedynamics.experiments.utils.metrics import (
    aggregate_receding_diagnostics,
    audit_configs,
    compare_runner_outputs,
    summarize_results,
    verify_results,
)
from genedynamics.experiments.framework.config import ExperimentConfig


def test_report_keeps_h1_physical_force_primary_separate_from_endpoints():
    from genedynamics.experiments.utils.metrics import _plot_metric_rows

    rows = [{"metric": name, "mean": value} for name, value in (
        ("humanoid_box_push_metrics.force_peak", 166.0),
        ("humanoid_box_push_metrics.physics_force_peak", 331.0),
        ("humanoid_box_push_metrics.physics_force_normalized_cvar95", 5.52),
        ("surface_scan_metrics.force_peak", 20.0),
        ("peg_insert_metrics.peak_lateral_force", 7.0),
    )]
    assert _plot_metric_rows(rows, "force_peak") == [rows[0], rows[3]]
    assert _plot_metric_rows(rows, "physics_force_peak") == [rows[1]]
    assert _plot_metric_rows(rows, "physics_force_normalized_cvar95") == [rows[2]]
    # Missing physical evidence must not silently fall back to endpoint data.
    assert _plot_metric_rows([rows[0]], "physics_force_peak") == []
    assert _plot_metric_rows([rows[0]], "physics_force_normalized_cvar95") == []
    assert _plot_metric_rows(rows, "peak_lateral_force") == [rows[4]]


def _result(root: Path, algorithm: str, seed: int, safe: float, force: float):
    path = root / algorithm / "level_id" / f"seed_{seed}" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "suite": "id", "level": "id", "seed": seed,
        "metrics": {"peg_insert_metrics": {
            "insertion_success": 1.0,
            "safe_insertion_success": safe,
            "peak_lateral_force": force,
        }},
    }), encoding="utf-8")


def test_formal_mga_configs_pass_checkpoint_and_causal_protocol_audit():
    root = Path(__file__).resolve().parents[2]
    report = audit_configs([
        str(root / "configs/arm/surface_scan"),
        str(root / "configs/arm/peg_insert"),
        str(root / "configs/humanoid/push_to_line"),
    ])
    assert report["ok"], report["errors"]
    assert report["config_count"] == 35
    assert report["run_count_by_task"] == {
        "manipulator_surface_scan": 1160,
        "manipulator_peg_insert": 300,
        "humanoid_box_push": 580,
    }
    assert report["formal_run_count"] == 2040
    assert report["warnings"]
    assert all("explicitly waived" in warning for warning in report["warnings"])


def test_protocol_pairs_episode_length_within_each_suite(tmp_path: Path):
    for algorithm in ("dial", "mga"):
        config = ExperimentConfig(
            name=algorithm, output_dir=tmp_path / algorithm,
            env_name="dummy", method=algorithm,
            suites=[{"name": "force"}, {"name": "walk", "n_steps": 300}],
            method_params={"Nsample": 64, "Hsample": 16},
        )
        config.to_yaml(tmp_path / f"{algorithm}.yaml")
    assert audit_configs([str(tmp_path)])["ok"]
    path = tmp_path / "mga.yaml"
    changed = ExperimentConfig.from_yaml(path)
    changed.suites[1]["n_steps"] = 301
    changed.to_yaml(path)
    report = audit_configs([str(tmp_path)])
    assert not report["ok"]
    assert any("paired protocol differs" in error for error in report["errors"])


def test_report_writes_paired_statistics_and_representative_seed(tmp_path: Path):
    root = tmp_path / "results"
    _result(root, "mga", 10, 1.0, 5.0)
    _result(root, "mga", 11, 1.0, 7.0)
    _result(root, "dial", 10, 0.0, 9.0)
    _result(root, "dial", 11, 1.0, 8.0)
    output = tmp_path / "report"
    report = summarize_results([str(root)], str(output))
    assert report["result_count"] == 4
    paired = json.loads((output / "paired_deltas.json").read_text())
    safe_rows = [row for row in paired if row["metric"].endswith("safe_insertion_success")]
    assert safe_rows[0]["n_pairs"] == 2
    assert "paired_sign_flip_p" in safe_rows[0]
    representative = json.loads((output / "representative_seeds.json").read_text())
    assert {row["algorithm"] for row in representative} == {"dial", "mga"}


def test_report_reads_legacy_algorithm_directory_as_mga(tmp_path: Path):
    root = tmp_path / "results"
    _result(root, "full_mdac", 10, 1.0, 5.0)
    _result(root, "dial", 10, 0.0, 9.0)
    output = tmp_path / "report"
    report = summarize_results([str(root)], str(output))
    assert report["result_count"] == 2
    rows = json.loads((output / "summary.json").read_text())
    assert {row["algorithm"] for row in rows} == {"dial", "mga"}
    paired = json.loads((output / "paired_deltas.json").read_text())
    assert paired
    assert {row["reference"] for row in paired} == {"mga"}


def test_aborted_h1_seed_counts_against_ssr_without_prefix_force_advantage(tmp_path):
    root = tmp_path / "results"
    for seed, aborted in ((110, False), (111, True)):
        path = root / "mga" / "level_p3" / f"seed_{seed}" / "results.json"
        path.parent.mkdir(parents=True)
        metrics = {"safe_success": 0. if aborted else 1.,
                   "task_success": 0. if aborted else 1.}
        payload = {"suite": "p3", "seed": seed,
                   "metrics": {"humanoid_box_push_metrics": metrics}}
        if aborted:
            payload.update({
                "execution_status": {"state": "aborted_unrecoverable"},
                "partial_metrics": {"humanoid_box_push_metrics": {"physics_force_peak": 0.1}},
            })
        else:
            metrics["physics_force_peak"] = 44.
        path.write_text(json.dumps(payload))
    output = tmp_path / "report"
    report = summarize_results([str(root)], str(output))
    assert report["result_count"] == 2
    rows = json.loads((output / "summary.json").read_text())
    safe = next(row for row in rows if row["metric"].endswith(".safe_success"))
    force = next(row for row in rows if row["metric"].endswith(".physics_force_peak"))
    assert safe["n"] == 2 and safe["mean"] == .5
    assert force["n"] == 1 and force["mean"] == 44.
    assert force["n_attempted"] == 2 and force["n_completed"] == 1
    aborts = next(row for row in rows if row["metric"] == "execution.abort_rate")
    assert aborts["n"] == 2 and aborts["mean"] == .5
    representative = json.loads((output / "representative_seeds.json").read_text())
    assert len(representative) == 1 and representative[0]["seed"] == 110


def test_legacy_equivalence_compares_actions_and_scalar_metrics(tmp_path: Path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([{
        "algorithm": "dial", "suite": "id", "seed": 3,
        "series": {"actions": [[0.1, -0.2], [0.0, 0.3]]},
        "insertion_success": 1.0,
        "peak_lateral_force": 4.5,
        "runtime": 2.0,
    }]), encoding="utf-8")
    legacy.with_name("manifest.json").write_text(json.dumps({
        "git_sha": "frozen",
        "config": {
            "task": "manipulator_peg_insert", "method": "dial",
            "n_steps": 2, "level": "wide",
            "env_params": {"clearance": 0.0015},
            "execution_env_params": {},
            "method_params": {"Nsample": 64},
        },
    }), encoding="utf-8")
    seed_dir = tmp_path / "current" / "level_id" / "seed_3"
    (seed_dir / "trajectory").mkdir(parents=True)
    result = seed_dir / "results.json"
    result.write_text(json.dumps({
        "metrics": {"peg_insert_metrics": {
            "insertion_success": 1.0,
            "peak_lateral_force": 4.5,
            "runtime": 3.0,
        }},
        "config_snapshot": {
            "env_name": "manipulator_peg_insert", "method": "dial",
            "n_steps": 2,
            "env_params": {"clearance": 0.0015, "level": "wide"},
            "execution_env_params": {},
            "method_params": {"Nsample": 64, "controller_method": "dial"},
        },
        "provenance": {"git_sha": "current"},
    }), encoding="utf-8")
    (seed_dir / "trajectory" / "trajectory.json").write_text(json.dumps({
        "actions": [[0.1, -0.2], [0.0, 0.3]],
    }), encoding="utf-8")
    report = compare_runner_outputs(
        str(legacy), str(result), "dial", "id", 3,
    )
    assert report["ok"] is True
    assert report["actions_equal"] is True
    assert report["matched_metrics"] == 2
    assert report["config_comparison"]["equal"] is True
    assert len(report["runtime_comparisons"]) == 1


def test_receding_diagnostics_preserve_legacy_aggregation_contract():
    report = aggregate_receding_diagnostics(
        [
            {
                "prior_accepted": 1.0,
                "prior_predicted_improvement": 0.4,
                "prior_risk_ok": 1.0,
                "prior_force_veto": 0.0,
                "prior_risk_refined": [0.0, 0.2, 0.0, 0.1],
                "emergency_selected": 0.0,
                "reliability_abstained": 1.0,
            },
            {
                "prior_accepted": 0.0,
                "prior_predicted_improvement": -0.2,
                "prior_risk_ok": 0.0,
                "prior_force_veto": 1.0,
                "prior_risk_refined": [1.0, 0.4, 1.0, 0.3],
                "emergency_selected": 1.0,
                "reliability_abstained": 0.0,
            },
        ],
        task="manipulator_peg_insert",
    )
    assert report["prior_acceptance_rate"] == 0.5
    assert report["prior_fallback_rate"] == 0.5
    assert report["prior_predicted_improvement_mean"] == pytest.approx(0.1)
    assert report["prior_risk_refined_force_violation"] == 0.5
    assert report["prior_risk_refined_torque_violation"] == pytest.approx(0.3)
    assert report["prior_risk_refined_jam"] == 0.5
    assert report["prior_risk_refined_force_mae"] == pytest.approx(0.2)
    assert report["emergency_selected"] == 0.5
    assert report["reliability_abstained"] == 0.5


def test_verifier_rebases_development_subset_and_detects_overwritten_summary(
    tmp_path: Path,
):
    config_path = tmp_path / "method.yaml"
    config_path.write_text(
        """
name: method
output_dir: results/mga_test/development_isolation
env_name: manipulator_surface_scan
method: dial
seeds: [10, 11]
n_steps: 2
metrics: []
visualizations: []
method_params: {Nsample: 4, Hsample: 4, Hnode: 2, Ndiffuse: 1, Ndiffuse_init: 1}
suites:
  - {name: kept, level: plane, n_steps: 3}
  - {name: skipped, level: cylinder}
""",
        encoding="utf-8",
    )
    development_root = tmp_path / "p7"
    cfg = ExperimentConfig.from_yaml(config_path)
    cfg.use_development_output_root(development_root)
    cfg.seeds = [0, 1]
    cfg.suites = [suite for suite in cfg.suites if suite["name"] == "kept"]
    cfg.output_dir.mkdir(parents=True)
    (cfg.output_dir / "protocol_manifest.json").write_text(json.dumps({
        "config": cfg.to_dict(),
    }), encoding="utf-8")
    level_dir = cfg.output_dir / "level_kept"
    level_dir.mkdir()
    (level_dir / "summary.json").write_text("{}", encoding="utf-8")
    for seed in cfg.seeds:
        seed_dir = level_dir / f"seed_{seed}"
        trajectory_dir = seed_dir / "trajectory"
        trajectory_dir.mkdir(parents=True)
        suite_cfg = cfg.for_suite(cfg.suites[0])
        snapshot = suite_cfg.to_dict()
        snapshot["seeds"] = [seed]
        (seed_dir / "results.json").write_text(json.dumps({
            "seed": seed,
            "suite": "kept",
            "config_snapshot": snapshot,
        }), encoding="utf-8")
        (trajectory_dir / "trajectory.json").write_text("{}", encoding="utf-8")
    overall_path = cfg.output_dir / "overall_summary.json"
    overall_path.write_text(json.dumps({"total_experiments": 2}), encoding="utf-8")

    kwargs = {
        "development_root": development_root,
        "seeds": [0, 1],
        "suites": ["kept"],
    }
    assert verify_results([str(config_path)], **kwargs)["ok"] is True

    result_path = level_dir / "seed_0" / "results.json"
    result = json.loads(result_path.read_text())
    result["config_snapshot"]["n_steps"] = 2
    result_path.write_text(json.dumps(result), encoding="utf-8")
    report = verify_results([str(config_path)], **kwargs)
    assert any("budget mismatch n_steps" in error for error in report["errors"])
    result["config_snapshot"]["n_steps"] = 3
    result_path.write_text(json.dumps(result), encoding="utf-8")

    manifest_path = cfg.output_dir / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["suites"][0]["n_steps"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_results([str(config_path)], **kwargs)
    assert any("resolved suite budget mismatch" in error for error in report["errors"])
    manifest["config"]["suites"][0]["n_steps"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    overall_path.write_text(json.dumps({"total_experiments": 1}), encoding="utf-8")
    report = verify_results([str(config_path)], **kwargs)
    assert report["ok"] is False
    assert any("run-count mismatch" in error for error in report["errors"])


@pytest.mark.parametrize("pattern", ["all_na", "mixed", "legacy"])
def test_receding_comparison_applicability_excludes_placeholders(pattern):
    def row(improvement, applicable):
        value = {
            "prior_accepted": 1.0, "prior_predicted_improvement": improvement,
            "prior_risk_ok": 1.0, "prior_force_veto": 0.0,
            "reliability_risk_incumbent": [improvement] * 4,
            "reliability_support_incumbent": improvement,
            "additive_prior_improvement": improvement,
            "emergency_fallback_recovery": float(not applicable),
        }
        if pattern != "legacy":
            value.update({
                "prior_comparison_applicable": float(applicable),
                "reliability_incumbent_applicable": float(applicable),
                "additive_prior_comparison_applicable": float(applicable),
            })
        return value
    rows = (
        [row(0.0, False), row(0.0, False)] if pattern == "all_na"
        else [row(0.0, False), row(4.0, True)]
    )
    result = aggregate_receding_diagnostics(rows, task="humanoid_box_push")
    keys = [
        "prior_predicted_improvement_mean", "additive_prior_improvement",
        "reliability_risk_incumbent_force_violation", "reliability_support_incumbent",
    ]
    counts = ["prior_comparison_count", "reliability_incumbent_count",
              "additive_prior_comparison_count"]
    if pattern == "all_na":
        assert all(key not in result for key in keys)
        assert all(result[key] == 0.0 for key in counts)
    else:
        expected = 4.0 if pattern == "mixed" else 2.0
        assert all(result[key] == expected for key in keys)
        if pattern == "mixed":
            assert all(result[key] == 1.0 for key in counts)
        else:
            assert all(key not in result for key in counts)
    # These are finite placeholders plus explicit applicability, not NaN JSON.
    json.dumps({"infos": rows, "aggregates": result}, allow_nan=False)


@pytest.mark.parametrize("task,labels", [
    ("humanoid_box_push", ("force_violation", "invalid_contact", "balance", "force_mae")),
    ("manipulator_peg_insert", ("force_violation", "torque_violation", "jam", "force_mae")),
    ("manipulator_surface_scan", ("force_violation", "contact_loss", "deformation", "force_mae")),
])
def test_receding_risk_names_follow_exact_task_semantics(task, labels):
    row = {
        "prior_accepted": 1.0, "prior_predicted_improvement": 1.0,
        "prior_risk_ok": 1.0, "prior_force_veto": 0.0,
        "prior_risk_refined": [1.0, 2.0, 3.0, 4.0],
    }
    result = aggregate_receding_diagnostics([row], task=task)
    actual = {key: value for key, value in result.items()
              if key.startswith("prior_risk_refined_")}
    assert actual == {f"prior_risk_refined_{label}": float(index + 1)
                      for index, label in enumerate(labels)}
