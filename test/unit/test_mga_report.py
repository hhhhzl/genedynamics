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
  - {name: kept, level: plane}
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

    overall_path.write_text(json.dumps({"total_experiments": 1}), encoding="utf-8")
    report = verify_results([str(config_path)], **kwargs)
    assert report["ok"] is False
    assert any("run-count mismatch" in error for error in report["errors"])
