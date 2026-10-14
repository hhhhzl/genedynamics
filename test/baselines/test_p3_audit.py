"""P3 must fail closed on missing, stale or partial evidence."""
import hashlib
import json

from scripts.baselines.p3 import audit, core_gate, physical_gate
from scripts.baselines import require_non_paper_output
import pytest


def test_core_report_rejects_changed_source_and_skips(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("qualified version")
    report = dict(status="source_core_parity_passed", tests=1, skipped=0, failures=0,
                  local_sha256={"model.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(report))
    assert core_gate(path, repository=tmp_path)["qualified"]
    source.write_text("changed version")
    assert not core_gate(path, repository=tmp_path)["qualified"]
    source.write_text("qualified version")
    report["skipped"] = 1; path.write_text(json.dumps(report))
    assert not core_gate(path, repository=tmp_path)["qualified"]


def test_success_without_trace_replay_does_not_qualify():
    report = dict(evaluation=dict(successes=1, status_mismatches=0),
                  replay=dict(matches_status=True, max_pose_error=0.))
    assert not physical_gate(report)
    report["replay"]["trace_exact"] = True
    assert physical_gate(report)
    report["evaluation"]["status_mismatches"] = 1
    assert not physical_gate(report)


def test_missing_native_assets_and_failed_attempt_are_retained(tmp_path):
    attempt = tmp_path / "results/arm/task/qualification/method/level_source/seed_1/attempt.json"
    attempt.parent.mkdir(parents=True)
    attempt.write_text(json.dumps(dict(status="failed", lifetime_accounting=dict(control_steps=7))))
    report = audit(tmp_path)
    assert not report["full_p3_qualified"]
    assert report["native"]["human2sim2robot_native"]["status"] == "assets_pending"
    assert report["reports"][0]["status"] == "failed"
    assert report["reports"][0]["lifetime_accounting"]["control_steps"] == 7


def test_paper_root_and_nested_report_paths_are_rejected():
    from scripts.baselines.p3 import ROOT
    for path in (ROOT / "latex", ROOT / "latex/cross_ebdm/../report.json"):
        with pytest.raises(ValueError, match="outside latex"):
            require_non_paper_output(path)


def test_resource_deferral_never_promotes_native_qualification(tmp_path):
    path = tmp_path / "_reports/route-decisions.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"routes": {"human2sim2robot_native": {"status": "deferred", "reason": "user has no assets"}}}))
    result = audit(tmp_path)
    assert result["deferred_routes"] == ["human2sim2robot_native"]
    assert "human2sim2robot_native" in result["native_closed_loop_pending"]
    assert "human2sim2robot_native" not in result["active_native_closed_loop_pending"]
    assert not result["full_p3_qualified"]


def test_failed_source_physical_audit_keeps_denominator_and_target_separate(tmp_path):
    path = tmp_path / "results/arm/keyed_insertion_recovery/qualification/acp_source/level_state/seed_1/physical_qualification.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(
        status="frozen_source_physical_audit_completed", source_only=True, target_calls=0,
        physical_gate_passed=False,
        evaluation=dict(requested_episodes=4, successes=0, outcomes={"overforce": 4}, status_mismatches=0),
    )))
    result = audit(tmp_path)
    item = result["reports"][0]
    assert item["evaluation"]["requested_episodes"] == 4
    assert item["evaluation"]["successes"] == 0
    assert item["source_only"] and item["target_calls"] == 0
    assert not item["physical_gate_passed"]
    assert not result["gates"]["physical_T3"]["qualified"]
    assert not result["full_p3_qualified"]


def test_runtime_exit_zero_annotation_retains_original_and_blocks_completion_claim(tmp_path):
    leaf = tmp_path / "results/dexterous/tight_insertion/qualification/play2perfect_native/level_isaacsim/seed_42"
    leaf.mkdir(parents=True)
    (leaf / "native_attempt.json").write_text(json.dumps(dict(status="complete", worker_returncode=0, control_calls=0)))
    (leaf / "runtime_failure_audit.json").write_text(json.dumps(dict(status="startup_failed_before_rollout", native_closed_loop_qualified=False)))
    result = audit(tmp_path)
    item = result["reports"][0]
    assert item["reported_status"] == "complete"
    assert item["status"] == "startup_failed_before_rollout"
    assert "play2perfect_native" in result["active_native_closed_loop_pending"]
    assert not result["full_p3_qualified"]
