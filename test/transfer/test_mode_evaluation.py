"""CPU-only contract checks from physical events, with no simulation rollout."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from genedynamics.tasks.transfer.episode import EpisodeRecord
from genedynamics.tasks.transfer.evaluation import (
    EVALUATION_VERSION,
    evaluate_episode,
    evaluate_records,
)
from genedynamics.tasks.transfer.guide_evaluation import (
    STRICT_GUIDE_VERSION,
    evaluate_strict_guide_episode,
)
from genedynamics.tasks.transfer.specs import task_spec


def _record(spec, prefix, *, mode=0):
    cfg = spec.config
    obstacle = spec.family == "obstacle"
    goal = cfg.goal_position if obstacle else cfg.goal
    hold = cfg.hold_steps if obstacle else round(.3 / cfg.control_dt)
    yaw = cfg.goal_yaw if obstacle else 0.
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = [[c, s, 0.], [s, -c, 0.], [0., 0., -1.]]
    positions = [*prefix, *([goal] * hold)]
    rows = []
    for index, position in enumerate(positions):
        rows.append({
            "tcp_position": list(position), "tcp_rotation": rotation,
            "valid": True, "finite": True, "joint_speed_peak": 0.,
            "joint_limit": False, "clearance": .1,
            "backend_status": 1 if index == len(positions) - 1 else 0,
            "normal_force": cfg.desired_force if not obstacle else 0.,
            "force_peak": cfg.desired_force if not obstacle else 0.,
            "lateral_force_peak": 0., "forbidden_collision": False,
            "axial_support": True, "loading": index >= len(prefix),
            "body_height": .3, "base_tilt": 0., "wbc_residual": 0.,
            "support_count": 4,
        })
    return EpisodeRecord(1, 2, 0, "fixture", traces=rows, state="ended", requested_mode=mode)


def _assert_success_fields(row, *, physical, constrained):
    assert row["physical_success"] is physical
    assert row["success"] is constrained
    assert row["constrained_success"] is constrained
    assert row["task_cost"] == (0. if constrained else 1.)
    assert row["status_mismatches"] == 0
    if not constrained:
        assert row["capped_normalized_completion_time"] == 1.


@pytest.mark.parametrize("alias,prefix", [
    ("T1", [(.5, 0., .7)]),
    ("T3", [(.44, -.20, .596), (.44, -.20, .610)]),
    ("T4", [(.44, -.20, .596), (.44, -.20, .610)]),
])
def test_flipping_requested_mode_preserves_physical_status_but_fails_task(alias, prefix):
    spec = task_spec(alias)
    record = _record(spec, prefix)
    matched = evaluate_episode(record, spec)
    mismatched = evaluate_episode(replace(record, requested_mode=1), spec)
    _assert_success_fields(matched, physical=True, constrained=True)
    _assert_success_fields(mismatched, physical=True, constrained=False)
    assert matched["evaluation_version"] == EVALUATION_VERSION
    assert matched["observed_mode"] == mismatched["observed_mode"] == 0
    assert matched["mode_match"] and not mismatched["mode_match"]
    assert matched["status"] == mismatched["status"] == 1
    assert mismatched["outcome"] == "mode_mismatch"
    assert not mismatched["constraint_violation"]


@pytest.mark.parametrize("alias", ["T1", "T3", "T4"])
def test_terminal_hold_without_route_events_is_unknown_and_unsuccessful(alias):
    spec = task_spec(alias)
    row = evaluate_episode(_record(spec, []), spec)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["observed_mode"] == -1 and not row["mode_match"]
    assert row["outcome"] == "mode_mismatch"


def test_t1_both_route_crossings_are_ambiguous_and_invalid_padding_is_ignored():
    spec = task_spec("T1")
    record = _record(spec, [(.5, 0., .7), (.5, -.2, .55)])
    row = evaluate_episode(record, spec)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["observed_mode"] == 2
    record.traces[1]["valid"] = False
    row = evaluate_episode(record, spec)
    _assert_success_fields(row, physical=True, constrained=True)
    assert row["observed_mode"] == 0


@pytest.mark.parametrize("alias", ["T3", "T4"])
def test_lateral_recovery_requires_actual_load_then_withdrawal(alias):
    spec = task_spec(alias)
    record = _record(spec, [(.44, -.2, .596), (.44, -.21, .600), (.44, -.21, .610)], mode=1)
    row = evaluate_episode(record, spec)
    _assert_success_fields(row, physical=True, constrained=True)
    assert row["observed_mode"] == 1
    for step in record.traces[:3]:
        step["normal_force"] = 0.
    row = evaluate_episode(record, spec)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["observed_mode"] == -1


def _guide_record():
    spec = task_spec("T2")
    return spec, _record(spec, [(.38, -.20, .596), (.40, -.20, .596),
                                (.44, -.19, .596), (.456, -.18, .596)])


def test_strict_t2_requires_both_contact_and_requested_crossing_order():
    spec, record = _guide_record()
    matched = evaluate_strict_guide_episode(record, spec, 0)
    mismatched = evaluate_strict_guide_episode(replace(record, requested_mode=1), spec, 1)
    _assert_success_fields(matched, physical=True, constrained=True)
    _assert_success_fields(mismatched, physical=True, constrained=False)
    assert matched["strict_contact_success"] and mismatched["strict_contact_success"]
    assert mismatched["outcome"] == "mode_mismatch"
    assert mismatched["group"] == "mode_1"
    assert not mismatched["constraint_violation"]
    summary = evaluate_records([record, replace(record, requested_mode=1)], spec)
    assert summary["evaluation_version"] == STRICT_GUIDE_VERSION
    assert summary["successes"] == 1 and summary["requested_episodes"] == 2
    assert summary["constrained_success_rate"] == .5
    assert summary["worst_group_constrained_success"] == 0.


def test_strict_t2_rejects_simultaneous_crossings_instead_of_assigning_mode_one():
    spec = task_spec("T2")
    record = _record(spec, [(.38, -.20, .596), (.40, -.19, .596)], mode=1)
    row = evaluate_strict_guide_episode(record, spec, 1)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["strict_contact_success"]
    assert row["observed_mode"] == 2 and row["outcome"] == "mode_mismatch"


def test_strict_t2_contact_failure_has_consistent_primary_metrics_and_reason():
    spec, record = _guide_record()
    for step in record.traces[:4]:
        step["axial_support"] = False
    row = evaluate_strict_guide_episode(record, spec, 0)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["mode_match"] and not row["strict_contact_success"]
    assert row["constraint_violation"]
    assert row["outcome"] == "guide_contact_not_established_before_outlet"
    assert row["legacy_success"] and row["status"] == 1


@pytest.mark.parametrize("missing_steps,expected", [(4, "guide_contact_coverage"), (8, "guide_contact_loss")])
def test_strict_t2_contact_coverage_and_gap_are_independent(missing_steps, expected):
    spec = task_spec("T2")
    before = [(.38, -.20, .596)] * (10 if missing_steps == 8 else 1)
    prefix = before + [(.40, -.20, .596)] * missing_steps + [(.456, -.19, .596)]
    record = _record(spec, prefix)
    for step in record.traces[len(before):len(before) + missing_steps]:
        step["axial_support"] = False
    row = evaluate_strict_guide_episode(record, spec, 0)
    _assert_success_fields(row, physical=True, constrained=False)
    assert row["outcome"] == expected and row["constraint_violation"]


def test_strict_t2_uses_physics_rate_contact_evidence_without_endpoint_aliasing():
    spec, record = _guide_record()
    for index, step in enumerate(record.traces):
        positions = np.repeat(
            np.asarray(step["tcp_position"], dtype=float)[None],
            spec.config.substeps, axis=0,
        )
        # Preserve the public crossing order at physics resolution.
        if index == len(record.traces) - 1:
            positions[:, 0] = np.linspace(.454, .456, spec.config.substeps)
        step["tcp_position_substeps"] = positions
        step["axial_support_substeps"] = np.ones(spec.config.substeps, dtype=bool)
        step["normal_force_substeps"] = np.full(spec.config.substeps, 5.0)
    # Endpoint samples still claim uninterrupted support, but the exact
    # evidence contains one complete 20 ms window without loaded support.
    record.traces[1]["normal_force_substeps"][:] = 0.0
    row = evaluate_strict_guide_episode(record, spec, 0)
    assert row["exact_physics_contact_evidence"]
    assert row["contact_evidence_resolution_seconds"] == spec.config.physics_dt
    assert row["maximum_contact_loss_seconds"] == pytest.approx(.02)
    # One 20 ms window is below the frozen 150 ms limit, so it remains valid.
    assert row["strict_contact_success"]


def test_physical_safety_failure_keeps_priority_over_mode_and_strict_contact():
    spec, record = _guide_record()
    record.requested_mode = 1
    record.traces[-1]["force_peak"] = spec.config.force_limit + 1.
    record.traces[-1]["backend_status"] = 9
    row = evaluate_strict_guide_episode(record, spec, 1)
    _assert_success_fields(row, physical=False, constrained=False)
    assert not row["mode_match"] and row["outcome"] == "overforce"
    assert row["constraint_violation"] and row["status"] == 9


@pytest.mark.parametrize("state", ["active", "rejected"])
def test_missing_physical_events_and_rejection_are_retained(state):
    spec = task_spec("T2")
    record = EpisodeRecord(1, 2, 0, "fixture", state=state, requested_mode=1)
    row = evaluate_strict_guide_episode(record, spec, 1)
    _assert_success_fields(row, physical=False, constrained=False)
    assert row["observed_mode"] == -1 and not row["mode_match"]
    assert row["physical_control_steps"] == 0
    assert row["outcome"] == ("rejected" if state == "rejected" else "incomplete")


def test_risk_set_empty_is_an_explicit_zero_action_failure_not_an_execution_error():
    spec = task_spec("T2")
    record = EpisodeRecord(
        1, 2, 0, "fixture", state="rejected", requested_mode=0,
        rejection_reason="risk_set_empty",
    )
    row = evaluate_strict_guide_episode(record, spec, 0)
    _assert_success_fields(row, physical=False, constrained=False)
    assert row["outcome"] == row["rejection_reason"] == "risk_set_empty"
    assert row["predeployment_rejection"]
    assert row["physical_control_steps"] == 0
    assert not row["constraint_violation"]


def test_disagreeing_request_identity_is_rejected_and_physical_mismatch_remains_independent():
    spec, record = _guide_record()
    with pytest.raises(ValueError, match="episode record"):
        evaluate_strict_guide_episode(record, spec, 1)
    record.traces[-1]["backend_status"] = 0
    row = evaluate_strict_guide_episode(record, spec, 0)
    assert row["success"] and row["status_mismatches"] == 1
    with pytest.raises(ValueError, match="0 or 1"):
        evaluate_episode(replace(record, requested_mode=2), spec)


def test_evaluator_import_does_not_load_physics_or_device_runtimes():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", """
import sys
from genedynamics.tasks.transfer.evaluation import evaluate_episode
from genedynamics.tasks.transfer.guide_evaluation import evaluate_strict_guide_episode
assert not ({'mujoco', 'jax', 'torch'} & set(sys.modules))
assert not any(name.startswith('genedynamics.envs') for name in sys.modules)
"""], capture_output=True, text=True, cwd=root,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu"))
    assert result.returncode == 0, result.stderr


def test_saved_development_counterexample_and_strict_guide_records():
    """Optional local evidence: never rerun physics or rewrite archived files."""
    root = Path(__file__).resolve().parents[2]
    diagnosis = root / "results/_protocol/four_task_transfer/confirmation_v1/diagnostics/contact_latency.json"
    if not diagnosis.exists():
        pytest.skip("archived local development records are not included in the checkout")
    audit = json.loads(diagnosis.read_text())
    reference = audit["mode_contract_audit"]["counterexample"]["saved_record"]
    cases = [("T3", "xarm7", 0, reference, True)]
    cases += [("T2", "panda" if case["domain"] == "source" else "xarm7",
               case["requested_mode"], case["saved_record"], case["domain"] == "source")
              for case in audit["T2"]["qualification_trace_checks"]]
    for alias, robot, mode, reference, expected_success in cases:
        payload = (root / reference["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
        data = json.loads(payload)[0]
        data["requested_mode"] = mode
        record = EpisodeRecord(**data)
        spec = task_spec(alias, robot=robot)
        evaluator = (lambda rec: evaluate_strict_guide_episode(rec, spec, rec.requested_mode)) if alias == "T2" else (lambda rec: evaluate_episode(rec, spec))
        matched = evaluator(record)
        _assert_success_fields(matched, physical=True, constrained=expected_success)
        assert matched["observed_mode"] == mode and matched["mode_match"]
        mismatched = evaluator(replace(record, requested_mode=1 - mode))
        _assert_success_fields(mismatched, physical=True, constrained=False)
        assert not mismatched["mode_match"]
        assert (root / reference["path"]).read_bytes() == payload
