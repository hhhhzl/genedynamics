"""Durable episode evidence can be audited without replaying a simulator."""
from collections import namedtuple
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from genedynamics.tasks.transfer.episode import (
    CONTRACT_VERSION,
    EpisodeRecord,
    audit_episode_evidence,
    controller_diagnostics,
    episode_evidence,
    load_episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.specs import task_spec


def _successful_record(spec):
    """Actual event evidence plus the task's terminal hold, with no backend."""
    cfg = spec.config
    prefixes = {
        "T1": [(0.5, 0., .7)],
        "T2": [(.38, -.20, .596), (.40, -.20, .596),
               (.44, -.19, .596), (.456, -.18, .596)],
        "T3": [(.44, -.20, .596), (.44, -.20, .610)],
        "T4": [(.44, -.20, .596), (.44, -.20, .610)],
    }
    obstacle = spec.family == "obstacle"
    goal = cfg.goal_position if obstacle else cfg.goal
    count = cfg.hold_steps if obstacle else round(.3 / cfg.control_dt)
    yaw = cfg.goal_yaw if obstacle else 0.
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[c, s, 0.], [s, -c, 0.], [0., 0., -1.]])
    positions = prefixes[spec.alias] + [goal] * count
    force = 0. if obstacle else cfg.desired_force
    rows = [{
        "tcp_position": np.asarray(position), "tcp_rotation": rotation.copy(),
        "valid": True, "finite": True, "joint_speed_peak": 0.,
        "joint_limit": False, "clearance": .1,
        "backend_status": int(index == len(positions) - 1),
        "normal_force": force, "force_peak": force, "lateral_force_peak": 0.,
        "forbidden_collision": False, "axial_support": True, "loading": True,
        "body_height": .3, "base_tilt": 0., "wbc_residual": 0., "support_count": 4,
    } for index, position in enumerate(positions)]
    record = EpisodeRecord(
        sample_id=17, seed=937017, generation=0, initial="synthetic_fixture",
        requested_mode=0, state="ended", traces=rows,
        requested_actions=[np.zeros(13, np.float32) for _ in rows],
        control_diagnostics=[{"stage": np.int64(1), "compensation": np.zeros(6)} for _ in rows],
    )
    record.reset_evidence = {
        "initial_public_pose": np.asarray(positions[0]),
        "sample_id": np.int64(record.sample_id),
    }
    return record


def _write(path, payload):
    # Reject JSON NaN rather than relying on a permissive JSON reader.
    encoded = json.dumps(payload, allow_nan=False).encode()
    path.write_bytes(gzip.compress(encoded) if path.suffix == ".gz" else encoded)
    return path


@pytest.mark.parametrize("alias", ("T1", "T2", "T3", "T4"))
def test_compressed_evidence_roundtrip_recomputes_mode_correct_outcomes(tmp_path, alias):
    spec = task_spec(alias)
    success = _successful_record(spec)
    mismatch = deepcopy(success)
    mismatch.sample_id += 1
    mismatch.requested_mode = 1
    records = [success, mismatch]
    report = evaluate_records(records, spec)
    assert report["successes"] == 1
    assert report["episodes"][1]["outcome"] == "mode_mismatch"
    path = _write(tmp_path / "episode_evidence.json.gz", episode_evidence(records, spec, report))

    restored, restored_spec, saved_report = load_episode_evidence(path)
    assert json.dumps(restored_spec.describe(), sort_keys=True) == json.dumps(spec.describe(), sort_keys=True)
    assert restored_spec.alias == alias
    assert [record.requested_mode for record in restored] == [0, 1]
    assert saved_report == report
    assert evaluate_records(restored, restored_spec) == report
    assert len(restored[0].requested_actions) == len(restored[0].traces) == len(restored[0].control_diagnostics)
    np.testing.assert_array_equal(restored[0].requested_actions, success.requested_actions)
    np.testing.assert_array_equal(restored[0].reset_evidence["initial_public_pose"], success.reset_evidence["initial_public_pose"])

    # Recorded evaluation is an auditable claim, never the replayed label.
    saved_report["successes"] = 999
    assert evaluate_records(restored, restored_spec)["successes"] == 1


def test_nonfinite_failed_episode_survives_strict_json_and_gzip(tmp_path):
    spec = task_spec("T3")
    record = _successful_record(spec)
    record.traces = record.traces[:1]
    record.traces[0].update(
        finite=False, backend_status=6,
        tcp_position=np.asarray([np.nan, -.2, .596]),
        normal_force=np.float64(np.nan), force_peak=np.inf,
    )
    record.requested_actions = [np.asarray([np.nan, np.inf, -np.inf] + [0.] * 10)]
    record.control_diagnostics = [{"compensation": np.asarray([np.nan, np.inf, -np.inf])}]
    report = evaluate_records([record], spec)
    assert report["episodes"][0]["outcome"] == "nonfinite"
    path = _write(tmp_path / "failed.json.gz", episode_evidence([record], spec, report))
    restored, restored_spec, saved_report = load_episode_evidence(path)
    action = np.asarray(restored[0].requested_actions[0])
    assert np.isnan(action[0]) and np.isposinf(action[1]) and np.isneginf(action[2])
    assert np.isnan(restored[0].traces[0]["tcp_position"][0])
    assert np.isposinf(restored[0].traces[0]["force_peak"])
    assert np.isnan(saved_report["episodes"][0]["final_position_error_m"])
    np.testing.assert_equal(restored[0].control_diagnostics[0]["compensation"], [np.nan, np.inf, -np.inf])
    replayed = evaluate_records(restored, restored_spec)
    assert replayed["successes"] == 0 and replayed["episodes"][0]["outcome"] == "nonfinite"
    assert audit_episode_evidence(path)["passed"]


@pytest.mark.parametrize("field", ("requested_actions", "traces", "control_diagnostics"))
def test_damaged_evidence_counts_are_rejected(tmp_path, field):
    spec = task_spec("T3")
    record = _successful_record(spec)
    payload = episode_evidence([record], spec, evaluate_records([record], spec))
    payload["records"][0][field].pop()
    with pytest.raises(ValueError, match="count"):
        load_episode_evidence(_write(tmp_path / "damaged.json.gz", payload))


@pytest.mark.parametrize("damage", ("missing_mode", "invalid_mode", "contract", "schema"))
def test_evidence_identity_must_be_explicit_and_compatible(tmp_path, damage):
    spec = task_spec("T2")
    record = _successful_record(spec)
    payload = episode_evidence([record], spec, evaluate_records([record], spec))
    if damage == "missing_mode":
        del payload["records"][0]["requested_mode"]
    elif damage == "invalid_mode":
        payload["records"][0]["requested_mode"] = 2
    elif damage == "contract":
        payload["contract_version"] = CONTRACT_VERSION + "-other"
    else:
        payload["schema_version"] = "unsupported-evidence"
    with pytest.raises(ValueError):
        load_episode_evidence(_write(tmp_path / "damaged.json", payload))


def test_empty_rejection_records_and_default_reset_evidence_are_supported(tmp_path):
    record = EpisodeRecord(1, 2, 0, "not_executed", state="rejected", requested_mode=1,
                           rejection_reason="source qualification failed")
    other = EpisodeRecord(2, 2, 0, "not_executed")
    record.reset_evidence["reset_attempted"] = False
    assert other.reset_evidence == {}
    spec = task_spec("T2")
    report = evaluate_records([record], spec)
    restored, restored_spec, saved = load_episode_evidence(
        _write(tmp_path / "rejected.json.gz", episode_evidence([record], spec, report))
    )
    assert saved == evaluate_records(restored, restored_spec) == report
    assert restored[0].requested_actions == restored[0].traces == restored[0].control_diagnostics == []
    assert restored[0].reset_evidence == {"reset_attempted": False}


def test_risk_set_empty_rejection_roundtrips_and_reaudits_without_physics(tmp_path):
    spec = task_spec("T2")
    record = EpisodeRecord(
        1, 2, 0, "risk_admission", state="rejected", requested_mode=0,
        rejection_reason="risk_set_empty",
    )
    report = evaluate_records([record], spec)
    assert report["episodes"][0]["outcome"] == "risk_set_empty"
    assert report["predeployment_rejections"] == 1
    path = _write(
        tmp_path / "risk_set_empty.json.gz",
        episode_evidence([record], spec, report),
    )
    restored, restored_spec, saved = load_episode_evidence(path)
    assert saved == evaluate_records(restored, restored_spec)
    assert restored[0].requested_actions == restored[0].traces == []
    audit = audit_episode_evidence(path)
    assert audit["passed"] and audit["physical_calls"] == 0


class _PublicController:
    def __init__(self):
        self.stage = 2
        self.compensation = np.asarray([.01, -.02, 0.])
        self.irreversible_network_calls = 7
        self.terminal_planar_distance_m = .003
        self.terminal_planar_speed_cap_active = True
        self.pre_descent_settling_active = True
        self.pre_descent_settling_complete = False
        self.pre_descent_settling_controls = 7
        self.pre_descent_settled_controls = 3
        self.pre_descent_reference_lead_norm_m = .0015
        self.pre_descent_planar_speed_mps = .003
        self.pre_descent_reference_unloading_active = True
        self.pre_descent_reference_unloading_speed_mps = .01
        self.public_reference_delay_estimate_controls = 3
        self.public_reference_delay_identification_samples = 8
        self.public_reference_delay_score_gap_mps = .002
        self.pre_descent_causal_unloading_limit_mps = .01
        self.task_dominance_blend_counts = np.asarray([5, 9])
        self.transport_admission_controls = 5
        self.task_only_fallback_controls = 9
        self.last_task_dominance_blend = 1.
        self.last_transport_task_proxy = .2
        self.last_task_only_task_proxy = .3
        memory_type = namedtuple("Memory", ("phase", "contact_count"))
        self.memory = memory_type(np.asarray([1]), np.asarray([4]))

    def get_state(self):
        raise AssertionError("diagnostics must not request a complete checkpoint")

    @property
    def weights(self):
        raise AssertionError("diagnostics must not read learned weights")

    @property
    def plant(self):
        raise AssertionError("diagnostics must not read plant state")


def test_controller_diagnostics_copy_public_memory_without_checkpoint_or_plant():
    actor = _PublicController()
    diagnostics = controller_diagnostics(actor)
    assert int(diagnostics["stage"]) == 2
    assert int(diagnostics["irreversible_network_calls"]) == 7
    assert diagnostics["terminal_planar_distance_m"] == pytest.approx(.003)
    assert bool(diagnostics["terminal_planar_speed_cap_active"])
    assert bool(diagnostics["pre_descent_settling_active"])
    assert not bool(diagnostics["pre_descent_settling_complete"])
    assert diagnostics["pre_descent_settling_controls"] == 7
    assert diagnostics["pre_descent_settled_controls"] == 3
    assert diagnostics["pre_descent_reference_lead_norm_m"] == pytest.approx(.0015)
    assert diagnostics["pre_descent_planar_speed_mps"] == pytest.approx(.003)
    assert bool(diagnostics["pre_descent_reference_unloading_active"])
    assert diagnostics[
        "pre_descent_reference_unloading_speed_mps"
    ] == pytest.approx(.01)
    assert diagnostics["public_reference_delay_estimate_controls"] == 3
    assert diagnostics["public_reference_delay_identification_samples"] == 8
    assert diagnostics["public_reference_delay_score_gap_mps"] == pytest.approx(
        .002
    )
    assert diagnostics[
        "pre_descent_causal_unloading_limit_mps"
    ] == pytest.approx(.01)
    np.testing.assert_array_equal(
        diagnostics["task_dominance_blend_counts"], [5, 9]
    )
    assert diagnostics["transport_admission_controls"] == 5
    assert diagnostics["task_only_fallback_controls"] == 9
    assert diagnostics["last_task_dominance_blend"] == pytest.approx(1.)
    assert diagnostics["last_transport_task_proxy"] == pytest.approx(.2)
    assert diagnostics["last_task_only_task_proxy"] == pytest.approx(.3)
    np.testing.assert_array_equal(diagnostics["memory"]["phase"], [1])
    assert "weights" not in diagnostics and "plant" not in diagnostics
    diagnostics["compensation"][0] = 99.
    diagnostics["memory"]["phase"][0] = 99
    assert actor.compensation[0] == .01 and actor.memory.phase[0] == 1


def test_forwarding_wrapper_does_not_duplicate_candidate_resource_counts():
    class Wrapper:
        def __init__(self, candidate):
            self.candidate = candidate
            self.fallback_controls = 3
            self.forwarded_attribute_reads = []

        def __getattr__(self, name):
            self.forwarded_attribute_reads.append(name)
            return getattr(self.candidate, name)

    actor = Wrapper(_PublicController())
    diagnostics = controller_diagnostics(actor)
    assert actor.forwarded_attribute_reads == []
    assert "irreversible_network_calls" not in diagnostics
    assert int(diagnostics["candidate"]["irreversible_network_calls"]) == 7
    assert int(diagnostics["fallback_controls"]) == 3
    assert "fallback_controls" not in diagnostics["candidate"]


def _auditable_payload():
    spec = task_spec("T3")
    record = _successful_record(spec)
    report = evaluate_records([record], spec)
    report["accounting"] = {
        "control_steps": len(record.traces), "preparation_control_steps": 0,
        "physics_steps": len(record.traces) * spec.config.substeps,
        "reset_calls": 1, "command_requests": len(record.traces),
    }
    report["lifetime_accounting"] = {
        **report["accounting"], "reset_calls": 4,
        "control_steps": len(record.traces) + 9,
    }
    return episode_evidence([record], spec, report)


def test_offline_audit_accepts_scoring_with_separate_adapter_accounting(tmp_path):
    payload = _auditable_payload()
    audit = audit_episode_evidence(_write(tmp_path / "valid.json.gz", payload))
    assert audit["passed"]
    assert audit["recorded_evaluation_matches"]
    assert audit["recorded_accounting_matches"]
    assert audit["evaluator_sources_match"]
    assert audit["physical_calls"] == audit["gpu_calls"] == 0
    assert audit["evaluation"]["successes"] == 1
    assert "accounting" not in audit["evaluation"]


@pytest.mark.parametrize("damage", ("aggregate_success", "episode_success", "control_charge", "source_hash"))
def test_offline_audit_rejects_tampered_score_charge_or_evaluator_source(tmp_path, damage):
    payload = _auditable_payload()
    if damage == "aggregate_success":
        payload["evaluation"]["successes"] = 0
    elif damage == "episode_success":
        payload["evaluation"]["episodes"][0]["success"] = False
    elif damage == "control_charge":
        payload["evaluation"]["accounting"]["control_steps"] += 1
    else:
        payload["evaluator_sources"]["evaluation.py"] = "0" * 64
    audit = audit_episode_evidence(_write(tmp_path / "tampered.json.gz", payload))
    assert not audit["passed"]
    assert audit["recorded_evaluation_matches"] == (damage not in {"aggregate_success", "episode_success"})
    assert audit["recorded_accounting_matches"] == (damage != "control_charge")
    assert audit["evaluator_sources_match"] == (damage != "source_hash")
    assert audit["evaluation"]["successes"] == 1
    assert audit["physical_calls"] == audit["gpu_calls"] == 0


def test_incomplete_raw_physical_transition_is_preserved_and_charged(tmp_path):
    spec = task_spec("T3")
    record = EpisodeRecord(18, 937018, 0, "synthetic_fixture", state="aborted", requested_mode=1)
    record.incomplete_transitions = [{
        "requested_action": np.arange(13, dtype=np.float32),
        "physical_trace": {
            "valid": np.bool_(True), "tcp_position": np.asarray([.44, -.21, .6]),
            "command": np.linspace(0., .01, 13), "normal_force": np.float64(np.nan),
        },
        "error_type": "RuntimeError", "reason": "post-transition measurement failed",
    }]
    report = evaluate_records([record], spec)
    report["accounting"] = {"control_steps": 1, "preparation_control_steps": 0}
    payload = episode_evidence([record], spec, report)
    path = _write(tmp_path / "incomplete.json.gz", payload)
    restored, _, saved = load_episode_evidence(path)
    assert restored[0].traces == restored[0].requested_actions == []
    assert saved["episodes"][0]["outcome"] == "incomplete"
    assert saved["episodes"][0]["physical_control_steps"] == 0
    retained = restored[0].incomplete_transitions[0]
    np.testing.assert_array_equal(retained["requested_action"], np.arange(13))
    np.testing.assert_array_equal(retained["physical_trace"]["tcp_position"], [.44, -.21, .6])
    assert np.isnan(retained["physical_trace"]["normal_force"])
    assert retained["error_type"] == "RuntimeError"
    assert audit_episode_evidence(path)["passed"]
    payload["evaluation"]["accounting"]["control_steps"] = 0
    audit = audit_episode_evidence(_write(tmp_path / "undercharged.json.gz", payload))
    assert audit["recorded_evaluation_matches"]
    assert not audit["recorded_accounting_matches"] and not audit["passed"]
