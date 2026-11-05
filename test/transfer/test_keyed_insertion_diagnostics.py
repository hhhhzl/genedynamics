"""Force and paired-event checks for keyed insertion evidence diagnostics."""

import pytest

from genedynamics.tasks.transfer.keyed_insertion_diagnostics import (
    diagnose_episode,
    first_force_violation,
    summarize_paired_diagnoses,
)


def trace(*, force=0.0, lateral=0.0, loading=True):
    return {
        "force_peak": force,
        "lateral_force_peak": lateral,
        "axial_support": True,
        "loading": loading,
        "applied_action": [0.003, 0.004, 0.008] + [0.0] * 10,
        "public_reference_lead": [0.001, 0.002, 0.0],
    }


def diagnostic(*, phase, admissions=0, blend=0.0, retreating=False):
    return {
        "base": {
            "retreating": [retreating],
            "observer": {"memory": {"phase": [phase]}},
        },
        "transport_admission_controls": admissions,
        "last_task_dominance_blend": blend,
    }


def episode(*, outcome="overforce", request="r0", mode=0):
    result = {
        "record_id": request,
        "instance_id": "instance_01",
        "outcome": outcome,
        "request": {
            "request_id": request,
            "requested_mode": mode,
            "mechanism_group": "nominal",
        },
    }
    evidence = {
        "task_spec": {
            "name": "keyed_insertion_recovery",
            "config": {"force_limit": 25.0, "lateral_force_limit": 10.0},
        },
        "records": [{
            "traces": [
                trace(force=4.0, lateral=2.0),
                trace(force=21.0, lateral=11.0, loading=False),
            ],
            "control_diagnostics": [
                diagnostic(phase=0, admissions=1, blend=4.0),
                diagnostic(phase=1, admissions=1, blend=0.0),
            ],
        }],
    }
    return result, evidence


def test_first_force_violation_matches_strict_evaluator_thresholds():
    rows = [
        trace(force=25.0, lateral=10.0),
        trace(force=25.1, lateral=9.0),
    ]
    event = first_force_violation(
        rows, force_limit_N=25.0, lateral_force_limit_N=10.0,
    )

    assert event.control_index == 1
    assert event.channel == "total"
    with pytest.raises(ValueError, match="finite positive force limits"):
        first_force_violation(rows, force_limit_N=0.0, lateral_force_limit_N=10.0)


def test_episode_diagnosis_exposes_delayed_transport_and_recovery_state():
    result, evidence = episode()

    row = diagnose_episode(result, evidence)

    assert row["force_violation_channel"] == "lateral"
    assert row["event"] == {
        "control_index": 1,
        "phase": 1,
        "retreating": False,
        "axial_support": True,
        "loading": False,
    }
    assert row["transport"]["admissions_before_or_at_event"] == 1
    assert row["transport"]["admitted_on_event_control"] is False
    assert row["command"]["applied_xy_speed_mps"] == pytest.approx(0.005)


def test_paired_summary_separates_common_failure_regression_and_gain():
    result, evidence = episode(request="common")
    common = diagnose_episode(result, evidence)
    regression_result, regression_evidence = episode(request="regression")
    regression = diagnose_episode(regression_result, regression_evidence)
    gain_result, gain_evidence = episode(outcome="success", request="gain")
    gain_evidence["records"][0]["traces"] = [trace(force=4.0, lateral=2.0)]
    gain_evidence["records"][0]["control_diagnostics"] = [diagnostic(phase=3)]
    gain = diagnose_episode(gain_result, gain_evidence)

    task_common = dict(common)
    task_regression = {**regression, "outcome": "success", "force_violation": None,
                       "force_violation_channel": None}
    task_gain = {**gain, "outcome": "overforce"}
    summary = summarize_paired_diagnoses(
        [common, regression, gain], [task_common, task_regression, task_gain],
    )

    assert summary["paired_episodes"] == 3
    assert summary["overforce"]["common_with_task_only"] == 1
    assert summary["overforce"]["ours_only_regressions"] == 1
    assert summary["overforce"]["ours_only_gains"] == 1
    assert summary["overforce"]["regressions_with_prior_transport"] == 1
