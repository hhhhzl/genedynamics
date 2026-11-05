"""The independent table must never score an unexecuted method as failure."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/paper/cross_ebdm/audit_main_table_admission.py"


def _audit():
    spec = importlib.util.spec_from_file_location("main_table_admission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit()


def test_independent_table_admission_distinguishes_missing_execution():
    report = _audit()
    assert report["required_execution_cells"] == 36
    assert 8 <= report["independent_table_eligible_cells"] <= 36
    assert report["statistics_contract"] == "four-task-independent-main-table-statistics-v5"
    assert report["frozen_independent_requests"]["request_count"] == 200
    assert report["frozen_independent_requests"]["status"] == "sealed_unaccessed"
    for cohort in report["independent_training_cohorts"].values():
        assert set(cohort["tasks"]) == {"T1", "T2", "T3", "T4"}
        assert all(row["instances"] == 8 and sum(row["states"].values()) == 8
                   for row in cohort["tasks"].values())
    asid = report["independent_training_cohorts"]["asid"]
    assert asid["tasks"]["T4"]["training_cohort"] == "independent_main_table_training_v2"
    assert asid["active_continuations"]["T4"]["retained_v1_failure"]["path"].endswith(
        "/independent_training_8804003/report.json"
    )
    assert all(asid["tasks"][task]["actor_package_qualification"]["status"]
               == "passed_development_only" for task in ("T1", "T3"))
    assert all(asid["tasks"][task]["source_nominal_qualification"]["status"]
               == "passed_development_only" for task in ("T1", "T3"))
    t1_asid = next(row for row in report["rows"]
                   if row["task"] == "T1" and row["method"] == "asid")
    assert t1_asid["admission_status"] in {
        "eight_instance_source_nominal_development_qualified",
        "independent_formal_cell_qualified",
    }
    assert report["behaviorally_screened_development_cells"] == 28
    assert report["shared_contract_verified_present_cells"] == (
        report["required_execution_cells"] - len(report["unexecuted_cells"])
    )
    assert all(check["physics_calls"] == 0
               for check in report["shared_contract_checks"])
    assert all(row["performance"] is None for row in report["rows"])
    screened = [row for row in report["rows"]
                if row.get("behavioral_development_evidence") is not None]
    assert len(screened) == 28
    assert all(row["behavioral_development_evidence"][cell]["requests"] == 4
               for row in screened for cell in ("source", "nominal"))
    assert all(row["method"] not in {"acp", "human2sim2robot"}
               for row in report["rows"])
    dial_rows = [row for row in report["rows"] if row["method"] == "dial_transfer"]
    assert len(dial_rows) == 4
    assert all(
        row["independent_source_nominal_qualification"] is None or
        row["independent_source_nominal_qualification"]["path"].endswith(
            "/independent_source_nominal_cohort_v2/independent_audit.json"
        ) for row in dial_rows
    )
    assert all(
        row["stale_source_nominal_qualification"] is None or
        row["stale_source_nominal_qualification"]["path"].endswith(
            "/independent_source_nominal_cohort_v1/independent_audit.json"
        ) for row in dial_rows
    )
    assert all(
        next(row for row in report["rows"] if row["task"] == task and
             row["method"] == method)["shared_training_lineage_qualification"] is not None
        for task in ("T1", "T2")
        for method in ("cross_ebdm", "task_only", "dial_transfer", "response_mpc")
    )
    for task, methods in {
        "T1": ("cross_ebdm", "task_only", "fada", "dial_transfer", "response_mpc"),
        "T2": ("cross_ebdm", "task_only", "fada"),
    }.items():
        for method in methods:
            row = next(row for row in report["rows"]
                       if row["task"] == task and row["method"] == method)
            assert row["admission_status"] == "independent_formal_cell_qualified"
            assert row["independent_table_eligible"] is True
            assert len(row["formal_configs"]) == 8
            assert row["performance"] is None
    for task in ("T1", "T3", "T4"):
        row = next(row for row in report["rows"]
                   if row["task"] == task and row["method"] == "asid")
        assert row["admission_status"] == "independent_formal_cell_qualified"
        assert row["independent_table_eligible"] is True
        assert row["formal_cohort_config"]["path"].endswith("/asid_cohort.json")
        assert row["performance"] is None
    assert all(
        next(row for row in report["rows"] if row["task"] == task and
             row["method"] == method)["current_actor_lineage_qualification"] is not None
        for task in ("T3", "T4")
        for method in ("cross_ebdm", "task_only", "dial_transfer", "response_mpc")
    )
    fixed = [row for row in report["rows"]
             if row["method"] in {"public_geometry", "public_source_prior"}]
    assert len(fixed) == 8
    assert all(row["independent_table_eligible"] is True
               and row["admission_status"] == "independent_formal_cell_qualified"
               and row["formal_config"] is not None for row in fixed)
    assert set(report["main_rows"]) == {
        "cross_ebdm", "task_only", "fada", "dial_transfer", "response_mpc",
        "public_geometry", "public_source_prior", "adaptsim", "asid",
    }
    excluded = report["excluded_methods"]
    assert len(excluded) == 8
    assert {row["method"] for row in excluded} == {"acp", "human2sim2robot"}
    assert all(row["table_status"] == "excluded_predeployment_rejection"
               and row["performance"] is None for row in excluded)
