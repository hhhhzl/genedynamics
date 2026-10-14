"""Audit P3 evidence without confusing availability, qualification and scores.

Only reads artifacts. Failure attempts remain in the inventory; this command
never reruns a charged target episode or promotes short training to paper results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_NAMES = {"qualification.json", "native_qualification.json", "report.json",
                "checkpoint_network.json", "attempt.json", "workloads.json",
                "accounting_correction.json", "method_fidelity_audit.json",
                "superseded_native_receipts.json", "training.json",
                "physical_qualification.json", "source_precision_closure.json",
                "native_closed_loop.json", "native_partial_closed_loop.json", "native_preparation.json", "native_attempt.json",
                "episode_split_post_execution_audit.json", "training_accounting_audit.json",
                "canonical_source_split_audit.json"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core_gate(path, *, repository=ROOT):
    if not path.is_file():
        return {"qualified": False, "reason": "missing source parity report"}
    report = json.loads(path.read_text())
    changed = [name for name, identity in report.get("local_sha256", {}).items()
               if not (repository / name).is_file() or digest(repository / name) != identity]
    qualified = (report.get("status") == "source_core_parity_passed"
                 and report.get("tests", 0) > 0 and report.get("skipped") == 0
                 and report.get("failures") == 0 and bool(report.get("local_sha256"))
                 and not changed)
    return dict(qualified=qualified, tests=report.get("tests"), skipped=report.get("skipped"),
                failures=report.get("failures"), changed_files=changed, report=str(path))


def physical_gate(report):
    evaluation = report.get("evaluation", report.get("report", {}))
    replay = report.get("replay", {})
    return (evaluation.get("successes", 0) > 0
            and evaluation.get("status_mismatches") == 0
            and replay.get("trace_exact") is True
            and replay.get("matches_status") is True
            and replay.get("max_pose_error") == 0.)


def audit(root):
    root = Path(root).resolve()
    reports, unreadable = [], []
    raw = []
    for path in sorted(root.rglob("*.json")):
        if path.name not in REPORT_NAMES or "source" in path.relative_to(root).parts:
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            unreadable.append(dict(path=str(path.relative_to(root)), error=str(error)))
            continue
        if not isinstance(document, dict):
            continue
        status = document.get("status", "execution_chain_with_open_performance_gates" if "execution_gates" in document else "evidence_document")
        item = dict(path=str(path.relative_to(root)), sha256=digest(path), status=status,
                    scope=document.get("scope", document.get("role")))
        evaluation = document.get("target_first_episode_evaluation", document.get("target_post_adaptation_evaluation",
                     document.get("target_evaluation", document.get("evaluation", document.get("report", {})))))
        if isinstance(evaluation, dict) and ("requested_episodes" in evaluation or "requested_trials" in evaluation):
            item["evaluation"] = {key: evaluation.get(key) for key in
                                  ("requested_episodes", "requested_trials", "completed_trials", "unfinished_trials",
                                   "successes", "outcomes", "status_mismatches", "full_denominator_success_rate") if key in evaluation}
        for key in ("lifetime_accounting", "source_interaction_accounting", "source_lifetime_calls",
                    "training_model_calls", "target_B_log", "native_record_duration_qualified",
                    "source_only", "target_calls", "target_interactions", "new_target_accounting",
                    "sampling_gate_passed", "physical_gate_passed", "complete_resume_exact",
                    "final_continuous_gate_passed", "final_decoder_variant_gate_passed",
                    "source_accounting", "source_accounting_this_attempt", "irreversible_source_accounting",
                    "frozen_physical_source_successes", "frozen_physical_source_requested",
                    "requested_trials", "completed_trials", "unfinished_trials", "successes",
                    "full_denominator_success_rate", "native_closed_loop_qualified", "Kit_started",
                    "attempt_status", "physical_qualification_complete"):
            if key in document:
                # Nested per-episode traces live in the original report, not in
                # an aggregate that could count the same calls twice.
                item[key] = document[key] if key != "target_B_log" else {
                    name: value for name, value in document[key].items()
                    if isinstance(value, (str, int, float, bool)) or name.endswith("accounting")}
        for annotation in ("method_fidelity_audit.json", "target_reference_audit.json", "post_run_correctness_revision.json", "post_run_correctness_revision_v3.json", "runtime_failure_audit.json"):
            sidecar = path.with_name(annotation)
            if path.name in {"qualification.json", "native_attempt.json"} and sidecar.is_file():
                item.setdefault("audit_annotations", {})[annotation] = json.loads(sidecar.read_text())
                if annotation == "runtime_failure_audit.json":
                    item["reported_status"] = item["status"]
                    item["status"] = item["audit_annotations"][annotation]["status"]
        if path.name == "accounting_correction.json":
            item["accounting_correction"] = document
        reports.append(item); raw.append((path, document))
    gates = {f"source_core_{device}": core_gate(root / "_reports/core" / device / "qualification.json")
             for device in ("cpu", "cuda")}
    tasks = {"T1": "obstacle_transfer", "T2": "guided_contact_alignment",
             "T3": "keyed_insertion_recovery", "T4": "mobile_keyed_insertion_recovery"}
    for alias, task in tasks.items():
        evidence = [str(path.relative_to(root)) for path, document in raw
                    if task in path.parts and path.name in {"report.json", "qualification.json"}
                    and physical_gate(document)
                    and (alias != "T4" or document.get("initial") == "jammed")]
        gates[f"physical_{alias}"] = dict(qualified=bool(evidence), evidence=evidence,
                                          scope="development reachability and exact CPU replay; no trained baseline claim")
    for method in ("fada_adapted", "human2sim2robot_adapted", "acp_adapted"):
        evidence = [dict(path=str(path.relative_to(root)), status=document.get("status"))
                    for path, document in raw if method in path.parts and path.name == "qualification.json"]
        gates[method] = dict(evidence=evidence, scope="inspect individual chain/resume, first target denominator and controller diagnostics")
    native = {method: [dict(path=str(path.relative_to(root)), status=document.get("status", "execution_chain_with_open_performance_gates"),
                            student_stability=document.get("student_stability"),
                            native_closed_loop_qualified=document.get("native_closed_loop_qualified", False))
                       for path, document in raw if method in path.parts
                       and path.name in {"native_qualification.json", "checkpoint_network.json", "native_preparation.json", "native_closed_loop.json"}]
              for method in ("fada_native", "acp_native", "play2perfect_native")}
    manifest = root / "_reports/human2sim2robot-native-assets-manifest.json"
    native["human2sim2robot_native"] = json.loads(manifest.read_text()) if manifest.exists() else {"status": "assets_pending"}
    # Full original-system qualification cannot be inferred from an actor
    # forward pass, a real-data loss check, or a project's state-only adapter.
    native_closed_loop = {method: any(item.get("native_closed_loop_qualified") is True for item in evidence)
                          for method, evidence in native.items() if isinstance(evidence, list)}
    native_closed_loop["human2sim2robot_native"] = native["human2sim2robot_native"].get("native_closed_loop_qualified") is True
    pending = [method for method, passed in native_closed_loop.items() if not passed]
    decisions_path = root / "_reports/route-decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.is_file() else {"routes": {}}
    deferred = [method for method, decision in decisions.get("routes", {}).items()
                if decision.get("status") == "deferred"]
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                status="p3_evidence_inventory_with_open_gates", full_p3_qualified=False,
                admission_scope="read-only inventory; route admission also requires review of fidelity and performance diagnostics",
                gates=gates, native=native, native_closed_loop_pending=pending,
                route_decisions=decisions, deferred_routes=deferred,
                active_native_closed_loop_pending=[method for method in pending if method not in deferred],
                reports=reports, unreadable_reports=unreadable,
                accounting_note="reports overlap: do not sum logical, lifetime, evaluator and replay counters; attempts are retained individually",
                performance_note="development qualification is not converged training, an original-paper score or a frozen P6 comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/_development/p3_20261014")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or args.root / "_reports/p3-audit.json").resolve()
    if output == ROOT / "latex" or ROOT / "latex" in output.parents:
        parser.error("development audit output belongs outside latex/")
    report = audit(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "full_p3_qualified": report["full_p3_qualified"],
                      "reports": len(report["reports"]), "active_native_routes": report["active_native_closed_loop_pending"],
                      "deferred_routes": report["deferred_routes"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
