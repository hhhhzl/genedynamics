#!/usr/bin/env python3
"""Independently audit a completed T2 execution-radius diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from genedynamics.tasks.transfer.episode import audit_episode_evidence

from run_t2_execution_radius_diagnosis import load_config, summarize


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    plan = json.loads((run / "plan.json").read_text())
    report = json.loads((run / "report.json").read_text())
    rows = [
        json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()
        if line
    ]
    planned_ids = [row["record_id"] for row in plan["jobs"]]
    result_ids = [row["record_id"] for row in rows]
    evidence_checks = []
    for row in rows:
        path = resolve(row["evidence"]["path"])
        evidence_checks.append(
            path.is_file()
            and sha(path) == row["evidence"]["sha256"]
            and audit_episode_evidence(path)["passed"]
        )
    config_path = resolve(plan["config"]["path"])
    config = load_config(config_path)
    recomputed = summarize(plan, rows, config)
    comparable = {key: report[key] for key in recomputed}
    passing = sorted({
        float(cell["scale"]) for cell in report["cells"]
        if all(
            peer["passed"] for peer in report["cells"]
            if float(peer["scale"]) == float(cell["scale"])
        )
    })
    expected_selection = max(passing) if passing else None
    checks = {
        "unique_record_ids": (
            len(planned_ids) == len(set(planned_ids))
            and len(result_ids) == len(set(result_ids))
        ),
        "results_exactly_cover_plan": set(result_ids) == set(planned_ids),
        "all_episode_evidence_reaudited": (
            len(evidence_checks) == len(rows) and all(evidence_checks)
        ),
        "summary_recomputed_exactly": comparable == recomputed,
        "reported_failure_is_faithful": (
            report["passed"] is False
            and report["status"] == "failed"
            and report["selected_execution_scale"] is None
        ),
        "largest_predeclared_passing_scale_selected": (
            report["selected_execution_scale"] == expected_selection
        ),
        "public_sentinel_passed": report["public_geometry_sentinel"]["passed"] is True,
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
        "plan_inputs_unchanged": all(
            sha(resolve(item["path"])) == item["sha256"]
            for item in (plan["config"], plan["runner"])
        ),
    }
    payload = {
        "schema_version": "t2-transport-execution-radius-independent-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "planned": len(planned_ids),
            "results": len(result_ids),
            "evidence_reaudited": sum(evidence_checks),
        },
        "audited_result_status": report["status"],
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
            }
            for name, path in {
                "plan": run / "plan.json",
                "progress": run / "progress.jsonl",
                "report": run / "report.json",
            }.items()
        },
    }
    output = run / "independent_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
