#!/usr/bin/env python3
"""Independently audit the complete T2 online substep-risk admission run."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from genedynamics.tasks.transfer.episode import audit_episode_evidence


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"], row["method_key"]].append(row)
    cells = []
    for (variant, method), values in sorted(grouped.items()):
        fractions = np.zeros(4, np.int64); calls = 0
        for row in values:
            if row["risk"] is not None:
                fractions += np.asarray(row["risk"]["fraction_counts"], np.int64)
                calls += row["risk"]["model_calls"]
        cells.append({
            "variant": variant, "method_key": method, "episodes": len(values),
            "successes": sum(row["success"] for row in values),
            "constraint_violations": sum(row["constraint_violation"] for row in values),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in values).items())),
            "risk_model_calls": calls, "risk_fraction_counts": fractions.tolist(),
        })
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
    config = json.loads((ROOT / plan["config"]["path"]).read_text())
    ids = [row["record_id"] for row in rows]
    planned = [row["record_id"] for row in plan["jobs"]]
    evidence = []
    for row in rows:
        path = ROOT / row["evidence"]["path"]
        evidence.append(
            sha(path) == row["evidence"]["sha256"]
            and audit_episode_evidence(path)["passed"]
        )
    cells = summarize(rows)
    lookup = {(row["variant"], row["method_key"]): row for row in cells}
    gates = config["gates"]
    recomputed_checks = {
        "risk_admitted_cross_ebdm_successes": lookup[("risk_admitted", "cross_ebdm")]["successes"] == gates["risk_admitted_cross_ebdm_successes"],
        "risk_admitted_task_only_successes": lookup[("risk_admitted", "task_only")]["successes"] == gates["risk_admitted_task_only_successes"],
        "risk_admitted_constraint_violations": sum(
            row["constraint_violations"] for row in cells if row["variant"] == "risk_admitted"
        ) <= gates["maximum_risk_admitted_constraint_violations"],
        "public_geometry_sentinel_successes": lookup[("sentinel", "public_geometry")]["successes"] == gates["public_geometry_sentinel_successes"],
        "nonzero_risk_model_calls": all(
            row["risk_model_calls"] > 0 for row in cells if row["variant"] == "risk_admitted"
        ),
        "all_evidence_audited": all(evidence),
    }
    expected_pass = all(recomputed_checks.values())
    checks = {
        "unique_record_ids": len(ids) == len(set(ids)),
        "results_exactly_cover_plan": set(ids) == set(planned),
        "all_evidence_independently_reaudited": len(evidence) == len(rows) and all(evidence),
        "cells_exactly_recomputed": cells == report["cells"],
        "checks_exactly_recomputed": recomputed_checks == report["checks"],
        "reported_status_is_faithful": (
            report["passed"] is expected_pass
            and report["status"] == ("passed" if expected_pass else "failed")
        ),
        "counts_exactly_recomputed": report["counts"] == {
            "planned": len(planned), "completed": len(rows),
            "control_steps": sum(row["control_steps"] for row in rows),
            "confirmation_states_opened": 0,
        },
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"]
            for item in (plan["config"], plan["runner"], plan["risk_qualification"])
        ),
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
    }
    audit = {
        "schema_version": "t2-substep-risk-admission-independent-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()), "audited_result_status": report["status"],
        "checks": checks,
        "counts": {"planned": len(planned), "results": len(rows),
                   "episodes_reaudited": sum(evidence)},
        "artifacts": {name: pointer(output / filename) for name, filename in {
            "plan": "plan.json", "progress": "progress.jsonl", "report": "report.json",
        }.items()},
    }
    write_json(output / "independent_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
