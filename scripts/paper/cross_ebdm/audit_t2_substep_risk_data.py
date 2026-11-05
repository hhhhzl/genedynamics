#!/usr/bin/env python3
"""Independently audit a complete T2 substep-risk data collection."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from genedynamics.solvers.single.cross_ebdm.learning.response import PUBLIC_FIELDS
from genedynamics.tasks.transfer.episode import audit_episode_evidence


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_npz(path: Path, horizon: int):
    with np.load(path, allow_pickle=False) as archive:
        public_names = {f"public.{name}" for name in PUBLIC_FIELDS}
        required = public_names | {
            "actions", "trace.force_peak", "trace.lateral_force_peak",
        }
        if set(archive.files) != required:
            raise ValueError(f"unexpected risk-data schema: {path}")
        count = len(archive["actions"])
        if (archive["actions"].shape != (count, 13)
                or any(len(archive[name]) != count + 1 for name in public_names)
                or len(archive["trace.force_peak"]) != count
                or len(archive["trace.lateral_force_peak"]) != count):
            raise ValueError(f"risk-data transition lengths differ: {path}")
        total = archive["trace.force_peak"].astype(float)
        lateral = archive["trace.lateral_force_peak"].astype(float)
        starts = np.arange(max(0, count - horizon + 1))
        indices = starts[:, None] + np.arange(horizon)[None, :]
        return {
            "control_steps": count,
            "windows": len(starts),
            "maximum_total_force_N": float(total.max(initial=0.0)),
            "maximum_lateral_force_N": float(lateral.max(initial=0.0)),
            "high_risk_windows": int(np.sum(
                (total[indices].max(axis=1) >= 20.0)
                | (lateral[indices].max(axis=1) >= 8.0)
            )) if len(starts) else 0,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output if arguments.output.is_absolute() else ROOT / arguments.output
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines() if line]
    valid = []
    summaries = []
    for row in rows:
        data = ROOT / row["data"]["path"]
        evidence = ROOT / row["evidence"]["path"]
        summary = validate_npz(data, 10)
        summaries.append(summary)
        valid.append(
            sha(data) == row["data"]["sha256"]
            and summary == row["data_summary"]
            and sha(evidence) == row["evidence"]["sha256"]
            and audit_episode_evidence(evidence)["passed"]
        )
    ids = [row["record_id"] for row in rows]
    planned = [row["record_id"] for row in plan["jobs"]]
    fit = defaultdict(int)
    for row in rows:
        if row["method_key"] != "public_geometry" and row["request"]["split"] == "fit":
            fit[row["instance"]] += row["data_summary"]["high_risk_windows"]
    sentinel = [row for row in rows if row["method_key"] == "public_geometry"]
    recomputed_counts = {
        "planned": len(plan["jobs"]), "completed": len(rows),
        "control_steps": sum(row["control_steps"] for row in summaries),
        "windows": sum(row["windows"] for row in summaries),
        "high_risk_windows": sum(row["high_risk_windows"] for row in summaries),
        "confirmation_states_opened": 0,
    }
    recomputed_tail_passed = all(
        fit[instance] >= report["required_fit_high_risk_windows_per_instance"]
        for instance in sorted(fit)
    )
    checks = {
        "unique_record_ids": len(ids) == len(set(ids)),
        "results_exactly_cover_plan": set(ids) == set(planned),
        "all_npz_and_evidence_reaudited": len(valid) == len(rows) and all(valid),
        "counts_recomputed_exactly": recomputed_counts == report["counts"],
        "outcomes_recomputed_exactly": dict(sorted(Counter(
            row["outcome"] for row in rows
        ).items())) == report["outcomes"],
        "fit_tail_counts_recomputed_exactly": dict(sorted(fit.items()))
        == report["fit_high_risk_windows_per_instance"],
        "reported_tail_status_is_faithful": (
            report["risk_tail_coverage_passed"] is recomputed_tail_passed
            and report["status"] == ("passed" if recomputed_tail_passed else "failed")
            and report["passed"] is recomputed_tail_passed
        ),
        "public_sentinel_passed": (
            len(sentinel) == len(plan["requests"])
            and all(row["success"] for row in sentinel)
            and report["public_geometry_sentinel"]["passed"] is True
        ),
        "old_invalid_outcomes_unused": report["old_invalid_outcomes_used"] is False,
        "confirmation_states_unopened": report["counts"]["confirmation_states_opened"] == 0,
        "plan_sources_unchanged": all(
            sha(ROOT / item["path"]) == item["sha256"]
            for item in (plan["config"], plan["runner"])
        ),
    }
    payload = {
        "schema_version": "t2-substep-risk-data-independent-audit-v1",
        "passed": all(checks.values()),
        "status": "passed" if all(checks.values()) else "failed",
        "audited_result_status": report["status"],
        "checks": checks,
        "counts": {"planned": len(planned), "results": len(rows), "episodes_reaudited": sum(valid)},
        "artifacts": {
            name: pointer(output / filename)
            for name, filename in {"plan": "plan.json", "progress": "progress.jsonl", "report": "report.json"}.items()
        },
    }
    write_json(output / "independent_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
