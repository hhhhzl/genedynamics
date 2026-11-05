#!/usr/bin/env python3
"""Offline mechanism analysis for the frozen T2 execution-radius diagnosis."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/arm/guided_contact_alignment/validation/transport_execution_radius"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_evidence(row: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / row["evidence"]["path"]
    if sha256(path) != row["evidence"]["sha256"]:
        raise ValueError(f"evidence changed: {path}")
    return json.load(gzip.open(path, "rt"))


def initial_residual(
    row: dict[str, Any], evidence: dict[str, Any], public: dict[str, Any],
) -> np.ndarray:
    scale = float(row["scale"])
    candidate = np.asarray(evidence["records"][0]["requested_actions"][0], dtype=float)
    reference = np.asarray(public["records"][0]["requested_actions"][0], dtype=float)
    return (candidate - reference) / scale


def terminal_transition(evidence: dict[str, Any]) -> dict[str, Any]:
    record = evidence["records"][0]
    traces = record["traces"]
    peak_index = int(np.argmax([float(row["force_peak"]) for row in traces]))
    peak = traces[peak_index]
    prior = traces[max(0, peak_index - 1)]
    diagnostic = record["control_diagnostics"][peak_index]
    base = diagnostic.get("base", diagnostic)
    return {
        "peak_tick": peak_index,
        "peak_force_N": float(peak["force_peak"]),
        "peak_lateral_force_N": float(peak["lateral_force_peak"]),
        "prior_force_N": float(prior["force_peak"]),
        "force_jump_N": float(peak["force_peak"] - prior["force_peak"]),
        "normal_force_at_peak_N": float(peak["normal_force"]),
        "tcp_position_m": [float(value) for value in peak["tcp_position"]],
        "reference_lead_m": [float(value) for value in peak["public_reference_lead"]],
        "issued_twist": [float(value) for value in peak["issued_action"][:6]],
        "base_stage": base.get("stage"),
        "base_contact_ticks": base.get("contact_ticks"),
    }


def mean_std(values: list[np.ndarray]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.stack(values)
    return {
        "count": len(array),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
    }


def analyze(input_dir: Path) -> dict[str, Any]:
    progress_path = input_dir / "progress.jsonl"
    report_path = input_dir / "report.json"
    audit_path = input_dir / "independent_audit.json"
    report = load_json(report_path)
    audit = load_json(audit_path)
    if not audit.get("passed") or audit.get("audited_result_status") != "failed":
        raise ValueError("radius diagnosis must be independently audited as a faithful failure")
    rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    public_rows = {
        row["request"]["request_id"]: row
        for row in rows if row["method_key"] == "public_geometry"
    }
    public_evidence = {key: load_evidence(row) for key, row in public_rows.items()}

    failures = []
    outcome_counts: dict[str, Counter[str]] = defaultdict(Counter)
    request_failures: Counter[str] = Counter()
    mode_failures: Counter[str] = Counter()
    seed_failures: Counter[str] = Counter()
    initial_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    pair_actions: dict[tuple[float, str, str], dict[str, np.ndarray]] = defaultdict(dict)

    for row in rows:
        method = row["method_key"]
        if method == "public_geometry":
            continue
        evidence = load_evidence(row)
        request_id = row["request"]["request_id"]
        residual = initial_residual(row, evidence, public_evidence[request_id])
        outcome_counts[f"{row['scale']}/{method}"][row["outcome"]] += 1
        initial_groups[f"{row['outcome']}/{method}/{row['scale']}"] .append(residual[:6])
        key = (float(row["scale"]), row["instance"], request_id)
        pair_actions[key][method] = np.asarray(
            evidence["records"][0]["requested_actions"], dtype=float,
        )
        if row["outcome"] != "success":
            request_failures[request_id] += 1
            mode_failures[str(row["request"]["requested_mode"])] += 1
            seed_failures[str(row["request"]["reset_seed"])] += 1
            failures.append({
                "record_id": row["record_id"],
                "method": method,
                "instance": row["instance"],
                "scale": row["scale"],
                "request_id": request_id,
                "reset_seed": row["request"]["reset_seed"],
                "requested_mode": row["request"]["requested_mode"],
                "outcome": row["outcome"],
                "initial_unscaled_twist_residual": residual[:6].tolist(),
                "terminal_transition": terminal_transition(evidence),
            })

    identical_pairs = []
    for key, methods in pair_actions.items():
        if set(methods) != {"cross_ebdm", "task_only"}:
            continue
        left, right = methods["cross_ebdm"], methods["task_only"]
        count = min(len(left), len(right))
        common = 0
        for index in range(count):
            if np.array_equal(left[index], right[index]):
                common += 1
            else:
                break
        if common:
            identical_pairs.append({
                "scale": key[0], "instance": key[1], "request_id": key[2],
                "identical_prefix_control_steps": common,
                "complete_trajectory_identical": bool(
                    len(left) == len(right) and np.array_equal(left, right)
                ),
            })

    smallest = min(
        float(value["scale"]) for value in report["cells"] if not value["passed"]
    )
    smallest_failures = [row for row in failures if float(row["scale"]) == smallest]
    failure_jumps = [row["terminal_transition"]["force_jump_N"] for row in smallest_failures]
    return {
        "schema_version": "t2-execution-radius-mechanism-analysis-v1",
        "status": "complete",
        "scope": "offline statistics only; no physics, GPU, training, or confirmation access",
        "inputs": {
            "progress": pointer(progress_path),
            "report": pointer(report_path),
            "independent_audit": pointer(audit_path),
            "analysis_script": pointer(Path(__file__).resolve()),
        },
        "counts": {
            "records": len(rows),
            "learned_records": sum(row["method_key"] != "public_geometry" for row in rows),
            "failures": len(failures),
            "smallest_radius_failures": len(smallest_failures),
            "physics_calls": 0,
            "gpu_calls": 0,
            "confirmation_states_opened": 0,
        },
        "outcomes": {key: dict(sorted(value.items())) for key, value in sorted(outcome_counts.items())},
        "failure_concentration": {
            "by_request": dict(sorted(request_failures.items())),
            "by_mode": dict(sorted(mode_failures.items())),
            "by_seed": dict(sorted(seed_failures.items())),
        },
        "initial_unscaled_twist_residual_statistics": {
            key: mean_std(value) for key, value in sorted(initial_groups.items())
        },
        "smallest_radius": smallest,
        "smallest_radius_terminal_force_jump_N": {
            "minimum": min(failure_jumps),
            "median": float(np.median(failure_jumps)),
            "maximum": max(failure_jumps),
        },
        "cross_vs_task_only_identical_action_prefixes": identical_pairs,
        "failed_records": failures,
        "findings": [
            "Every smallest-radius failure is mode 0 and terminates by overforce; the matched public route succeeds for every request.",
            "Shrinking the fixed residual radius reduces but does not remove violations, so a smaller constant radius is not a qualified repair.",
            "The failure is a one-control-cell force impulse after a safe preceding sample, so a 50 Hz measured-wrench threshold alone cannot certify the 1 ms substeps.",
            "Transport currently perturbs angular velocity as well as tangent translation even though T2 orientation is part of the contact frame maintained by the task scaffold.",
        ],
        "next_revision_hypothesis": {
            "change": "Make T2 orientation task-owned and transport only the two tangent linear velocity coordinates; keep the common executor and task-owned normal/impedance coordinates unchanged.",
            "falsification": "Run a frozen CPU coordinate-scope matrix with current checkpoints, matched requests and an unchanged strict evaluator. The translation-only scope must eliminate violations for every learned instance before any GPU retraining.",
            "if_falsified": "Instrument response-model candidate/base predictions and test a separately versioned online admission rule; do not weaken the common task contract or evaluator.",
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    arguments = parser.parse_args()
    input_dir = arguments.input.resolve()
    report = analyze(input_dir)
    output = input_dir / "mechanism_analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "status": report["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
