#!/usr/bin/env python3
"""Audit complete-episode T3 admission-budget physical forks on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from scripts.paper.cross_ebdm.audit_t3_single_pulse import (
    checked_pointer, read_evidence, same_array, sha,
)


BRANCHES = ("task_only", "one_admission", "two_admissions", "full_ours")


def _check_budgeted_actions(record, branch: str, prefix: int) -> int:
    cap = {"task_only": 0, "one_admission": 1, "two_admissions": 2}[branch]
    selected = 0
    for action, diagnostic in zip(
        record.requested_actions[prefix:],
        record.control_diagnostics[prefix:], strict=True,
    ):
        if diagnostic["admission_budget_branch"] != branch:
            raise ValueError("branch diagnostic identity differs")
        admitted = bool(diagnostic["candidate_admitted"])
        chosen = admitted and selected < cap
        if diagnostic["selected_candidate"] is not chosen:
            raise ValueError("admission-budget selection differs")
        if chosen:
            selected += 1
        expected = (diagnostic["candidate_requested_action"] if chosen
                    else diagnostic["base_requested_action"])
        if expected is None or not same_array(action, expected):
            raise ValueError("recorded action differs from frozen budget rule")
        if diagnostic["selected_transport_admissions"] != selected:
            raise ValueError("selected transport admission counter differs")
        if chosen and diagnostic["candidate_requested_action"] is None:
            raise ValueError("selected candidate is missing")
    return selected


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    report_path = ROOT / config["output"] / "report.json"
    report = json.loads(report_path.read_text())
    if (config["version"] != "t3-admission-budget-episode-v1"
            or report["version"] != config["version"]
            or report["status"] != "complete_development_diagnostic"
            or report["requests"] != config["requests"]
            or len(report["results"]) != len(config["requests"]) == 16
            or report["branches"] != list(BRANCHES)
            or report["maximum_prefix_controls"] != 750
            or report["branch_horizon_controls"] != 750
            or report["maximum_physical_control_steps"] != 60000
            or report["formal_performance_evidence"] is not False):
        raise ValueError("full-episode report or frozen contract differs")
    if report["config"] != {"path": str(config_path.relative_to(ROOT)), "sha256": sha(config_path)}:
        raise ValueError("full-episode config binding changed")
    for name in ("config", "runner", "evidence_writer", "actor_roster", "native_library"):
        checked_pointer(report[name], name)
    if (report["actor_roster"] != config["actor_roster"]
            or report["native_library"] != config["native_library"]):
        raise ValueError("full-episode actor or native ABI binding changed")

    audited = []
    charged = evidence_count = forked = no_admission = 0
    for request, row in zip(config["requests"], report["results"], strict=True):
        if row["request"] != request:
            raise ValueError("full-episode request order changed")
        prefix = row["prefix_control_steps"]
        if not 0 <= prefix <= 750:
            raise ValueError("full-episode prefix length changed")
        if row["status"] == "no_admission_in_episode":
            record, _ = read_evidence(row["prefix_evidence"])
            if (len(record.traces) != prefix or row["branches"] != {}
                    or any(d.get("transport_admission_controls", 0)
                           for d in record.control_diagnostics)):
                raise ValueError("no-admission episode receipt differs")
            step_charge = prefix
            evidence_count += 1
            no_admission += 1
            audited.append({"request_id": request["request_id"],
                            "scene_cell": request["scene_cell"],
                            "requested_mode": request["requested_mode"],
                            "status": row["status"],
                            "shared_outcome": row["prefix_evidence"]["outcome"]})
        elif row["status"] == "four_branch_episode_evidence_audited":
            if row["first_admission_control_index"] != prefix or set(row["branches"]) != set(BRANCHES):
                raise ValueError("full-episode first admission or branch roster differs")
            records = {}
            outcomes = {}
            selected_counts = {}
            tail_steps = 0
            reset_hashes = set()
            for branch in BRANCHES:
                receipt = row["branches"][branch]
                record, evaluation = read_evidence(receipt["evidence"])
                tail = record.traces[prefix:]
                if not 0 < len(tail) == receipt["control_steps"] <= 750:
                    raise ValueError("full-episode branch length differs")
                if (not same_array(record.requested_actions[prefix], receipt["first_requested_action"])
                        or not same_array(tail[0]["issued_action"], receipt["first_issued_action"])
                        or max(float(t["force_peak"]) for t in tail) != receipt["maximum_substep_force_N"]
                        or max(float(t["lateral_force_peak"]) for t in tail)
                        != receipt["maximum_substep_lateral_force_N"]
                        or evaluation["episodes"][-1]["outcome"] != receipt["outcome"]):
                    raise ValueError("full-episode branch receipt differs from evidence")
                if branch != "full_ours":
                    selected = _check_budgeted_actions(record, branch, prefix)
                else:
                    counts = [int(d["transport_admission_controls"])
                              for d in record.control_diagnostics[prefix:]]
                    if not counts or counts[0] != 1 or any(b < a for a, b in zip(counts, counts[1:])):
                        raise ValueError("full method transport admission history differs")
                    selected = counts[-1]
                if selected != receipt["selected_transport_admissions"]:
                    raise ValueError("selected admission count differs from branch receipt")
                records[branch] = record
                outcomes[branch] = receipt["outcome"]
                selected_counts[branch] = selected
                tail_steps += len(tail)
                reset_hashes.add(receipt["evidence"]["reset_physical_hash"])
                evidence_count += 1
            if len(reset_hashes) != 1:
                raise ValueError("forked branches have different physical resets")
            common = records[BRANCHES[0]].traces[:prefix]
            if any(records[branch].traces[:prefix] != common for branch in BRANCHES[1:]):
                raise ValueError("forked branches have different physical prefixes")
            base = row["task_only_requested_action"]
            candidate = row["candidate_requested_action"]
            for branch in BRANCHES:
                expected = base if branch == "task_only" else candidate
                if not same_array(records[branch].requested_actions[prefix], expected):
                    raise ValueError("first branch request differs from frozen candidate or base")
            first = records["one_admission"].traces[prefix]
            if (records["two_admissions"].traces[prefix] != first
                    or records["full_ours"].traces[prefix] != first):
                raise ValueError("first candidate physical transition differs across branches")
            one = records["one_admission"]
            two = records["two_admissions"]
            shared_length = min(len(one.requested_actions), len(two.requested_actions))
            divergences = [index for index in range(prefix, shared_length)
                           if not same_array(one.requested_actions[index], two.requested_actions[index])]
            if divergences and two.control_diagnostics[divergences[0]][
                    "selected_transport_admissions"] != 2:
                raise ValueError("one/two branches first differ outside second admission")
            if selected_counts["two_admissions"] < 2 and divergences:
                raise ValueError("one/two branches differ without second admission")
            step_charge = prefix + tail_steps
            forked += 1
            audited.append({"request_id": request["request_id"],
                            "scene_cell": request["scene_cell"],
                            "requested_mode": request["requested_mode"],
                            "status": row["status"],
                            "prefix_control_steps": prefix,
                            "outcomes": outcomes,
                            "control_steps": {branch: row["branches"][branch]["control_steps"]
                                              for branch in BRANCHES},
                            "selected_transport_admissions": selected_counts,
                            "first_one_two_divergence_offset": (
                                None if not divergences else divergences[0] - prefix
                            ),
                            "evidence_sha256": {branch: row["branches"][branch]["evidence"]["sha256"]
                                                for branch in BRANCHES}})
        else:
            raise ValueError("full-episode run retained an execution failure")
        if step_charge != row["physical_control_steps_charged"]:
            raise ValueError("full-episode request control ledger differs")
        charged += step_charge
    if (charged != report["physical_control_steps_charged"]
            or charged > report["maximum_physical_control_steps"]
            or forked != report["four_branch_requests"]
            or no_admission != report["no_admission_requests"]):
        raise ValueError("full-episode total control ledger differs")
    return {
        "version": "t3-admission-budget-episode-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "audit_source_sha256": sha(Path(__file__)),
        "shared_evidence_reader_sha256": sha(
            ROOT / "scripts/paper/cross_ebdm/audit_t3_single_pulse.py"
        ),
        "request_count": len(audited), "forked_requests": forked,
        "no_admission_requests": no_admission,
        "physical_evidence_verified": evidence_count,
        "physical_control_steps_reconciled": charged,
        "outcomes_by_branch": {
            branch: dict(Counter(row["outcomes"][branch] for row in audited
                                 if "outcomes" in row)) for branch in BRANCHES
        },
        "rows": audited, "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = audit(config_path)
    output = ROOT / json.loads(config_path.read_text())["output"] / "independent_audit.json"
    atomic_json(output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "request_count", "forked_requests", "no_admission_requests",
        "physical_evidence_verified", "physical_control_steps_reconciled",
        "outcomes_by_branch",
    )}, indent=2))


if __name__ == "__main__":
    main()
