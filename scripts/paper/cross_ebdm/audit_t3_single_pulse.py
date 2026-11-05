#!/usr/bin/env python3
"""Independently check frozen T3 single-pulse physical evidence on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.episode import audit_episode_evidence, load_episode_evidence


BRANCHES = ("pulse_0", "pulse_0p25", "pulse_0p5", "pulse_1", "full_ours")
FRACTIONS = (0., .25, .5, 1.)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_pointer(value: dict, name: str) -> Path:
    path = ROOT / value["path"]
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{name} source or evidence hash changed")
    return path


def same_array(first, second) -> bool:
    """A recorded single action omits the actor's leading batch dimension."""
    return np.array_equal(
        np.asarray(first, dtype=np.float32).reshape(-1),
        np.asarray(second, dtype=np.float32).reshape(-1),
    )


def read_evidence(value: dict):
    path = checked_pointer(value, "physical evidence")
    audit = audit_episode_evidence(path)
    if not audit["passed"] or audit["reset_physical_hashes"][-1] != value["reset_physical_hash"]:
        raise ValueError("physical evidence failed task-owned CPU replay")
    records, _, evaluation = load_episode_evidence(path)
    if len(records) != 1:
        raise ValueError("expected one physical episode per branch")
    record = records[0]
    if not len(record.traces) == len(record.requested_actions) == len(record.control_diagnostics):
        raise ValueError("physical, requested, and diagnostic trace counts differ")
    if evaluation["episodes"][-1]["outcome"] != value["outcome"]:
        raise ValueError("receipt outcome differs from CPU scoring")
    return record, evaluation


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    report_path = ROOT / config["output"] / "report.json"
    report = json.loads(report_path.read_text())
    if config["version"] != "t3-first-intervention-single-pulse-v1":
        raise ValueError("single-pulse protocol version changed")
    if (report["status"] != "complete_development_diagnostic"
            or report["requests"] != config["requests"]
            or len(report["results"]) != len(config["requests"]) == 16
            or report["branch_names"] != list(BRANCHES)
            or report["pulse_fractions"] != list(FRACTIONS)
            or report["maximum_prefix_controls"] != 160
            or report["branch_horizon_controls"] != 32
            or report["maximum_physical_control_steps"] != 5120
            or report["formal_performance_evidence"] is not False):
        raise ValueError("frozen single-pulse contract or roster changed")
    if report["config"] != {"path": str(config_path.relative_to(ROOT)), "sha256": sha(config_path)}:
        raise ValueError("single-pulse config binding changed")
    for name in ("config", "runner", "evidence_writer", "actor_roster", "native_library"):
        checked_pointer(report[name], name)
    if report["actor_roster"] != config["actor_roster"] or report["native_library"] != config["native_library"]:
        raise ValueError("actor or native ABI binding changed")

    results = []
    evidence_count = charged = admitted = no_admission = 0
    for request, row in zip(config["requests"], report["results"], strict=True):
        if row["request"] != request:
            raise ValueError("request order or physical reset changed")
        prefix = row["prefix_control_steps"]
        if not 0 <= prefix <= 160:
            raise ValueError("prefix control count out of contract")
        if row["status"] == "no_admission_in_prefix":
            record, _ = read_evidence(row["prefix_evidence"])
            if (len(record.traces) != prefix or row["branches"] != {}
                    or any(d.get("transport_admission_controls", 0)
                           for d in record.control_diagnostics)):
                raise ValueError("no-admission evidence contradicts receipt")
            row_charged = prefix
            evidence_count += 1
            no_admission += 1
            results.append({"request_id": request["request_id"],
                            "scene_cell": request["scene_cell"],
                            "status": "no_admission_in_prefix",
                            "prefix_control_steps": prefix})
        elif row["status"] == "five_branch_physical_evidence_audited":
            if row["first_admission_control_index"] != prefix or set(row["branches"]) != set(BRANCHES):
                raise ValueError("first admission or branch roster changed")
            records = {}
            outcomes = {}
            branch_steps = 0
            for branch in BRANCHES:
                receipt = row["branches"][branch]
                record, evaluation = read_evidence(receipt["evidence"])
                tail = record.traces[prefix:]
                if not 0 < len(tail) == receipt["control_steps"] <= 32:
                    raise ValueError("branch physical horizon changed")
                if (not same_array(record.requested_actions[prefix], receipt["first_requested_action"])
                        or not same_array(tail[0]["issued_action"], receipt["first_issued_action"])
                        or max(float(t["force_peak"]) for t in tail) != receipt["maximum_substep_force_N"]
                        or max(float(t["lateral_force_peak"]) for t in tail)
                        != receipt["maximum_substep_lateral_force_N"]
                        or evaluation["episodes"][-1]["outcome"] != receipt["outcome"]):
                    raise ValueError("branch action, force, or outcome receipt differs")
                records[branch] = record
                outcomes[branch] = receipt["outcome"]
                branch_steps += len(tail)
                evidence_count += 1
            common = records[BRANCHES[0]].traces[:prefix]
            if any(records[branch].traces[:prefix] != common for branch in BRANCHES[1:]):
                raise ValueError("five branches do not share identical physical prefix")
            base = np.asarray(row["task_only_requested_action"], dtype=np.float32)
            candidate = np.asarray(row["candidate_requested_action"], dtype=np.float32)
            for branch, alpha in zip(BRANCHES[:4], FRACTIONS, strict=True):
                expected = (base if alpha == 0. else candidate if alpha == 1.
                            else (base + np.float32(alpha) * (candidate - base)).astype(np.float32))
                if not same_array(records[branch].requested_actions[prefix], expected):
                    raise ValueError(f"first requested action differs from pulse blend: {branch}")
            if not same_array(records["full_ours"].requested_actions[prefix], candidate):
                raise ValueError("full method first requested action differs from candidate")
            pulse = records["pulse_1"]
            full = records["full_ours"]
            if pulse.traces[prefix] != full.traces[prefix]:
                raise ValueError("full pulse and continued method first physical transition differ")
            shared_length = min(len(pulse.requested_actions), len(full.requested_actions))
            divergences = [
                index - prefix for index in range(prefix, shared_length)
                if not same_array(pulse.requested_actions[index], full.requested_actions[index])
            ]
            first_divergence = divergences[0] if divergences else None
            if first_divergence is None:
                raise ValueError("continued method never differs from one-pulse continuation")
            admissions_at_divergence = full.control_diagnostics[
                prefix + first_divergence
            ]["transport_admission_controls"]
            if admissions_at_divergence != 2:
                raise ValueError("first policy divergence is not the second transport admission")
            previous_admissions = (full.control_diagnostics[prefix - 1].get(
                "transport_admission_controls", 0
            ) if prefix else 0)
            later_admissions = (
                full.control_diagnostics[-1]["transport_admission_controls"]
                - previous_admissions - 1
            )
            if later_admissions < 1:
                raise ValueError("continued method has no later transport admission")
            row_charged = prefix + branch_steps
            admitted += 1
            results.append({"request_id": request["request_id"],
                            "scene_cell": request["scene_cell"],
                            "status": "five_branch_physical_evidence_audited",
                            "prefix_control_steps": prefix, "outcomes": outcomes,
                            "first_later_policy_divergence_offset": first_divergence,
                            "transport_admissions_after_first": later_admissions,
                            "evidence_sha256": {
                                branch: row["branches"][branch]["evidence"]["sha256"]
                                for branch in BRANCHES
                            }})
        else:
            raise ValueError("physical run retained an execution failure")
        if row_charged != row["physical_control_steps_charged"]:
            raise ValueError("request physical control ledger differs")
        charged += row_charged
    if (charged != report["physical_control_steps_charged"]
            or charged > report["maximum_physical_control_steps"]
            or admitted != report["five_branch_requests"]
            or no_admission != report["no_admission_requests"]):
        raise ValueError("total physical control ledger differs")
    paired = [row for row in results if "outcomes" in row]
    return {
        "version": "t3-first-intervention-single-pulse-independent-audit-v2",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "audit_source_sha256": sha(Path(__file__)),
        "request_count": len(results), "admitted_requests": admitted,
        "no_admission_requests": no_admission,
        "physical_evidence_verified": evidence_count,
        "physical_control_steps_reconciled": charged,
        "outcomes_by_branch": {
            branch: dict(Counter(row["outcomes"][branch] for row in paired))
            for branch in BRANCHES
        },
        "rows": results,
        "physical_calls": 0, "gpu_calls": 0,
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
        "status", "request_count", "admitted_requests", "no_admission_requests",
        "physical_evidence_verified", "physical_control_steps_reconciled", "outcomes_by_branch",
    )}, indent=2))


if __name__ == "__main__":
    main()
