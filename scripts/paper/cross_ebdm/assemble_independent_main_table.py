#!/usr/bin/env python3
"""CPU-rebuild the nine-method table from complete formal physical audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.main_table_evaluation import (
    ROOT, _roster, formal_plan, audit_formal,
)
from genedynamics.tasks.transfer.main_table_configs import (
    config_paths as _config_paths,
    configs as _configs,
)
from genedynamics.tasks.transfer.main_table_statistics import (
    METHODS, TASKS, VERSION,
    aggregate_independent_main_table, bootstrap_independent_main_table,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}


def inventory(revision: str = "v1") -> dict:
    """Report missing task-local configs/audits without opening formal physics."""
    rows = []
    for task in TASKS:
        for method in METHODS:
            if not all(path.is_file() for path in _config_paths(task, method, revision)):
                rows.append({"task": task, "method": method,
                             "status": "formal_cell_config_pending"})
                continue
            try:
                entries = _configs(task, method, revision)
            except ValueError:
                rows.append({"task": task, "method": method,
                             "status": "independent_instance_roster_pending"})
                continue
            missing = []
            for _, config in entries:
                instance_id = config["instance_id"]
                try:
                    plan = formal_plan(config)
                except (ValueError, FileNotFoundError, KeyError) as error:
                    missing.append({"instance_id": instance_id,
                                    "reason": f"admission: {error}"})
                    continue
                output = ROOT / plan["output"]
                if not (output / "independent_audit.json").is_file():
                    missing.append({"instance_id": instance_id,
                                    "reason": "formal execution/audit pending"})
            rows.append({
                "task": task, "method": method,
                "status": "fully_audited" if not missing else "formal_outcomes_pending",
                "missing_instances": missing,
            })
    return {
        "version": f"transfer-independent-main-table-assembly-{revision}",
        "status": "all_cells_audited" if all(
            row["status"] == "fully_audited" for row in rows
        ) else "pending",
        "cell_count": len(rows),
        "fully_audited_cells": sum(row["status"] == "fully_audited" for row in rows),
        "rows": rows,
        "physics_calls": 0, "gpu_calls": 0,
    }


def assemble(output: Path, revision: str = "v1") -> dict:
    state = inventory() if revision == "v1" else inventory(revision)
    if state["status"] != "all_cells_audited":
        raise ValueError("independent main table is incomplete; inspect --inventory")
    if output.exists() or not output.is_relative_to(
        ROOT / "results/_analysis/four_task_transfer"
    ):
        raise ValueError("assembled table output must be fresh and analysis-local")
    requests = {}
    full_requests = {}
    request_pointers = {}
    for task in TASKS:
        pointer, roster = _roster(task, revision)
        request_pointers[task] = pointer
        full_requests[task] = roster["requests"]
        requests[task] = [{
            "request_id": row["request_id"],
            "mechanism_group": row["mechanism_group"],
            "requested_mode": row["requested_mode"],
            "body_support_group": row["body_support_group"],
        } for row in roster["requests"]]
    protocol = {
        "version": VERSION, "task_order": list(TASKS),
        "method_order": list(METHODS),
        "requests": requests, "cells": {},
    }
    if revision == "v2":
        protocol["formal_cohort_revision"] = revision
    outcomes = []
    audit_pointers = []
    config_pointers = []
    seen_config_paths = set()
    for method in METHODS:
        protocol["cells"][method] = {}
        for task in TASKS:
            entries = _configs(task, method, revision)
            ids = [config["instance_id"] for _, config in entries]
            for path, _ in entries:
                if path not in seen_config_paths:
                    config_pointers.append(_pointer(path))
                    seen_config_paths.add(path)
            protocol["cells"][method][task] = {
                "status": "executable", "instance_ids": ids,
                "method_identity": entries[0][1]["method_identity"],
            }
            for _, config in entries:
                instance_id = config["instance_id"]
                plan = formal_plan(config)
                run_output = ROOT / plan["output"]
                checked = audit_formal(config, write=False)
                audit_path = run_output / "independent_audit.json"
                saved = json.loads(audit_path.read_text())
                verified_path = run_output / "verified_outcomes.json"
                if (checked != saved or checked["status"] != "passed"
                        or checked["formal_performance_evidence"] is not True
                        or _sha(verified_path) != checked["verified_outcomes_sha256"]):
                    raise ValueError(f"{task}/{method}/{instance_id} formal evidence differs")
                selected = json.loads(verified_path.read_text())
                expected = 80 if task == "T4" else 40
                if len(selected) != expected:
                    raise ValueError("formal actor has incomplete physical outcomes")
                outcomes.extend(selected)
                audit_pointers.append(_pointer(audit_path))
    table = aggregate_independent_main_table(protocol, outcomes)
    if (table["status"] != "complete_outcome_matrix"
            or table["physical_outcomes"] != 11600):
        raise ValueError("nine-method formal physical matrix is incomplete")
    table["formal_performance_evidence"] = True
    if revision == "v2":
        table["formal_cohort_revision"] = revision
    table["evidence_audit_required_before_publication"] = False
    table["all_cells_cpu_reaudited"] = True
    table["frozen_request_rosters"] = request_pointers
    table["actor_configs"] = config_pointers
    table["formal_audits"] = audit_pointers
    uncertainty = bootstrap_independent_main_table(
        protocol, outcomes, full_requests, resamples=10000, seed=267100,
    )
    uncertainty["formal_performance_evidence"] = True
    if revision == "v2":
        uncertainty["formal_cohort_revision"] = revision
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "verified_outcomes.json", outcomes)
    atomic_json(output / "main_table.json", table)
    atomic_json(output / "uncertainty.json", uncertainty)
    return {
        "version": f"transfer-independent-main-table-assembly-{revision}",
        "status": "complete_formal_physical_evidence",
        "output": str(output.relative_to(ROOT)),
        "protocol": _pointer(output / "protocol.json"),
        "verified_outcomes": _pointer(output / "verified_outcomes.json"),
        "main_table": _pointer(output / "main_table.json"),
        "uncertainty": _pointer(output / "uncertainty.json"),
        "physical_outcomes": table["physical_outcomes"],
        "physics_calls": 0, "gpu_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inventory", action="store_true")
    action.add_argument("--assemble", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    if args.inventory:
        report = inventory(args.revision)
        print(json.dumps({"status": report["status"],
                          "fully_audited_cells": report["fully_audited_cells"],
                          "cell_count": report["cell_count"]}, indent=2))
    else:
        if args.output is None:
            parser.error("--assemble requires --output")
        output = args.output if args.output.is_absolute() else ROOT / args.output
        print(json.dumps(assemble(output, args.revision), indent=2))


if __name__ == "__main__":
    main()
