#!/usr/bin/env python3
"""Freeze, collect, or audit T3 first-intervention physical force-risk data."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
from genedynamics.tasks.transfer.episode import audit_episode_evidence, load_episode_evidence
from genedynamics.tasks.transfer.method_evaluation import (
    _checked, _gpu1, _task, evaluation_plan, manifest_roster_config,
)
from genedynamics.tasks.transfer.paired_force_risk import collect_first_intervention


VERSION = "t3-first-intervention-force-risk-data-v1"
EXPECTED_SPLITS = {"fit": 24, "calibration": 12, "validation": 12,
                   "mechanism_holdout": 16}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked_config(path: Path) -> tuple[dict, Path, Path]:
    config = json.loads(path.read_text())
    requests = config.get("requests", [])
    if (config.get("version") != VERSION or config.get("task") != "T3"
            or config.get("methods") != ["cross_ebdm", "task_only"]
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False
            or config.get("split_counts") != EXPECTED_SPLITS
            or config.get("maximum_prefix_controls") != 160
            or config.get("branch_horizon_controls") != 32
            or len(requests) != 64
            or config.get("maximum_physical_control_steps") != 64 * (160 + 2 * 32)):
        raise ValueError("T3 first-intervention risk data contract changed")
    if config["execution"] != {
        "device": "cuda", "cuda_visible_devices": "1",
        "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
        "maximum_gpu_memory_gib": 2, "maximum_wall_seconds": 2400,
    }:
        raise ValueError("T3 risk data GPU1 budget changed")
    roster = _checked(config["actor_roster"], "T3 actor roster")
    native = _checked(config["native_library"], "T3 native ABI")
    if json.loads(roster.read_text())["execution"]["native_library"] != config["native_library"]:
        raise ValueError("T3 risk data and actor roster native ABI differ")
    seeds = [int(request["reset_seed"]) for request in requests]
    if seeds != list(range(3301600000, 3301600064)):
        raise ValueError("T3 risk data seed roster changed")
    splits = Counter(request["split"] for request in requests)
    if dict(splits) != EXPECTED_SPLITS:
        raise ValueError("T3 risk data split counts differ")
    if {r["scene_cell"] for r in requests if r["split"] == "mechanism_holdout"} != {
        "delay_gain_shift", "payload_shift",
    } or {r["scene_cell"] for r in requests if r["split"] != "mechanism_holdout"} != {
        "source", "nominal",
    }:
        raise ValueError("T3 mechanism holdout overlaps in-domain splits")
    if any(r["requested_mode"] != 0 or r["robot"] != "xarm7"
           or r["body_support_parameters"] != {}
           or not r["request_id"].startswith("T3/FIRST-INTERVENTION-RISK/")
           for r in requests):
        raise ValueError("T3 first-intervention request contract changed")
    return config, roster, native


def build_plan(path: Path, config: dict, roster: Path) -> tuple[dict, dict]:
    plans = {name: evaluation_plan(manifest_roster_config(roster, name))
             for name in config["methods"]}
    core = ROOT / "genedynamics/tasks/transfer/paired_force_risk.py"
    result = {
        "version": VERSION, "status": "planned_unexecuted",
        "config": pointer(path), "runner": pointer(Path(__file__).resolve()),
        "collector": pointer(core), "actor_roster": config["actor_roster"],
        "native_library": config["native_library"],
        "methods": config["methods"],
        "method_identity": {name: plans[name]["method_identity"]
                            for name in config["methods"]},
        "requests": config["requests"],
        "split_counts": config["split_counts"],
        "maximum_prefix_controls": config["maximum_prefix_controls"],
        "branch_horizon_controls": config["branch_horizon_controls"],
        "maximum_physical_control_steps": config["maximum_physical_control_steps"],
        "physical_gpu_uuid": config["execution"]["physical_uuid"],
        "formal_performance_evidence": False,
    }
    return result, plans


def run(path: Path, config: dict, roster: Path, native: Path) -> dict:
    plan, plans = build_plan(path, config, roster)
    configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
    if configured is not None and Path(configured).resolve() != native.resolve():
        raise ValueError("configured native ABI differs from frozen T3 pointer")
    library = ctypes.CDLL(str(native))
    library.cross_ebdm_abi_version.restype = ctypes.c_int
    if library.cross_ebdm_abi_version() != 1:
        raise ValueError("T3 native ABI version differs")
    os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native)
    _gpu1(maximum_gpu_memory_gib=2)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    experiments = {name: ExperimentConfig.from_yaml(ROOT / plans[name]["task_config"])
                   for name in config["methods"]}
    rows = []
    for request in config["requests"]:
        task = None
        try:
            task = _task(experiments["cross_ebdm"], {**request, "task": "T3"})
            public = task.reset((request["physical_initial_slot"],),
                                seed=request["reset_seed"], initial="jammed",
                                modes=[request["requested_mode"]])
            actors = {}
            for name in config["methods"]:
                parameters = deepcopy(experiments[name].method_params)
                parameters.update({
                    "mode": request["requested_mode"], "device": "cuda",
                    "actor_binding": deepcopy(plans[name]["actor_binding"]),
                })
                actors[name] = create_transfer_actor(task, name, parameters)
            row = collect_first_intervention(
                task, actors, public, request, output, ROOT,
                maximum_prefix_controls=config["maximum_prefix_controls"],
                branch_horizon_controls=config["branch_horizon_controls"],
            )
            row["lifetime_accounting"] = dict(task.lifetime_accounting)
        except BaseException as error:
            row = {"request": request, "status": "retained_execution_failure",
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(),
                   "lifetime_accounting": {} if task is None else dict(task.lifetime_accounting)}
        rows.append(row)
        atomic_json(output / "progress.json", {
            "status": "running", "completed": len(rows),
            "requested": len(config["requests"]), "latest": row,
        })
    charged = sum(row.get("physical_control_steps_charged", 0) for row in rows)
    complete = all(row["status"] in {"paired_fork_audited", "no_admission_in_prefix"}
                   for row in rows)
    report = {**plan,
              "status": "complete_development_data" if complete else "retained_execution_failures",
              "results": rows, "physical_control_steps_charged": charged,
              "pair_count": sum(row["status"] == "paired_fork_audited" for row in rows),
              "no_admission_count": sum(row["status"] == "no_admission_in_prefix"
                                        for row in rows)}
    if charged > config["maximum_physical_control_steps"]:
        raise ValueError("T3 risk data charged control steps exceed frozen budget")
    atomic_json(output / "report.json", report)
    return report


def _read_evidence(pointer_value: dict) -> tuple[list, dict]:
    path = _checked(pointer_value, "T3 risk evidence")
    audit = audit_episode_evidence(path)
    if not audit["passed"]:
        raise ValueError(f"T3 risk evidence failed independent scoring: {path}")
    records, _, evaluation = load_episode_evidence(path)
    if len(records) != 1 or len(records[0].traces) != len(records[0].control_diagnostics):
        raise ValueError("T3 risk evidence record or diagnostic count differs")
    return records[0], evaluation


def audit(path: Path, config: dict, roster: Path) -> dict:
    report_path = ROOT / config["output"] / "report.json"
    report = json.loads(report_path.read_text())
    plan, _ = build_plan(path, config, roster)
    for name in ("config", "runner", "collector", "actor_roster", "native_library"):
        if report[name] != plan[name]:
            raise ValueError(f"T3 risk data {name} SHA binding differs")
        _checked(report[name], f"T3 risk data {name}")
    if (report["version"] != VERSION or report["status"] != "complete_development_data"
            or report["requests"] != config["requests"]
            or len(report["results"]) != len(config["requests"])):
        raise ValueError("T3 risk data report incomplete or request roster differs")
    audited = []
    charged = 0
    evidence_count = 0
    for request, row in zip(config["requests"], report["results"], strict=True):
        if row["request"] != request:
            raise ValueError("T3 risk data request order differs")
        prefix = int(row["prefix_control_steps"])
        if not 0 <= prefix <= config["maximum_prefix_controls"]:
            raise ValueError("T3 risk data prefix length differs")
        if row["status"] == "no_admission_in_prefix":
            record, evaluation = _read_evidence(row["prefix_evidence"])
            if (len(record.traces) != prefix or row["branches"] != {}
                    or any(d["transport_admission_controls"] for d in record.control_diagnostics)):
                raise ValueError("T3 no-admission physical prefix differs")
            result = {"status": row["status"], "outcome": evaluation["episodes"][-1]["outcome"],
                      "evidence_sha256": row["prefix_evidence"]["sha256"]}
            row_charge = prefix
            evidence_count += 1
        elif row["status"] == "paired_fork_audited":
            if (row["first_admission_control_index"] != prefix
                    or row["same_state_public_restore"] is not True
                    or row["first_action_exact_replay"] is not True
                    or set(row["branches"]) != {"cross_ebdm", "task_only"}):
                raise ValueError("T3 paired fork state receipt differs")
            records = {}
            branch_results = {}
            for name, branch in row["branches"].items():
                record, evaluation = _read_evidence(branch["evidence"])
                tail = record.traces[prefix:]
                if not 0 < len(tail) == branch["control_steps"] <= config["branch_horizon_controls"]:
                    raise ValueError("T3 paired fork branch length differs")
                force = max(float(trace["force_peak"]) for trace in tail)
                lateral = max(float(trace["lateral_force_peak"]) for trace in tail)
                if (force != branch["maximum_substep_force_N"]
                        or lateral != branch["maximum_substep_lateral_force_N"]
                        or evaluation["episodes"][-1]["outcome"] != branch["evidence"]["outcome"]):
                    raise ValueError("T3 paired fork physical risk label differs")
                records[name] = record
                branch_results[name] = {"outcome": branch["evidence"]["outcome"],
                                        "control_steps": len(tail),
                                        "maximum_substep_force_N": force,
                                        "maximum_substep_lateral_force_N": lateral,
                                        "evidence_sha256": branch["evidence"]["sha256"]}
                evidence_count += 1
            ours, task = records["cross_ebdm"], records["task_only"]
            if ours.traces[:prefix] != task.traces[:prefix]:
                raise ValueError("T3 paired fork physical prefix differs")
            requested_delta = float(np.max(np.abs(
                np.asarray(ours.requested_actions[prefix], dtype=float)
                - np.asarray(task.requested_actions[prefix], dtype=float)
            )))
            issued_delta = float(np.max(np.abs(
                np.asarray(ours.traces[prefix]["issued_action"], dtype=float)
                - np.asarray(task.traces[prefix]["issued_action"], dtype=float)
            )))
            if (not np.isclose(requested_delta, row["first_requested_action_max_abs_difference"],
                               rtol=0., atol=1e-7)
                    or not np.isclose(issued_delta, row["first_issued_action_max_abs_difference"],
                                      rtol=0., atol=1e-7)):
                raise ValueError("T3 paired fork requested/issued action delta differs")
            row_charge = prefix + sum(b["control_steps"] for b in branch_results.values())
            result = {"status": row["status"], "branches": branch_results,
                      "first_requested_action_max_abs_difference": requested_delta,
                      "first_issued_action_max_abs_difference": issued_delta}
        else:
            raise ValueError("T3 risk data retained an unqualified execution failure")
        if row_charge != row["physical_control_steps_charged"]:
            raise ValueError("T3 risk data charged control ledger differs")
        charged += row_charge
        audited.append({"request_id": request["request_id"], "split": request["split"],
                        "scene_cell": request["scene_cell"], "prefix_control_steps": prefix,
                        "physical_control_steps_charged": row_charge, **result})
    if (charged != report["physical_control_steps_charged"]
            or charged > config["maximum_physical_control_steps"]):
        raise ValueError("T3 risk data total charged controls differ")
    summary = {}
    for split in EXPECTED_SPLITS:
        rows = [r for r in audited if r["split"] == split]
        paired = [r for r in rows if r["status"] == "paired_fork_audited"]
        summary[split] = {
            "requests": len(rows), "paired_forks": len(paired),
            "no_admission_in_prefix": len(rows) - len(paired),
            "ours_only_overforce": sum(
                r["branches"]["cross_ebdm"]["outcome"] == "overforce"
                and r["branches"]["task_only"]["outcome"] != "overforce" for r in paired
            ),
            "task_only_only_overforce": sum(
                r["branches"]["cross_ebdm"]["outcome"] != "overforce"
                and r["branches"]["task_only"]["outcome"] == "overforce" for r in paired
            ),
            "both_overforce": sum(
                r["branches"]["cross_ebdm"]["outcome"] == "overforce"
                and r["branches"]["task_only"]["outcome"] == "overforce" for r in paired
            ),
        }
    return {
        "version": "t3-first-intervention-force-risk-independent-audit-v1",
        "status": "passed_development_data_only", "report_sha256": sha(report_path),
        "audit_runner": pointer(Path(__file__).resolve()),
        "request_count": len(audited), "physical_evidence_verified": evidence_count,
        "physical_control_steps_reconciled": charged,
        "split_summary": summary, "rows": audited,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config, roster, native = checked_config(path)
    if args.plan_only:
        result, _ = build_plan(path, config, roster)
    elif args.run:
        result = run(path, config, roster, native)
    else:
        result = audit(path, config, roster)
        atomic_json(ROOT / config["output"] / "independent_audit.json", result)
    print(json.dumps({"status": result["status"],
                      "output": config["output"],
                      "requests": len(config["requests"])}, indent=2))
    if result["status"] in {"retained_execution_failures"}:
        raise RuntimeError("T3 risk data contains retained failures; inspect report")


if __name__ == "__main__":
    main()
