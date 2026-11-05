#!/usr/bin/env python3
"""Compare one transported T3 control pulse with continued transport."""
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
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence, controller_diagnostics, load_episode_evidence,
)
from genedynamics.tasks.transfer.method_evaluation import (
    _checked, _gpu1, _task, evaluation_plan, manifest_roster_config,
)
from genedynamics.tasks.transfer.paired_force_risk import _same_public, _save_evidence


VERSION = "t3-first-intervention-single-pulse-v1"
BRANCHES = ("pulse_0", "pulse_0p25", "pulse_0p5", "pulse_1", "full_ours")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked_config(path: Path) -> tuple[dict, Path, Path]:
    config = json.loads(path.read_text())
    requests = config.get("requests", [])
    if (config.get("version") != VERSION or config.get("task") != "T3"
            or config.get("methods") != ["cross_ebdm", "task_only"]
            or config.get("pulse_fractions") != [0.0, 0.25, 0.5, 1.0]
            or config.get("full_ours_branch") is not True
            or config.get("maximum_prefix_controls") != 160
            or config.get("branch_horizon_controls") != 32
            or len(requests) != 16
            or config.get("maximum_physical_control_steps") != 16 * (160 + 5 * 32)
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T3 single-pulse diagnostic contract changed")
    if config["execution"] != {
        "device": "cuda", "cuda_visible_devices": "1",
        "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
        "maximum_gpu_memory_gib": 2, "maximum_wall_seconds": 900,
    }:
        raise ValueError("T3 single-pulse GPU1 budget changed")
    seeds = [int(row["reset_seed"]) for row in requests]
    if seeds != list(range(3301800000, 3301800016)):
        raise ValueError("T3 single-pulse seed roster changed")
    if Counter(row["scene_cell"] for row in requests) != {"source": 8, "nominal": 8}:
        raise ValueError("T3 single-pulse mechanism coverage changed")
    if any(row["robot"] != "xarm7" or row["requested_mode"] != 0
           or row["body_support_parameters"] != {}
           or not row["request_id"].startswith("T3/FIRST-INTERVENTION-PULSE/")
           for row in requests):
        raise ValueError("T3 single-pulse request changed")
    roster = _checked(config["actor_roster"], "T3 actor roster")
    native = _checked(config["native_library"], "T3 native ABI")
    if json.loads(roster.read_text())["execution"]["native_library"] != config["native_library"]:
        raise ValueError("T3 single-pulse native ABI differs from actor roster")
    return config, roster, native


def build_plan(path: Path, config: dict, roster: Path) -> tuple[dict, dict]:
    plans = {name: evaluation_plan(manifest_roster_config(roster, name))
             for name in config["methods"]}
    collector = ROOT / "genedynamics/tasks/transfer/paired_force_risk.py"
    return {
        "version": VERSION, "status": "planned_unexecuted",
        "config": pointer(path), "runner": pointer(Path(__file__).resolve()),
        "evidence_writer": pointer(collector),
        "actor_roster": config["actor_roster"],
        "native_library": config["native_library"],
        "method_identity": {name: plans[name]["method_identity"] for name in config["methods"]},
        "requests": config["requests"],
        "pulse_fractions": config["pulse_fractions"],
        "branch_names": list(BRANCHES),
        "maximum_prefix_controls": config["maximum_prefix_controls"],
        "branch_horizon_controls": config["branch_horizon_controls"],
        "maximum_physical_control_steps": config["maximum_physical_control_steps"],
        "physical_gpu_uuid": config["execution"]["physical_uuid"],
        "formal_performance_evidence": False,
    }, plans


def collect_one(request: dict, plans: dict, experiments: dict, output: Path,
                config: dict) -> dict:
    task = _task(experiments["cross_ebdm"], {**request, "task": "T3"})
    public = task.reset((request["physical_initial_slot"],),
                        seed=request["reset_seed"], initial="jammed", modes=[0])
    actors = {}
    for name in config["methods"]:
        parameters = deepcopy(experiments[name].method_params)
        parameters.update({"mode": 0, "device": "cuda",
                           "actor_binding": deepcopy(plans[name]["actor_binding"])})
        actors[name] = create_transfer_actor(task, name, parameters)
    prefix = 0
    first = None
    for tick in range(config["maximum_prefix_controls"]):
        if bool(np.asarray(task.state.status)[0]):
            break
        checkpoint = task.get_checkpoint()
        states = {name: actor.get_state() for name, actor in actors.items()}
        actions = {name: np.asarray(actor.act(public), dtype=np.float32)
                   for name, actor in actors.items()}
        diagnostics = {name: controller_diagnostics(actor)
                       for name, actor in actors.items()}
        if diagnostics["cross_ebdm"]["transport_admission_controls"] > 0:
            first = (tick, checkpoint, states, actions, deepcopy(public))
            break
        if not np.array_equal(actions["cross_ebdm"], actions["task_only"]):
            raise ValueError(f"pre-admission requests differ at {tick}")
        public = task.step(actions["cross_ebdm"]).observation
        task.record_control_diagnostics(diagnostics["cross_ebdm"])
        prefix += 1
    result = {"request": request, "prefix_control_steps": prefix,
              "first_admission_control_index": None,
              "branches": {}, "physical_control_steps_charged": prefix}
    if first is None:
        result["status"] = "no_admission_in_prefix"
        result["prefix_evidence"] = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / "prefix" / "evidence.json.gz", ROOT,
        )
        return result
    tick, checkpoint, states, actions, first_public = first
    if tick != prefix:
        raise ValueError("single-pulse prefix/admission count differs")
    result["first_admission_control_index"] = tick
    result["candidate_requested_action"] = actions["cross_ebdm"].tolist()
    result["task_only_requested_action"] = actions["task_only"].tolist()
    for branch in BRANCHES:
        restored = task.restore_checkpoint(deepcopy(checkpoint))
        if not _same_public(restored, first_public):
            raise ValueError(f"single-pulse {branch} physical public restore differs")
        actor_name = "cross_ebdm" if branch == "full_ours" else "task_only"
        actor = actors[actor_name]
        actor.restore(deepcopy(states[actor_name]))
        start = len(task.records[-1].traces)
        for step in range(config["branch_horizon_controls"]):
            if bool(np.asarray(task.state.status)[0]):
                break
            common_action = np.asarray(actor.act(restored), dtype=np.float32)
            action = common_action
            if step == 0:
                if not np.array_equal(common_action, actions[actor_name]):
                    raise ValueError(f"single-pulse {branch} actor replay differs")
                if branch != "full_ours":
                    fraction = config["pulse_fractions"][BRANCHES.index(branch)]
                    if fraction == 0.:
                        action = actions["task_only"].copy()
                    elif fraction == 1.:
                        action = actions["cross_ebdm"].copy()
                    else:
                        action = (actions["task_only"] + np.float32(fraction) *
                                  (actions["cross_ebdm"] - actions["task_only"])).astype(np.float32)
            diagnostic = controller_diagnostics(actor)
            diagnostic["single_pulse_branch"] = branch
            restored = task.step(action).observation
            task.record_control_diagnostics(diagnostic)
        traces = task.records[-1].traces[start:]
        if not traces:
            raise ValueError(f"single-pulse {branch} has no physical transition")
        evidence = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / branch / "evidence.json.gz", ROOT,
        )
        result["branches"][branch] = {
            "control_steps": len(traces),
            "outcome": evidence["outcome"],
            "terminal_status": int(np.asarray(task.state.status)[0]),
            "first_requested_action": np.asarray(
                task.records[-1].requested_actions[start], dtype=np.float32
            ).tolist(),
            "first_issued_action": np.asarray(
                traces[0]["issued_action"], dtype=np.float32
            ).tolist(),
            "maximum_substep_force_N": max(float(t["force_peak"]) for t in traces),
            "maximum_substep_lateral_force_N": max(
                float(t["lateral_force_peak"]) for t in traces
            ),
            "evidence": evidence,
        }
        result["physical_control_steps_charged"] += len(traces)
    result["status"] = "five_branch_physical_evidence_audited"
    return result


def run(path: Path, config: dict, roster: Path, native: Path) -> dict:
    plan, plans = build_plan(path, config, roster)
    configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
    if configured is not None and Path(configured).resolve() != native.resolve():
        raise ValueError("single-pulse native ABI path differs")
    library = ctypes.CDLL(str(native))
    library.cross_ebdm_abi_version.restype = ctypes.c_int
    if library.cross_ebdm_abi_version() != 1:
        raise ValueError("single-pulse native ABI version differs")
    os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native)
    _gpu1(maximum_gpu_memory_gib=2)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    experiments = {name: ExperimentConfig.from_yaml(ROOT / plans[name]["task_config"])
                   for name in config["methods"]}
    rows = []
    for request in config["requests"]:
        try:
            row = collect_one(request, plans, experiments, output, config)
        except BaseException as error:
            row = {"request": request, "status": "retained_execution_failure",
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        rows.append(row)
        atomic_json(output / "progress.json", {
            "status": "running", "completed": len(rows),
            "requested": len(config["requests"]), "latest": row,
        })
    charged = sum(row.get("physical_control_steps_charged", 0) for row in rows)
    if charged > config["maximum_physical_control_steps"]:
        raise ValueError("single-pulse physical budget exceeded")
    complete = all(row["status"] in {
        "five_branch_physical_evidence_audited", "no_admission_in_prefix",
    } for row in rows)
    report = {**plan,
              "status": "complete_development_diagnostic" if complete
              else "retained_execution_failures",
              "results": rows,
              "physical_control_steps_charged": charged,
              "five_branch_requests": sum(
                  row["status"] == "five_branch_physical_evidence_audited" for row in rows
              ),
              "no_admission_requests": sum(
                  row["status"] == "no_admission_in_prefix" for row in rows
              )}
    atomic_json(output / "report.json", report)
    return report


def _read(pointer_value: dict) -> tuple[object, dict]:
    path = _checked({"path": pointer_value["path"], "sha256": pointer_value["sha256"]},
                    "single-pulse evidence")
    audited = audit_episode_evidence(path)
    if not audited["passed"]:
        raise ValueError("single-pulse physical evidence failed CPU scoring")
    records, _, evaluation = load_episode_evidence(path)
    if len(records) != 1 or len(records[0].traces) != len(records[0].control_diagnostics):
        raise ValueError("single-pulse physical/diagnostic trace count differs")
    return records[0], evaluation


def audit(path: Path, config: dict, roster: Path) -> dict:
    report_path = ROOT / config["output"] / "report.json"
    report = json.loads(report_path.read_text())
    plan, _ = build_plan(path, config, roster)
    for name in ("config", "runner", "evidence_writer", "actor_roster", "native_library"):
        if report[name] != plan[name]:
            raise ValueError(f"single-pulse {name} binding changed")
        _checked(report[name], f"single-pulse {name}")
    if (report["status"] != "complete_development_diagnostic"
            or report["requests"] != config["requests"]
            or len(report["results"]) != len(config["requests"])):
        raise ValueError("single-pulse report incomplete or request roster changed")
    rows = []
    charged = 0
    evidence_count = 0
    for request, row in zip(config["requests"], report["results"], strict=True):
        if row["request"] != request:
            raise ValueError("single-pulse request order differs")
        prefix = int(row["prefix_control_steps"])
        if not 0 <= prefix <= config["maximum_prefix_controls"]:
            raise ValueError("single-pulse prefix length differs")
        if row["status"] == "no_admission_in_prefix":
            record, _ = _read(row["prefix_evidence"])
            if (len(record.traces) != prefix or row["branches"] != {}
                    or any(d["transport_admission_controls"] for d in record.control_diagnostics)):
                raise ValueError("single-pulse no-admission prefix differs")
            result = {"status": row["status"],
                      "prefix_evidence_sha256": row["prefix_evidence"]["sha256"]}
            row_charge = prefix
            evidence_count += 1
        elif row["status"] == "five_branch_physical_evidence_audited":
            if row["first_admission_control_index"] != prefix or set(row["branches"]) != set(BRANCHES):
                raise ValueError("single-pulse five-branch receipt differs")
            records = {}
            outcomes = {}
            branch_steps = 0
            for branch in BRANCHES:
                info = row["branches"][branch]
                record, evaluation = _read(info["evidence"])
                tail = record.traces[prefix:]
                if not 0 < len(tail) == info["control_steps"] <= config["branch_horizon_controls"]:
                    raise ValueError("single-pulse branch physical length differs")
                if (record.requested_actions[prefix] != info["first_requested_action"]
                        or tail[0]["issued_action"] != info["first_issued_action"]
                        or max(float(t["force_peak"]) for t in tail) != info["maximum_substep_force_N"]
                        or max(float(t["lateral_force_peak"]) for t in tail)
                        != info["maximum_substep_lateral_force_N"]
                        or evaluation["episodes"][-1]["outcome"] != info["outcome"]):
                    raise ValueError("single-pulse branch action, force, or outcome differs")
                records[branch] = record
                outcomes[branch] = info["outcome"]
                branch_steps += len(tail)
                evidence_count += 1
            common = records[BRANCHES[0]].traces[:prefix]
            if any(records[branch].traces[:prefix] != common for branch in BRANCHES[1:]):
                raise ValueError("single-pulse branches differ before first intervention")
            base = np.asarray(row["task_only_requested_action"], dtype=np.float32)
            candidate = np.asarray(row["candidate_requested_action"], dtype=np.float32)
            for branch, fraction in zip(BRANCHES[:4], config["pulse_fractions"], strict=True):
                expected = (base if fraction == 0. else candidate if fraction == 1.
                            else (base + np.float32(fraction) * (candidate - base)).astype(np.float32))
                actual = np.asarray(records[branch].requested_actions[prefix], dtype=np.float32)
                if not np.array_equal(expected, actual):
                    raise ValueError(f"single-pulse requested blend differs: {branch}")
            if not np.array_equal(np.asarray(records["full_ours"].requested_actions[prefix], dtype=np.float32),
                                  candidate):
                raise ValueError("single-pulse full method first request differs")
            row_charge = prefix + branch_steps
            result = {"status": row["status"], "outcomes": outcomes,
                      "first_issued_action": {branch: records[branch].traces[prefix]["issued_action"]
                                              for branch in BRANCHES},
                      "evidence_sha256": {branch: row["branches"][branch]["evidence"]["sha256"]
                                          for branch in BRANCHES}}
        else:
            raise ValueError("single-pulse run retained an execution failure")
        if row_charge != row["physical_control_steps_charged"]:
            raise ValueError("single-pulse charged control ledger differs")
        charged += row_charge
        rows.append({"request_id": request["request_id"], "scene_cell": request["scene_cell"],
                     "prefix_control_steps": prefix,
                     "physical_control_steps_charged": row_charge, **result})
    if (charged != report["physical_control_steps_charged"]
            or charged > config["maximum_physical_control_steps"]):
        raise ValueError("single-pulse total charged controls differ")
    paired = [row for row in rows if row["status"] == "five_branch_physical_evidence_audited"]
    outcomes = {branch: Counter(row["outcomes"][branch] for row in paired)
                for branch in BRANCHES}
    return {
        "version": "t3-first-intervention-single-pulse-independent-audit-v1",
        "status": "passed_development_only", "report_sha256": sha(report_path),
        "audit_runner": pointer(Path(__file__).resolve()),
        "request_count": len(rows), "five_branch_requests": len(paired),
        "physical_evidence_verified": evidence_count,
        "physical_control_steps_reconciled": charged,
        "outcomes_by_branch": {branch: dict(counts) for branch, counts in outcomes.items()},
        "rows": rows, "physical_calls": 0, "gpu_calls": 0,
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
    print(json.dumps({"status": result["status"], "output": config["output"],
                      "requests": len(config["requests"])}, indent=2))
    if result["status"] == "retained_execution_failures":
        raise RuntimeError("single-pulse diagnostic retained execution failures")


if __name__ == "__main__":
    main()
