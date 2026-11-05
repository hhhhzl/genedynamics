#!/usr/bin/env python3
"""Collect a bounded, exact-state T3 substep-force fork preflight.

Both actors see the same public prefix. At the first admitted transport action,
the complete task state is restored separately for Cross-EBDM and task-only.
Every physical branch is saved as independently re-scoreable episode evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import gzip
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
    audit_episode_evidence, controller_diagnostics, episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.method_evaluation import (
    _checked, _gpu1, _task, evaluation_plan, manifest_roster_config,
)


VERSION = "t3-paired-force-fork-preflight-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if (config.get("version") != VERSION or config.get("task") != "T3"
            or config.get("methods") != ["cross_ebdm", "task_only"]
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False
            or config.get("maximum_prefix_controls") != 160
            or config.get("branch_horizon_controls") != 32
            or config.get("maximum_physical_control_steps") !=
            len(config.get("requests", ())) * (160 + 2 * 32)):
        raise ValueError("T3 paired force fork preflight contract changed")
    execution = config["execution"]
    if execution != {
        "device": "cuda", "cuda_visible_devices": "1",
        "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
        "maximum_gpu_memory_gib": 2, "maximum_wall_seconds": 300,
    }:
        raise ValueError("physical GPU1 resource contract changed")
    actor_roster = _checked(config["actor_roster"], "T3 actor roster")
    native = _checked(config["native_library"], "T3 native ABI")
    if json.loads(actor_roster.read_text())["execution"]["native_library"] != config["native_library"]:
        raise ValueError("T3 actor roster and fork native ABI differ")
    if len(config["requests"]) != 2 or len({r["reset_seed"] for r in config["requests"]}) != 2:
        raise ValueError("expected two fresh unique T3 requests")
    for request in config["requests"]:
        if (request["robot"] != "xarm7" or request["requested_mode"] != 0
                or request["body_support_parameters"] != {}
                or not request["request_id"].startswith("T3/PAIRED-FORCE-FORK-PREFLIGHT/")):
            raise ValueError("T3 fork request changed")
    config["_actor_roster_path"] = actor_roster
    config["_native_path"] = native
    return config


def same_public(first: dict, second: dict) -> bool:
    return first.keys() == second.keys() and all(
        np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
        for key in first
    )


def save_evidence(task, path: Path) -> dict[str, object]:
    evaluation = evaluate_records(task.records, task.spec)
    payload = episode_evidence(task.records, task.spec, evaluation)
    path.parent.mkdir(parents=True, exist_ok=False)
    compressed = gzip.compress(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0,
    )
    path.write_bytes(compressed)
    audit = audit_episode_evidence(path)
    if not audit["passed"]:
        raise ValueError(f"independent fork evidence audit failed: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "audit_passed": True, "outcome": evaluation["episodes"][-1]["outcome"]}


def collect_one(request: dict, plans: dict, experiments: dict, output: Path, horizon: int) -> dict:
    task = _task(experiments["cross_ebdm"], {**request, "task": "T3"})
    public = task.reset((request["physical_initial_slot"],), seed=request["reset_seed"],
                        initial="jammed", modes=[request["requested_mode"]])
    actors = {}
    for name in ("cross_ebdm", "task_only"):
        parameters = deepcopy(experiments[name].method_params)
        parameters.update({
            "mode": request["requested_mode"], "device": "cuda",
            "actor_binding": deepcopy(plans[name]["actor_binding"]),
        })
        actors[name] = create_transfer_actor(task, name, parameters)
    prefix = 0
    first = None
    for tick in range(160):
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
            raise ValueError(f"pre-admission issued actions differ at control {tick}")
        public = task.step(actions["cross_ebdm"]).observation
        task.record_control_diagnostics(diagnostics["cross_ebdm"])
        prefix += 1
    result = {
        "request": request,
        "prefix_control_steps": prefix,
        "first_admission_control_index": None,
        "first_action_max_abs_difference": None,
        "same_state_public_restore": None,
        "first_action_exact_replay": None,
        "branches": {},
        "physical_control_steps_charged": prefix,
    }
    if first is None:
        result["status"] = "no_admission_observed"
        return result
    tick, checkpoint, states, actions, first_public = first
    if tick != prefix:
        raise ValueError("prefix count and first admission differ")
    result["first_admission_control_index"] = tick
    result["first_action_max_abs_difference"] = float(np.max(np.abs(
        actions["cross_ebdm"] - actions["task_only"]
    )))
    for name in ("cross_ebdm", "task_only"):
        restored = task.restore_checkpoint(deepcopy(checkpoint))
        actors[name].restore(deepcopy(states[name]))
        if not same_public(restored, first_public):
            raise ValueError(f"{name} public checkpoint restore differs")
        start = len(task.records[-1].traces)
        for step in range(horizon):
            if bool(np.asarray(task.state.status)[0]):
                break
            action = np.asarray(actors[name].act(restored), dtype=np.float32)
            if step == 0 and not np.array_equal(action, actions[name]):
                raise ValueError(f"{name} actor first action replay differs")
            diagnostic = controller_diagnostics(actors[name])
            restored = task.step(action).observation
            task.record_control_diagnostics(diagnostic)
        traces = task.records[-1].traces[start:]
        if not traces:
            raise ValueError(f"{name} fork has no physical transition")
        evidence = save_evidence(task, output / "episodes" / str(request["reset_seed"])
                                 / name / "evidence.json.gz")
        result["branches"][name] = {
            "control_steps": len(traces),
            "terminal_status": int(np.asarray(task.state.status)[0]),
            "maximum_substep_force_N": max(float(row["force_peak"]) for row in traces),
            "maximum_substep_lateral_force_N": max(
                float(row["lateral_force_peak"]) for row in traces
            ),
            "transport_admission_controls": int(getattr(
                actors[name], "transport_admission_controls", 0
            )),
            "evidence": evidence,
        }
        result["physical_control_steps_charged"] += len(traces)
    result["same_state_public_restore"] = True
    result["first_action_exact_replay"] = True
    result["status"] = "paired_fork_audited"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = checked_config(path)
    roster = config.pop("_actor_roster_path")
    native = config.pop("_native_path")
    plans = {name: evaluation_plan(manifest_roster_config(roster, name))
             for name in config["methods"]}
    plan = {
        "version": VERSION,
        "status": "planned_unexecuted",
        "config": pointer(path), "runner": pointer(Path(__file__).resolve()),
        "actor_roster": config["actor_roster"],
        "native_library": config["native_library"],
        "methods": config["methods"],
        "method_identity": {name: plans[name]["method_identity"] for name in config["methods"]},
        "requests": config["requests"],
        "maximum_physical_control_steps": config["maximum_physical_control_steps"],
        "physical_gpu_uuid": config["execution"]["physical_uuid"],
        "formal_performance_evidence": False,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if os.environ.get("CROSS_EBDM_NATIVE_LIBRARY") not in (None, str(native)):
        raise ValueError("configured T3 native ABI differs from the frozen pointer")
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
        try:
            row = collect_one(request, plans, experiments, output,
                              config["branch_horizon_controls"])
        except BaseException as error:
            row = {"request": request, "status": "retained_execution_failure",
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        rows.append(row)
        atomic_json(output / "progress.json", {"completed": len(rows),
                                                 "requested": len(config["requests"]),
                                                 "latest": row})
    charged = sum(row.get("physical_control_steps_charged", 0) for row in rows)
    report = {**plan,
              "status": "complete_development_preflight" if all(
                  row["status"] in {"paired_fork_audited", "no_admission_observed"}
                  for row in rows
              ) else "retained_execution_failure",
              "results": rows, "physical_control_steps_charged": charged,
              "gpu_calls": sum(row["status"] == "paired_fork_audited" for row in rows)}
    if charged > config["maximum_physical_control_steps"]:
        raise ValueError("charged physical controls exceed frozen preflight limit")
    atomic_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output / "report.json"),
                      "physical_control_steps_charged": charged}, indent=2))
    if report["status"] != "complete_development_preflight":
        raise RuntimeError("T3 paired fork retained a failure; inspect report")


if __name__ == "__main__":
    main()
