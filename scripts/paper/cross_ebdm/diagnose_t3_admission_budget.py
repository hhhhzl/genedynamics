#!/usr/bin/env python3
"""Fork complete T3 episodes at first transport admission with fixed budgets."""
from __future__ import annotations

import argparse
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
from genedynamics.tasks.transfer.episode import controller_diagnostics
from genedynamics.tasks.transfer.method_evaluation import (
    _checked, _gpu1, _task, evaluation_plan, manifest_roster_config,
)
from genedynamics.tasks.transfer.paired_force_risk import _same_public, _save_evidence


VERSION = "t3-admission-budget-episode-v1"
BRANCHES = ("task_only", "one_admission", "two_admissions", "full_ours")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def checked_config(path: Path) -> tuple[dict, Path, Path]:
    config = json.loads(path.read_text())
    requests = config.get("requests", [])
    if (config.get("version") != VERSION or config.get("task") != "T3"
            or config.get("methods") != ["cross_ebdm", "task_only"]
            or config.get("branches") != list(BRANCHES)
            or config.get("maximum_prefix_controls") != 750
            or config.get("branch_horizon_controls") != 750
            or config.get("maximum_physical_control_steps") != 16 * 5 * 750
            or len(requests) != 16
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T3 full-episode admission-budget contract changed")
    if config["execution"] != {
        "device": "cuda", "cuda_visible_devices": "1",
        "physical_uuid": "GPU-58b54684-3577-11d3-f039-0e8469190739",
        "maximum_gpu_memory_gib": 2, "maximum_wall_seconds": 1800,
    }:
        raise ValueError("T3 full-episode GPU1 budget changed")
    if [int(row["reset_seed"]) for row in requests] != list(range(3301900000, 3301900016)):
        raise ValueError("T3 full-episode seed roster changed")
    for scene in ("source", "nominal"):
        for mode in (0, 1):
            subset = [row for row in requests if row["scene_cell"] == scene
                      and row["requested_mode"] == mode]
            if len(subset) != 4 or sorted(row["physical_initial_slot"] for row in subset) != [0, 0, 1, 1]:
                raise ValueError("T3 full-episode scene, mode, or physical-slot coverage changed")
    if any(row["robot"] != "xarm7" or row["body_support_parameters"] != {}
           or not row["request_id"].startswith("T3/ADMISSION-BUDGET/")
           or row["mechanism_parameters"] != (
               {"servo_gain": .85, "payload_mass": .65, "delay_steps": 0}
               if row["scene_cell"] == "source" else
               {"servo_gain": 1., "payload_mass": .5, "delay_steps": 0}
           ) for row in requests):
        raise ValueError("T3 full-episode request mechanism changed")
    roster = _checked(config["actor_roster"], "T3 actor roster")
    native = _checked(config["native_library"], "T3 native ABI")
    if json.loads(roster.read_text())["execution"]["native_library"] != config["native_library"]:
        raise ValueError("T3 full-episode native ABI differs from actor roster")
    return config, roster, native


def build_plan(path: Path, config: dict, roster: Path) -> tuple[dict, dict]:
    plans = {name: evaluation_plan(manifest_roster_config(roster, name))
             for name in config["methods"]}
    return {
        "version": VERSION, "status": "planned_unexecuted",
        "config": pointer(path), "runner": pointer(Path(__file__).resolve()),
        "evidence_writer": pointer(ROOT / "genedynamics/tasks/transfer/paired_force_risk.py"),
        "actor_roster": config["actor_roster"],
        "native_library": config["native_library"],
        "method_identity": {name: plans[name]["method_identity"] for name in config["methods"]},
        "requests": config["requests"], "branches": list(BRANCHES),
        "maximum_prefix_controls": config["maximum_prefix_controls"],
        "branch_horizon_controls": config["branch_horizon_controls"],
        "maximum_physical_control_steps": config["maximum_physical_control_steps"],
        "physical_gpu_uuid": config["execution"]["physical_uuid"],
        "formal_performance_evidence": False,
    }, plans


def _branch_step(branch: str, actors: dict, public: dict, selected: int) -> tuple[np.ndarray, dict, int]:
    base = np.asarray(actors["task_only"].act(public), dtype=np.float32)
    base_diag = controller_diagnostics(actors["task_only"])
    candidate = None
    candidate_diag = None
    if branch == "full_ours" or (branch == "two_admissions" and selected < 2):
        candidate = np.asarray(actors["cross_ebdm"].act(public), dtype=np.float32)
        candidate_diag = controller_diagnostics(actors["cross_ebdm"])
    elif branch == "one_admission" and selected == 0:
        candidate = np.asarray(actors["cross_ebdm"].act(public), dtype=np.float32)
        candidate_diag = controller_diagnostics(actors["cross_ebdm"])
    admitted = (candidate_diag is not None and
                int(candidate_diag["transport_admission_controls"]) > selected)
    choose_candidate = branch == "full_ours" or (
        branch != "task_only" and admitted and selected <
        (1 if branch == "one_admission" else 2)
    )
    action = candidate if choose_candidate else base
    if action is None:
        raise ValueError("budgeted branch has no selected action")
    if choose_candidate and admitted:
        selected += 1
    diagnostic = deepcopy(candidate_diag if choose_candidate else base_diag)
    diagnostic["admission_budget_branch"] = branch
    diagnostic["selected_transport_admissions"] = selected
    diagnostic["selected_candidate"] = bool(choose_candidate)
    diagnostic["base_requested_action"] = base.tolist()
    diagnostic["candidate_requested_action"] = (
        None if candidate is None else candidate.tolist()
    )
    diagnostic["candidate_admitted"] = bool(admitted)
    return action, diagnostic, selected


def collect_one(request: dict, plans: dict, experiments: dict, output: Path,
                config: dict) -> dict:
    task = _task(experiments["cross_ebdm"], {**request, "task": "T3"})
    public = task.reset((request["physical_initial_slot"],),
                        seed=request["reset_seed"], initial="jammed",
                        modes=[request["requested_mode"]])
    actors = {}
    for name in config["methods"]:
        parameters = deepcopy(experiments[name].method_params)
        parameters.update({"mode": request["requested_mode"], "device": "cuda",
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
        if int(diagnostics["cross_ebdm"]["transport_admission_controls"]) > 0:
            first = (tick, checkpoint, states, actions, deepcopy(public))
            break
        if not np.array_equal(actions["cross_ebdm"], actions["task_only"]):
            raise ValueError(f"pre-admission requested actions differ at {tick}")
        public = task.step(actions["cross_ebdm"]).observation
        task.record_control_diagnostics(diagnostics["cross_ebdm"])
        prefix += 1
    row = {"request": request, "prefix_control_steps": prefix,
           "first_admission_control_index": None, "branches": {},
           "physical_control_steps_charged": prefix}
    if first is None:
        row["status"] = "no_admission_in_episode"
        row["prefix_evidence"] = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / "prefix" / "evidence.json.gz", ROOT,
        )
        return row
    tick, checkpoint, states, actions, first_public = first
    if tick != prefix:
        raise ValueError("prefix/admission control index differs")
    row["first_admission_control_index"] = tick
    row["candidate_requested_action"] = actions["cross_ebdm"].tolist()
    row["task_only_requested_action"] = actions["task_only"].tolist()
    for branch in BRANCHES:
        restored = task.restore_checkpoint(deepcopy(checkpoint))
        if not _same_public(restored, first_public):
            raise ValueError(f"{branch} physical/public checkpoint restore differs")
        for name, actor in actors.items():
            actor.restore(deepcopy(states[name]))
        start = len(task.records[-1].traces)
        selected = 0
        for step in range(config["branch_horizon_controls"]):
            if bool(np.asarray(task.state.status)[0]):
                break
            if branch == "full_ours":
                action = np.asarray(actors["cross_ebdm"].act(restored), dtype=np.float32)
                diagnostic = controller_diagnostics(actors["cross_ebdm"])
                diagnostic["admission_budget_branch"] = branch
                diagnostic["selected_candidate"] = bool(
                    diagnostic["transport_admission_controls"] > selected
                )
                selected = int(diagnostic["transport_admission_controls"])
                diagnostic["selected_transport_admissions"] = selected
            else:
                action, diagnostic, selected = _branch_step(
                    branch, actors, restored, selected
                )
            if step == 0:
                expected = (actions["task_only"] if branch == "task_only"
                            else actions["cross_ebdm"])
                if not np.array_equal(action, expected):
                    raise ValueError(f"{branch} first actor replay differs")
            restored = task.step(action).observation
            task.record_control_diagnostics(diagnostic)
        record = task.records[-1]
        traces = record.traces[start:]
        if not traces:
            raise ValueError(f"{branch} has no physical transition")
        evidence = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / branch / "evidence.json.gz", ROOT,
        )
        row["branches"][branch] = {
            "control_steps": len(traces), "outcome": evidence["outcome"],
            "selected_transport_admissions": selected,
            "terminal_status": int(np.asarray(task.state.status)[0]),
            "first_requested_action": np.asarray(
                record.requested_actions[start], dtype=np.float32
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
        row["physical_control_steps_charged"] += len(traces)
    row["status"] = "four_branch_episode_evidence_audited"
    return row


def run(path: Path, config: dict, roster: Path, native: Path) -> dict:
    plan, plans = build_plan(path, config, roster)
    configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
    if configured is not None and Path(configured).resolve() != native.resolve():
        raise ValueError("T3 full-episode native ABI path differs")
    library = ctypes.CDLL(str(native))
    library.cross_ebdm_abi_version.restype = ctypes.c_int
    if library.cross_ebdm_abi_version() != 1:
        raise ValueError("T3 full-episode native ABI version differs")
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
        raise ValueError("T3 full-episode physical budget exceeded")
    complete = all(row["status"] in {
        "four_branch_episode_evidence_audited", "no_admission_in_episode",
    } for row in rows)
    report = {**plan,
              "status": "complete_development_diagnostic" if complete
              else "retained_execution_failures",
              "results": rows, "physical_control_steps_charged": charged,
              "four_branch_requests": sum(row["status"] ==
                  "four_branch_episode_evidence_audited" for row in rows),
              "no_admission_requests": sum(row["status"] ==
                  "no_admission_in_episode" for row in rows)}
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config, roster, native = checked_config(path)
    result = build_plan(path, config, roster)[0] if args.plan_only else run(
        path, config, roster, native
    )
    print(json.dumps({"status": result["status"], "output": config["output"],
                      "requests": len(config["requests"])}, indent=2))
    if result["status"] == "retained_execution_failures":
        raise RuntimeError("T3 full-episode diagnostic retained execution failures")


if __name__ == "__main__":
    main()
