"""Task-owned exact-state T3 force-risk collection from public actors.

This module records physical substep peaks, not a policy or solver risk model.
Each request branches only at its first admitted transport action so the
matched task-only actor has followed the same public and physical prefix.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .episode import audit_episode_evidence, controller_diagnostics, episode_evidence
from .evaluation import evaluate_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_public(first: dict, second: dict) -> bool:
    return first.keys() == second.keys() and all(
        np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
        for key in first
    )


def _save_evidence(task, path: Path, root: Path) -> dict:
    evaluation = evaluate_records(task.records, task.spec)
    payload = episode_evidence(task.records, task.spec, evaluation)
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_bytes(gzip.compress(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    audit = audit_episode_evidence(path)
    if not audit["passed"]:
        raise ValueError(f"physical evidence failed CPU reevaluation: {path}")
    return {
        "path": str(path.relative_to(root)), "sha256": _sha(path),
        "audit_passed": True,
        "reset_physical_hash": audit["reset_physical_hashes"][-1],
        "outcome": evaluation["episodes"][-1]["outcome"],
    }


def collect_first_intervention(
    task, actors: dict, public: dict, request: dict, output: Path, root: Path,
    *, maximum_prefix_controls: int, branch_horizon_controls: int,
) -> dict:
    """Collect a single charged prefix and at most two exact-state branches."""
    if set(actors) != {"cross_ebdm", "task_only"} or request["requested_mode"] != 0:
        raise ValueError("first-intervention risk data requires the matched T3 mode-zero actors")
    if maximum_prefix_controls < 1 or branch_horizon_controls < 1:
        raise ValueError("positive finite fork budget required")
    prefix = 0
    first = None
    for tick in range(maximum_prefix_controls):
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
            raise ValueError(f"pre-admission policy requests differ at control {tick}")
        public = task.step(actions["cross_ebdm"]).observation
        task.record_control_diagnostics(diagnostics["cross_ebdm"])
        prefix += 1
    result = {
        "request": request,
        "prefix_control_steps": prefix,
        "prefix_terminated": bool(np.asarray(task.state.status)[0]),
        "first_admission_control_index": None,
        "first_requested_action_max_abs_difference": None,
        "first_issued_action_max_abs_difference": None,
        "same_state_public_restore": None,
        "first_action_exact_replay": None,
        "branches": {},
        "physical_control_steps_charged": prefix,
    }
    if first is None:
        result["status"] = "no_admission_in_prefix"
        result["prefix_evidence"] = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / "prefix" / "evidence.json.gz", root,
        )
        return result
    tick, checkpoint, states, actions, first_public = first
    if tick != prefix:
        raise ValueError("prefix length differs from first admission")
    result["first_admission_control_index"] = tick
    result["first_requested_action_max_abs_difference"] = float(np.max(np.abs(
        actions["cross_ebdm"] - actions["task_only"]
    )))
    physical_first = {}
    for name in ("cross_ebdm", "task_only"):
        restored = task.restore_checkpoint(deepcopy(checkpoint))
        actors[name].restore(deepcopy(states[name]))
        if not _same_public(restored, first_public):
            raise ValueError(f"{name} public checkpoint restore differs")
        start = len(task.records[-1].traces)
        for step in range(branch_horizon_controls):
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
            raise ValueError(f"{name} branch has no physical transition")
        physical_first[name] = np.asarray(traces[0]["issued_action"], dtype=float)
        evidence = _save_evidence(
            task, output / "episodes" / str(request["reset_seed"])
            / name / "evidence.json.gz", root,
        )
        result["branches"][name] = {
            "control_steps": len(traces),
            "terminal_status": int(np.asarray(task.state.status)[0]),
            "maximum_substep_force_N": max(float(row["force_peak"]) for row in traces),
            "maximum_substep_lateral_force_N": max(
                float(row["lateral_force_peak"]) for row in traces
            ),
            "evidence": evidence,
        }
        result["physical_control_steps_charged"] += len(traces)
    result["first_issued_action_max_abs_difference"] = float(np.max(np.abs(
        physical_first["cross_ebdm"] - physical_first["task_only"]
    )))
    result["same_state_public_restore"] = True
    result["first_action_exact_replay"] = True
    result["status"] = "paired_fork_audited"
    return result
