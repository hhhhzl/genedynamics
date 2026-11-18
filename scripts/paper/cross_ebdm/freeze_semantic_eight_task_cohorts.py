#!/usr/bin/env python3
"""Audit mechanism gains and seal semantic cohorts for the eight-task table."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper.cross_ebdm.prepare_confirmation import prior_seed_conflicts

VERSION = "semantic-eight-task-formal-cohort-v1"
OUTPUT = ROOT / "results/_analysis/eight_task_transfer/semantic_main_table_freeze_v1"
METHODS = ("icdt", "task_only", "asid", "adaptsim", "response_mpc", "dial_mpc")
METRICS = ("task_success", "key_event_completion", "peak_force_N",
           "failure_penalized_normalized_time")
REPORTS = {
    "aperture_brake": ROOT / "results/arm/obstacle_transfer/training/aperture_brake_semantic_phase/report.json",
    "corner_guided_seating": ROOT / "results/arm/guided_contact_alignment/validation/corner_guided_seating_paired_development/report.json",
    "compliant_fixture_insertion": ROOT / "results/arm/compliant_keyed_insertion_recovery/validation/compliant_fixture_insertion_paired_development/report.json",
    "reposition_then_dock": ROOT / "results/quadruped/mobile_keyed_insertion_recovery/validation/reposition_then_dock_paired_pilot/report.json",
}
OLD_REQUESTS = {
    "obstacle_avoidance": ROOT / "configs/arm/obstacle_transfer/protocol/independent_main_table_requests_v2.json",
    "dual_channel_contact": ROOT / "configs/arm/guided_contact_alignment/protocol/independent_main_table_requests_v2.json",
    "keyed_insertion_recovery": ROOT / "configs/arm/keyed_insertion_recovery/protocol/independent_main_table_requests_v2.json",
    "mobile_keyed_insertion_recovery": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/protocol/independent_main_table_requests_v2.json",
}
COHORT_PATHS = {
    "aperture_brake": ROOT / "configs/arm/obstacle_transfer/protocol/aperture_brake_formal_cohort_v1.json",
    "corner_guided_seating": ROOT / "configs/arm/guided_contact_alignment/protocol/corner_guided_seating_formal_cohort_v1.json",
    "compliant_fixture_insertion": ROOT / "configs/arm/compliant_keyed_insertion_recovery/protocol/compliant_fixture_insertion_formal_cohort_v1.json",
    "reposition_then_dock": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/protocol/reposition_then_dock_formal_cohort_v1.json",
}
CONTRACTS = {
    "aperture_brake": ROOT / "configs/arm/obstacle_transfer/protocol/subtasks/aperture_brake_v1.json",
    "corner_guided_seating": ROOT / "configs/arm/guided_contact_alignment/protocol/subtasks/corner_guided_seating_v1.json",
    "compliant_fixture_insertion": ROOT / "configs/arm/compliant_keyed_insertion_recovery/protocol/subtasks/compliant_fixture_insertion_v1.json",
    "reposition_then_dock": ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/protocol/subtasks/reposition_then_dock_v1.json",
}
TRAIN_ANCHORS = {"aperture_brake": 12_810_000, "corner_guided_seating": 12_820_000,
                 "compliant_fixture_insertion": 12_830_000, "reposition_then_dock": 12_840_000}
EVAL_ANCHORS = {"aperture_brake": 12_811_000, "corner_guided_seating": 12_821_000,
                "compliant_fixture_insertion": 12_831_000, "reposition_then_dock": 12_841_000}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value): return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def pointer(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def mechanism_audit():
    reports = {name: json.loads(path.read_text()) for name, path in REPORTS.items()}
    cells = []
    for task in ("corner_guided_seating", "compliant_fixture_insertion"):
        report = reports[task]
        if (report.get("status") != "complete_development_only"
                or report.get("pairing_passed") is not True
                or report.get("all_evidence_reaudited") is not True):
            raise ValueError(f"{task} paired evidence is not auditable")
        grouped = defaultdict(list)
        for row in report["rows"]: grouped[(row["cell"], row["method"])].append(row)
        for cell in sorted({key[0] for key in grouped}):
            full = {(r["seed"], r["mode"]): r for r in grouped[(cell, "full")]}
            task_only = {(r["seed"], r["mode"]): r for r in grouped[(cell, "task_only")]}
            if set(full) != set(task_only) or len(full) < 2:
                raise ValueError(f"{cell} lacks two paired seeds")
            comparisons = []
            for key in sorted(full):
                ours, base = full[key], task_only[key]
                direction = (
                    int(ours["success"]) > int(base["success"])
                    or (ours["success"] == base["success"]
                        and ours["capped_normalized_completion_time"]
                        < base["capped_normalized_completion_time"])
                )
                comparisons.append({"seed": key[0], "mode": key[1],
                                    "icdt_success": ours["success"],
                                    "task_only_success": base["success"],
                                    "icdt_normalized_time": ours["capped_normalized_completion_time"],
                                    "task_only_normalized_time": base["capped_normalized_completion_time"],
                                    "directional_gain": direction})
            cells.append({"task": task, "cell": cell, "paired_seeds": len(full),
                          "all_seed_directions_positive": all(r["directional_gain"] for r in comparisons),
                          "comparisons": comparisons})
    passing = [row for row in cells if row["all_seed_directions_positive"]]
    eligible = len(passing) >= 2
    return {"criterion": "ICDT beats task-only lexicographically on success then failure-penalized normalized time for every paired seed",
            "required_directionally_consistent_contact_cells": 2,
            "passing_contact_cells": len(passing), "cells": cells,
            "formal_freeze_eligible": eligible,
            "evidence": {name: pointer(path) for name, path in REPORTS.items()}}


def requests(task):
    rows = []
    anchor = EVAL_ANCHORS[task]
    if task == "aperture_brake":
        cells = [("nominal_short_brake", {"servo_gain": 1., "delay_steps": 0, "payload_mass": .5}, (0, 1))]
    elif task == "corner_guided_seating":
        cells = [("sustained_corner_load_high_friction", {"contact_friction": .8, "contact_time_constant": .04}, (0,)),
                 ("sustained_corner_load_slow_response", {"contact_friction": .4, "contact_time_constant": .08}, (0,))]
    elif task == "compliant_fixture_insertion":
        cells = [("moving_fixture_nominal", {"fixture_mass_kg": .5, "fixture_stiffness_Npm": 800., "fixture_damping_Nspm": 12.}, (0, 1)),
                 ("moving_fixture_underdamped", {"fixture_mass_kg": .8, "fixture_stiffness_Npm": 800., "fixture_damping_Nspm": 4.}, (0, 1))]
    else:
        cells = [("nominal_reposition_dock", {"servo_gain": 1., "delay_steps": 0, "payload_mass": .5, "contact_time_constant": .02}, (0, 1))]
    for cell, mechanism, modes in cells:
        for slot in range(5):
            for mode in modes:
                rows.append({"request_id": f"SEMANTIC-EIGHT-TASK/{task}/{cell}/i{slot:02d}/m{mode}",
                             "task": task, "cell": cell, "mechanism_parameters": mechanism,
                             "physical_initial_slot": slot, "requested_mode": mode,
                             "reset_seed": anchor + len(rows), "formal_policy_episodes_accessed": 0})
    return rows


def cohort(task):
    train = [{"instance_id": f"instance_{index+1:02d}",
              "calibration_seed": TRAIN_ANCHORS[task] + index,
              "maximum_target_interaction_seconds": 120,
              "budget_curve_seconds": [0, 30, 60, 120],
              "selection_or_evaluation_seed_reuse": False}
             for index in range(8)]
    return {"version": VERSION, "status": "sealed_unaccessed", "task": task,
            "contract": pointer(CONTRACTS[task]), "readiness_evidence": pointer(REPORTS[task]),
            "methods": list(METHODS), "training_instances": train,
            "evaluation_requests": requests(task), "metrics": list(METRICS),
            "clean_ablation_treatments": ["no_calibration", "no_response_anchoring", "unconditional_response_anchoring"],
            "failure_replacement": False, "formal_policy_episodes_accessed": 0,
            "device": {"cuda_visible_devices": "0", "physical_uuid": "GPU-253f0813-4432-2d51-a149-3e73b38bf77a"}}


def build_receipt(audit, cohorts):
    order = ("obstacle_avoidance", "aperture_brake", "dual_channel_contact",
             "corner_guided_seating", "keyed_insertion_recovery",
             "compliant_fixture_insertion", "mobile_keyed_insertion_recovery",
             "reposition_then_dock")
    task_files = {}
    for task in order:
        task_files[task] = pointer(OLD_REQUESTS[task] if task in OLD_REQUESTS else COHORT_PATHS[task])
    applicability = {task: {method: True for method in METHODS} for task in order}
    applicability["corner_guided_seating"]["dial_mpc"] = False
    return {"version": VERSION, "status": "sealed_unaccessed",
            "freeze_basis": audit, "task_order": list(order), "method_order": list(METHODS),
            "task_files": task_files, "method_applicability": applicability,
            "inapplicable_cells": {"corner_guided_seating/dial_mpc": "legacy T2 DIAL cost requires corridor observer memory absent from the semantic corner planner; report N/A, never a performance failure"},
            "metrics": list(METRICS), "formal_policy_episodes_accessed": 0,
            "failure_replacement": False, "development_results_promoted_to_formal": False}


def freeze():
    audit = mechanism_audit()
    if not audit["formal_freeze_eligible"]: raise ValueError("mechanism gain did not pass the two-cell gate")
    if OUTPUT.exists() or any(path.exists() for path in COHORT_PATHS.values()):
        raise FileExistsError("semantic cohort freeze already exists")
    values = {task: cohort(task) for task in COHORT_PATHS}
    seeds = {row["calibration_seed"] for value in values.values() for row in value["training_instances"]}
    seeds |= {row["reset_seed"] for value in values.values() for row in value["evaluation_requests"]}
    if len(seeds) != 82 or prior_seed_conflicts(seeds):
        raise ValueError("semantic cohort seeds overlap or denominator changed")
    for task, path in COHORT_PATHS.items(): path.write_text(canonical(values[task]))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "decision.json").write_text(canonical(audit))
    receipt = build_receipt(audit, values)
    (OUTPUT / "receipt.json").write_text(canonical(receipt))
    return receipt


def verify():
    receipt = json.loads((OUTPUT / "receipt.json").read_text())
    decision = json.loads((OUTPUT / "decision.json").read_text())
    if (receipt.get("version") != VERSION or receipt.get("status") != "sealed_unaccessed"
            or receipt.get("formal_policy_episodes_accessed") != 0
            or decision != mechanism_audit()):
        raise ValueError("semantic freeze receipt or decision changed")
    for task, path in COHORT_PATHS.items():
        if json.loads(path.read_text()) != cohort(task): raise ValueError(f"{task} cohort changed")
    if receipt != build_receipt(decision, {task: cohort(task) for task in COHORT_PATHS}):
        raise ValueError("eight-task receipt changed")
    return receipt


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true"); group.add_argument("--verify", action="store_true")
    args = parser.parse_args(); print(canonical(freeze() if args.freeze else verify()), end="")


if __name__ == "__main__": main()
