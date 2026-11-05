#!/usr/bin/env python3
"""Independently audit T3 ASID pre-overforce physical evidence and information."""
from __future__ import annotations

import argparse
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_episode(receipt: dict, output: Path):
    path = ROOT / receipt["evidence"]["path"]
    if not path.is_relative_to(output) or sha(path) != receipt["evidence"]["sha256"]:
        raise ValueError("T3 ASID evidence path or SHA differs")
    audit = audit_episode_evidence(path)
    if (not audit["passed"]
            or audit["reset_physical_hashes"][-1] != receipt["evidence"]["reset_physical_hash"]):
        raise ValueError("T3 ASID physical evidence failed CPU rescoring")
    records, _, evaluation = load_episode_evidence(path)
    if len(records) != 1:
        raise ValueError("T3 ASID evidence must contain one episode")
    record = records[0]
    first_force = next((index for index, trace in enumerate(record.traces)
                        if float(trace["force_peak"]) > 1.), None)
    if (len(record.traces) != receipt["physical_control_steps"]
            or evaluation["episodes"][-1]["outcome"] != receipt["outcome"]
            or bool(evaluation["episodes"][-1]["constraint_violation"])
            != receipt["constraint_violation"]
            or first_force != receipt["first_force_over_1_N_control"]):
        raise ValueError("T3 ASID physical evidence outcome or contact onset differs")
    positions = np.asarray([trace["tcp_position"] for trace in record.traces],
                           dtype=np.float64) * 1000.
    if len(positions) < 32:
        positions = np.concatenate((positions, np.repeat(
            positions[-1:], 32 - len(positions), axis=0,
        )))
    return record, positions


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    recipe_path = output / "recipe.json"
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if (json.loads(recipe_path.read_text()) != config
            or report.get("version") != "asid-t3-preoverforce-information-v1"
            or report.get("status") != "complete_development_probe"
            or report.get("recipe_sha256") != sha(recipe_path)
            or report.get("formal_performance_evidence") is not False
            or report.get("failure_replacement") is not False
            or len(report.get("rows", [])) != 6
            or report.get("wall_seconds", 1e9) > 900):
        raise ValueError("T3 ASID recipe, scope, or completed report differs")
    if report["source_bindings"] != config["source_bindings"]:
        raise ValueError("T3 ASID source roster differs")
    for binding in config["source_bindings"]:
        if sha(output / "source_snapshot" / binding["path"]) != binding["sha256"]:
            raise ValueError("T3 ASID source snapshot differs")
    episodes = controls = 0
    cells = []
    for row, (policy, mode, seed) in zip(report["rows"], (
            (policy, mode, seed) for policy in config["policies"]
            for mode, seed in zip(config["modes"], config["seeds"], strict=True)
            ), strict=True):
        if row["policy"] != policy or row["mode"] != mode or row["reset_seed"] != seed:
            raise ValueError("T3 ASID policy/mode/seed roster differs")
        source, _ = read_episode(row["source"], output)
        episodes += 1
        controls += len(source.traces)
        actions = np.asarray(source.requested_actions, dtype=np.float32)
        if (hashlib.sha256(actions.tobytes()).hexdigest() != row["source_action_sha256"]
                or row["source"]["mechanism"] != config["source_mechanism"]
                or row["source"]["mode"] != mode
                or row["source"]["reset_seed"] != seed):
            raise ValueError("T3 ASID source mechanism or action stream differs")
        perturbations = row["perturbations"]
        if len(perturbations) not in (0, 4):
            raise ValueError("T3 ASID central difference needs four replays")
        positions = []
        expected_theta = ((.95, .65), (.75, .65), (.85, .75), (.85, .55))
        for receipt, (gain, mass) in zip(perturbations, expected_theta):
            record, xyz = read_episode(receipt, output)
            if (not np.isclose(receipt["mechanism"]["servo_gain"], gain)
                    or not np.isclose(receipt["mechanism"]["payload_mass"], mass)
                    or receipt["mechanism"]["delay_steps"] != 0
                    or receipt["mechanism"]["contact_time_constant"] != .02
                    or receipt["mode"] != mode or receipt["reset_seed"] != seed):
                raise ValueError("T3 ASID perturbation mechanism or request differs")
            replay = np.asarray(record.requested_actions, dtype=np.float32)
            if not np.array_equal(replay, actions[:len(replay)]):
                raise ValueError("T3 ASID perturbation did not replay source requests")
            positions.append(xyz)
            episodes += 1
            controls += len(record.traces)
        valid = (len(source.traces) == 32
                 and not row["source"]["constraint_violation"]
                 and len(perturbations) == 4
                 and all(receipt["physical_control_steps"] == 32
                         and not receipt["constraint_violation"]
                         for receipt in perturbations))
        if row["valid_full_horizon_and_no_violation"] is not valid:
            raise ValueError("T3 ASID information validity flag differs")
        if not perturbations:
            if row["information_score"] is not None:
                raise ValueError("T3 ASID skipped source retained information score")
            position_error = None
        else:
            jacobian = np.column_stack((
                ((positions[0] - positions[1]) / .2).reshape(-1),
                ((positions[2] - positions[3]) / .2).reshape(-1),
            ))
            norms = np.linalg.norm(jacobian, axis=0)
            reported = np.asarray(row["position_column_norms"], dtype=np.float64)
            singular = np.asarray(row["full_singular_values"], dtype=np.float64)
            score = float(row["information_score"])
            position_error = float(np.max(np.abs(norms - reported)))
            if (reported.shape != (2,) or singular.shape != (2,)
                    or not np.all(np.isfinite([*reported, *singular, score]))
                    or not np.allclose(norms, reported, rtol=2e-5, atol=3e-4)
                    or score + 1e-4 < float(np.sum(norms**2))
                    or not np.isclose(score, float(np.sum(singular**2)),
                                      rtol=1e-6, atol=1e-4)):
                raise ValueError("T3 ASID physical position information or score differs")
        cells.append({
            "policy": policy["name"], "mode": mode,
            "source_outcome": row["source"]["outcome"],
            "valid_full_horizon_and_no_violation": valid,
            "first_force_over_1_N_control": row["source"]["first_force_over_1_N_control"],
            "information_score": row["information_score"],
            "full_singular_values": row["full_singular_values"],
            "position_norm_max_error": position_error,
        })
    if (episodes != report["physical_episodes_charged"]
            or controls != report["physical_control_steps_charged"]
            or episodes > 30 or controls > 960):
        raise ValueError("T3 ASID physical budget or report ledger differs")
    return {
        "version": "asid-t3-preoverforce-information-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "recipe_sha256": sha(recipe_path),
        "audit_source_sha256": sha(Path(__file__)),
        "source_snapshots_verified": len(config["source_bindings"]),
        "physical_evidence_verified": episodes,
        "physical_control_steps_reconciled": controls,
        "position_jacobian_recomputed_from_physical_traces": True,
        "wrench_time_series_independently_rescored": False,
        "cells": cells,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = audit(config_path)
    output = ROOT / json.loads(config_path.read_text())["output"]
    atomic_json(output / "independent_audit.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "physical_evidence_verified", "physical_control_steps_reconciled",
        "cells",
    )}, indent=2))


if __name__ == "__main__":
    main()
