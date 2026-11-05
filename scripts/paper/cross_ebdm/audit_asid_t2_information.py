#!/usr/bin/env python3
"""Independently audit T2 ASID information evidence and public positions."""
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


def read_evidence(receipt: dict, output: Path):
    path = ROOT / receipt["evidence"]["path"]
    if not path.is_relative_to(output) or sha(path) != receipt["evidence"]["sha256"]:
        raise ValueError("T2 ASID evidence location or SHA differs")
    audit = audit_episode_evidence(path)
    if (not audit["passed"]
            or audit["reset_physical_hashes"][-1] != receipt["evidence"]["reset_physical_hash"]):
        raise ValueError("T2 ASID physical evidence failed CPU rescoring")
    records, _, evaluation = load_episode_evidence(path)
    if len(records) != 1:
        raise ValueError("T2 ASID evidence must contain one episode")
    record = records[0]
    first_force = next((index for index, trace in enumerate(record.traces)
                        if float(trace["force_peak"]) > 1.), None)
    if (len(record.traces) != receipt["physical_control_steps"]
            or evaluation["episodes"][-1]["outcome"] != receipt["outcome"]
            or bool(evaluation["episodes"][-1]["constraint_violation"])
            != receipt["constraint_violation"]
            or first_force != receipt["first_force_over_1_N_control"]):
        raise ValueError("T2 ASID evidence outcome, length, or contact onset differs")
    positions = np.asarray([trace["tcp_position"] for trace in record.traces], dtype=np.float64)
    if len(positions) < 320:
        positions = np.concatenate((positions, np.repeat(
            positions[-1:], 320 - len(positions), axis=0,
        )))
    return record, positions * 1000.


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    recipe_path = output / "recipe.json"
    report_path = output / "report.json"
    if json.loads(recipe_path.read_text()) != config:
        raise ValueError("T2 ASID frozen recipe differs from current config")
    report = json.loads(report_path.read_text())
    if (report.get("version") != "asid-t2-public-information-probe-v1"
            or report.get("status") != "complete_development_probe"
            or report.get("recipe_sha256") != sha(recipe_path)
            or len(report.get("rows", [])) != 6
            or report.get("formal_performance_evidence") is not False
            or report.get("maximum_physics_episodes") != 30
            or report.get("maximum_physical_control_steps") != 9600):
        raise ValueError("T2 ASID report status, recipe, or budget differs")
    if report["source_bindings"] != config["source_bindings"]:
        raise ValueError("T2 ASID bound source roster differs")
    for binding in config["source_bindings"]:
        snapshot = output / "source_snapshot" / binding["path"]
        if sha(snapshot) != binding["sha256"]:
            raise ValueError("T2 ASID source snapshot differs")
    expected = [(policy["name"], mode, seed)
                for policy in config["policies"]
                for mode, seed in zip(config["requested_modes"], config["seeds"], strict=True)]
    verified = []
    episodes = controls = 0
    for row, (name, mode, seed) in zip(report["rows"], expected, strict=True):
        if (row["policy"] != next(p for p in config["policies"] if p["name"] == name)
                or row["mode"] != mode or row["seed"] != seed):
            raise ValueError("T2 ASID policy, mode, or fresh seed changed")
        source, source_positions = read_evidence(row["source"], output)
        episodes += 1
        controls += len(source.traces)
        source_actions = np.asarray(source.requested_actions, dtype=np.float32)
        if hashlib.sha256(source_actions.tobytes()).hexdigest() != row["source_action_sha256"]:
            raise ValueError("T2 ASID source action stream SHA differs")
        if (row["source"]["mechanism"] != config["source_mechanism"]
                or row["source"]["reset_seed"] != seed
                or row["source"]["mode"] != mode):
            raise ValueError("T2 ASID source mechanism or request differs")
        perturbations = row["perturbations"]
        if len(perturbations) not in (0, 4):
            raise ValueError("T2 ASID central differences need four simulations")
        expected_theta = [
            (.95, .65), (.75, .65), (.85, .75), (.85, .55),
        ]
        positions = []
        for receipt, (gain, mass) in zip(perturbations, expected_theta, strict=False):
            record, xyz = read_evidence(receipt, output)
            if (receipt["mechanism"] != {
                    "servo_gain": gain, "payload_mass": mass,
                    "delay_steps": 0, "contact_time_constant": .04,
                } or receipt["mode"] != mode or receipt["reset_seed"] != seed):
                raise ValueError("T2 ASID perturbation parameter or request differs")
            if not np.array_equal(
                    np.asarray(record.requested_actions, dtype=np.float32),
                    source_actions[:len(record.requested_actions)]):
                raise ValueError("T2 ASID perturbation does not replay source requests")
            positions.append(xyz)
            episodes += 1
            controls += len(record.traces)
        expected_valid = (len(source.traces) == 320 and not row["source"]["constraint_violation"]
                          and len(perturbations) == 4 and all(
                              receipt["physical_control_steps"] == 320
                              and not receipt["constraint_violation"]
                              for receipt in perturbations))
        if row["valid_full_horizon_and_no_constraint_violation"] is not expected_valid:
            raise ValueError("T2 ASID information validity flag differs")
        if len(perturbations) == 0:
            if len(source.traces) == 320 and not row["source"]["constraint_violation"]:
                raise ValueError("T2 ASID skipped valid source without perturbations")
            if row["information_score"] is not None:
                raise ValueError("T2 ASID invalid source retained information score")
            position_error = None
        else:
            if len(source_positions) != 320:
                raise ValueError("T2 ASID source position horizon differs")
            jacobian = np.column_stack((
                ((positions[0] - positions[1]) / .2).reshape(-1),
                ((positions[2] - positions[3]) / .2).reshape(-1),
            ))
            observed = np.linalg.norm(jacobian, axis=0)
            reported = np.asarray(row["position_column_norms"], dtype=np.float64)
            position_error = float(np.max(np.abs(observed - reported)))
            if not np.allclose(observed, reported, rtol=1e-6, atol=1e-4):
                raise ValueError("T2 ASID public-position Jacobian differs from physical traces")
            total = np.asarray(row["parameter_column_norms"], dtype=np.float64)
            wrench = np.asarray(row["wrench_column_norms"], dtype=np.float64)
            singular = np.asarray(row["full_singular_values"], dtype=np.float64)
            pre = np.asarray(row["precontact_singular_values"], dtype=np.float64)
            contact = np.asarray(row["contact_window_singular_values"], dtype=np.float64)
            score = float(row["information_score"])
            if (total.shape != (2,) or wrench.shape != (2,) or singular.shape != (2,)
                    or not np.all(np.isfinite([*total, *wrench, *singular, *pre, *contact, score]))
                    or not np.allclose(total**2, reported**2 + wrench**2, rtol=1e-6, atol=1e-4)
                    or not np.isclose(score, np.sum(total**2), rtol=1e-6, atol=1e-4)
                    or not np.isclose(score, np.sum(singular**2), rtol=1e-6, atol=1e-4)
                    or not np.isclose(score, np.sum(pre**2) + np.sum(contact**2), rtol=1e-6, atol=1e-4)):
                raise ValueError("T2 ASID information score or singular spectrum is internally inconsistent")
        verified.append({"policy": name, "mode": mode,
                         "valid": expected_valid,
                         "source_contact_onset": row["source"]["first_force_over_1_N_control"],
                         "position_column_norm_max_error": position_error,
                         "information_score": row["information_score"],
                         "position_column_norms": row["position_column_norms"],
                         "wrench_column_norms": row["wrench_column_norms"],
                         "full_singular_values": row["full_singular_values"]})
    if (episodes != report["physical_episodes_charged"]
            or controls != report["physical_control_steps_charged"]
            or episodes > report["maximum_physics_episodes"]
            or controls > report["maximum_physical_control_steps"]):
        raise ValueError("T2 ASID physical episode or control ledger differs")
    return {
        "version": "asid-t2-public-information-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "audit_source_sha256": sha(Path(__file__)),
        "physical_evidence_verified": episodes,
        "physical_control_steps_reconciled": controls,
        "public_position_jacobian_recomputed": True,
        "wrench_jacobian_evidence": "run-bound public observations and internal algebra; full wrench series absent from physical trace",
        "cells": verified, "physical_calls": 0, "gpu_calls": 0,
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
        "status", "physical_evidence_verified", "physical_control_steps_reconciled",
        "public_position_jacobian_recomputed", "wrench_jacobian_evidence",
    )}, indent=2))


if __name__ == "__main__":
    main()
