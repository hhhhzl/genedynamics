#!/usr/bin/env python3
"""Independently reconcile T2 ASID development budgets, ledgers, and evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reward(row: dict) -> float:
    controls = row["control_steps"]
    error = row["final_position_error_m"]
    if row["success"]:
        return 1. - .3 * controls / 1000
    if row["constraint_violation"]:
        return -1. - min(error, 1.)
    return -min(error, 1.) - .2 * controls / 1000


def audit(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    report_path = output / "report.json"
    recipe_path = output / "recipe.json"
    report = json.loads(report_path.read_text())
    plan = json.loads((output / "plan.json").read_text())
    if (json.loads(recipe_path.read_text()) != config
            or report.get("version") != "asid-t2-complete-chain-development-v1"
            or report.get("status") != "complete_development_chain"
            or report.get("recipe_sha256") != sha(recipe_path)
            or report.get("formal_performance_evidence") is not False
            or plan.get("formal_performance_evidence") is not False
            or plan.get("failure_replacement") is not False
            or report.get("wall_seconds", 1e9) > 4800):
        raise ValueError("ASID T2 frozen recipe, scope, or completed status differs")
    if (report["source_bindings"] != config["source_bindings"]
            or len(config["source_bindings"]) != 23):
        raise ValueError("ASID T2 source roster differs")
    for binding in config["source_bindings"]:
        if sha(output / "source_snapshot" / binding["path"]) != binding["sha256"]:
            raise ValueError(f"ASID T2 source snapshot differs: {binding['path']}")
    if sha(ROOT / config["task_config"]["path"]) != config["task_config"]["sha256"]:
        raise ValueError("ASID T2 frozen task contract differs")

    observed = Counter()
    accounted = Counter()
    outcomes = Counter()
    evidence_cache = {}
    physical_replays = 0
    ledger = report["interaction_ledger"]
    for row in ledger:
        stage = row["stage"]
        target = row["target_interaction"]
        if target != (stage in {"target_exploration", "target_deployment"}):
            raise ValueError("ASID T2 simulator/target stage misclassified")
        receipt = {key: value for key, value in row.items()
                   if key not in {"stage", "target_interaction"}}
        observed[(stage, canonical(receipt))] += 1
        controls = int(receipt.get("physical_control_steps", receipt.get("control_steps", 0)))
        if not 1 <= controls <= (1000 if stage in {"task_policy_training", "target_deployment"} else 320):
            raise ValueError("ASID T2 physical control count outside stage horizon")
        group = "target" if target else "simulator"
        accounted[f"{group}_episodes"] += 1
        accounted[f"{group}_control_steps"] += controls
        if "evidence" in receipt:
            path = ROOT / receipt["evidence"]["path"]
            if (not path.is_relative_to(output)
                    or sha(path) != receipt["evidence"]["sha256"]):
                raise ValueError("ASID T2 physical evidence path or SHA differs")
            audit_row = audit_episode_evidence(path)
            if (not audit_row["passed"] or audit_row["reset_physical_hashes"][-1]
                    != receipt["evidence"]["reset_physical_hash"]):
                raise ValueError("ASID T2 physical evidence failed CPU rescoring")
            records, _, evaluation = load_episode_evidence(path)
            if len(records) != 1 or len(records[0].traces) != controls:
                raise ValueError("ASID T2 physical evidence episode length differs")
            episode = evaluation["episodes"][-1]
            if (episode["outcome"] != receipt["outcome"]
                    or bool(episode["constraint_violation"]) != receipt["constraint_violation"]
                    or ("success" in receipt and bool(episode["success"]) != receipt["success"])):
                raise ValueError("ASID T2 physical outcome differs")
            evidence_cache[canonical(receipt)] = records[0]
            physical_replays += 1
        elif stage != "task_policy_training":
            raise ValueError("ASID T2 non-training interaction has no physical evidence")
        if "reward" in receipt and not np.isclose(reward(receipt), receipt["reward"], atol=1e-12):
            raise ValueError("ASID T2 task reward differs")
        outcomes[receipt["outcome"]] += 1
    if (dict(accounted) != report["accounting"]
            or accounted["simulator_episodes"] > 232
            or accounted["target_episodes"] != 3
            or sum(value for key, value in accounted.items() if key.endswith("control_steps")) > 103760):
        raise ValueError("ASID T2 simulator/target physical budget differs")

    expected = Counter()
    position_lower_bounds = []
    exploration = report["exploration"]["ledger"]
    if len(exploration) != 8:
        raise ValueError("ASID T2 Fisher exploration candidate roster differs")
    for candidate in exploration:
        groups = defaultdict(list)
        for index, receipt in enumerate(candidate["episodes"]):
            new_mode = (index == 0 or
                        (receipt["mode"], receipt["reset_seed"]) !=
                        (candidate["episodes"][index - 1]["mode"],
                         candidate["episodes"][index - 1]["reset_seed"]))
            stage = "exploration_source" if new_mode else "exploration_perturbation"
            expected[(stage, canonical(receipt))] += 1
            groups[(receipt["mode"], receipt["reset_seed"])].append(receipt)
        if len(groups) != 2 or len(candidate["mode_information_scores"]) != 2:
            raise ValueError("ASID T2 exploration must retain both modes")
        for group, mode_score in zip(groups.values(), candidate["mode_information_scores"], strict=True):
            source = evidence_cache[canonical(group[0])]
            actions = np.asarray(source.requested_actions, dtype=np.float32)
            for perturb in group[1:]:
                replay = evidence_cache[canonical(perturb)]
                physical_actions = np.asarray(replay.requested_actions, dtype=np.float32)
                if not np.array_equal(physical_actions, actions[:len(physical_actions)]):
                    raise ValueError("ASID T2 Fisher perturbation did not replay source requests")
            if len(group) == 5 and mode_score >= 0:
                xyz = [np.asarray([trace["tcp_position"] for trace in
                                   evidence_cache[canonical(row)].traces], dtype=np.float64) * 1000.
                       for row in group[1:]]
                jacobian = np.column_stack((
                    ((xyz[0] - xyz[1]) / .2 / 100.).reshape(-1),
                    ((xyz[2] - xyz[3]) / .2 / 100.).reshape(-1),
                ))
                lower = min(float(np.sum(jacobian * jacobian)), 1000.)
                if mode_score + 1e-4 < lower or mode_score > 1000. + 1e-8:
                    raise ValueError("ASID T2 Fisher reward below physical position information")
                position_lower_bounds.append(lower)
            elif len(group) not in (1, 5):
                raise ValueError("ASID T2 Fisher central-difference roster incomplete")
        if not np.isclose(candidate["score"], np.mean(candidate["mode_information_scores"]), atol=1e-8):
            raise ValueError("ASID T2 Fisher candidate score differs")
    best_exploration = max(exploration, key=lambda row: row["score"])
    if (report["exploration"]["best_policy"] != best_exploration["candidate"]
            or not np.isclose(report["exploration"]["best_information_score"],
                              best_exploration["score"], atol=1e-8)):
        raise ValueError("ASID T2 exploration selection differs")

    target = report["target_exploration"]
    target_receipt = target["episode"]
    expected[("target_exploration", canonical(target_receipt))] += 1
    target_record = evidence_cache[canonical(target_receipt)]
    target_actions = np.asarray(target_record.requested_actions, dtype=np.float32)
    if hashlib.sha256(target_actions.tobytes()).hexdigest() != target["requested_actions_sha256"]:
        raise ValueError("ASID T2 target exploration action SHA differs")
    public = np.asarray(target["public_response"], dtype=np.float64)
    target_xyz = np.asarray([trace["tcp_position"] for trace in target_record.traces]) * 1000.
    if (public.shape != (320, 6)
            or not np.allclose(public[:, :3], target_xyz, rtol=1e-5, atol=1e-3)):
        raise ValueError("ASID T2 target public position differs from physical trace")

    identification = report["identification"]
    id_rows = identification["grid_and_local_ledger"]
    local = identification["local_search_ledger"]
    if len(id_rows) != 112 or len(local) != 12:
        raise ValueError("ASID T2 identification grid/CEM roster differs")
    for row in id_rows:
        receipt = row["episode"]
        expected[("identification", canonical(receipt))] += 1
        replay = evidence_cache[canonical(receipt)]
        actions = np.asarray(replay.requested_actions, dtype=np.float32)
        if not np.array_equal(actions, target_actions[:len(actions)]):
            raise ValueError("ASID T2 identification did not replay target requests")
        if (receipt["mode"] != target_receipt["mode"]
                or receipt["reset_seed"] != target_receipt["reset_seed"]
                or [receipt["mechanism"][key] for key in (
                    "servo_gain", "payload_mass", "delay_steps")]
                != row["parameters"]):
            raise ValueError("ASID T2 identification physical parameter differs")
        xyz = np.asarray([trace["tcp_position"] for trace in replay.traces]) * 1000.
        if len(xyz) < 320:
            xyz = np.concatenate((xyz, np.repeat(xyz[-1:], 320 - len(xyz), axis=0)))
        position_mse_lower = float(np.sum((xyz - public[:, :3]) ** 2) / (320 * 6))
        if row["score"] + 1e-5 < position_mse_lower:
            raise ValueError("ASID T2 public MSE below physical position residual")
    best_grid = min(id_rows[:100], key=lambda row: row["score"])
    best_local = min(local, key=lambda row: row["score"])
    if (identification["best_grid"] != best_grid
            or identification["selected_delay_steps"] != best_grid["parameters"][2]
            or identification["best_parameters"] != [*best_local["candidate"],
                                                      best_grid["parameters"][2]]
            or not np.isclose(identification["best_public_mse"],
                              best_local["score"], atol=1e-10)):
        raise ValueError("ASID T2 grid/CEM identification selection differs")

    task = report["task_policy"]
    if len(task["ledger"]) != 10:
        raise ValueError("ASID T2 downstream policy candidate roster differs")
    for candidate in task["ledger"]:
        if len(candidate["episodes"]) != 4:
            raise ValueError("ASID T2 task candidate needs four simulated episodes")
        for row in candidate["episodes"]:
            stripped = {key: value for key, value in row.items() if key != "mechanism"}
            expected[("task_policy_training", canonical(stripped))] += 1
        if not np.isclose(candidate["score"], np.mean([
                row["reward"] for row in candidate["episodes"]]), atol=1e-12):
            raise ValueError("ASID T2 task-policy candidate reward differs")
    best_task = max(task["ledger"], key=lambda row: row["score"])
    if (task["best_policy"] != best_task["candidate"]
            or not np.isclose(task["best_simulated_reward"], best_task["score"], atol=1e-12)):
        raise ValueError("ASID T2 trained task-policy selection differs")
    if len(report["deployment"]) != 2:
        raise ValueError("ASID T2 target deployment mode roster differs")
    for mode, row in enumerate(report["deployment"]):
        if row["mode"] != mode or row["reset_seed"] != config["target_deployment_seeds"][mode]:
            raise ValueError("ASID T2 target deployment request differs")
        expected[("target_deployment", canonical(row))] += 1
    if expected != observed:
        raise ValueError("ASID T2 nested stage ledgers disagree with charged interactions")
    checkpoint = output / "checkpoint.json"
    receipt = json.loads((output / "checkpoint_receipt.json").read_text())
    if (receipt["checkpoint"]["sha256"] != sha(checkpoint)
            or receipt["report"]["sha256"] != sha(report_path)
            or json.loads(checkpoint.read_text())["policy"] != task["best_policy"]):
        raise ValueError("ASID T2 checkpoint/report binding differs")
    return {
        "schema_version": "asid-t2-full-chain-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": sha(report_path),
        "recipe_sha256": sha(recipe_path),
        "checkpoint_sha256": sha(checkpoint),
        "audit_source_sha256": sha(Path(__file__)),
        "source_snapshots_verified": len(config["source_bindings"]),
        "physical_evidence_verified": physical_replays,
        "simulated_training_episodes_ledger_only": accounted["simulator_episodes"] - physical_replays + 3,
        "physical_position_fisher_lower_bounds_checked": len(position_lower_bounds),
        "accounting": dict(accounted),
        "outcomes": dict(sorted(outcomes.items())),
        "wrench_time_series_independently_rescored": False,
        "formal_performance_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    result = audit(path)
    output = ROOT / json.loads(path.read_text())["output"]
    atomic_json(output / "independent_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
