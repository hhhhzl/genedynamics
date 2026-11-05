#!/usr/bin/env python3
"""Plan, run, or independently audit a T3 AdaptSim feedback diagnosis."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.contact import ContactMechanismSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.mating_policy_search import MatingPolicySearchTask


ROOT = Path(__file__).resolve().parents[3]
VERSION = "adaptsim-t3-feedback-diagnosis-v2"
POPULATION_VERSION = "adaptsim-t3-feedback-population-diagnosis-v3"
POPULATION_TRAINING_VERSION = "adaptsim-mating-feedback-population-development-v2"
GPU1_UUID = "58b54684-3577-11d3-f039-0e8469190739"
METHODS = ("adaptsim_development", "public_anchor")
SOURCE_PATHS = {
    "scripts/paper/cross_ebdm/diagnose_adaptsim_feedback.py",
    "genedynamics/tasks/transfer/mating_policy_search.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
}
POPULATION_SOURCE_PATHS = SOURCE_PATHS | {
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
    "genedynamics/core/control/transfer.py",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(config: dict) -> dict:
    version = config.get("version")
    if version not in {VERSION, POPULATION_VERSION} or config.get("task") != "T3" or config.get("robot") != "xarm7":
        raise ValueError("T3 AdaptSim diagnosis identity differs")
    population = version == POPULATION_VERSION
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": "GPU-" + GPU1_UUID,
        "maximum_wall_seconds": 600, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("T3 diagnosis execution budget differs")
    if config.get("reset_slots_per_cell") != 8 or config.get("requested_modes") != [0, 1]:
        raise ValueError("T3 diagnosis roster differs")
    if (config.get("seed_anchor") != (3200400000 if population else 3100400000)
            or config.get("failure_replacement") is not False):
        raise ValueError("T3 diagnosis seed or failure policy differs")
    cells = config.get("scene_cells")
    if not isinstance(cells, list) or [c.get("id") for c in cells] != [
        "source", "nominal", "latency_gain_shift",
    ]:
        raise ValueError("T3 diagnosis cells differ")
    expected_mechanisms = [(0.85, 0.65, 0), (1.0, 0.5, 0), (0.7, 0.5, 3)]
    for cell, expected in zip(cells, expected_mechanisms, strict=True):
        mechanism = ContactMechanismSpec(**cell["mechanism"])
        if (mechanism.servo_gain, mechanism.payload_mass, mechanism.delay_steps) != expected:
            raise ValueError("T3 diagnosis mechanism differs")
    for key in ("training_report", "checkpoint"):
        pointer = config[key]
        if set(pointer) != {"path", "sha256"} or _digest(ROOT / pointer["path"]) != pointer["sha256"]:
            raise ValueError(f"T3 diagnosis {key} changed")
    sources = config.get("source_bindings")
    expected_sources = POPULATION_SOURCE_PATHS if population else SOURCE_PATHS
    if not isinstance(sources, list) or {row.get("path") for row in sources} != expected_sources:
        raise ValueError("T3 diagnosis source roster differs")
    for row in sources:
        if set(row) != {"path", "sha256"} or _digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"T3 diagnosis source changed: {row.get('path')}")
    report = json.loads((ROOT / config["training_report"]["path"]).read_text())
    if (report["status"] != "complete_development"
            or (population and report["version"] != POPULATION_TRAINING_VERSION)):
        raise ValueError("T3 training is incomplete")
    recipe = json.loads((ROOT / config["training_report"]["path"]).with_name("recipe.json").read_text())
    if recipe["task"] != "T3" or recipe["robot"] != "xarm7":
        raise ValueError("T3 training recipe differs")
    if (ROOT / config["output"]).exists():
        raise FileExistsError(config["output"])
    return {
        "status": "planned_unexecuted", "version": version,
        "physical_episodes": 3 * 8 * 2 * len(METHODS),
        "paired_resets": 3 * 8 * 2,
        "scope": "single trained instance, held-out development diagnosis; not formal main-table performance",
    }


def run(config: dict) -> dict:
    scheduled = plan(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T3 diagnosis requires GPU1 only")
    if str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") != GPU1_UUID:
        raise RuntimeError("visible GPU UUID differs")
    recipe = json.loads((ROOT / config["training_report"]["path"]).with_name("recipe.json").read_text())
    checkpoint = torch.load(ROOT / config["checkpoint"]["path"], map_location="cpu", weights_only=True)
    policies = {
        "adaptsim_development": checkpoint["final_policy"],
        "public_anchor": recipe["task_policy_search"]["anchor"],
    }
    search = MatingPolicySearchTask(robot="xarm7", feedback_base=recipe["feedback_base"],
                                    shared_execution=recipe["shared_execution"])
    for policy in policies.values():
        search.feedback(policy)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")
    for row in config["source_bindings"]:
        destination = output / "sources" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], destination)
    started = time.monotonic()
    rows: list[dict] = []
    result = {**scheduled, "status": "running", "formal_performance_evidence": False}
    try:
        for cell_index, cell in enumerate(config["scene_cells"]):
            mechanism = ContactMechanismSpec(**cell["mechanism"])
            for slot in range(config["reset_slots_per_cell"]):
                for mode in config["requested_modes"]:
                    seed = config["seed_anchor"] + cell_index * 1000 + slot * 2 + mode
                    for method in METHODS:
                        if time.monotonic() - started > 600:
                            raise TimeoutError("T3 diagnosis wall limit")
                        task = instantiate_task("T3", robot="xarm7", mechanism_spec=mechanism,
                                                seed=seed, shared_execution=recipe["shared_execution"])
                        observation = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([mode]))
                        actor = PublicLivenessMatingFeedbackPolicy(
                            task.spec, feedback=search.feedback(policies[method]), modes=(mode,), seed=seed,
                        )
                        for _ in range(task.config.timeout_steps):
                            step = task.step(actor.act(observation))
                            observation = step.observation
                            if bool(step.done[0]):
                                break
                        evaluation = task.evaluate()
                        episode = evaluation["episodes"][-1]
                        path = output / "evidence" / f"{cell_index}_{slot}_{mode}_{method}.json.gz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        payload = episode_evidence(task.records, task.spec, evaluation)
                        path.write_bytes(gzip.compress(json.dumps(payload, sort_keys=True,
                                                                   allow_nan=False).encode(), mtime=0))
                        checked = audit_episode_evidence(path)
                        if not checked["passed"]:
                            raise ValueError("T3 evidence self-check failed")
                        row = {
                            "cell": cell["id"], "slot": slot, "mode": mode, "method": method,
                            "reset_seed": seed, "success": bool(episode["success"]),
                            "outcome": episode["outcome"],
                            "constraint_violation": bool(episode["constraint_violation"]),
                            "control_steps": int(episode["physical_control_steps"]),
                            "force_peak_N": float(episode["force_peak_N"]),
                            "evidence": {"path": str(path.relative_to(ROOT)), "sha256": checked["sha256"]},
                            "reset_physical_hash": checked["reset_physical_hashes"][0],
                        }
                        rows.append(row)
                        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
                        print(json.dumps(row), flush=True)
        for i in range(0, len(rows), 2):
            left, right = rows[i:i + 2]
            if (left["cell"], left["slot"], left["mode"], left["reset_seed"], left["reset_physical_hash"]) != (
                right["cell"], right["slot"], right["mode"], right["reset_seed"], right["reset_physical_hash"]
            ):
                raise ValueError("T3 paired physical reset differs")
        result.update({"status": "complete_development_diagnosis", "rows": rows,
                       "physical_control_steps": sum(row["control_steps"] for row in rows),
                       "wall_seconds": time.monotonic() - started, "all_failures_retained": True})
    except BaseException as error:
        result.update({"status": "failed_retained", "error": str(error),
                       "traceback": traceback.format_exc(), "rows": rows,
                       "wall_seconds": time.monotonic() - started})
        raise
    finally:
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(config: dict) -> dict:
    output = ROOT / config["output"]
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    rows = report["rows"]
    expected = 3 * config["reset_slots_per_cell"] * 2 * len(METHODS)
    if report["status"] != "complete_development_diagnosis" or len(rows) != expected:
        raise ValueError("T3 diagnosis roster incomplete")
    checks = []
    for row in rows:
        checked = audit_episode_evidence(ROOT / row["evidence"]["path"])
        episode = checked["evaluation"]["episodes"][-1]
        if not (checked["passed"] and checked["sha256"] == row["evidence"]["sha256"]
                and checked["reset_physical_hashes"] == [row["reset_physical_hash"]]
                and episode["outcome"] == row["outcome"]
                and bool(episode["success"]) == row["success"]
                and bool(episode["constraint_violation"]) == row["constraint_violation"]
                and int(episode["physical_control_steps"]) == row["control_steps"]):
            raise ValueError("T3 diagnosis evidence disagrees with report")
        checks.append(checked["sha256"])
    for i in range(0, len(rows), 2):
        left, right = rows[i:i + 2]
        if (left["cell"], left["slot"], left["mode"], left["reset_seed"], left["reset_physical_hash"]) != (
            right["cell"], right["slot"], right["mode"], right["reset_seed"], right["reset_physical_hash"]
        ) or {left["method"], right["method"]} != set(METHODS):
            raise ValueError("T3 diagnosis paired roster differs")
    for row in config["source_bindings"]:
        if _digest(output / "sources" / row["path"]) != row["sha256"]:
            raise ValueError("T3 diagnosis source snapshot changed")
    result = {
        "schema_version": "adaptsim-t3-feedback-diagnosis-audit-v1", "status": "passed",
        "report_sha256": _digest(report_path), "episode_count": len(checks),
        "paired_resets": len(rows) // 2, "source_snapshots": len(config["source_bindings"]),
        "evidence_sha256": checks,
        "outcomes": {cell: {method: {
            outcome: sum(row["cell"] == cell and row["method"] == method and row["outcome"] == outcome
                         for row in rows)
            for outcome in sorted({row["outcome"] for row in rows})
        } for method in METHODS} for cell in ("source", "nominal", "latency_gain_shift")},
    }
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = plan(config) if args.plan_only else run(config) if args.run else audit(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
