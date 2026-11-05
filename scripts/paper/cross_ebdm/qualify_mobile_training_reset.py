#!/usr/bin/env python3
"""Qualify a fresh T4 ASID training reset pool before any policy episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.contact import MobileContactMechanismSpec
from genedynamics.tasks.transfer.mobile_mating_policy_search import MOBILE_CONFIG, GPU1_UUID


VERSION = "asid-t4-independent-training-reset-feasibility-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(config: dict) -> dict:
    pointer = config.get("task_config", {})
    if (config.get("version") != VERSION or config.get("task") != "T4"
            or config.get("robot") != "go2_z1"
            or config.get("seed_start") != 3404000000
            or config.get("seed_count") != 64
            or config.get("physical_initial_slot") != 0
            or config.get("modes") != [0, 1]
            or config.get("failure_replacement") is not False
            or config.get("formal_performance_evidence") is not False
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_reset_calls": 256, "maximum_wall_seconds": 900,
                "maximum_gpu_memory_gib": 2,
            }
            or set(pointer) != {"path", "sha256"}
            or pointer["path"] != "configs/quadruped/mobile_keyed_insertion_recovery/_base.yaml"
            or sha(ROOT / pointer["path"]) != pointer["sha256"]):
        raise ValueError("T4 independent reset preflight contract differs")
    base = ExperimentConfig.from_yaml(ROOT / pointer["path"])
    if base.env_params.get("shared_execution") != config.get("shared_execution"):
        raise ValueError("T4 independent reset preflight shared execution differs")
    expected = {
        "source": {"servo_gain": .85, "payload_mass": .65,
                   "delay_steps": 0, "body_servo_gain": 1.},
        "target": {"servo_gain": .7, "payload_mass": .5,
                   "delay_steps": 3, "body_servo_gain": .9},
    }
    if config.get("mechanisms") != expected:
        raise ValueError("T4 independent reset preflight mechanisms differ")
    output = ROOT / config.get("output", "")
    if (config.get("output", "").split("/")[:5] != [
            "results", "quadruped", "mobile_keyed_insertion_recovery",
            "baseline", "asid",
            ] or output.exists()):
        raise ValueError("T4 independent reset preflight output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted", "task": "T4",
        "output": config["output"], "seed_start": config["seed_start"],
        "seed_count": 64, "maximum_reset_calls": 256,
        "maximum_wall_seconds": 900, "failure_replacement": False,
        "formal_performance_evidence": False,
    }


def run(config: dict) -> dict:
    import torch

    frozen = plan(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("T4 reset preflight bound to physical GPU1")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    started = time.monotonic()
    rows = []
    failure = None
    try:
        for seed in range(config["seed_start"], config["seed_start"] + 64):
            cells = []
            row = {"seed": seed, "all_reset_passed": False, "cells": cells}
            rows.append(row)
            for mechanism_name, mechanism in config["mechanisms"].items():
                for mode in config["modes"]:
                    if time.monotonic() - started > 900:
                        raise TimeoutError("T4 independent reset preflight wall ceiling exceeded")
                    cell = {"mechanism": mechanism_name, "mode": mode}
                    try:
                        task = instantiate_task(
                            "T4", robot="go2_z1",
                            mechanism_spec=MobileContactMechanismSpec(**mechanism),
                            seed=seed, shared_execution=config["shared_execution"],
                            config_overrides=MOBILE_CONFIG,
                        )
                        task.reset((0,), seed=seed, initial="jammed",
                                   modes=np.asarray([mode]))
                        cell["reset_passed"] = True
                    except Exception as exc:
                        cell.update(reset_passed=False, error_type=type(exc).__name__,
                                    error=str(exc))
                    cells.append(cell)
            row["all_reset_passed"] = all(cell["reset_passed"] for cell in cells)
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)}
    report = {
        "version": VERSION,
        "status": "failed_retained" if failure else "complete", "task": "T4",
        "recipe_sha256": sha(output / "recipe.json"),
        "seed_count": 64, "reset_calls": sum(len(row["cells"]) for row in rows),
        "safe_seed_count": sum(row["all_reset_passed"] for row in rows),
        "rows": rows, "wall_seconds": time.monotonic() - started,
        "policy_episodes": 0, "formal_performance_evidence": False,
        "failure": failure,
    }
    atomic_json(output / "report.json", report)
    return report


def audit(config: dict) -> dict:
    output = ROOT / config["output"]
    destination = output / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    rows = report.get("rows", [])
    expected_seeds = list(range(config["seed_start"], config["seed_start"] + 64))
    if (json.loads((output / "recipe.json").read_text()) != config
            or report.get("version") != VERSION or report.get("status") != "complete"
            or report.get("recipe_sha256") != sha(output / "recipe.json")
            or report.get("reset_calls") != 256 or len(rows) != 64
            or [row["seed"] for row in rows] != expected_seeds
            or report.get("policy_episodes") != 0
            or report.get("formal_performance_evidence") is not False):
        raise ValueError("T4 independent reset ledger differs")
    expected_cells = {(name, mode) for name in config["mechanisms"] for mode in (0, 1)}
    for row in rows:
        cells = row["cells"]
        if (len(cells) != 4
                or {(cell["mechanism"], cell["mode"]) for cell in cells} != expected_cells
                or row["all_reset_passed"] != all(cell["reset_passed"] for cell in cells)):
            raise ValueError("T4 independent reset cell ledger differs")
    safe = [row["seed"] for row in rows if row["all_reset_passed"]]
    if report["safe_seed_count"] != len(safe):
        raise ValueError("T4 independent safe seed count differs")
    result = {
        "version": "asid-t4-independent-training-reset-ledger-audit-v1",
        "status": "passed" if len(safe) == 64 else "failed_retained",
        "report_sha256": sha(report_path),
        "recipe_sha256": sha(output / "recipe.json"),
        "reset_calls_accounted": 256, "safe_seed_count": len(safe),
        "safe_seed_pool": safe, "physical_calls": 256,
        "gpu_calls": 0, "task_policy_episodes": 0,
        "formal_performance_evidence": False,
    }
    atomic_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    result = plan(config) if args.plan_only else run(config) if args.run else audit(config)
    print(json.dumps(result, indent=2))
    if args.run and result["status"] != "complete":
        raise RuntimeError("T4 independent reset preflight retained a failure")
    if args.audit and result["status"] != "passed":
        raise RuntimeError("T4 independent reset pool was not fully safe")


if __name__ == "__main__":
    main()
