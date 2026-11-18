#!/usr/bin/env python3
"""Run a fresh paired development pilot for one semantic transfer subtask."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.controllers.bindings import (
    create_transfer_actor, resolve_transfer_binding,
)
from genedynamics.tasks.transfer.episode import (
    audit_episode_evidence, episode_evidence,
)
from genedynamics.tasks.transfer.specs import mechanism_spec
from genedynamics.tasks.transfer.subtask_contracts import load_subtask_contract


VERSION = "semantic-task-paired-development-pilot-v1"
FORMAL_VERSION = "semantic-task-formal-comparison-v1"
COMMON_METHODS = ("cross_ebdm", "task_only", "adaptsim", "asid")
TASKS = {
    "aperture_brake": {
        "alias": "T1", "targeted": "response_mpc", "robot": "xarm7",
        "family": "arm/obstacle_transfer",
    },
    "sequential_aperture_reorientation": {
        "alias": "T1", "targeted": "response_mpc", "robot": "xarm7",
        "family": "arm/obstacle_transfer",
    },
    "reposition_then_dock": {
        "alias": "T4", "targeted": "dial_transfer", "robot": "go2_z1",
        "family": "quadruped/mobile_keyed_insertion_recovery",
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(pointer, label):
    if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA pointer")
    path = ROOT / pointer["path"]
    if not path.is_file() or sha256(path) != pointer["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def pointer(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def method_binding(path, method, alias, mode):
    config = ExperimentConfig.from_yaml(path)
    if config.method != method:
        raise ValueError(f"{method} config identity differs")
    binding = deepcopy(config.method_params["actor_binding"])
    if binding.get("manifest") is not None:
        binding.update(family="main", key=method, instance="instance_01")
        binding = resolve_transfer_binding(
            {"actor_binding": binding, "mode": mode},
            task_alias=alias, instance_id="instance_01",
        )
    return config, binding


def summarize(rows):
    latencies = [row["actor_latency_seconds"]["p95"] for row in rows]
    return {
        "episodes": len(rows),
        "constrained_success_rate": sum(row["success"] for row in rows) / len(rows),
        "constraint_violation_rate": sum(row["constraint_violation"] for row in rows) / len(rows),
        "failure_penalized_normalized_time": sum(
            row["capped_normalized_completion_time"] for row in rows
        ) / len(rows),
        "median_episode_actor_p95_latency_seconds": statistics.median(latencies),
    }


def run(config_path, *, plan_only=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    expected = {
        "version", "status", "task", "methods", "method_configs",
        "contract", "source_qualification", "native_library", "requests",
        "execution", "output", "policy",
    }
    optional = {"response_transport_checkpoint"}
    if (set(config) - optional != expected
            or config["version"] not in {VERSION, FORMAL_VERSION}):
        raise ValueError("semantic paired-pilot config schema differs")
    task_name = config["task"]
    if task_name not in TASKS:
        raise ValueError("unsupported semantic pilot task")
    task_info = TASKS[task_name]
    methods = (*COMMON_METHODS, task_info["targeted"])
    if tuple(config["methods"]) != methods or set(config["method_configs"]) != set(methods):
        raise ValueError("paired-pilot method roster differs")
    formal = config["policy"].get("formal_performance_evidence")
    if formal not in {True, False}:
        raise ValueError("explicit formal-performance evidence tier required")
    if config["version"] != (FORMAL_VERSION if formal else VERSION):
        raise ValueError("comparison version disagrees with evidence tier")
    if config["status"] != "frozen_unexecuted" or config["policy"] != {
        "formal_performance_evidence": formal,
        "failure_replacement": False,
        "all_failures_retained": True,
        "confirmation_states_opened": 0,
        "task_adaptation_is_reported": True,
    }:
        raise ValueError("paired-pilot evidence policy differs")
    if config["execution"] != {
        "cuda_visible_devices": "0",
        "physical_uuid": "GPU-253f0813-4432-2d51-a149-3e73b38bf77a",
        "actor_device": "cuda",
        "physics_device": "cpu",
        "maximum_wall_seconds": 7200 if formal else 1800,
    }:
        raise ValueError("paired-pilot execution contract differs")
    modes = [int(row["mode"]) for row in config["requests"]]
    if (set(modes) != {0, 1} or modes.count(0) != modes.count(1)
            or (not formal and len(modes) != 2)
            or (formal and len(modes) < 10)):
        raise ValueError("balanced paired modes and the declared evidence tier are required")
    if len({row["seed"] for row in config["requests"]}) != len(config["requests"]):
        raise ValueError("comparison reset seeds must be distinct")

    contract_path = checked(config["contract"], "semantic contract")
    contract = load_subtask_contract(task_name, require_runnable=True)
    if contract.path.resolve() != contract_path.resolve():
        raise ValueError("semantic contract path differs")
    source_path = checked(config["source_qualification"], "source qualification")
    source = json.loads(source_path.read_text())
    if source.get("passed") is not True or source.get("formal_performance_evidence") is not False:
        raise ValueError("passed development source qualification required")
    native = checked(config["native_library"], "native ICDT ABI")
    response_checkpoint_path = None
    response_checkpoint = None
    if task_name in {
            "aperture_brake", "sequential_aperture_reorientation"}:
        response_checkpoint_path = checked(
            config.get("response_transport_checkpoint"),
            "response transport checkpoint",
        )
        response_checkpoint = json.loads(response_checkpoint_path.read_text())
        from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import (
            validate_checkpoint,
        )
        validate_checkpoint(response_checkpoint, task=task_name)
    method_paths = {
        name: checked(value, f"{name} method config")
        for name, value in config["method_configs"].items()
    }
    resolved = {}
    for mode in (0, 1):
        for method in methods:
            _, binding = method_binding(
                method_paths[method], method, task_info["alias"], mode,
            )
            resolved[(method, mode)] = binding
    plan = {
        "version": config["version"],
        "status": "planned_unexecuted",
        "task": task_name,
        "methods": list(methods),
        "requests": config["requests"],
        "planned_episodes": len(methods) * len(config["requests"]),
        "execution": config["execution"],
        "policy": config["policy"],
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__)),
            "contract": pointer(contract_path),
            "source_qualification": pointer(source_path),
            "native_library": pointer(native),
            **({"response_transport_checkpoint": pointer(response_checkpoint_path)}
               if response_checkpoint_path is not None else {}),
            "method_configs": {name: pointer(path) for name, path in method_paths.items()},
        },
    }
    if plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("paired pilot requires physical GPU0 visibility")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("paired pilot requires exactly one visible CUDA device")
    visible_uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    if visible_uuid != config["execution"]["physical_uuid"]:
        raise RuntimeError("visible CUDA device UUID differs")

    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    rows = []
    reset_hashes = {request["seed"]: set() for request in config["requests"]}
    for request in config["requests"]:
        mode, seed = int(request["mode"]), int(request["seed"])
        for method in methods:
            if time.monotonic() - started > config["execution"]["maximum_wall_seconds"]:
                raise TimeoutError("paired pilot exceeded its frozen wall limit")
            mechanism = mechanism_spec(
                task_name, request["mechanism"],
                body_support_parameters=request.get("body_support_parameters"),
            )
            task_config = ExperimentConfig.from_yaml(method_paths[method])
            shared = (
                task_config.env_params.get("shared_execution")
                if task_name == "reposition_then_dock" else None
            )
            task = instantiate_task(
                task_name, robot=task_info["robot"], mechanism_spec=mechanism,
                seed=seed, config_overrides=request.get("config_overrides"),
                shared_execution=shared,
            )
            public = task.reset((seed,), seed=seed, initial=request["initial"], modes=[mode])
            if (method == "cross_ebdm" and task_name in {
                    "aperture_brake", "sequential_aperture_reorientation"}):
                from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import (
                    PublicSemanticPhaseTransport,
                )
                from genedynamics.tasks.transfer.aperture import (
                    PublicApertureBrakePolicy, PublicSequentialAperturePolicy,
                )
                base_type = (
                    PublicApertureBrakePolicy if task_name == "aperture_brake"
                    else PublicSequentialAperturePolicy
                )
                actor = PublicSemanticPhaseTransport(
                    base_type(task.spec, mode=mode), response_checkpoint,
                    task=task_name,
                )
            else:
                actor = create_transfer_actor(task, method, {
                    "actor_binding": resolved[(method, mode)],
                    "mode": mode, "device": config["execution"]["actor_device"],
                })
                if task_name in {
                        "aperture_brake", "sequential_aperture_reorientation"}:
                    from genedynamics.tasks.transfer.aperture import bind_aperture_public_planner
                    actor = bind_aperture_public_planner(actor, task.spec, mode=mode)
                else:
                    from genedynamics.tasks.transfer.reposition_dock import (
                        PublicDockingMethodExecution, PublicRepositionDockExecution,
                    )
                    actor = PublicDockingMethodExecution(actor)
                    actor = PublicRepositionDockExecution(task.spec, actor, mode=mode)
            latency = []
            for _ in range(task.config.timeout_steps):
                if task.state.status[0] != 0:
                    break
                before = time.perf_counter()
                if task_name == "reposition_then_dock":
                    action, body = actor.act_with_body(public)
                else:
                    action, body = actor.act(public), None
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency.append(time.perf_counter() - before)
                step = (
                    task.step(action) if body is None
                    else task.step_contact(action, body_twist=body)
                )
                public = step.observation
            evaluation = task.evaluate()
            episode = evaluation["episodes"][-1]
            destination = output / "evidence" / method / f"seed_{seed}.json.gz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(gzip.compress(json.dumps(
                episode_evidence(task.records, task.spec, evaluation),
                sort_keys=True, allow_nan=False,
            ).encode(), mtime=0))
            audit = audit_episode_evidence(destination)
            reset_hash = audit["reset_physical_hashes"][0]
            reset_hashes[seed].add(reset_hash)
            values = sorted(latency)
            row = {
                "task": task_name, "method": method, "mode": mode,
                "seed": seed, "mechanism_group": request["mechanism_group"],
                "success": bool(episode["success"]),
                "outcome": episode["outcome"],
                "constraint_violation": bool(episode["constraint_violation"]),
                "capped_normalized_completion_time": float(
                    episode["capped_normalized_completion_time"]
                ),
                "physical_control_steps": int(episode["physical_control_steps"]),
                "reset_physical_hash": reset_hash,
                "evidence": pointer(destination),
                "evidence_audit_passed": bool(audit["passed"]),
                "actor_latency_seconds": {
                    "scope": "synchronous actor call plus CUDA synchronization; excludes CPU physics",
                    "count": len(values),
                    "median": statistics.median(values),
                    "p95": values[min(len(values) - 1, int(.95 * len(values)))],
                },
            }
            rows.append(row)
            (output / "progress.json").write_text(json.dumps(
                {"completed": len(rows), "rows": rows}, indent=2,
            ) + "\n")
            print(json.dumps({key: row[key] for key in (
                "task", "method", "mode", "seed", "success", "outcome",
                "physical_control_steps",
            )}), flush=True)
    pairing = all(len(values) == 1 for values in reset_hashes.values())
    audits = all(row["evidence_audit_passed"] for row in rows)
    by_method = {
        method: summarize([row for row in rows if row["method"] == method])
        for method in methods
    }
    report = {
        "version": config["version"],
        "status": (
            "complete_formal" if formal and pairing and audits else
            "complete_development_only" if pairing and audits else
            "audit_failed"
        ),
        "task": task_name, "methods": list(methods),
        "rows": rows, "summary": by_method,
        "pairing": {
            "passed": pairing,
            "reset_physical_hashes": {
                str(seed): sorted(values) for seed, values in reset_hashes.items()
            },
        },
        "all_evidence_reaudited": audits,
        "formal_performance_evidence": bool(formal),
        "task_adaptation": (
            "shared semantic aperture planner; method response models retained; ICDT uses its SHA-bound learned response checkpoint"
            if task_name in {"aperture_brake", "sequential_aperture_reorientation"} else
            "shared public body-target/reposition/settle prephase; T4 docking actors retained"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "plan_sha256": sha256(output / "plan.json"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "summary": report["summary"],
        "report": str((output / "report.json").relative_to(ROOT)),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args.config, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
