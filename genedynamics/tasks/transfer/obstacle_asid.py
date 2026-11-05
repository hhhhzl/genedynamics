"""T1 task adaptation of ASID's exploration, identification, and retraining chain.

All MuJoCo construction, public observations, target interaction, and episode
accounting stay task-owned. The ASID solver owns only the published Fisher
objective and independent bounded search. This is a development adaptation:
the author's released rod-specific SAC trainer is not reproduced here.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.asid import ASIDSolver
from genedynamics.solvers.single.asid.learning import ASIDBoundedCEM
from genedynamics.tasks.transfer.controllers.obstacle_feedback import OBSTACLE_BOUNDS
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.obstacle_policy_search import ObstaclePolicySearchTask
from genedynamics.tasks.transfer.runtime import MechanismSpec


VERSION = "asid-obstacle-full-chain-development-v1"
GRID_VERSION = "asid-obstacle-grid-seeded-development-v2"
INDEPENDENT_COHORT = "independent_main_table_training_v1"
NOMINAL_VERSION = "asid-obstacle-nominal-diagnosis-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_COMMIT = "cf2cb670fb983f67cf2af156d4e17b4a690787e2"


def _checked(pointer: dict[str, str]) -> Path:
    if set(pointer) != {"path", "sha256"}:
        raise ValueError("source binding must have path and SHA256")
    path = ROOT / pointer["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pointer["sha256"]:
        raise ValueError(f"source binding changed: {path}")
    return path


def _search(spec: dict[str, Any], *, maximize: bool) -> ASIDBoundedCEM:
    return ASIDBoundedCEM(
        spec["lower"], spec["upper"], spec["anchor"],
        population=int(spec["population"]), elite=int(spec["elite"]),
        iterations=int(spec["iterations"]), maximize=maximize,
    )


def plan(config: dict[str, Any]) -> dict[str, Any]:
    version = config.get("version")
    if (version not in {VERSION, GRID_VERSION} or config.get("task") != "T1"
            or config.get("robot") != "xarm7"
            or config.get("source_commit") != SOURCE_COMMIT
            or config.get("delay_steps") != 0):
        raise ValueError("pinned xArm7 T1 ASID development chain required")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 600, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("ASID development resource ceiling changed")
    if config.get("exploration_controls") != 60 or config.get("delta") != .1:
        raise ValueError("ASID exploration horizon or finite difference changed")
    if config.get("source_mechanism") != {
        "servo_gain": .85, "payload_mass": .65, "delay_steps": 0,
    } or config.get("target_mechanism") != {
        "servo_gain": .7, "payload_mass": .5, "delay_steps": 0,
    }:
        raise ValueError("T1 source/target mechanisms changed")
    if config.get("exploration") != {
        "lower": [.52, .6, .58, .55, .65, .65],
        "upper": [.62, .73, .7, .69, 1.15, 1.15],
        "anchor": [.5678590883444584, .6581064010126911,
                   .6270149599348386, .6097529670246064, 1., 1.],
        "population": 7, "elite": 2, "iterations": 2,
    }:
        raise ValueError("T1 ASID exploration search changed")
    if config.get("identification") != {
        "lower": [.4, .2], "upper": [1.4, 1.2], "anchor": [.85, .65],
        "population": 10, "elite": 3, "iterations": 3,
    }:
        raise ValueError("T1 ASID identification search changed")
    grid = config.get("identification_grid")
    if version == GRID_VERSION:
        if grid != {
            "servo_gain": [.4, .5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4],
            "payload_mass": [.2, .3, .4, .5, .6, .7, .8, .9, 1., 1.1, 1.2],
        }:
            raise ValueError("ASID global identification grid changed")
    elif grid is not None:
        raise ValueError("v1 ASID identification has no global grid")
    if config.get("task_policy") != {
        **config["exploration"], "population": 7, "elite": 2,
        "iterations": 2, "simulator_samples_per_candidate": 2,
    }:
        raise ValueError("T1 ASID downstream policy search changed")
    for spec, maximize in ((config["exploration"], True),
                           (config["identification"], False),
                           (config["task_policy"], True)):
        _search(spec, maximize=maximize)
    if (not isinstance(config.get("seed"), int) or config["seed"] < 0
            or config.get("exploration_modes") != [0, 1]
            or config.get("target_interaction_budget") != {
                "exploration_episodes": 1, "deployment_episodes": 2,
            }):
        raise ValueError("ASID mode roster or target budget changed")
    cohort = config.get("training_cohort")
    if cohort not in (None, INDEPENDENT_COHORT) or (cohort and version != GRID_VERSION):
        raise ValueError("ASID T1 independent training cohort contract differs")
    bindings = config.get("source_bindings")
    required = {
        "genedynamics/solvers/single/asid/asid.py",
        "genedynamics/solvers/single/asid/backends/asid_numpy.py",
        "genedynamics/solvers/single/asid/learning/search.py",
        "genedynamics/tasks/transfer/obstacle_asid.py",
        "genedynamics/tasks/transfer/obstacle_policy_search.py",
        "scripts/paper/cross_ebdm/train_asid_development.py",
    }
    if cohort == INDEPENDENT_COHORT:
        required |= {
            "genedynamics/core/control/transfer.py",
            "genedynamics/envs/adapters/transfer_task.py",
            "genedynamics/tasks/transfer/controllers/asid.py",
            "genedynamics/tasks/transfer/controllers/bindings.py",
            "genedynamics/tasks/transfer/controllers/obstacle_path.py",
            "genedynamics/tasks/transfer/controllers/obstacle_scaled_path.py",
            "genedynamics/tasks/transfer/episode.py",
            "genedynamics/tasks/transfer/evaluation.py",
            "genedynamics/tasks/transfer/runtime.py",
            "genedynamics/tasks/transfer/specs.py",
        }
    if not isinstance(bindings, list) or {row.get("path") for row in bindings} != required:
        raise ValueError("ASID chain requires all six exact source bindings")
    for row in bindings:
        _checked(row)
    if not (ROOT / "baselines/ASID/README.md").is_file():
        raise FileNotFoundError("pinned ASID author source checkout missing")
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"ASID output exists: {output}")
    exploration = config["exploration"]
    identification = config["identification"]
    task = config["task_policy"]
    maximum_simulator_episodes = (
        len(config["exploration_modes"]) * exploration["population"]
        * exploration["iterations"] * 5
        + identification["population"] * identification["iterations"]
        + task["population"] * task["iterations"]
        * task["simulator_samples_per_candidate"] * 2
    )
    if version == GRID_VERSION:
        maximum_simulator_episodes += len(grid["servo_gain"]) * len(grid["payload_mass"])
    return {
        "version": version, "status": "planned_unexecuted", "task": "T1",
        "scope": (
            "source-snapshotted independent ASID training instance; task-adapted CEM, not author-native SAC or formal performance"
            if cohort == INDEPENDENT_COHORT else
            "task-adapted full ASID chain; development, not author-native SAC or paper main table"
        ),
        "training_cohort": cohort,
        "source_commit": SOURCE_COMMIT, "source_bindings": bindings,
        "exploration_controls": config["exploration_controls"],
        "target_interaction_budget": config["target_interaction_budget"],
        "maximum_simulator_episodes": maximum_simulator_episodes,
        "maximum_target_episodes": 3,
        "maximum_wall_seconds": 600,
        "output": config["output"], "formal_performance_evidence": False,
        "failure_replacement": False,
    }


def _trajectory(policy: np.ndarray | None, mechanism: MechanismSpec,
                mode: int, seed: int, controls: int, *, robot: str,
                fixed_actions: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    task = instantiate_task("T1", robot=robot, mechanism_spec=mechanism, seed=seed)
    public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
    actor = None if fixed_actions is not None else PublicObstaclePathPolicy(
        task.spec, modes=[mode], path=policy[:4],
    )
    values, issued = [], []
    for tick in range(controls):
        if fixed_actions is None:
            action = np.asarray(actor.act(public), dtype=np.float32)
            action[:, :3] = np.clip(
                action[:, :3] * policy[4 + mode],
                -OBSTACLE_BOUNDS[:3], OBSTACLE_BOUNDS[:3],
            )
        else:
            action = np.asarray(fixed_actions[tick], dtype=np.float32)[None]
        issued.append(action[0].copy())
        transition = task.step(action)
        public = transition.observation
        values.append(np.r_[public["tcp_position"][0], public["tcp_twist"][0, :3]])
        if bool(transition.done[0]):
            break
    episode = task.evaluate()["episodes"][-1]
    summary = {
        "mode": mode, "seed": seed, "mechanism": {
            "servo_gain": mechanism.servo_gain,
            "payload_mass": mechanism.payload_mass,
            "delay_steps": mechanism.delay_steps,
        },
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "control_steps": len(values),
    }
    return np.asarray(values, dtype=np.float64), np.asarray(issued), summary


def _features(trace: np.ndarray) -> np.ndarray:
    """Public TCP position and one-control displacement, in millimetres."""
    return np.column_stack((trace[:, :3] * 1000., trace[:, 3:] * 20.))


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    frozen = plan(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("ASID development bound to physical GPU1")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    started = time.monotonic()
    seed = int(config["seed"])
    controls = int(config["exploration_controls"])
    source = MechanismSpec(**config["source_mechanism"])
    target = MechanismSpec(**config["target_mechanism"])
    sim_count = target_count = sim_controls = target_controls = 0
    interaction_ledger: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": frozen["version"], "status": "running", "plan": frozen,
                              "interaction_ledger": interaction_ledger}

    def charged_trace(policy, mechanism, mode, reset_seed, horizon, *, target_call=False,
                      fixed_actions=None, stage="exploration"):
        nonlocal sim_count, target_count, sim_controls, target_controls
        if time.monotonic() - started > frozen["maximum_wall_seconds"]:
            raise TimeoutError("ASID development wall ceiling exceeded")
        if target_call:
            if target_count >= frozen["maximum_target_episodes"]:
                raise RuntimeError("ASID target interaction budget exceeded")
            target_count += 1
        else:
            if sim_count >= frozen["maximum_simulator_episodes"]:
                raise RuntimeError("ASID simulator budget exceeded")
            sim_count += 1
        try:
            trace, actions, summary = _trajectory(
                policy, mechanism, mode, reset_seed, horizon, robot=config["robot"],
                fixed_actions=fixed_actions,
            )
        except BaseException as exc:
            interaction_ledger.append({"stage": stage,
                "target_interaction": target_call, "mode": mode,
                "seed": reset_seed, "error": str(exc)})
            raise
        if target_call:
            target_controls += len(trace)
        else:
            sim_controls += len(trace)
        interaction_ledger.append({"stage": stage,
            "target_interaction": target_call, **summary})
        return trace, actions, summary

    try:
        if config.get("training_cohort") == INDEPENDENT_COHORT:
            snapshots = []
            for pointer in config["source_bindings"]:
                snapshot_source = _checked(pointer)
                snapshot_target = output / "source_snapshot" / pointer["path"]
                snapshot_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(snapshot_source, snapshot_target)
                digest = hashlib.sha256(snapshot_target.read_bytes()).hexdigest()
                if digest != pointer["sha256"]:
                    raise ValueError(f"ASID T1 source changed during snapshot: {snapshot_source}")
                snapshots.append({"path": pointer["path"], "sha256": digest})
            report["source_snapshot_manifest"] = snapshots
        exploration_rows = []
        for mode in config["exploration_modes"]:
            search = _search(config["exploration"], maximize=True)

            def evaluate_exploration(candidate, iteration, index):
                reset_seed = seed + mode * 1000 + iteration * 100 + index
                source_trace, actions, source_row = charged_trace(
                    candidate, source, mode, reset_seed, controls,
                )
                rows = [source_row]
                if len(source_trace) != controls or source_row["constraint_violation"]:
                    return -1e9, {"episodes": rows, "reason": "source_exploration_unsafe"}

                def transition(state, action_stream, theta):
                    mechanism = MechanismSpec(
                        servo_gain=float(theta[0]), payload_mass=float(theta[1]),
                        delay_steps=0,
                    )
                    trace, _, row = charged_trace(
                        None, mechanism, mode, reset_seed, controls,
                        fixed_actions=action_stream,
                    )
                    rows.append(row)
                    if len(trace) < controls:
                        trace = np.concatenate((trace, np.repeat(trace[-1:], controls-len(trace), axis=0)))
                    return _features(trace).reshape(-1)

                solver = ASIDSolver(transition, delta=config["delta"], normalization=100.)
                jacobian = solver.observation_jacobian(
                    None, actions, np.asarray([source.servo_gain, source.payload_mass]),
                )
                score = float(np.clip(np.sum(jacobian * jacobian), 0., 1000.))
                if any(row["constraint_violation"] or row["control_steps"] != controls
                       for row in rows):
                    score = -1e9
                return score, {"episodes": rows,
                               "singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist()}

            result = search.run(evaluate_exploration, seed=seed + mode * 1000 + 13)
            exploration_rows.append({"mode": mode, "best_policy": result.best.tolist(),
                                     "best_information_score": result.best_score,
                                     "ledger": list(result.ledger)})
        selected = max(exploration_rows, key=lambda row: row["best_information_score"])
        if selected["best_information_score"] <= 0:
            raise RuntimeError("no safe informative T1 exploration policy found")
        mode = int(selected["mode"])
        exploration_policy = np.asarray(selected["best_policy"])
        target_trace, target_actions, target_row = charged_trace(
            exploration_policy, target, mode, seed + 40000, controls,
            target_call=True, stage="target_exploration",
        )
        report["target_exploration"] = {
            "episode": target_row,
            "public_trace": target_trace.tolist(),
            "issued_actions": target_actions.tolist(),
        }
        if len(target_trace) < 20 or target_row["constraint_violation"]:
            raise RuntimeError("target exploration unsafe or too short for identification")

        def evaluate_id(theta, iteration, index):
            mechanism = MechanismSpec(
                servo_gain=float(theta[0]), payload_mass=float(theta[1]), delay_steps=0,
            )
            trace, _, row = charged_trace(
                None, mechanism, mode, seed + 40000, len(target_trace),
                fixed_actions=target_actions, stage="identification",
            )
            length = min(len(trace), len(target_trace))
            error = float(np.mean((_features(trace[:length]) - _features(target_trace[:length]))**2))
            if len(trace) != len(target_trace) or row["constraint_violation"]:
                error += 1e6
            return error, {"episodes": [row], "public_mse_mm2": error}

        grid_ledger = []
        search_config = dict(config["identification"])
        if frozen["version"] == GRID_VERSION:
            for gain in config["identification_grid"]["servo_gain"]:
                for mass in config["identification_grid"]["payload_mass"]:
                    theta = np.asarray([gain, mass], dtype=np.float64)
                    score, details = evaluate_id(theta, -1, len(grid_ledger))
                    grid_ledger.append({"parameters": theta.tolist(),
                                        "score": score, **details})
            best_grid = min(grid_ledger, key=lambda row: row["score"])
            search_config["anchor"] = best_grid["parameters"]
        identification = _search(search_config, maximize=False)
        fit = identification.run(evaluate_id, seed=seed + 40013)
        fitted = np.clip(fit.best, identification.lower, identification.upper)
        distribution_std = np.maximum(fit.std, [.025, .025])
        task_search = _search(config["task_policy"], maximize=True)
        task_binding = ObstaclePolicySearchTask(robot=config["robot"], delay_steps=0)
        task_rng = np.random.default_rng(seed + 50001)
        simulation_parameters = [
            np.clip(task_rng.normal(fitted, distribution_std, size=(2, 2)),
                    identification.lower, identification.upper)
            for _ in range(task_search.iterations)
        ]

        def evaluate_task(policy, iteration, index):
            nonlocal sim_count, sim_controls
            rows, rewards = [], []
            for parameter_index, theta in enumerate(simulation_parameters[iteration]):
                mechanism = MechanismSpec(
                    servo_gain=float(theta[0]), payload_mass=float(theta[1]), delay_steps=0,
                )
                for task_mode in (0, 1):
                    if sim_count >= frozen["maximum_simulator_episodes"]:
                        raise RuntimeError("ASID simulator budget exceeded")
                    sim_count += 1
                    task_seed = (seed + 60000 + iteration * 10000
                                 + parameter_index * 1000 + task_mode)
                    try:
                        reward, _, steps, row = task_binding._episode(
                            policy, mechanism, task_mode, task_seed,
                        )
                    except BaseException as exc:
                        interaction_ledger.append({"stage": "task_policy_training",
                            "target_interaction": False, "mode": task_mode,
                            "seed": task_seed, "error": str(exc)})
                        raise
                    sim_controls += steps
                    interaction_ledger.append({"stage": "task_policy_training",
                        "target_interaction": False, **row})
                    rows.append({"mechanism": {
                        "servo_gain": mechanism.servo_gain,
                        "payload_mass": mechanism.payload_mass,
                        "delay_steps": 0,
                    }, **row})
                    rewards.append(reward)
            return float(np.mean(rewards)), {"episodes": rows}

        learned = task_search.run(evaluate_task, seed=seed + 50013)
        deployment = []
        for task_mode in (0, 1):
            if target_count >= frozen["maximum_target_episodes"]:
                raise RuntimeError("ASID target interaction budget exceeded")
            target_count += 1
            task_seed = seed + 70000 + task_mode
            try:
                reward, _, steps, row = task_binding._episode(
                    learned.best, target, task_mode, task_seed,
                )
            except BaseException as exc:
                interaction_ledger.append({"stage": "deployment",
                    "target_interaction": True, "mode": task_mode,
                    "seed": task_seed, "error": str(exc)})
                raise
            target_controls += steps
            interaction_ledger.append({"stage": "deployment",
                "target_interaction": True, **row})
            deployment.append({"reward": reward, **row})

        report.update({
            "status": "complete_development_chain",
            "exploration": {"candidates": exploration_rows,
                            "selected_mode": mode, "target_episode": target_row,
                            "target_actions_sha256": hashlib.sha256(target_actions.tobytes()).hexdigest()},
            "identification": {"best_parameters": fitted.tolist(),
                               "best_public_mse_mm2": fit.best_score,
                               "parameter_distribution_std": distribution_std.tolist(),
                               "global_grid_ledger": grid_ledger,
                               "ledger": list(fit.ledger)},
            "task_policy": {"best_policy": learned.best.tolist(),
                            "best_simulated_reward": learned.best_score,
                            "ledger": list(learned.ledger)},
            "deployment": deployment,
            "deployment_outcomes": dict(Counter(row["outcome"] for row in deployment)),
            "method_identity": ("ASID Fisher objective plus task-adapted CEM exploration, "
                                "predeclared grid-seeded CEM identification, and CEM "
                                "task-policy training; not author SAC"
                                if frozen["version"] == GRID_VERSION else
                                "ASID Fisher objective plus task-adapted CEM exploration, "
                                "CEM identification, and CEM task-policy training; not author SAC"),
            "formal_performance_evidence": False,
        })
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
                       "traceback": traceback.format_exc()})
        raise
    finally:
        report.update({"accounting": {
            "simulator_episodes": sim_count, "simulator_control_steps": sim_controls,
            "target_episodes": target_count, "target_control_steps": target_controls,
        }, "wall_seconds": time.monotonic() - started})
        atomic_json(output / "report.json", report)
    if config.get("training_cohort") == INDEPENDENT_COHORT:
        report_path = output / "report.json"
        fitted = report["identification"]["best_parameters"]
        checkpoint = {
            "schema_version": "asid-obstacle-public-policy-checkpoint-v1",
            "task": "T1", "robot": "xarm7",
            "policy": report["task_policy"]["best_policy"],
            "identified_mechanism": {
                "servo_gain": fitted[0], "payload_mass": fitted[1],
                "delay_steps": 0,
            },
            "source_report": {
                "path": str(report_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
            "method_identity": report["method_identity"],
            "training_target_exploration_episodes": 1,
            "training_target_exploration_controls": controls,
            "development_only": True,
        }
        atomic_json(output / "checkpoint.json", checkpoint)
    return report


def nominal_plan(config: dict[str, Any]) -> dict[str, Any]:
    if (config.get("version") != NOMINAL_VERSION or config.get("task") != "T1"
            or config.get("robot") != "xarm7"
            or config.get("requested_modes") != [0, 1]
            or config.get("reset_slots_per_cell") != 4
            or config.get("seed_anchor") != 3000200000):
        raise ValueError("frozen T1 ASID nominal roster changed")
    expected_cells = [
        {"id": "source", "mechanism": {
            "servo_gain": .85, "payload_mass": .65, "delay_steps": 0}},
        {"id": "deployment_nominal", "mechanism": {
            "servo_gain": 1., "payload_mass": .5, "delay_steps": 0}},
        {"id": "target_gain_shift", "mechanism": {
            "servo_gain": .7, "payload_mass": .5, "delay_steps": 0}},
    ]
    if config.get("scene_cells") != expected_cells:
        raise ValueError("ASID nominal scene cells changed")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 240, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("ASID nominal resource ceiling changed")
    checkpoint_path = _checked(config["checkpoint"])
    checkpoint = json.loads(checkpoint_path.read_text())
    report_path = _checked(config["training_report"])
    training = json.loads(report_path.read_text())
    if (checkpoint.get("schema_version") != "asid-obstacle-public-policy-checkpoint-v1"
            or checkpoint.get("task") != "T1" or checkpoint.get("robot") != "xarm7"
            or checkpoint.get("source_report") != config["training_report"]
            or checkpoint.get("development_only") is not True
            or training.get("status") != "complete_development_chain"
            or checkpoint.get("policy") != training["task_policy"]["best_policy"]):
        raise ValueError("ASID checkpoint does not match completed development training")
    policy = np.asarray(checkpoint["policy"], dtype=np.float64)
    anchor = np.asarray(config.get("public_anchor"), dtype=np.float64)
    if (policy.shape != (6,) or anchor.shape != (6,)
            or not np.isfinite(policy).all() or not np.isfinite(anchor).all()
            or not np.all(policy[4:] > 0) or not np.all(anchor[4:] > 0)):
        raise ValueError("ASID and public path vectors must both be finite six-dimensional policies")
    bindings = config.get("source_bindings")
    expected_paths = {
        "genedynamics/tasks/transfer/obstacle_asid.py",
        "genedynamics/tasks/transfer/controllers/obstacle_path.py",
        "scripts/paper/cross_ebdm/diagnose_asid_nominal.py",
    }
    if not isinstance(bindings, list) or {row.get("path") for row in bindings} != expected_paths:
        raise ValueError("ASID nominal diagnosis needs three source bindings")
    for row in bindings:
        _checked(row)
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"ASID nominal output exists: {output}")
    requests = []
    for cell_index, cell in enumerate(expected_cells):
        for slot in range(config["reset_slots_per_cell"]):
            for mode in config["requested_modes"]:
                reset_seed = config["seed_anchor"] + cell_index * 1000 + slot * 2 + mode
                requests.append({"scene_cell": cell["id"],
                    "mechanism": cell["mechanism"], "physical_initial_slot": slot,
                    "requested_mode": mode, "reset_seed": reset_seed})
    return {"version": NOMINAL_VERSION, "status": "planned_unexecuted", "task": "T1",
            "scope": "T1 held-out development nominal/source/target diagnosis; no main-table evidence",
            "checkpoint": config["checkpoint"], "training_report": config["training_report"],
            "source_bindings": bindings, "requests": requests,
            "methods": ["asid", "public_anchor"],
            "planned_physics_episodes": 2 * len(requests),
            "maximum_wall_seconds": 240, "output": config["output"],
            "formal_performance_evidence": False, "failure_replacement": False}


def run_nominal(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    frozen = nominal_plan(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("ASID nominal diagnosis bound to physical GPU1")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    checkpoint = json.loads(_checked(config["checkpoint"]).read_text())
    policies = {"asid": np.asarray(checkpoint["policy"], dtype=np.float64),
                "public_anchor": np.asarray(config["public_anchor"], dtype=np.float64)}
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    started = time.monotonic()
    rows = []
    report: dict[str, Any] = {"version": NOMINAL_VERSION, "status": "running",
                              "plan": frozen, "rows": rows}
    try:
        for index, request in enumerate(frozen["requests"]):
            for method, policy in policies.items():
                if time.monotonic() - started > frozen["maximum_wall_seconds"]:
                    raise TimeoutError("ASID nominal wall ceiling exceeded")
                row = {"request_index": index, "method": method,
                       "request": request, "status": "running"}
                try:
                    mechanism = MechanismSpec(**request["mechanism"])
                    task = instantiate_task("T1", robot=config["robot"],
                                            mechanism_spec=mechanism,
                                            seed=request["reset_seed"])
                    mode = int(request["requested_mode"])
                    public = task.reset((request["physical_initial_slot"],),
                                        seed=request["reset_seed"], modes=np.asarray([mode]))
                    actor = PublicObstaclePathPolicy(task.spec, modes=[mode], path=policy[:4])
                    for _ in range(task.config.timeout_steps):
                        action = np.asarray(actor.act(public), dtype=np.float32)
                        action[:, :3] = np.clip(
                            action[:, :3] * policy[4 + mode],
                            -OBSTACLE_BOUNDS[:3], OBSTACLE_BOUNDS[:3],
                        )
                        transition = task.step(action)
                        public = transition.observation
                        if bool(transition.done[0]):
                            break
                    evaluation = task.evaluate()
                    evidence = episode_evidence(task.records, task.spec, evaluation)
                    path = output / "episodes" / f"{index:02d}_{method}" / "evidence.json.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
                    path.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
                    audit = audit_episode_evidence(path)
                    if audit.get("passed") is not True:
                        raise RuntimeError("ASID nominal episode evidence audit failed")
                    episode = evaluation["episodes"][-1]
                    reset_sha = evidence["records"][0]["reset_evidence"]["physical_state_sha256"]
                    row.update({"status": "complete", "outcome": episode["outcome"],
                        "success": bool(episode["constrained_success"]),
                        "constraint_violation": bool(episode["constraint_violation"]),
                        "control_steps": int(episode["physical_control_steps"]),
                        "force_peak_N": episode["force_peak_N"],
                        "reset_sha256": reset_sha,
                        "evidence": {"path": str(path.relative_to(ROOT)),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
                except BaseException as exc:
                    row.update({"status": "failed_retained", "error": str(exc),
                                "traceback": traceback.format_exc()})
                rows.append(row)
                atomic_json(output / "progress.json", {"rows": rows})
        report.update({"status": "complete" if all(row["status"] == "complete" for row in rows)
                       else "incomplete_retained",
                       "counts": {"completed": sum(row["status"] == "complete" for row in rows),
                                  "outcomes": dict(Counter(row.get("outcome", "execution_error") for row in rows))}})
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
                       "traceback": traceback.format_exc()})
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "report.json", report)
    return report


__all__ = ["plan", "run", "nominal_plan", "run_nominal", "VERSION", "GRID_VERSION", "NOMINAL_VERSION"]
