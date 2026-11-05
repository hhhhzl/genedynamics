"""T2 task-owned public-trajectory ASID exploration and development chain.

The ASID solver supplies the published finite-difference Fisher objective.
This module owns the T2 simulator, public policy chart, evidence, and budget.
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

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.asid import ASIDSolver
from genedynamics.solvers.single.asid.learning import ASIDBoundedCEM
from genedynamics.tasks.transfer.contact import ContactMechanismSpec
from genedynamics.tasks.transfer.controllers.adaptsim import (
    GUIDE_POLICY_FIELDS, GUIDE_POLICY_LOWER, GUIDE_POLICY_UPPER,
    public_guide_actor,
)
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.guide_policy_search import GuidePolicySearchTask, _task_config


VERSION = "asid-t2-public-information-probe-v1"
TRAINING_VERSION = "asid-t2-complete-chain-development-v1"
INDEPENDENT_COHORT = "independent_main_table_training_v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
SOURCE_COMMIT = "cf2cb670fb983f67cf2af156d4e17b4a690787e2"
HORIZON = 320
FEATURES = ("tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "wrench_fx_N",
            "wrench_fy_N", "wrench_fz_N")
BASE_SOURCES = {
    "genedynamics/core/control/transfer.py",
    "genedynamics/solvers/single/asid/asid.py",
    "genedynamics/solvers/single/asid/backends/asid_numpy.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/contact_runtime.py",
    "genedynamics/tasks/transfer/guide_asid.py",
    "genedynamics/tasks/transfer/guide_policy_search.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/contact_reference.py",
    "genedynamics/tasks/transfer/runtime.py",
    "genedynamics/tasks/transfer/specs.py",
    "genedynamics/tasks/transfer/guide_evaluation.py",
    "genedynamics/tasks/transfer/routes.py",
    "genedynamics/tasks/transfer/controllers/adaptsim.py",
    "genedynamics/tasks/transfer/controllers/guide_contact.py",
    "genedynamics/tasks/transfer/controllers/guide_feedback.py",
    "genedynamics/tasks/transfer/controllers/guide_lift_route.py",
    "genedynamics/tasks/transfer/controllers/guide_pose_feedback.py",
    "genedynamics/tasks/transfer/controllers/guide_wrench_projection.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
}
PROBE_SOURCES = BASE_SOURCES | {
    "scripts/paper/cross_ebdm/diagnose_asid_t2_information.py",
}
TRAINING_SOURCES = BASE_SOURCES | {
    "genedynamics/solvers/single/asid/learning/search.py",
    "scripts/paper/cross_ebdm/train_asid_t2_development.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bindings(config: dict, root: Path) -> None:
    expected = PROBE_SOURCES
    bindings = config.get("source_bindings")
    if (not isinstance(bindings, list) or len(bindings) != len(expected)
            or {row.get("path") for row in bindings} != expected):
        raise ValueError("T2 ASID information source roster changed")
    for row in bindings:
        if set(row) != {"path", "sha256"} or _sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T2 ASID source binding changed: {row.get('path')}")


def plan(config: dict, root: Path) -> dict:
    if (config.get("version") != VERSION or config.get("task") != "T2"
            or config.get("robot") != "xarm7"
            or config.get("source_commit") != SOURCE_COMMIT
            or config.get("source_mechanism") != {
                "servo_gain": .85, "payload_mass": .65, "delay_steps": 0,
                "contact_time_constant": .04,
            }
            or config.get("delta") != .1
            or config.get("horizon_controls") != HORIZON
            or config.get("feature_fields") != list(FEATURES)
            or config.get("policy_fields") != list(GUIDE_POLICY_FIELDS)
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 900, "maximum_gpu_memory_gib": 2,
            }
            or config.get("maximum_physics_episodes") != 30
            or config.get("maximum_physical_control_steps") != 30 * HORIZON
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T2 ASID public information contract changed")
    _task_config(config, root)
    if not (root / "baselines/ASID/README.md").is_file():
        raise FileNotFoundError("pinned ASID author source checkout missing")
    _source_bindings(config, root)
    policies = config.get("policies", [])
    if len(policies) != 3 or [row.get("name") for row in policies] != [
            "anchor", "conservative", "aggressive"]:
        raise ValueError("T2 ASID public policy roster changed")
    for row in policies:
        values = np.asarray(row.get("values"), dtype=np.float64)
        if (set(row) != {"name", "values"} or values.shape != (3,)
                or not np.isfinite(values).all()
                or np.any(values < GUIDE_POLICY_LOWER)
                or np.any(values > GUIDE_POLICY_UPPER)):
            raise ValueError("T2 ASID public policy outside frozen support")
    if config.get("requested_modes") != [0, 1] or config.get("seeds") != [
            3302000000, 3302000001]:
        raise ValueError("T2 ASID fresh public mode/seed roster changed")
    output = config.get("output", "")
    if (output.split("/")[:5] != [
            "results", "arm", "guided_contact_alignment", "baseline", "asid"]
            or (root / output).exists()):
        raise ValueError("T2 ASID output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted",
        "task": "T2", "source_commit": SOURCE_COMMIT,
        "source_bindings": config["source_bindings"],
        "maximum_physics_episodes": 30,
        "maximum_physical_control_steps": 30 * HORIZON,
        "maximum_wall_seconds": 900,
        "output": output, "failure_replacement": False,
        "formal_performance_evidence": False,
        "scope": "ASID T2 public exploration information probe; no target interaction, identification, or trained task actor",
    }


def _public_feature(public: dict) -> np.ndarray:
    position = np.asarray(public["tcp_position"][0], dtype=np.float64)
    wrench = np.asarray(public["measured_wrench"][0, :3], dtype=np.float64)
    return np.r_[position * 1000., wrench]


def _trajectory(config: dict, root: Path, policy: np.ndarray | None,
                mechanism: ContactMechanismSpec, mode: int, seed: int,
                output: Path, *, fixed_actions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    overrides, shared = _task_config(config, root)
    task = instantiate_task(
        "T2", robot="xarm7", seed=seed, mechanism_spec=mechanism,
        config_overrides=overrides, shared_execution=shared,
    )
    public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
    actor = None if fixed_actions is not None else public_guide_actor(
        task, policy, mode=mode
    )
    values, actions = [], []
    for tick in range(HORIZON):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = (np.asarray(actor.act(public), dtype=np.float32)
                  if fixed_actions is None else
                  np.asarray(fixed_actions[tick], dtype=np.float32)[None])
        actions.append(action[0].copy())
        public = task.step(action).observation
        values.append(_public_feature(public))
    if not values:
        raise ValueError("T2 ASID trajectory has no physical transitions")
    evaluation = task.evaluate()
    evidence = episode_evidence(task.records, task.spec, evaluation)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_bytes(gzip.compress(
        json.dumps(evidence, sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    audit = audit_episode_evidence(output)
    if not audit["passed"]:
        raise ValueError("T2 ASID physical evidence failed CPU reevaluation")
    episode = evaluation["episodes"][-1]
    summary = {
        "mode": mode, "reset_seed": seed,
        "mechanism": {
            "servo_gain": mechanism.servo_gain,
            "payload_mass": mechanism.payload_mass,
            "delay_steps": mechanism.delay_steps,
            "contact_time_constant": mechanism.contact_time_constant,
        },
        "physical_control_steps": len(values),
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "first_force_over_1_N_control": next(
            (i for i, trace in enumerate(task.records[-1].traces)
             if float(trace["force_peak"]) > 1.), None,
        ),
        "evidence": {"path": str(output.relative_to(root)),
                     "sha256": _sha(output),
                     "reset_physical_hash": audit["reset_physical_hashes"][-1]},
    }
    response = np.asarray(values, dtype=np.float64)
    if len(response) < HORIZON:
        response = np.concatenate((response, np.repeat(
            response[-1:], HORIZON - len(response), axis=0,
        )))
    return response, np.asarray(actions, dtype=np.float32), summary


def run(config: dict, root: Path) -> dict:
    import torch

    frozen = plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T2 ASID information probe requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T2 ASID information probe saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    for binding in config["source_bindings"]:
        source = root / binding["path"]
        snapshot = output / "source_snapshot" / binding["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        if _sha(snapshot) != binding["sha256"]:
            raise ValueError("T2 ASID source snapshot changed during launch")
    started = time.monotonic()
    charged = 0
    episodes = 0
    rows = []
    result = {**frozen, "status": "running"}
    source = ContactMechanismSpec(**config["source_mechanism"])

    def charged_trajectory(policy, mechanism, mode, seed, path, *, fixed_actions=None):
        nonlocal charged, episodes
        if (time.monotonic() - started > frozen["maximum_wall_seconds"]
                or episodes >= frozen["maximum_physics_episodes"]):
            raise TimeoutError("T2 ASID information physical or wall ceiling reached")
        episodes += 1
        response, actions, row = _trajectory(
            config, root, policy, mechanism, mode, seed, path,
            fixed_actions=fixed_actions,
        )
        charged += row["physical_control_steps"]
        if charged > frozen["maximum_physical_control_steps"]:
            raise ValueError("T2 ASID physical control budget exceeded")
        return response, actions, row

    try:
        for policy in config["policies"]:
            for mode, seed in zip(config["requested_modes"], config["seeds"], strict=True):
                base = output / "episodes" / f"{policy['name']}_m{mode}"
                response, actions, source_row = charged_trajectory(
                    np.asarray(policy["values"], dtype=np.float64), source,
                    mode, seed, base / "source" / "evidence.json.gz",
                )
                perturbations = []
                if (source_row["physical_control_steps"] < HORIZON
                        or source_row["constraint_violation"]):
                    rows.append({
                        "policy": policy, "mode": mode, "seed": seed,
                        "source": source_row, "perturbations": perturbations,
                        "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                        "valid_full_horizon_and_no_constraint_violation": False,
                        "invalid_reason": "source_exploration_terminated_or_violated_constraint",
                        "information_score": None,
                        "precontact_singular_values": None,
                        "contact_window_singular_values": None,
                        "full_singular_values": None,
                        "parameter_column_norms": None,
                        "position_column_norms": None,
                        "wrench_column_norms": None,
                    })
                    atomic_json(output / "progress.json", {
                        "status": "running", "completed_policy_mode_cells": len(rows),
                        "physical_episodes": episodes, "physical_control_steps": charged,
                    })
                    continue

                def transition(state, action_stream, theta):
                    del state
                    parameter = np.asarray(theta, dtype=np.float64)
                    if parameter.shape != (2,):
                        raise ValueError("ASID T2 Fisher parameter shape changed")
                    mechanism = ContactMechanismSpec(
                        servo_gain=float(parameter[0]),
                        payload_mass=float(parameter[1]), delay_steps=0,
                        contact_time_constant=.04,
                    )
                    index = len(perturbations)
                    trace, _, row = charged_trajectory(
                        None, mechanism, mode, seed,
                        base / f"perturb_{index:02d}" / "evidence.json.gz",
                        fixed_actions=action_stream,
                    )
                    perturbations.append(row)
                    return trace.reshape(-1)

                solver = ASIDSolver(transition, delta=config["delta"])
                jacobian = solver.observation_jacobian(
                    None, actions, np.asarray([source.servo_gain, source.payload_mass]),
                ).reshape(HORIZON, len(FEATURES), 2)
                valid = (source_row["physical_control_steps"] == HORIZON
                         and not source_row["constraint_violation"]
                         and all(row["physical_control_steps"] == HORIZON
                                 and not row["constraint_violation"]
                                 for row in perturbations))
                pre = jacobian[:220].reshape(-1, 2)
                contact = jacobian[220:].reshape(-1, 2)
                rows.append({
                    "policy": policy, "mode": mode, "seed": seed,
                    "source": source_row, "perturbations": perturbations,
                    "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "valid_full_horizon_and_no_constraint_violation": bool(valid),
                    "information_score": float(np.sum(jacobian * jacobian)),
                    "precontact_singular_values": np.linalg.svd(pre, compute_uv=False).tolist(),
                    "contact_window_singular_values": np.linalg.svd(contact, compute_uv=False).tolist(),
                    "full_singular_values": np.linalg.svd(jacobian.reshape(-1, 2), compute_uv=False).tolist(),
                    "parameter_column_norms": np.linalg.norm(jacobian.reshape(-1, 2), axis=0).tolist(),
                    "position_column_norms": np.linalg.norm(jacobian[:, :3].reshape(-1, 2), axis=0).tolist(),
                    "wrench_column_norms": np.linalg.norm(jacobian[:, 3:].reshape(-1, 2), axis=0).tolist(),
                })
                atomic_json(output / "progress.json", {
                    "status": "running", "completed_policy_mode_cells": len(rows),
                    "physical_episodes": episodes, "physical_control_steps": charged,
                })
        result.update({"status": "complete_development_probe", "rows": rows})
    except BaseException as error:
        result.update({"status": "failed_retained", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc(),
                       "rows": rows})
    finally:
        result.update({"physical_episodes_charged": episodes,
                       "physical_control_steps_charged": charged,
                       "wall_seconds": time.monotonic() - started,
                       "recipe_sha256": _sha(output / "recipe.json")})
        atomic_json(output / "report.json", result)
    return result


def training_plan(config: dict, root: Path) -> dict:
    """Freeze a complete exploration, identification, and retraining chain."""
    cohort = config.get("training_cohort")
    seed = config.get("training_seed")
    if cohort is None:
        seeds_valid = (
            seed == 8764601
            and config.get("target_exploration_seed") == 3302100000
            and config.get("target_deployment_seeds") == [3302101000, 3302101001]
        )
    elif cohort == INDEPENDENT_COHORT:
        target_seed = 3402000000 + (seed - 8800000) * 10 if isinstance(seed, int) else -1
        seeds_valid = (
            isinstance(seed, int) and 8802001 <= seed <= 8802008
            and config.get("target_exploration_seed") == target_seed
            and config.get("target_deployment_seeds") == [target_seed + 1, target_seed + 2]
            and config.get("output", "").endswith(f"/independent_training_{seed}")
        )
    else:
        seeds_valid = False
    if (config.get("version") != TRAINING_VERSION or config.get("task") != "T2"
            or config.get("robot") != "xarm7"
            or config.get("source_commit") != SOURCE_COMMIT
            or config.get("source_mechanism") != {
                "servo_gain": .85, "payload_mass": .65, "delay_steps": 0,
                "contact_time_constant": .04,
            }
            or config.get("target_mechanism") != {
                "servo_gain": .8, "payload_mass": .6, "delay_steps": 1,
                "contact_time_constant": .04,
            }
            or not seeds_valid
            or config.get("target_exploration_mode") != 0
            or config.get("exploration_controls") != HORIZON
            or config.get("feature_fields") != list(FEATURES)
            or config.get("policy_fields") != list(GUIDE_POLICY_FIELDS)
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 4800, "maximum_gpu_memory_gib": 2,
            }
            or config.get("target_interaction_budget") != {
                "exploration_episodes": 1, "deployment_episodes": 2,
            }
            or config.get("maximum_simulator_episodes") != 232
            or config.get("maximum_target_episodes") != 3
            or config.get("maximum_physical_control_steps") != 103760
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T2 ASID complete-chain development contract changed")
    _task_config(config, root)
    if not (root / "baselines/ASID/README.md").is_file():
        raise FileNotFoundError("pinned ASID author source checkout missing")
    if (len(config.get("source_bindings", [])) != len(TRAINING_SOURCES)
            or {row.get("path") for row in config["source_bindings"]} != TRAINING_SOURCES):
        raise ValueError("T2 ASID training source roster changed")
    for row in config["source_bindings"]:
        if set(row) != {"path", "sha256"} or _sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T2 ASID training source SHA changed: {row.get('path')}")
    exploration = config.get("exploration")
    if exploration != {
        "lower": GUIDE_POLICY_LOWER.tolist(),
        "upper": GUIDE_POLICY_UPPER.tolist(),
        "anchor": [.02, .008, .03],
        "population": 4, "elite": 2, "iterations": 2,
        "modes": [0, 1], "normalization": 100.,
    }:
        raise ValueError("T2 ASID exploration chart or Fisher normalization changed")
    if config.get("identification_grid") != {
        "servo_gain": [.6, .7, .8, .9, 1.],
        "payload_mass": [.4, .5, .6, .7, .8],
        "delay_steps": [0, 1, 2, 3],
    }:
        raise ValueError("T2 ASID discrete-delay identification grid changed")
    if config.get("identification") != {
        "lower": [.4, .2], "upper": [1.4, 1.2],
        "population": 6, "elite": 2, "iterations": 2,
        "delta": .1,
    }:
        raise ValueError("T2 ASID local continuous identification budget changed")
    task = config.get("task_policy")
    if task != {
        "lower": GUIDE_POLICY_LOWER.tolist(),
        "upper": GUIDE_POLICY_UPPER.tolist(),
        "anchor": [.02, .008, .03],
        "population": 5, "elite": 2, "iterations": 2,
        "simulator_samples_per_candidate": 2,
    }:
        raise ValueError("T2 ASID downstream task-policy search changed")
    for spec, maximize in ((exploration, True), (task, True)):
        ASIDBoundedCEM(
            spec["lower"], spec["upper"], spec["anchor"],
            population=spec["population"], elite=spec["elite"],
            iterations=spec["iterations"], maximize=maximize,
        )
    output = config.get("output", "")
    if (output.split("/")[:5] != [
            "results", "arm", "guided_contact_alignment", "baseline", "asid"]
            or (root / output).exists()):
        raise ValueError("T2 ASID training output must be fresh and task-local")
    return {
        "version": TRAINING_VERSION, "status": "planned_unexecuted",
        "task": "T2", "source_commit": SOURCE_COMMIT,
        "training_cohort": cohort,
        "source_bindings": config["source_bindings"],
        "maximum_simulator_episodes": 232,
        "maximum_target_episodes": 3,
        "maximum_physical_control_steps": 103760,
        "maximum_wall_seconds": 4800,
        "target_interaction_budget": config["target_interaction_budget"],
        "output": output, "failure_replacement": False,
        "formal_performance_evidence": False,
        "method_identity": "ASID_Fisher_exploration_grid_and_CEM_identification_CEM_task_policy",
        "scope": "complete task-adapted ASID chain using the published Fisher objective; task CEM replaces the unreleased T2 SAC training, development only",
    }


def _full_task_episode(config: dict, root: Path, policy: np.ndarray,
                       mechanism: ContactMechanismSpec, mode: int, seed: int,
                       evidence_path: Path) -> tuple[float, dict]:
    overrides, shared = _task_config(config, root)
    task = instantiate_task(
        "T2", robot="xarm7", mechanism_spec=mechanism, seed=seed,
        config_overrides=overrides, shared_execution=shared,
    )
    public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
    actor = public_guide_actor(task, policy, mode=mode)
    for _ in range(task.config.timeout_steps):
        if bool(np.asarray(task.state.status)[0]):
            break
        public = task.step(actor.act(public)).observation
    evaluation = task.evaluate()
    episode = evaluation["episodes"][-1]
    evidence_path.parent.mkdir(parents=True, exist_ok=False)
    evidence_path.write_bytes(gzip.compress(
        json.dumps(episode_evidence(task.records, task.spec, evaluation),
                   sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    audit = audit_episode_evidence(evidence_path)
    if not audit["passed"]:
        raise ValueError("T2 ASID deployment evidence failed CPU rescoring")
    controls = int(episode["physical_control_steps"])
    error = float(episode["final_position_error_m"])
    if bool(episode["success"]):
        reward = 1. - .3 * controls / task.config.timeout_steps
    elif bool(episode["constraint_violation"]):
        reward = -1. - min(error, 1.)
    else:
        reward = -min(error, 1.) - .2 * controls / task.config.timeout_steps
    return reward, {
        "mode": mode, "reset_seed": seed,
        "mechanism": {
            "servo_gain": mechanism.servo_gain,
            "payload_mass": mechanism.payload_mass,
            "delay_steps": mechanism.delay_steps,
            "contact_time_constant": mechanism.contact_time_constant,
        },
        "outcome": episode["outcome"],
        "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "control_steps": controls,
        "final_position_error_m": error, "reward": reward,
        "evidence": {"path": str(evidence_path.relative_to(root)),
                     "sha256": _sha(evidence_path),
                     "reset_physical_hash": audit["reset_physical_hashes"][-1]},
    }


def run_training(config: dict, root: Path) -> dict:
    """Execute one budgeted T2 ASID chain with distinct simulator/target ledgers."""
    import torch

    frozen = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T2 ASID training requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T2 ASID training saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    for binding in config["source_bindings"]:
        source_path = root / binding["path"]
        snapshot = output / "source_snapshot" / binding["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot)
        if _sha(snapshot) != binding["sha256"]:
            raise ValueError("T2 ASID training source snapshot changed during launch")
    started = time.monotonic()
    sim_episodes = target_episodes = sim_controls = target_controls = 0
    interaction_ledger = []
    report = {**frozen, "status": "running", "interaction_ledger": interaction_ledger}
    seed = int(config["training_seed"])
    source = ContactMechanismSpec(**config["source_mechanism"])
    target = ContactMechanismSpec(**config["target_mechanism"])

    def progress(stage: str) -> None:
        atomic_json(output / "progress.json", {
            "status": "running", "stage": stage,
            "wall_seconds": time.monotonic() - started,
            "simulator_episodes": sim_episodes,
            "simulator_control_steps": sim_controls,
            "target_episodes": target_episodes,
            "target_control_steps": target_controls,
        })

    progress("source_snapshot_complete")

    def charge(stage, summary, *, target_call):
        nonlocal sim_controls, target_controls
        controls = int(summary.get("physical_control_steps", summary.get("control_steps", 0)))
        if target_call:
            target_controls += controls
        else:
            sim_controls += controls
        interaction_ledger.append({"stage": stage, "target_interaction": target_call,
                                   **summary})
        if sim_controls + target_controls > frozen["maximum_physical_control_steps"]:
            raise ValueError("T2 ASID physical control budget exceeded")
        progress(stage)

    def check_budget(*, target_call):
        nonlocal sim_episodes, target_episodes
        if time.monotonic() - started > frozen["maximum_wall_seconds"]:
            raise TimeoutError("T2 ASID development wall ceiling reached")
        if target_call:
            if target_episodes >= frozen["maximum_target_episodes"]:
                raise RuntimeError("T2 ASID target interaction budget exceeded")
            target_episodes += 1
        else:
            if sim_episodes >= frozen["maximum_simulator_episodes"]:
                raise RuntimeError("T2 ASID simulator episode budget exceeded")
            sim_episodes += 1

    def charged_trajectory(stage, policy, mechanism, mode, reset_seed, evidence_path,
                           *, fixed_actions=None, target_call=False):
        check_budget(target_call=target_call)
        response, actions, summary = _trajectory(
            config, root, policy, mechanism, mode, reset_seed,
            evidence_path, fixed_actions=fixed_actions,
        )
        charge(stage, summary, target_call=target_call)
        return response, actions, summary

    def charged_task(stage, policy, mechanism, mode, reset_seed, task_binding):
        check_budget(target_call=False)
        reward, _, controls, summary = task_binding._episode(
            policy, mechanism, mode, reset_seed,
        )
        if controls != summary["control_steps"]:
            raise ValueError("T2 ASID task-policy simulator ledger differs")
        charge(stage, summary, target_call=False)
        return reward, summary

    try:
        search_cfg = config["exploration"]
        explorer = ASIDBoundedCEM(
            search_cfg["lower"], search_cfg["upper"], search_cfg["anchor"],
            population=search_cfg["population"], elite=search_cfg["elite"],
            iterations=search_cfg["iterations"], maximize=True,
        )

        def evaluate_exploration(candidate, iteration, index):
            rows, mode_scores = [], []
            for mode in search_cfg["modes"]:
                reset_seed = seed + iteration * 10000 + index * 10 + mode
                base = output / "episodes" / "exploration" / f"i{iteration}_c{index}_m{mode}"
                _, actions, source_row = charged_trajectory(
                    "exploration_source", candidate, source, mode, reset_seed,
                    base / "source" / "evidence.json.gz",
                )
                rows.append(source_row)
                if (source_row["physical_control_steps"] != HORIZON
                        or source_row["constraint_violation"]):
                    mode_scores.append(-1e9)
                    continue
                perturbations = []

                def transition(state, action_stream, theta):
                    del state
                    mechanism = ContactMechanismSpec(
                        servo_gain=float(theta[0]),
                        payload_mass=float(theta[1]),
                        delay_steps=0, contact_time_constant=.04,
                    )
                    response, _, summary = charged_trajectory(
                        "exploration_perturbation", None, mechanism, mode,
                        reset_seed, base / f"perturb_{len(perturbations):02d}" / "evidence.json.gz",
                        fixed_actions=action_stream,
                    )
                    rows.append(summary)
                    perturbations.append(summary)
                    return response.reshape(-1)

                fisher = ASIDSolver(
                    transition, delta=.1,
                    normalization=search_cfg["normalization"],
                    max_reward=1000.,
                )
                score = fisher.information_reward(
                    None, actions,
                    np.asarray([source.servo_gain, source.payload_mass]),
                )
                if any(row["physical_control_steps"] != HORIZON
                       or row["constraint_violation"] for row in perturbations):
                    score = -1e9
                mode_scores.append(score)
            return float(np.mean(mode_scores)), {
                "episodes": rows, "mode_information_scores": mode_scores,
                "safe_both_modes": all(score >= 0. for score in mode_scores),
            }

        explored = explorer.run(evaluate_exploration, seed=seed + 13)
        report["exploration"] = {
            "best_policy": explored.best.tolist(),
            "best_information_score": explored.best_score,
            "normalization": search_cfg["normalization"],
            "ledger": list(explored.ledger),
        }
        if explored.best_score <= 0:
            raise RuntimeError("no safe informative two-mode T2 ASID exploration policy")

        target_response, target_actions, target_row = charged_trajectory(
            "target_exploration", explored.best, target,
            config["target_exploration_mode"],
            config["target_exploration_seed"],
            output / "episodes" / "target_exploration" / "evidence.json.gz",
            target_call=True,
        )
        report["target_exploration"] = {
            "episode": target_row,
            "public_response": target_response.tolist(),
            "requested_actions_sha256": hashlib.sha256(target_actions.tobytes()).hexdigest(),
        }
        if (target_row["physical_control_steps"] != HORIZON
                or target_row["constraint_violation"]):
            raise RuntimeError("T2 ASID target exploration was unsafe or ended early")

        identification_rows = []

        def evaluate_id(theta, delay, label):
            mechanism = ContactMechanismSpec(
                servo_gain=float(theta[0]), payload_mass=float(theta[1]),
                delay_steps=int(delay), contact_time_constant=.04,
            )
            response, _, summary = charged_trajectory(
                "identification", None, mechanism,
                config["target_exploration_mode"],
                config["target_exploration_seed"],
                output / "episodes" / "identification" / label / "evidence.json.gz",
                fixed_actions=target_actions,
            )
            mse = float(np.mean((response - target_response) ** 2))
            if summary["physical_control_steps"] != HORIZON or summary["constraint_violation"]:
                mse += 1e6
            identification_rows.append({"label": label,
                "parameters": [float(theta[0]), float(theta[1]), int(delay)],
                "score": mse, "episode": summary})
            return mse, {"episodes": [summary], "public_mse": mse}

        grid = config["identification_grid"]
        for gain in grid["servo_gain"]:
            for mass in grid["payload_mass"]:
                for delay in grid["delay_steps"]:
                    evaluate_id(
                        np.asarray([gain, mass], dtype=np.float64), delay,
                        f"grid_g{gain:.1f}_m{mass:.1f}_d{delay}",
                    )
        best_grid = min(identification_rows, key=lambda row: row["score"])
        selected_delay = int(best_grid["parameters"][2])
        id_spec = config["identification"]
        identifier = ASIDBoundedCEM(
            id_spec["lower"], id_spec["upper"],
            best_grid["parameters"][:2],
            population=id_spec["population"], elite=id_spec["elite"],
            iterations=id_spec["iterations"], maximize=False,
        )

        def evaluate_local(theta, iteration, index):
            return evaluate_id(theta, selected_delay,
                               f"cem_i{iteration}_c{index}")

        identified = identifier.run(evaluate_local, seed=seed + 40013)
        fitted = np.asarray(identified.best, dtype=np.float64)
        report["identification"] = {
            "best_grid": best_grid,
            "selected_delay_steps": selected_delay,
            "best_parameters": [*fitted.tolist(), selected_delay],
            "best_public_mse": identified.best_score,
            "local_search_std": identified.std.tolist(),
            "grid_and_local_ledger": identification_rows,
            "local_search_ledger": list(identified.ledger),
        }

        task_cfg = config["task_policy"]
        trainer = ASIDBoundedCEM(
            task_cfg["lower"], task_cfg["upper"], task_cfg["anchor"],
            population=task_cfg["population"], elite=task_cfg["elite"],
            iterations=task_cfg["iterations"], maximize=True,
        )
        overrides, shared = _task_config(config, root)
        task_binding = GuidePolicySearchTask(
            config_overrides=overrides, shared_execution=shared,
            deadline=started + frozen["maximum_wall_seconds"],
        )
        rng = np.random.default_rng(seed + 50001)
        std = np.maximum(identified.std, [.05, .05])
        training_tasks = []
        for iteration in range(task_cfg["iterations"]):
            sample = np.clip(
                rng.normal(fitted, std, size=(2, 2)),
                id_spec["lower"], id_spec["upper"],
            )
            training_tasks.append([{
                "mechanism": {
                    "servo_gain": float(theta[0]),
                    "payload_mass": float(theta[1]),
                    "delay_steps": selected_delay,
                    "contact_time_constant": .04,
                },
                "seeds": [int(value) for value in rng.integers(0, 2**31, size=2)],
            } for theta in sample])

        def evaluate_task(candidate, iteration, index):
            rows, rewards = [], []
            for task_row in training_tasks[iteration]:
                mechanism = ContactMechanismSpec(**task_row["mechanism"])
                for mode, reset_seed in enumerate(task_row["seeds"]):
                    reward, summary = charged_task(
                        "task_policy_training", candidate, mechanism,
                        mode, reset_seed, task_binding,
                    )
                    rows.append({"mechanism": task_row["mechanism"], **summary})
                    rewards.append(reward)
            return float(np.mean(rewards)), {"episodes": rows}

        learned = trainer.run(evaluate_task, seed=seed + 50013)
        report["task_policy"] = {
            "best_policy": learned.best.tolist(),
            "best_simulated_reward": learned.best_score,
            "training_tasks": training_tasks,
            "ledger": list(learned.ledger),
        }
        deployment = []
        for mode, reset_seed in enumerate(config["target_deployment_seeds"]):
            check_budget(target_call=True)
            reward, summary = _full_task_episode(
                config, root, learned.best, target, mode, reset_seed,
                output / "episodes" / "target_deployment" / f"mode_{mode}" / "evidence.json.gz",
            )
            if reward != summary["reward"]:
                raise ValueError("T2 ASID target reward receipt differs")
            charge("target_deployment", summary, target_call=True)
            deployment.append(summary)
        report.update({
            "status": "complete_development_chain",
            "deployment": deployment,
            "deployment_outcomes": dict(Counter(row["outcome"] for row in deployment)),
            "formal_performance_evidence": False,
        })
    except BaseException as error:
        report.update({"status": "failed_retained",
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc()})
    finally:
        report.update({
            "accounting": {
                "simulator_episodes": sim_episodes,
                "simulator_control_steps": sim_controls,
                "target_episodes": target_episodes,
                "target_control_steps": target_controls,
            },
            "wall_seconds": time.monotonic() - started,
            "recipe_sha256": _sha(output / "recipe.json"),
        })
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "status": report["status"],
            **report["accounting"],
            "wall_seconds": report["wall_seconds"],
        })
    if report["status"] == "complete_development_chain":
        checkpoint = {
            "schema_version": "asid-guide-public-policy-checkpoint-v1",
            "task": "T2", "robot": "xarm7", "development_only": True,
            "policy": report["task_policy"]["best_policy"],
            "identified_mechanism": report["identification"]["best_parameters"],
            "source_report": {"path": str((output / "report.json").relative_to(root)),
                              "sha256": _sha(output / "report.json")},
        }
        atomic_json(output / "checkpoint.json", checkpoint)
        atomic_json(output / "checkpoint_receipt.json", {
            "report": checkpoint["source_report"],
            "checkpoint": {"path": str((output / "checkpoint.json").relative_to(root)),
                           "sha256": _sha(output / "checkpoint.json")},
        })
    return report
