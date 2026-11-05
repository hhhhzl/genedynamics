"""Task-owned T3 pre-overforce information probe for ASID exploration."""
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
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.mating_policy_search import (
    MatingPolicySearchTask, POLICY_FIELDS,
)


VERSION = "asid-t3-preoverforce-information-v1"
TRAINING_VERSION = "asid-t3-complete-chain-development-v1"
INDEPENDENT_COHORT = "independent_main_table_training_v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
HORIZON = 32
FEATURES = ("tcp_x_mm", "tcp_y_mm", "tcp_z_mm", "wrench_fx_N",
            "wrench_fy_N", "wrench_fz_N")
POLICIES = (
    {"name": "anchor", "values": [.25, 6., 2., .008, .004, 1., 0.]},
    {"name": "guarded", "values": [.5, 4., .8, .004, .002, .8, .08]},
    {"name": "fast", "values": [.1, 8., 2., .01, .006, 1.2, 0.]},
)
SOURCE_FILES = {
    "genedynamics/core/control/transfer.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/contact_runtime.py",
    "genedynamics/solvers/single/asid/asid.py",
    "genedynamics/solvers/single/asid/backends/asid_numpy.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/contact_reference.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/__init__.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
    "genedynamics/tasks/transfer/mating_asid.py",
    "genedynamics/tasks/transfer/mating_policy_search.py",
    "genedynamics/tasks/transfer/routes.py",
    "genedynamics/tasks/transfer/runtime.py",
    "genedynamics/tasks/transfer/specs.py",
    "scripts/paper/cross_ebdm/diagnose_asid_t3_information.py",
}
TRAINING_SOURCES = (SOURCE_FILES - {
    "scripts/paper/cross_ebdm/diagnose_asid_t3_information.py",
}) | {
    "genedynamics/solvers/single/asid/learning/search.py",
    "scripts/paper/cross_ebdm/train_asid_t3_development.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(config: dict, root: Path) -> dict:
    if (config.get("version") != VERSION or config.get("task") != "T3"
            or config.get("robot") != "xarm7"
            or config.get("source_mechanism") != {
                "servo_gain": .85, "payload_mass": .65,
                "delay_steps": 0, "contact_time_constant": .02,
            }
            or config.get("delta") != .1
            or config.get("horizon_controls") != HORIZON
            or config.get("feature_fields") != list(FEATURES)
            or config.get("policy_fields") != list(POLICY_FIELDS)
            or config.get("policies") != list(POLICIES)
            or config.get("modes") != [0, 1]
            or config.get("seeds") != [3302300000, 3302300001]
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 900, "maximum_gpu_memory_gib": 2,
            }
            or config.get("maximum_physics_episodes") != 30
            or config.get("maximum_physical_control_steps") != 960
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T3 ASID pre-overforce information contract changed")
    pointer = config.get("task_config", {})
    if (set(pointer) != {"path", "sha256"}
            or pointer["path"] != "configs/arm/keyed_insertion_recovery/_base.yaml"
            or sha(root / pointer["path"]) != pointer["sha256"]):
        raise ValueError("T3 ASID task contract binding changed")
    from genedynamics.experiments.framework import ExperimentConfig
    task_config = ExperimentConfig.from_yaml(root / pointer["path"])
    if (task_config.env_name != "keyed_insertion_recovery"
            or task_config.env_params["shared_execution"] != config.get("shared_execution")):
        raise ValueError("T3 ASID shared execution differs")
    basis_pointer = config.get("public_policy_basis", {})
    if (set(basis_pointer) != {"path", "sha256"}
            or basis_pointer["path"] !=
            "configs/arm/keyed_insertion_recovery/training/adaptsim_feedback_population_development.json"
            or sha(root / basis_pointer["path"]) != basis_pointer["sha256"]):
        raise ValueError("T3 ASID common public-policy basis changed")
    basis = json.loads((root / basis_pointer["path"]).read_text())
    if (basis["feedback_base"] != config["feedback_base"]
            or basis["shared_execution"] != config["shared_execution"]
            or basis["policy_fields"] != config["policy_fields"]):
        raise ValueError("T3 ASID public policy chart differs from frozen basis")
    search = MatingPolicySearchTask(
        robot="xarm7", feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
    )
    for policy in POLICIES:
        search.feedback(policy["values"])
    bindings = config.get("source_bindings", [])
    if (len(bindings) != len(SOURCE_FILES)
            or {row.get("path") for row in bindings} != SOURCE_FILES):
        raise ValueError("T3 ASID source roster differs")
    for row in bindings:
        if set(row) != {"path", "sha256"} or sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T3 ASID source SHA differs: {row.get('path')}")
    output = config.get("output", "")
    if (output.split("/")[:5] != [
            "results", "arm", "keyed_insertion_recovery", "baseline", "asid"]
            or (root / output).exists()):
        raise ValueError("T3 ASID diagnostic output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted",
        "task": "T3", "output": output,
        "maximum_physics_episodes": 30,
        "maximum_physical_control_steps": 960,
        "maximum_wall_seconds": 900,
        "source_bindings": bindings,
        "failure_replacement": False,
        "formal_performance_evidence": False,
        "scope": "T3 public pre-overforce Fisher information only; no identification, task training, or target evaluation",
    }


def _feature(public: dict) -> np.ndarray:
    return np.r_[
        np.asarray(public["tcp_position"][0], dtype=np.float64) * 1000.,
        np.asarray(public["measured_wrench"][0, :3], dtype=np.float64),
    ]


def _trajectory(config: dict, root: Path, policy: np.ndarray | None,
                mechanism: ContactMechanismSpec, mode: int, seed: int,
                evidence_path: Path, *, fixed_actions: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    task = instantiate_task(
        "T3", robot="xarm7", mechanism_spec=mechanism, seed=seed,
        shared_execution=config["shared_execution"],
    )
    public = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([mode]))
    actor = None
    if fixed_actions is None:
        search = MatingPolicySearchTask(
            robot="xarm7", feedback_base=config["feedback_base"],
            shared_execution=config["shared_execution"],
        )
        actor = PublicLivenessMatingFeedbackPolicy(
            task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
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
        values.append(_feature(public))
    if not values:
        raise ValueError("T3 ASID trajectory has no physical transitions")
    evaluation = task.evaluate()
    evidence_path.parent.mkdir(parents=True, exist_ok=False)
    evidence_path.write_bytes(gzip.compress(
        json.dumps(episode_evidence(task.records, task.spec, evaluation),
                   sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    evidence_audit = audit_episode_evidence(evidence_path)
    if not evidence_audit["passed"]:
        raise ValueError("T3 ASID physical evidence failed CPU reevaluation")
    episode = evaluation["episodes"][-1]
    response = np.asarray(values, dtype=np.float64)
    if len(response) < HORIZON:
        response = np.concatenate((response, np.repeat(
            response[-1:], HORIZON - len(response), axis=0,
        )))
    return response, np.asarray(actions, dtype=np.float32), {
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
            (index for index, trace in enumerate(task.records[-1].traces)
             if float(trace["force_peak"]) > 1.), None,
        ),
        "evidence": {"path": str(evidence_path.relative_to(root)),
                     "sha256": sha(evidence_path),
                     "reset_physical_hash": evidence_audit["reset_physical_hashes"][-1]},
    }


def run(config: dict, root: Path) -> dict:
    import torch

    frozen = plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T3 ASID probe requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T3 ASID probe saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    for binding in config["source_bindings"]:
        snapshot = output / "source_snapshot" / binding["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / binding["path"], snapshot)
        if sha(snapshot) != binding["sha256"]:
            raise ValueError("T3 ASID source snapshot differs")
    started = time.monotonic()
    source = ContactMechanismSpec(**config["source_mechanism"])
    report = {**frozen, "status": "running", "rows": []}
    episodes = controls = 0

    def charged(policy, mechanism, mode, seed, path, *, fixed_actions=None):
        nonlocal episodes, controls
        if (time.monotonic() - started > 900 or episodes >= 30):
            raise TimeoutError("T3 ASID physical or wall budget reached")
        episodes += 1
        response, actions, row = _trajectory(
            config, root, policy, mechanism, mode, seed, path,
            fixed_actions=fixed_actions,
        )
        controls += row["physical_control_steps"]
        if controls > 960:
            raise ValueError("T3 ASID control budget exceeded")
        atomic_json(output / "progress.json", {
            "status": "running", "episodes": episodes,
            "physical_control_steps": controls,
            "completed_policy_mode_cells": len(report["rows"]),
        })
        return response, actions, row

    try:
        for policy in POLICIES:
            for mode, seed in zip(config["modes"], config["seeds"], strict=True):
                base = output / "episodes" / f"{policy['name']}_m{mode}"
                _, actions, source_row = charged(
                    np.asarray(policy["values"], dtype=np.float64), source,
                    mode, seed, base / "source" / "evidence.json.gz",
                )
                perturbations = []
                if (source_row["physical_control_steps"] == HORIZON
                        and not source_row["constraint_violation"]):
                    def transition(state, action_stream, theta):
                        del state
                        mechanism = ContactMechanismSpec(
                            servo_gain=float(theta[0]),
                            payload_mass=float(theta[1]),
                            delay_steps=0, contact_time_constant=.02,
                        )
                        trace, _, row = charged(
                            None, mechanism, mode, seed,
                            base / f"perturb_{len(perturbations):02d}" / "evidence.json.gz",
                            fixed_actions=action_stream,
                        )
                        perturbations.append(row)
                        return trace.reshape(-1)

                    jacobian = ASIDSolver(transition, delta=.1).observation_jacobian(
                        None, actions, np.asarray([source.servo_gain, source.payload_mass]),
                    ).reshape(HORIZON, len(FEATURES), 2)
                    valid = all(row["physical_control_steps"] == HORIZON
                                and not row["constraint_violation"] for row in perturbations)
                    singular = np.linalg.svd(jacobian.reshape(-1, 2), compute_uv=False).tolist()
                    position_norms = np.linalg.norm(
                        jacobian[:, :3].reshape(-1, 2), axis=0,
                    ).tolist()
                    score = float(np.sum(jacobian * jacobian))
                else:
                    valid = False
                    singular = position_norms = score = None
                report["rows"].append({
                    "policy": policy, "mode": mode, "reset_seed": seed,
                    "source": source_row, "perturbations": perturbations,
                    "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "valid_full_horizon_and_no_violation": valid,
                    "information_score": score,
                    "full_singular_values": singular,
                    "position_column_norms": position_norms,
                })
        report["status"] = "complete_development_probe"
    except BaseException as error:
        report.update({"status": "failed_retained",
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc()})
    finally:
        report.update({
            "physical_episodes_charged": episodes,
            "physical_control_steps_charged": controls,
            "wall_seconds": time.monotonic() - started,
            "recipe_sha256": sha(output / "recipe.json"),
        })
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "status": report["status"],
            "episodes": episodes, "physical_control_steps": controls,
            "completed_policy_mode_cells": len(report["rows"]),
        })
    return report


def training_plan(config: dict, root: Path) -> dict:
    """Freeze one bounded T3 information, identification, and policy chain."""
    cohort = config.get("training_cohort")
    seed = config.get("training_seed")
    if cohort is None:
        seeds_valid = (
            seed == 8764701
            and config.get("target_exploration_seed") == 3302400000
            and config.get("target_deployment_seeds") == [3302401000, 3302401001]
        )
    elif cohort == INDEPENDENT_COHORT:
        target_seed = 3403000000 + (seed - 8800000) * 10 if isinstance(seed, int) else -1
        seeds_valid = (
            isinstance(seed, int) and 8803001 <= seed <= 8803008
            and config.get("target_exploration_seed") == target_seed
            and config.get("target_deployment_seeds") == [target_seed + 1, target_seed + 2]
            and config.get("output", "").endswith(f"/independent_training_{seed}")
        )
    else:
        seeds_valid = False
    if (config.get("version") != TRAINING_VERSION
            or config.get("task") != "T3" or config.get("robot") != "xarm7"
            or config.get("source_commit") != "cf2cb670fb983f67cf2af156d4e17b4a690787e2"
            or config.get("source_mechanism") != {
                "servo_gain": .85, "payload_mass": .65,
                "delay_steps": 0, "contact_time_constant": .02,
            }
            or config.get("target_mechanism") != {
                "servo_gain": .7, "payload_mass": .5,
                "delay_steps": 3, "contact_time_constant": .02,
            }
            or not seeds_valid
            or config.get("target_exploration_mode") != 0
            or config.get("exploration_controls") != HORIZON
            or config.get("feature_fields") != list(FEATURES)
            or config.get("policy_fields") != list(POLICY_FIELDS)
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 2400, "maximum_gpu_memory_gib": 2,
            }
            or config.get("target_interaction_budget") != {
                "exploration_episodes": 1, "deployment_episodes": 2,
            }
            or config.get("maximum_simulator_episodes") != 232
            or config.get("maximum_target_episodes") != 3
            or config.get("maximum_physical_control_steps") != 37676
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T3 ASID complete-chain development contract changed")
    pointer = config.get("task_config", {})
    if (set(pointer) != {"path", "sha256"}
            or pointer["path"] != "configs/arm/keyed_insertion_recovery/_base.yaml"
            or sha(root / pointer["path"]) != pointer["sha256"]):
        raise ValueError("T3 ASID training task contract changed")
    from genedynamics.experiments.framework import ExperimentConfig
    task_config = ExperimentConfig.from_yaml(root / pointer["path"])
    if (task_config.env_name != "keyed_insertion_recovery"
            or task_config.env_params["shared_execution"] != config.get("shared_execution")
            or task_config.n_steps != 750):
        raise ValueError("T3 ASID training task or horizon differs")
    basis_pointer = config.get("public_policy_basis", {})
    if (set(basis_pointer) != {"path", "sha256"}
            or basis_pointer["path"] !=
            "configs/arm/keyed_insertion_recovery/training/adaptsim_feedback_population_development.json"
            or sha(root / basis_pointer["path"]) != basis_pointer["sha256"]):
        raise ValueError("T3 ASID common public-policy basis changed")
    basis = json.loads((root / basis_pointer["path"]).read_text())
    if (basis["feedback_base"] != config["feedback_base"]
            or basis["shared_execution"] != config["shared_execution"]
            or basis["policy_fields"] != config["policy_fields"]):
        raise ValueError("T3 ASID common public-policy chart changed")
    if not (root / "baselines/ASID/README.md").is_file():
        raise FileNotFoundError("pinned ASID author source checkout missing")
    bindings = config.get("source_bindings", [])
    if (len(bindings) != len(TRAINING_SOURCES)
            or {row.get("path") for row in bindings} != TRAINING_SOURCES):
        raise ValueError("T3 ASID training source roster differs")
    for row in bindings:
        if set(row) != {"path", "sha256"} or sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T3 ASID training source SHA differs: {row.get('path')}")
    bounds = basis["task_policy_search"]
    expected_chart = {key: bounds[key] for key in ("lower", "upper", "anchor")}
    if config.get("exploration") != {
            **expected_chart, "population": 4, "elite": 2, "iterations": 2,
            "modes": [0, 1], "normalization": 10.,
            }:
        raise ValueError("T3 ASID Fisher exploration chart changed")
    if config.get("identification_grid") != {
            "servo_gain": [.5, .6, .7, .8, .9],
            "payload_mass": [.3, .4, .5, .6, .7],
            "delay_steps": [0, 1, 2, 3],
            }:
        raise ValueError("T3 ASID discrete-delay grid changed")
    if config.get("identification") != {
            "lower": [.4, .2], "upper": [1.4, 1.2],
            "population": 6, "elite": 2, "iterations": 2,
            "delta": .1,
            }:
        raise ValueError("T3 ASID local parameter search changed")
    if config.get("task_policy") != {
            **expected_chart, "population": 5, "elite": 2,
            "iterations": 2, "simulator_samples_per_candidate": 2,
            }:
        raise ValueError("T3 ASID downstream policy search changed")
    for name in ("exploration", "task_policy"):
        search = config[name]
        ASIDBoundedCEM(
            search["lower"], search["upper"], search["anchor"],
            population=search["population"], elite=search["elite"],
            iterations=search["iterations"], maximize=True,
        )
    output = config.get("output", "")
    if (output.split("/")[:5] != [
            "results", "arm", "keyed_insertion_recovery", "baseline", "asid"]
            or (root / output).exists()):
        raise ValueError("T3 ASID training output must be fresh and task-local")
    return {
        "version": TRAINING_VERSION, "status": "planned_unexecuted",
        "task": "T3", "output": output,
        "training_cohort": cohort,
        "source_bindings": bindings,
        "maximum_simulator_episodes": 232,
        "maximum_target_episodes": 3,
        "maximum_physical_control_steps": 37676,
        "maximum_wall_seconds": 2400,
        "target_interaction_budget": config["target_interaction_budget"],
        "failure_replacement": False,
        "formal_performance_evidence": False,
        "method_identity": "ASID_Fisher_32step_grid_CEM_mating_policy",
        "scope": "complete task-adapted ASID T3 chain; bounded public-policy CEM replaces unreleased T3 SAC, development only",
    }


def _full_task_episode(config: dict, root: Path, policy: np.ndarray,
                       mechanism: ContactMechanismSpec, mode: int, seed: int,
                       evidence_path: Path) -> tuple[float, dict]:
    search = MatingPolicySearchTask(
        robot="xarm7", feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
    )
    task = instantiate_task(
        "T3", robot="xarm7", mechanism_spec=mechanism, seed=seed,
        shared_execution=config["shared_execution"],
    )
    public = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([mode]))
    actor = PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )
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
    evidence_audit = audit_episode_evidence(evidence_path)
    if not evidence_audit["passed"]:
        raise ValueError("T3 ASID deployment evidence failed CPU rescoring")
    controls = int(episode["physical_control_steps"])
    error = float(episode["final_position_error_m"])
    if bool(episode["success"]):
        reward = 1. - .4 * controls / task.config.timeout_steps
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
                     "sha256": sha(evidence_path),
                     "reset_physical_hash": evidence_audit["reset_physical_hashes"][-1]},
    }


def run_training(config: dict, root: Path) -> dict:
    """Execute one budgeted T3 ASID chain with distinct simulator/target ledgers."""
    import torch

    frozen = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T3 ASID training requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T3 ASID training saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    for binding in config["source_bindings"]:
        snapshot = output / "source_snapshot" / binding["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / binding["path"], snapshot)
        if sha(snapshot) != binding["sha256"]:
            raise ValueError("T3 ASID training source snapshot changed")
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
            raise ValueError("T3 ASID physical control budget exceeded")
        progress(stage)

    def check_budget(*, target_call):
        nonlocal sim_episodes, target_episodes
        if time.monotonic() - started > frozen["maximum_wall_seconds"]:
            raise TimeoutError("T3 ASID development wall ceiling reached")
        if target_call:
            if target_episodes >= frozen["maximum_target_episodes"]:
                raise RuntimeError("T3 ASID target interaction budget exceeded")
            target_episodes += 1
        else:
            if sim_episodes >= frozen["maximum_simulator_episodes"]:
                raise RuntimeError("T3 ASID simulator episode budget exceeded")
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

    def charged_task(policy, mechanism, mode, reset_seed, task_binding):
        check_budget(target_call=False)
        reward, _, controls, summary = task_binding._episode(
            policy, mechanism, mode, reset_seed,
        )
        if controls != summary["control_steps"]:
            raise ValueError("T3 ASID task-policy simulator ledger differs")
        charge("task_policy_training", summary, target_call=False)
        return reward, summary

    try:
        exploration = config["exploration"]
        explorer = ASIDBoundedCEM(
            exploration["lower"], exploration["upper"], exploration["anchor"],
            population=exploration["population"], elite=exploration["elite"],
            iterations=exploration["iterations"], maximize=True,
        )

        def evaluate_exploration(candidate, iteration, index):
            rows, mode_scores = [], []
            for mode in exploration["modes"]:
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
                        servo_gain=float(theta[0]), payload_mass=float(theta[1]),
                        delay_steps=0, contact_time_constant=.02,
                    )
                    response, _, summary = charged_trajectory(
                        "exploration_perturbation", None, mechanism, mode,
                        reset_seed, base / f"perturb_{len(perturbations):02d}" / "evidence.json.gz",
                        fixed_actions=action_stream,
                    )
                    rows.append(summary)
                    perturbations.append(summary)
                    return response.reshape(-1)

                solver = ASIDSolver(
                    transition, delta=.1,
                    normalization=exploration["normalization"],
                    max_reward=1000.,
                )
                score = solver.information_reward(
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
            "normalization": exploration["normalization"],
            "ledger": list(explored.ledger),
        }
        if explored.best_score <= 0:
            raise RuntimeError("no safe informative two-mode T3 ASID exploration policy")

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
            raise RuntimeError("T3 ASID target exploration was unsafe or ended early")

        identification_rows = []

        def evaluate_id(theta, delay, label):
            mechanism = ContactMechanismSpec(
                servo_gain=float(theta[0]), payload_mass=float(theta[1]),
                delay_steps=int(delay), contact_time_constant=.02,
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
            identification_rows.append({
                "label": label,
                "parameters": [float(theta[0]), float(theta[1]), int(delay)],
                "score": mse, "episode": summary,
            })
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
        task_binding = MatingPolicySearchTask(
            robot="xarm7", feedback_base=config["feedback_base"],
            shared_execution=config["shared_execution"],
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
                    "contact_time_constant": .02,
                },
                "seeds": [int(value) for value in rng.integers(0, 2**31, size=2)],
            } for theta in sample])

        def evaluate_task(candidate, iteration, index):
            rows, rewards = [], []
            for task_row in training_tasks[iteration]:
                mechanism = ContactMechanismSpec(**task_row["mechanism"])
                for mode, reset_seed in enumerate(task_row["seeds"]):
                    reward, summary = charged_task(
                        candidate, mechanism, mode, reset_seed, task_binding,
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
                raise ValueError("T3 ASID target reward receipt differs")
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
            "recipe_sha256": sha(output / "recipe.json"),
        })
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "status": report["status"],
            **report["accounting"],
            "wall_seconds": report["wall_seconds"],
        })
    if report["status"] == "complete_development_chain":
        checkpoint = {
            "schema_version": "asid-mating-public-policy-checkpoint-v1",
            "task": "T3", "robot": "xarm7", "development_only": True,
            "policy": report["task_policy"]["best_policy"],
            "identified_mechanism": report["identification"]["best_parameters"],
            "source_report": {
                "path": str((output / "report.json").relative_to(root)),
                "sha256": sha(output / "report.json"),
            },
        }
        atomic_json(output / "checkpoint.json", checkpoint)
        atomic_json(output / "checkpoint_receipt.json", {
            "report": checkpoint["source_report"],
            "checkpoint": {
                "path": str((output / "checkpoint.json").relative_to(root)),
                "sha256": sha(output / "checkpoint.json"),
            },
        })
    return report
