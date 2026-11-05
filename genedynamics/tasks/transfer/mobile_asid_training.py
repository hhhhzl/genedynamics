"""Task-owned, bounded ASID exploration and downstream policy training for T4."""

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
from genedynamics.tasks.transfer.contact import MobileContactMechanismSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.mobile_mating_asid import (
    GPU1_UUID, HORIZON, _trajectory,
)
from genedynamics.tasks.transfer.mobile_mating_policy_search import (
    MOBILE_CONFIG, MobileMatingPolicySearchTask,
)
from genedynamics.tasks.transfer.mating_policy_search import POLICY_FIELDS


VERSION = "asid-t4-complete-chain-development-v2"
INDEPENDENT_COHORT = "independent_main_table_training_v1"
INDEPENDENT_COHORT_V2 = "independent_main_table_training_v2"
SOURCES = {
    "genedynamics/core/control/transfer.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/solvers/single/asid/asid.py",
    "genedynamics/solvers/single/asid/backends/asid_numpy.py",
    "genedynamics/solvers/single/asid/learning/search.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
    "genedynamics/tasks/transfer/mobile_mating_policy_search.py",
    "genedynamics/tasks/transfer/mobile_mating_asid.py",
    "genedynamics/tasks/transfer/mobile_asid_training.py",
    "scripts/paper/cross_ebdm/train_asid_mobile_development.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _simulator_reset_rejection(error: BaseException) -> bool:
    """Classify only known zero-control simulator reset failures."""
    return (
        isinstance(error, ValueError)
        and str(error).startswith("Go2+Z1 source pose unreachable:")
    ) or (
        isinstance(error, RuntimeError)
        and str(error).startswith("floating reset stance preparation failed:")
    )


def training_plan(config: dict, root: Path) -> dict:
    source = {"servo_gain": .85, "payload_mass": .65,
              "delay_steps": 0, "body_servo_gain": 1.}
    target = {"servo_gain": .7, "payload_mass": .5,
              "delay_steps": 3, "body_servo_gain": .9}
    cohort = config.get("training_cohort")
    seed = config.get("training_seed")
    if cohort is None:
        seeds_valid = (
            seed == 8764802
            and config.get("target_exploration_seed") == 3302700044
            and config.get("target_deployment_seeds") == [3302700045, 3302700046]
        )
    elif cohort in {INDEPENDENT_COHORT, INDEPENDENT_COHORT_V2}:
        first_seed = 8804001 if cohort == INDEPENDENT_COHORT else 8804101
        target_start = 3404000000 if cohort == INDEPENDENT_COHORT else 3404000024
        target_seed = target_start + 3 * (seed - first_seed) if isinstance(seed, int) else -1
        reset_pointer = config.get("reset_seed_audit")
        seeds_valid = (
            isinstance(seed, int) and first_seed <= seed <= first_seed + 7
            and config.get("target_exploration_seed") == target_seed
            and config.get("target_deployment_seeds") == [target_seed + 1, target_seed + 2]
            and config.get("output", "").endswith(f"/independent_training_{seed}")
            and isinstance(reset_pointer, dict)
            and reset_pointer.get("path") == (
                "results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/"
                "diagnostics/independent_training_reset_feasibility_3404000000/"
                "independent_audit.json"
            )
        )
    else:
        seeds_valid = False
    if (config.get("version") != VERSION or config.get("task") != "T4"
            or config.get("robot") != "go2_z1"
            or config.get("source_mechanism") != source
            or config.get("target_mechanism") != target
            or not seeds_valid
            or config.get("exploration_controls") != HORIZON
            or config.get("policy_fields") != list(POLICY_FIELDS)
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 3600, "maximum_gpu_memory_gib": 2,
            }
            or config.get("target_interaction_budget") != {
                "exploration_episodes": 1, "deployment_episodes": 2,
            }
            or config.get("maximum_simulator_episodes") != 308
            or config.get("maximum_target_episodes") != 3
            or config.get("maximum_physical_control_steps") != 40108
            or config.get("failure_replacement") is not False
            or config.get("formal_performance_evidence") is not False):
        raise ValueError("T4 ASID complete-chain training contract differs")
    from genedynamics.experiments.framework import ExperimentConfig

    task_pointer = config.get("task_config", {})
    basis_pointer = config.get("public_policy_basis", {})
    info_pointer = config.get("information_probe_audit", {})
    reset_pointer = config.get("reset_seed_audit", {})
    for pointer in (task_pointer, basis_pointer, info_pointer, reset_pointer):
        if (set(pointer) != {"path", "sha256"}
                or not (root / pointer["path"]).is_file()
                or _sha(root / pointer["path"]) != pointer["sha256"]):
            raise ValueError("T4 ASID task, policy, information or reset binding differs")
    task = ExperimentConfig.from_yaml(root / task_pointer["path"])
    basis = json.loads((root / basis_pointer["path"]).read_text())
    info = json.loads((root / info_pointer["path"]).read_text())
    resets = json.loads((root / reset_pointer["path"]).read_text())
    safe = resets["safe_seed_pool"]
    if (task.env_name != "mobile_keyed_insertion_recovery"
            or task.n_steps != 750
            or task.env_params["shared_execution"] != config.get("shared_execution")
            or basis.get("version") != "adaptsim-mobile-mating-population-development-v2"
            or basis["feedback_base"] != config.get("feedback_base")
            or basis["shared_execution"] != config["shared_execution"]
            or info.get("status") != "passed_development_only"
            or info.get("physical_evidence_verified") != 42
            or resets.get("status") != "passed"
            or config.get("reset_seed_pool") != safe
            or len(safe) != 64
            or not set([config["target_exploration_seed"],
                        *config["target_deployment_seeds"]]) <= set(safe)):
        raise ValueError("T4 ASID frozen public or safe-reset basis differs")
    chart = basis["task_policy_search"]
    expected_chart = {key: chart[key] for key in ("lower", "upper", "anchor")}
    if (config.get("exploration") != {
                **expected_chart, "population": 4, "elite": 2,
                "iterations": 2, "modes": [0, 1], "normalization": 100.,
            }
            or config.get("identification_grid") != {
                "servo_gain": [.6, .7, .8, .9],
                "payload_mass": [.4, .5, .6],
                "body_servo_gain": [.8, .9, 1.],
                "delay_steps": [0, 1, 2, 3],
            }
            or config.get("identification") != {
                "lower": [.4, .2, .7], "upper": [1.4, 1.2, 1.3],
                "population": 6, "elite": 2, "iterations": 2, "delta": .1,
            }
            or config.get("task_policy") != {
                **expected_chart, "population": 5, "elite": 2,
                "iterations": 2, "simulator_samples_per_candidate": 2,
            }):
        raise ValueError("T4 ASID search charts or charged budgets differ")
    for name in ("exploration", "task_policy"):
        spec = config[name]
        ASIDBoundedCEM(
            spec["lower"], spec["upper"], spec["anchor"],
            population=spec["population"], elite=spec["elite"],
            iterations=spec["iterations"], maximize=True,
        )
    MobileMatingPolicySearchTask(
        feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
        reset_seed_pool=tuple(safe),
    ).feedback(chart["anchor"])
    bindings = config.get("source_bindings", [])
    if len(bindings) != len(SOURCES) or {row.get("path") for row in bindings} != SOURCES:
        raise ValueError("T4 ASID complete-chain source roster differs")
    for row in bindings:
        if set(row) != {"path", "sha256"} or _sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T4 ASID complete-chain source SHA differs: {row.get('path')}")
    output = config.get("output", "")
    if (output.split("/")[:5] != [
        "results", "quadruped", "mobile_keyed_insertion_recovery",
        "baseline", "asid",
    ] or (root / output).exists()):
        raise ValueError("T4 ASID output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted", "task": "T4",
        "training_cohort": cohort,
        "output": output, "source_bindings": bindings,
        "maximum_simulator_episodes": 308, "maximum_target_episodes": 3,
        "maximum_physical_control_steps": 40108,
        "maximum_wall_seconds": 3600,
        "target_interaction_budget": config["target_interaction_budget"],
        "failure_replacement": False, "formal_performance_evidence": False,
        "method_identity": "ASID_Fisher_32step_4parameter_grid_CEM_mobile_policy",
        "scope": "complete task-adapted ASID T4 chain; bounded public-policy CEM replaces unreleased T4 SAC; development only",
    }


def _full_task_episode(config: dict, root: Path, policy: np.ndarray,
                       mechanism: MobileContactMechanismSpec, mode: int,
                       seed: int, evidence_path: Path):
    search = MobileMatingPolicySearchTask(
        feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
        reset_seed_pool=tuple(config["reset_seed_pool"]),
    )
    task = instantiate_task(
        "T4", robot="go2_z1", mechanism_spec=mechanism, seed=seed,
        shared_execution=config["shared_execution"], config_overrides=MOBILE_CONFIG,
    )
    public = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([mode]))
    actor = PublicLivenessMatingFeedbackPolicy(
        task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
    )
    for _ in range(task.config.timeout_steps):
        if bool(np.asarray(task.state.status)[0]):
            break
        body_intent = task.public_body_intent(public)
        if body_intent is None:
            raise RuntimeError("T4 ASID public body intent missing")
        public = task.step_contact(actor.act(public), body_twist=body_intent).observation
    evaluation = task.evaluate()
    episode = evaluation["episodes"][-1]
    evidence_path.parent.mkdir(parents=True, exist_ok=False)
    evidence_path.write_bytes(gzip.compress(
        json.dumps(episode_evidence(task.records, task.spec, evaluation),
                   sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    checked = audit_episode_evidence(evidence_path)
    if not checked["passed"]:
        raise ValueError("T4 ASID deployment evidence failed CPU rescoring")
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
            "body_servo_gain": mechanism.body_servo_gain,
        },
        "outcome": episode["outcome"], "success": bool(episode["success"]),
        "constraint_violation": bool(episode["constraint_violation"]),
        "control_steps": controls, "final_position_error_m": error,
        "reward": reward,
        "evidence": {"path": str(evidence_path.relative_to(root)),
                     "sha256": _sha(evidence_path),
                     "reset_physical_hash": checked["reset_physical_hashes"][-1]},
    }


def run_training(config: dict, root: Path) -> dict:
    """Execute one charged T4 ASID exploration, ID, task search and deployment."""
    import torch

    frozen = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T4 ASID complete chain requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T4 ASID complete chain saw wrong physical GPU")
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
        if _sha(snapshot) != binding["sha256"]:
            raise ValueError("T4 ASID training source snapshot differs")
    started = time.monotonic()
    sim_episodes = target_episodes = sim_controls = target_controls = 0
    ledger = []
    report = {**frozen, "status": "running", "interaction_ledger": ledger}
    seed = int(config["training_seed"])
    source = MobileContactMechanismSpec(**config["source_mechanism"])
    target = MobileContactMechanismSpec(**config["target_mechanism"])
    pool = config["reset_seed_pool"]

    def progress(stage: str):
        atomic_json(output / "progress.json", {
            "status": "running", "stage": stage,
            "wall_seconds": time.monotonic() - started,
            "simulator_episodes": sim_episodes,
            "simulator_control_steps": sim_controls,
            "target_episodes": target_episodes,
            "target_control_steps": target_controls,
        })

    progress("source_snapshot_complete")

    def check_budget(*, target_call: bool):
        nonlocal sim_episodes, target_episodes
        if time.monotonic() - started > frozen["maximum_wall_seconds"]:
            raise TimeoutError("T4 ASID wall ceiling reached")
        if target_call:
            if target_episodes >= frozen["maximum_target_episodes"]:
                raise RuntimeError("T4 ASID target interaction budget exceeded")
            target_episodes += 1
        else:
            if sim_episodes >= frozen["maximum_simulator_episodes"]:
                raise RuntimeError("T4 ASID simulator episode budget exceeded")
            sim_episodes += 1

    def charge(stage, summary, *, target_call: bool):
        nonlocal sim_controls, target_controls
        controls = int(summary.get("physical_control_steps", summary.get("control_steps", 0)))
        if target_call:
            target_controls += controls
        else:
            sim_controls += controls
        ledger.append({"stage": stage, "target_interaction": target_call, **summary})
        if sim_controls + target_controls > frozen["maximum_physical_control_steps"]:
            raise ValueError("T4 ASID physical control budget exceeded")
        progress(stage)

    def charged_trajectory(stage, policy, mechanism, mode, reset_seed, path,
                           *, fixed_actions=None, fixed_body_intents=None,
                           target_call=False):
        check_budget(target_call=target_call)
        try:
            response, actions, body, summary = _trajectory(
                config, root, policy, mechanism, mode, reset_seed, path,
                fixed_actions=fixed_actions, fixed_body_intents=fixed_body_intents,
            )
        except (ValueError, RuntimeError) as exc:
            if not _simulator_reset_rejection(exc) or target_call:
                raise
            response = np.zeros((HORIZON, 9), dtype=np.float64)
            actions = np.zeros((0, 13), dtype=np.float32)
            body = np.zeros((0, 6), dtype=np.float32)
            summary = {
                "mode": mode, "reset_seed": reset_seed,
                "mechanism": {
                    "servo_gain": mechanism.servo_gain,
                    "payload_mass": mechanism.payload_mass,
                    "delay_steps": mechanism.delay_steps,
                    "body_servo_gain": mechanism.body_servo_gain,
                },
                "physical_control_steps": 0,
                "outcome": "reset_unreachable", "constraint_violation": False,
                "reset_error": str(exc),
            }
        charge(stage, summary, target_call=target_call)
        return response, actions, body, summary

    task_binding = MobileMatingPolicySearchTask(
        feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
        reset_seed_pool=tuple(pool),
    )

    def charged_task(policy, mechanism, mode, reset_seed):
        check_budget(target_call=False)
        try:
            reward, _, controls, summary = task_binding._episode(
                policy, mechanism, mode, reset_seed,
            )
        except RuntimeError as exc:
            if not _simulator_reset_rejection(exc):
                raise
            reward, controls = -2., 0
            summary = {
                "mode": mode, "reset_seed": reset_seed,
                "success": False, "constraint_violation": False,
                "outcome": "reset_unreachable", "control_steps": 0,
                "final_position_error_m": None, "reward": reward,
                "reset_error": str(exc),
            }
        if controls != summary["control_steps"]:
            raise ValueError("T4 ASID task-policy simulator ledger differs")
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
            rows, scores = [], []
            for mode in exploration["modes"]:
                reset_seed = pool[iteration * 8 + index * 2 + mode]
                base = output / "episodes" / "exploration" / f"i{iteration}_c{index}_m{mode}"
                _, actions, body, source_row = charged_trajectory(
                    "exploration_source", candidate, source, mode, reset_seed,
                    base / "source" / "evidence.json.gz",
                )
                rows.append(source_row)
                if (source_row["physical_control_steps"] != HORIZON
                        or source_row["constraint_violation"]):
                    scores.append(-1e9)
                    continue
                perturbations = []

                def transition(state, action_stream, theta):
                    del state
                    mechanism = MobileContactMechanismSpec(
                        servo_gain=float(theta[0]),
                        payload_mass=float(theta[1]),
                        delay_steps=0,
                        body_servo_gain=float(theta[2]),
                    )
                    response, _, _, summary = charged_trajectory(
                        "exploration_perturbation", None, mechanism,
                        mode, reset_seed,
                        base / f"perturb_{len(perturbations):02d}" / "evidence.json.gz",
                        fixed_actions=action_stream, fixed_body_intents=body,
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
                    np.asarray([source.servo_gain, source.payload_mass,
                                source.body_servo_gain]),
                )
                if any(row["physical_control_steps"] != HORIZON
                       or row["constraint_violation"] for row in perturbations):
                    score = -1e9
                scores.append(score)
            return float(np.mean(scores)), {
                "episodes": rows, "mode_information_scores": scores,
                "safe_both_modes": all(score >= 0 for score in scores),
            }

        explored = explorer.run(evaluate_exploration, seed=seed + 13)
        report["exploration"] = {
            "best_policy": explored.best.tolist(),
            "best_information_score": explored.best_score,
            "normalization": exploration["normalization"],
            "ledger": list(explored.ledger),
        }
        if explored.best_score <= 0:
            raise RuntimeError("no safe informative two-mode T4 ASID exploration policy")

        target_response, target_actions, target_body, target_row = charged_trajectory(
            "target_exploration", explored.best, target, 0,
            config["target_exploration_seed"],
            output / "episodes" / "target_exploration" / "evidence.json.gz",
            target_call=True,
        )
        report["target_exploration"] = {
            "episode": target_row,
            "public_response": target_response.tolist(),
            "requested_actions_sha256": hashlib.sha256(target_actions.tobytes()).hexdigest(),
            "body_intent_sha256": hashlib.sha256(target_body.tobytes()).hexdigest(),
        }
        if (target_row["physical_control_steps"] != HORIZON
                or target_row["constraint_violation"]):
            raise RuntimeError("T4 ASID target exploration unsafe or ended early")

        identification_rows = []

        def evaluate_id(theta, delay, label):
            mechanism = MobileContactMechanismSpec(
                servo_gain=float(theta[0]), payload_mass=float(theta[1]),
                delay_steps=int(delay), body_servo_gain=float(theta[2]),
            )
            response, _, _, summary = charged_trajectory(
                "identification", None, mechanism, 0,
                config["target_exploration_seed"],
                output / "episodes" / "identification" / label / "evidence.json.gz",
                fixed_actions=target_actions, fixed_body_intents=target_body,
            )
            mse = float(np.mean((response - target_response) ** 2))
            if summary["physical_control_steps"] != HORIZON or summary["constraint_violation"]:
                mse += 1e6
            identification_rows.append({
                "label": label,
                "parameters": [float(theta[0]), float(theta[1]),
                               float(theta[2]), int(delay)],
                "score": mse, "episode": summary,
            })
            return mse, {"episodes": [summary], "public_mse": mse}

        grid = config["identification_grid"]
        for gain in grid["servo_gain"]:
            for mass in grid["payload_mass"]:
                for body_gain in grid["body_servo_gain"]:
                    for delay in grid["delay_steps"]:
                        evaluate_id(
                            np.asarray([gain, mass, body_gain]), delay,
                            f"grid_g{gain:.1f}_m{mass:.1f}_b{body_gain:.1f}_d{delay}",
                        )
        best_grid = min(identification_rows, key=lambda row: row["score"])
        selected_delay = int(best_grid["parameters"][3])
        id_spec = config["identification"]
        identifier = ASIDBoundedCEM(
            id_spec["lower"], id_spec["upper"],
            best_grid["parameters"][:3],
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
        rng = np.random.default_rng(seed + 50001)
        std = np.maximum(identified.std, [.05, .05, .05])
        selected_seeds = rng.choice(np.asarray(pool[24:40]), size=8, replace=False)
        training_tasks = []
        for iteration in range(task_cfg["iterations"]):
            sample = np.clip(
                rng.normal(fitted, std, size=(2, 3)),
                id_spec["lower"], id_spec["upper"],
            )
            training_tasks.append([{
                "mechanism": {
                    "servo_gain": float(theta[0]),
                    "payload_mass": float(theta[1]),
                    "delay_steps": selected_delay,
                    "body_servo_gain": float(theta[2]),
                },
                "seeds": [int(x) for x in selected_seeds[iteration * 4 + j * 2:
                                                      iteration * 4 + j * 2 + 2]],
            } for j, theta in enumerate(sample)])

        def evaluate_task(candidate, iteration, index):
            rows, rewards = [], []
            for task_row in training_tasks[iteration]:
                mechanism = MobileContactMechanismSpec(**task_row["mechanism"])
                for mode, reset_seed in enumerate(task_row["seeds"]):
                    reward, summary = charged_task(
                        candidate, mechanism, mode, reset_seed,
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
                raise ValueError("T4 ASID target reward receipt differs")
            charge("target_deployment", summary, target_call=True)
            deployment.append(summary)
        report.update({
            "status": "complete_development_chain",
            "deployment": deployment,
            "deployment_outcomes": dict(Counter(row["outcome"] for row in deployment)),
            "formal_performance_evidence": False,
        })
    except BaseException as exc:
        report.update({"status": "failed_retained",
                       "error_type": type(exc).__name__, "error": str(exc),
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
            "status": report["status"], **report["accounting"],
            "wall_seconds": report["wall_seconds"],
        })
    if report["status"] == "complete_development_chain":
        checkpoint = {
            "schema_version": "asid-mobile-public-policy-checkpoint-v2",
            "task": "T4", "robot": "go2_z1", "development_only": True,
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


def audit_training(config: dict, root: Path) -> dict:
    """CPU re-score saved physical episodes and reconcile the whole ASID ledger."""
    if config.get("version") != VERSION:
        raise ValueError("unknown T4 ASID complete-chain audit recipe")
    output = root / config["output"]
    recipe_path = output / "recipe.json"
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if (json.loads(recipe_path.read_text()) != config
            or report.get("status") != "complete_development_chain"
            or report.get("recipe_sha256") != _sha(recipe_path)
            or report.get("formal_performance_evidence") is not False
            or report.get("wall_seconds", 3601) > 3600):
        raise ValueError("T4 ASID completed report or recipe differs")
    for binding in config["source_bindings"]:
        snapshot = output / "source_snapshot" / binding["path"]
        if _sha(snapshot) != binding["sha256"]:
            raise ValueError("T4 ASID source snapshot differs")
    ledger = report["interaction_ledger"]
    sim = [row for row in ledger if not row["target_interaction"]]
    target = [row for row in ledger if row["target_interaction"]]
    accounting = report["accounting"]
    if (len(sim) != accounting["simulator_episodes"]
            or len(target) != accounting["target_episodes"] == 3
            or len(sim) > 308
            or sum(int(row.get("physical_control_steps", row.get("control_steps", 0)))
                   for row in sim) != accounting["simulator_control_steps"]
            or sum(int(row.get("physical_control_steps", row.get("control_steps", 0)))
                   for row in target) != accounting["target_control_steps"]
            or accounting["simulator_control_steps"] + accounting["target_control_steps"]
            > 40108):
        raise ValueError("T4 ASID simulator/target episode accounting differs")
    stages = Counter(row["stage"] for row in ledger)
    if (stages["target_exploration"] != 1
            or stages["target_deployment"] != 2
            or stages["task_policy_training"] != 40
            or stages["identification"] != 156
            or stages["exploration_source"] != 16
            or stages["exploration_perturbation"] > 96):
        raise ValueError("T4 ASID stage roster incomplete")
    evidence_count = 0
    for row in ledger:
        if row["stage"] == "task_policy_training":
            if "evidence" in row:
                raise ValueError("T4 task-policy training ledger unexpectedly claims physical audit")
            continue
        pointer = row.get("evidence")
        if pointer is None:
            if (row.get("outcome") != "reset_unreachable"
                    or row.get("physical_control_steps") != 0
                    or row.get("constraint_violation") is not False
                    or not str(row.get("reset_error", "")).startswith((
                        "Go2+Z1 source pose unreachable:",
                        "floating reset stance preparation failed:",
                    ))):
                raise ValueError("T4 ASID unaudited interaction is not a reset failure")
            continue
        if not isinstance(pointer, dict):
            raise ValueError("T4 ASID physical interaction missing evidence")
        path = root / pointer["path"]
        if not path.is_relative_to(output):
            raise ValueError("T4 ASID physical evidence outside training output")
        checked = audit_episode_evidence(path)
        scored = checked["evaluation"]["episodes"][-1]
        controls = int(row.get("physical_control_steps", row.get("control_steps", 0)))
        if (not checked["passed"] or checked["sha256"] != pointer["sha256"]
                or checked["reset_physical_hashes"][-1] != pointer["reset_physical_hash"]
                or scored["outcome"] != row["outcome"]
                or int(scored["physical_control_steps"]) != controls):
            raise ValueError("T4 ASID physical evidence re-score differs")
        evidence_count += 1
    reset_failures = sum(row.get("outcome") == "reset_unreachable" for row in ledger)
    physical_reset_failures = sum(
        row.get("outcome") == "reset_unreachable"
        for row in ledger if row["stage"] != "task_policy_training"
    )
    if evidence_count != len(ledger) - 40 - physical_reset_failures:
        raise ValueError("T4 ASID physical evidence count differs")
    exploration = report["exploration"]
    identification = report["identification"]
    policy = report["task_policy"]
    if (len(exploration["ledger"]) != 8
            or len(identification["grid_and_local_ledger"]) != 156
            or len(identification["local_search_ledger"]) != 12
            or len(identification["best_parameters"]) != 4
            or len(policy["ledger"]) != 10
            or len(policy["best_policy"]) != len(POLICY_FIELDS)
            or len(report["deployment"]) != 2):
        raise ValueError("T4 ASID search or deployment structure differs")
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if (checkpoint.get("schema_version") != "asid-mobile-public-policy-checkpoint-v2"
            or checkpoint.get("policy") != policy["best_policy"]
            or checkpoint.get("identified_mechanism") != identification["best_parameters"]
            or checkpoint.get("source_report", {}).get("sha256") != _sha(report_path)):
        raise ValueError("T4 ASID checkpoint/report identity differs")
    audit = {
        "schema_version": "asid-t4-full-chain-independent-audit-v2",
        "status": "passed_development_only", "task": "T4",
        "report_sha256": _sha(report_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "source_snapshots_verified": len(config["source_bindings"]),
        "physical_evidence_verified": evidence_count,
        "reset_unreachable_attempts": reset_failures,
        "task_simulator_episodes_ledger_only": 40,
        "simulator_episodes": len(sim), "target_episodes": len(target),
        "stage_counts": dict(sorted(stages.items())),
        "full_wrench_and_body_intent_replay_independently_verified": False,
        "physical_calls": 0, "gpu_calls": 0,
        "formal_performance_evidence": False,
    }
    atomic_json(output / "independent_audit.json", audit)
    return audit
