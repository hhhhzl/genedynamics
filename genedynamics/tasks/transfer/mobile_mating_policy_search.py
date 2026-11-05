"""Task-owned public-feedback feasibility for AdaptSim on mobile mating."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.contact import MobileContactMechanismSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies import PublicBiasFeedbackSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.mating_policy_search import POLICY_FIELDS, POLICY_LOWER, POLICY_UPPER
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence


VERSION = "adaptsim-mobile-mating-feasibility-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
MOBILE_CONFIG = {
    "execution_profile": "joint_margin_public_reference",
    "reset_profile": "folded_stance",
    "reset_joint_noise": .002,
}


def _feedback(base: dict, policy: list[float]) -> PublicBiasFeedbackSpec:
    values = np.asarray(policy, dtype=np.float64)
    if (values.shape != (len(POLICY_FIELDS),) or not np.isfinite(values).all()
            or np.any(values < POLICY_LOWER) or np.any(values > POLICY_UPPER)):
        raise ValueError("mobile feedback policy is outside declared support")
    return PublicBiasFeedbackSpec(**{**base, **dict(zip(POLICY_FIELDS, values.tolist(), strict=True))})


def feasibility_plan(config: dict, root: Path) -> dict:
    """Freeze a two-cell, two-policy screen without simulator resets."""
    if config.get("version") != VERSION or config.get("task") != "T4" or config.get("robot") != "go2_z1":
        raise ValueError("expected mobile mating feasibility recipe")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 900, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("mobile feasibility execution budget differs")
    if (config.get("policy_fields") != list(POLICY_FIELDS)
            or len(config.get("policies", [])) != 2
            or len(config.get("requests", [])) != 4):
        raise ValueError("expected two declared policies and four frozen requests")
    for policy in config["policies"]:
        if set(policy) != {"name", "values"}:
            raise ValueError("policy row differs")
        _feedback(config["feedback_base"], policy["values"])
    if {policy["name"] for policy in config["policies"]} != {
        "anchor", "conservative_recovery",
    }:
        raise ValueError("expected the two named public-feedback probes")
    allowed_cells = {
        "source", "nominal", "latency_gain_shift", "payload_shift",
    }
    cells = {row.get("scene_cell") for row in config["requests"]}
    if cells not in ({"source", "nominal"}, {"latency_gain_shift", "payload_shift"}):
        raise ValueError("expected source/nominal or held-out-shift cell pair")
    for row in config["requests"]:
        if set(row) != {"scene_cell", "mode", "seed", "mechanism"} or row["mode"] not in (0, 1):
            raise ValueError("request row differs")
        if row["scene_cell"] not in allowed_cells:
            raise ValueError("unknown development scene cell")
        MobileContactMechanismSpec(**row["mechanism"])
    if len({(r["scene_cell"], r["mode"]) for r in config["requests"]}) != 4:
        raise ValueError("every declared cell/mode must be present once")
    from genedynamics.tasks.transfer.specs import task_spec, validate_shared_execution
    validate_shared_execution(
        task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG),
        MobileContactMechanismSpec(),
        config["shared_execution"],
    )
    if config["output"].split("/")[:4] != [
        "results", "quadruped", "mobile_keyed_insertion_recovery", "baseline",
    ] or (root / config["output"]).exists():
        raise ValueError("output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted", "task": "T4",
        "maximum_physics_episodes": 8, "maximum_wall_seconds": 900,
        "failure_replacement": False, "formal_performance_evidence": False,
        "scope": "AdaptSim T4 public-feedback policy feasibility; no training or target adaptation",
        "scene_cells": sorted(cells),
        "output": config["output"],
    }


def run_feasibility(config: dict, root: Path) -> dict:
    import torch

    plan = feasibility_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("physical GPU1 must be the only visible CUDA device")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    rows = []
    status = "complete_feasibility"
    error = None
    try:
        for policy in config["policies"]:
            feedback = _feedback(config["feedback_base"], policy["values"])
            for request in config["requests"]:
                if time.monotonic() - started > plan["maximum_wall_seconds"]:
                    raise TimeoutError("mobile feasibility wall ceiling exceeded")
                task = instantiate_task(
                    "T4", robot="go2_z1",
                    mechanism_spec=MobileContactMechanismSpec(**request["mechanism"]),
                    seed=request["seed"], shared_execution=config["shared_execution"],
                    config_overrides=MOBILE_CONFIG,
                )
                public = task.reset(
                    (0,), seed=request["seed"], initial="jammed",
                    modes=np.asarray([request["mode"]]),
                )
                actor = PublicLivenessMatingFeedbackPolicy(
                    task.spec, feedback=feedback, modes=(request["mode"],),
                    seed=request["seed"],
                )
                for _ in range(task.config.timeout_steps):
                    action = actor.act(public)
                    body_intent = task.public_body_intent(public)
                    if body_intent is None:
                        raise RuntimeError("T4 public body intent missing")
                    transition = task.step_contact(action, body_twist=body_intent)
                    public = transition.observation
                    if bool(transition.done[0]):
                        break
                evaluation = task.evaluate()
                evaluation["accounting"] = dict(task.accounting)
                episode = evaluation["episodes"][-1]
                valid = [trace for trace in task.records[-1].traces if trace.get("valid")]
                final_trace = valid[-1] if valid else None
                evidence_dir = output / "episodes" / (
                    f"{policy['name']}_{request['scene_cell']}_m{request['mode']}"
                )
                evidence_dir.mkdir(parents=True, exist_ok=False)
                evidence_path = evidence_dir / "evidence.json.gz"
                evidence = episode_evidence(task.records, task.spec, evaluation)
                evidence_path.write_bytes(gzip.compress(
                    json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
                    compresslevel=6, mtime=0,
                ))
                evidence_audit = audit_episode_evidence(evidence_path)
                if not evidence_audit["passed"]:
                    raise RuntimeError("mobile episode evidence failed CPU reevaluation")
                row = {
                    "policy": policy["name"], **request,
                    "success": bool(episode["success"]),
                    "outcome": episode["outcome"],
                    "constraint_violation": bool(episode["constraint_violation"]),
                    "physical_control_steps": int(episode["physical_control_steps"]),
                    "minimum_support_count": min(
                        (int(trace.get("support_count", 0)) for trace in valid), default=0,
                    ),
                    "final_limiting_joint": (
                        final_trace.get("limiting_joint") if final_trace else None
                    ),
                    "final_minimum_joint_margin_rad": (
                        float(final_trace["minimum_joint_margin_rad"])
                        if final_trace and final_trace.get("minimum_joint_margin_rad") is not None
                        else None
                    ),
                    "final_body_position_m": np.asarray(public["body_position"])[0].tolist(),
                    "evidence": {
                        "path": str(evidence_path.relative_to(root)),
                        "sha256": evidence_audit["sha256"],
                        "cpu_reevaluation_passed": True,
                    },
                    "final_position_error_m": float(episode["final_position_error_m"]),
                }
                rows.append(row)
                atomic_json(output / "progress.json", {"status": "running", "episodes": rows})
    except Exception as exc:
        status = "failed_retained"
        error = f"{type(exc).__name__}: {exc}"
    report = {
        **plan, "status": status, "error": error, "episodes": rows,
        "wall_seconds": time.monotonic() - started,
        "recipe_sha256": hashlib.sha256((output / "recipe.json").read_bytes()).hexdigest(),
    }
    atomic_json(output / "report.json", report)
    return report


def audit_feasibility(config: dict, root: Path) -> dict:
    """Recompute every retained episode from durable evidence, without physics."""
    if config.get("version") != VERSION:
        raise ValueError("unknown mobile feasibility recipe")
    output = root / config["output"]
    saved_recipe = output / "recipe.json"
    report_path = output / "report.json"
    if json.loads(saved_recipe.read_text()) != config:
        raise ValueError("run recipe differs from requested audit recipe")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "complete_feasibility"
            or report.get("recipe_sha256") != hashlib.sha256(saved_recipe.read_bytes()).hexdigest()
            or len(report.get("episodes", [])) != 8):
        raise ValueError("completed mobile screen report or recipe differs")
    expected = {
        (policy["name"], request["scene_cell"], request["mode"], request["seed"])
        for policy in config["policies"] for request in config["requests"]
    }
    observed = set()
    audited = []
    for row in report["episodes"]:
        key = (row["policy"], row["scene_cell"], row["mode"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("mobile evidence request roster differs")
        observed.add(key)
        pointer = row["evidence"]
        evidence_path = root / pointer["path"]
        if not evidence_path.is_relative_to(output):
            raise ValueError("mobile evidence outside task-local run")
        result = audit_episode_evidence(evidence_path)
        episode = result["evaluation"]["episodes"][-1]
        if (not result["passed"] or result["sha256"] != pointer["sha256"]
                or bool(episode["success"]) != row["success"]
                or episode["outcome"] != row["outcome"]
                or int(episode["physical_control_steps"]) != row["physical_control_steps"]):
            raise ValueError(f"mobile evidence reevaluation differs: {key}")
        audited.append({"request": key, "evidence_sha256": result["sha256"],
                        "reset_physical_hash": result["reset_physical_hashes"][-1],
                        "outcome": episode["outcome"]})
    if observed != expected:
        raise ValueError("mobile evidence request coverage differs")
    audit = {
        "version": VERSION, "status": "passed", "physical_calls": 0,
        "gpu_calls": 0, "episodes": audited,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    atomic_json(output / "independent_audit.json", audit)
    return audit


TRAINING_VERSION = "adaptsim-mobile-mating-population-development-v2"


def _simulator_reset_rejection(error: BaseException) -> bool:
    """Recognize only known T4 zero-control simulator reset failures."""
    return (
        isinstance(error, ValueError)
        and str(error).startswith("Go2+Z1 source pose unreachable:")
    ) or (
        isinstance(error, RuntimeError)
        and str(error).startswith("floating reset stance preparation failed:")
    )


@dataclass(frozen=True)
class MobileMatingSearchTask:
    mechanism: MobileContactMechanismSpec
    mode_seeds: tuple[int, int]


class MobileMatingPolicySearchTask:
    """AdaptSim's public task-policy chart with T4's shared body intent."""

    def __init__(self, *, feedback_base: dict, shared_execution: dict,
                 reset_seed_pool: tuple[int, ...] = ()) -> None:
        from genedynamics.tasks.transfer.specs import task_spec, validate_shared_execution

        self.feedback_base = dict(feedback_base)
        self.shared_execution = dict(shared_execution)
        self.reset_seed_pool = tuple(int(value) for value in reset_seed_pool)
        if self.reset_seed_pool and (
            len(self.reset_seed_pool) < 16
            or len(set(self.reset_seed_pool)) != len(self.reset_seed_pool)
        ):
            raise ValueError("T4 training reset seed pool needs 16 distinct seeds")
        validate_shared_execution(
            task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG),
            MobileContactMechanismSpec(), self.shared_execution,
        )
        _feedback(self.feedback_base, [.25, 6., 2., .008, .004, 1., 0.])

    def feedback(self, policy):
        return _feedback(self.feedback_base, policy)

    def sample_tasks(self, distribution, rng, count):
        if distribution.kind not in {"gaussian", "uniform"} or len(distribution.left) != 4:
            raise ValueError("T4 AdaptSim distribution needs gain, mass, delay, body gain")
        tasks = []
        for gain, mass, delay, body_gain in distribution.sample(rng, count):
            mechanism = MobileContactMechanismSpec(
                servo_gain=float(np.clip(gain, .4, 1.4)),
                payload_mass=float(np.clip(mass, .2, 1.2)),
                delay_steps=int(np.clip(np.rint(delay), 0, 3)),
                body_servo_gain=float(np.clip(body_gain, .7, 1.3)),
            )
            seeds = tuple(
                self.reset_seed_pool[int(rng.integers(0, len(self.reset_seed_pool)))]
                if self.reset_seed_pool else int(rng.integers(0, 2**31))
                for _ in range(2)
            )
            tasks.append(MobileMatingSearchTask(mechanism, seeds))
        return tuple(tasks)

    def _episode(self, policy, mechanism, mode, seed, *, simulator=False):
        task = instantiate_task(
            "T4", robot="go2_z1", mechanism_spec=mechanism, seed=seed,
            shared_execution=self.shared_execution, config_overrides=MOBILE_CONFIG,
        )
        try:
            observation = task.reset(
                (0,), seed=seed, initial="jammed", modes=np.asarray([mode]),
            )
        except (ValueError, RuntimeError) as exc:
            if not simulator or not _simulator_reset_rejection(exc):
                raise
            return -2., np.zeros(4, dtype=np.float32), 0, {
                "mode": mode, "reset_seed": seed,
                "success": False, "constraint_violation": False,
                "outcome": "reset_unreachable", "control_steps": 0,
                "final_position_error_m": None, "reward": -2.,
                "reset_error": str(exc),
            }
        actor = PublicLivenessMatingFeedbackPolicy(
            task.spec, feedback=self.feedback(policy), modes=(mode,), seed=seed,
        )
        for _ in range(task.config.timeout_steps):
            body_intent = task.public_body_intent(observation)
            if body_intent is None:
                raise RuntimeError("T4 shared public body intent missing")
            transition = task.step_contact(actor.act(observation), body_twist=body_intent)
            observation = transition.observation
            if bool(transition.done[0]):
                break
        episode = task.evaluate()["episodes"][-1]
        controls = int(episode["physical_control_steps"])
        error = float(episode["final_position_error_m"])
        if bool(episode["success"]):
            reward = 1. - .4 * controls / task.config.timeout_steps
        elif bool(episode["constraint_violation"]):
            reward = -1. - min(error, 1.)
        else:
            reward = -min(error, 1.) - .2 * controls / task.config.timeout_steps
        public_error = (
            np.asarray(observation["goal_position"][0], dtype=np.float64)
            - np.asarray(observation["tcp_position"][0], dtype=np.float64)
        )
        features = np.r_[np.clip(public_error / .25, -2., 2.),
                         controls / task.config.timeout_steps].astype(np.float32)
        summary = {
            "mode": mode, "reset_seed": seed,
            "success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "outcome": episode["outcome"], "control_steps": controls,
            "final_position_error_m": error, "reward": reward,
        }
        return reward, features, controls, summary

    def evaluate_policy(self, policy, tasks, seed):
        from genedynamics.solvers.single.adaptsim.learning import PolicySearchEvaluation

        del seed
        if not tasks:
            raise ValueError("at least one T4 simulated task required")
        scores, controls, episodes = [], 0, []
        for task in tasks:
            for mode in (0, 1):
                reward, _, steps, summary = self._episode(
                    policy, task.mechanism, mode, task.mode_seeds[mode],
                    simulator=True,
                )
                scores.append(reward)
                controls += steps
                episodes.append({
                    "mechanism": {
                        "servo_gain": task.mechanism.servo_gain,
                        "payload_mass": task.mechanism.payload_mass,
                        "delay_steps": task.mechanism.delay_steps,
                        "body_servo_gain": task.mechanism.body_servo_gain,
                    }, **summary,
                })
        return PolicySearchEvaluation(float(np.mean(scores)), controls, tuple(episodes))

    def evaluate_target(self, policy, mechanism, seed, *, simulator=False):
        from genedynamics.solvers.single.adaptsim.learning import AdaptationEvaluation

        scores, features, summaries = [], [], []
        controls = 0
        for mode in (0, 1):
            reset_seed = (
                self.reset_seed_pool[(seed + mode) % len(self.reset_seed_pool)]
                if self.reset_seed_pool else seed + mode
            )
            reward, feature, steps, summary = self._episode(
                policy, mechanism, mode, reset_seed, simulator=simulator,
            )
            scores.append(reward)
            features.append(feature)
            controls += steps
            summaries.append(summary)
        return (AdaptationEvaluation(float(np.mean(scores)),
                                     np.concatenate(features), controls),
                tuple(summaries))


def training_plan(config: dict, root: Path) -> dict:
    """Validate the complete mobile population/target budget before a reset."""
    from genedynamics.solvers.single.adaptsim import ParameterDistribution, ParameterSpec
    from genedynamics.solvers.single.adaptsim.learning import CEMTaskPolicySearch

    expected_execution = {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 7200, "maximum_gpu_memory_gib": 2,
    }
    expected_meta = {
        "outer_iterations": 2, "inner_episodes": 2,
        "adaptation_steps": 2, "kill_frequency": 1,
        "spawn_reward_threshold": .65, "meta_spawn_perturbation": .1,
        "inner_spawn_perturbation": .05, "batch_size": 2,
        "update_every": 1, "gradient_steps": 1,
        "exploration_probability": .5,
    }
    expected_search = {"population": 5, "elite": 2, "iterations": 2,
                       "tasks_per_candidate": 2}
    if (config.get("version") != TRAINING_VERSION or config.get("task") != "T4"
            or config.get("robot") != "go2_z1"
            or config.get("execution") != expected_execution
            or config.get("meta_training") != expected_meta
            or config.get("deployment_target", {}).get("updates") != 2
            or config["deployment_target"].get("select_particles") is not True
            or config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False
            or config.get("policy_fields") != list(POLICY_FIELDS)):
        raise ValueError("T4 AdaptSim training contract differs")
    pool = config.get("reset_seed_pool")
    if (not isinstance(pool, list) or len(pool) < 16
            or len(set(pool)) != len(pool)
            or any(not isinstance(seed, int) for seed in pool)):
        raise ValueError("T4 training requires a frozen, distinct reset seed pool")
    qualification = config.get("reset_seed_qualification")
    if not isinstance(qualification, dict) or set(qualification) != {"path", "sha256"}:
        raise ValueError("T4 reset seed pool needs a bound feasibility report")
    qualification_path = root / qualification["path"]
    if (not qualification_path.is_file()
            or hashlib.sha256(qualification_path.read_bytes()).hexdigest() != qualification["sha256"]):
        raise ValueError("T4 reset feasibility report missing or changed")
    feasibility = json.loads(qualification_path.read_text())
    passed = {row["seed"] for row in feasibility["rows"] if row["all_reset_passed"]}
    audit_pointer = config.get("reset_seed_audit")
    if not isinstance(audit_pointer, dict) or set(audit_pointer) != {"path", "sha256"}:
        raise ValueError("T4 reset seed pool needs a bound CPU audit")
    audit_path = root / audit_pointer["path"]
    if (not audit_path.is_file()
            or hashlib.sha256(audit_path.read_bytes()).hexdigest() != audit_pointer["sha256"]):
        raise ValueError("T4 reset seed audit missing or changed")
    seed_audit = json.loads(audit_path.read_text())
    if (feasibility.get("status") != "complete"
            or feasibility.get("policy_episodes") != 0
            or not set(pool) <= passed
            or seed_audit.get("status") != "passed"
            or seed_audit.get("report_sha256") != qualification["sha256"]
            or seed_audit.get("safe_seed_pool") != pool):
        raise ValueError("T4 training seed pool differs from reset-only qualification")
    search_cfg = config["task_policy_search"]
    if {key: search_cfg.get(key) for key in expected_search} != expected_search:
        raise ValueError("T4 AdaptSim task-policy search budget differs")
    task = MobileMatingPolicySearchTask(
        feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
        reset_seed_pool=tuple(pool),
    )
    task.feedback(search_cfg["anchor"])
    CEMTaskPolicySearch(
        search_cfg["lower"], search_cfg["upper"], search_cfg["anchor"],
        task.sample_tasks, task.evaluate_policy, **expected_search,
    )
    specs = [ParameterSpec(**row) for row in config["parameters"]]
    if [row.name for row in specs] != [
        "servo_gain", "payload_mass", "delay_steps", "body_servo_gain",
    ]:
        raise ValueError("T4 AdaptSim parameter order differs")
    if len(config.get("particles", [])) != 2:
        raise ValueError("T4 AdaptSim requires two real initial particles")
    for row in config["particles"]:
        distribution = ParameterDistribution(
            np.asarray(row["left"]), np.asarray(row["right"]), row["kind"],
        )
        if distribution.kind != "gaussian" or len(distribution.left) != 4:
            raise ValueError("T4 particles must be four-dimensional Gaussians")
    MobileContactMechanismSpec(**config["deployment_target"]["mechanism"])
    required = {
        "genedynamics/core/control/transfer.py",
        "genedynamics/envs/adapters/transfer_task.py",
        "genedynamics/solvers/single/adaptsim/adaptsim.py",
        "genedynamics/solvers/single/adaptsim/backends/branching_q_torch.py",
        "genedynamics/solvers/single/adaptsim/learning/adaptation.py",
        "genedynamics/solvers/single/adaptsim/learning/meta_training.py",
        "genedynamics/solvers/single/adaptsim/learning/task_policy_search.py",
        "genedynamics/tasks/transfer/contact.py",
        "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
        "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
        "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
        "genedynamics/tasks/transfer/controllers/mating_feedback.py",
        "genedynamics/tasks/transfer/episode.py",
        "genedynamics/tasks/transfer/evaluation.py",
        "genedynamics/tasks/transfer/mobile_mating_policy_search.py",
        "scripts/paper/cross_ebdm/train_adaptsim_development.py",
    }
    bindings = config.get("source_bindings")
    if not isinstance(bindings, list) or {row.get("path") for row in bindings} != required:
        raise ValueError("T4 AdaptSim requires exact task/solver source bindings")
    for row in bindings:
        path = root / row["path"]
        if (set(row) != {"path", "sha256"} or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError(f"T4 AdaptSim source binding differs: {row.get('path')}")
    output = root / config["output"]
    if (output.exists() or config["output"].split("/")[:5] != [
        "results", "quadruped", "mobile_keyed_insertion_recovery",
        "baseline", "adaptsim",
    ]):
        raise ValueError("T4 AdaptSim training output must be fresh and task-local")
    policy_calls = 21
    simulated_target_rollouts = 16
    deployment_target_rollouts = 5
    candidate_episodes = policy_calls * 5 * 2 * 2 * 2
    maximum_episodes = candidate_episodes + 2 * (
        simulated_target_rollouts + deployment_target_rollouts
    )
    if maximum_episodes != 882:
        raise RuntimeError("T4 AdaptSim population arithmetic differs")
    return {
        "version": TRAINING_VERSION, "status": "planned_unexecuted",
        "task": "T4", "scope": "single-instance mobile AdaptSim development",
        "source_bindings": bindings,
        "policy_training_calls": policy_calls,
        "simulated_target_rollouts": simulated_target_rollouts,
        "deployment_target_rollouts": deployment_target_rollouts,
        "candidate_physics_episodes": candidate_episodes,
        "maximum_physics_episodes": maximum_episodes,
        "maximum_wall_seconds": expected_execution["maximum_wall_seconds"],
        "output": config["output"], "all_failures_retained": True,
        "formal_performance_evidence": False,
    }


def run_training(config: dict, root: Path) -> dict:
    """Train and deploy one T4 AdaptSim instance with a charged episode ledger."""
    import torch
    from genedynamics.solvers.single.adaptsim import (
        AdaptSimSolver, ParameterDistribution, ParameterSpec,
    )
    from genedynamics.solvers.single.adaptsim.learning import (
        AdaptSimAdaptation, AdaptSimMetaTraining, CEMTaskPolicySearch,
    )

    plan = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T4 AdaptSim training requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T4 AdaptSim training saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    for row in config["source_bindings"]:
        snapshot = output / "source_snapshot" / row["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / row["path"], snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("T4 AdaptSim source snapshot differs")
    started = time.monotonic()
    training_ledger, simulated_ledger, deployment_ledger = [], [], []
    report = {"version": TRAINING_VERSION, "task": "T4", "status": "running"}

    def check_time():
        if time.monotonic() - started > plan["maximum_wall_seconds"]:
            raise TimeoutError("T4 AdaptSim wall ceiling exceeded")

    def write_progress():
        atomic_json(output / "progress.json", {
            "version": TRAINING_VERSION, "status": "running",
            "wall_seconds": time.monotonic() - started,
            "policy_training_calls": len(training_ledger),
            "candidate_physics_episodes": sum(
                len(candidate["episodes"]) for training in training_ledger
                for candidate in training["candidates"]
            ),
            "simulated_target_rollouts": len(simulated_ledger),
            "deployment_target_rollouts": len(deployment_ledger),
        })

    try:
        seed = int(config["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        task = MobileMatingPolicySearchTask(
            feedback_base=config["feedback_base"],
            shared_execution=config["shared_execution"],
            reset_seed_pool=tuple(config["reset_seed_pool"]),
        )
        search_cfg = config["task_policy_search"]
        search = CEMTaskPolicySearch(
            search_cfg["lower"], search_cfg["upper"], search_cfg["anchor"],
            task.sample_tasks, task.evaluate_policy,
            **{key: search_cfg[key] for key in (
                "population", "elite", "iterations", "tasks_per_candidate",
            )},
        )
        solver = AdaptSimSolver(
            [ParameterSpec(**row) for row in config["parameters"]],
            feature_dim=8, hidden=(32, 32), gamma=.9, tau=.01, device="cuda:0",
        )
        particles = [ParameterDistribution(
            np.asarray(row["left"]), np.asarray(row["right"]), row["kind"],
        ) for row in config["particles"]]

        def train(distribution, warm_start, call_seed):
            check_time()
            trained = search(distribution, warm_start, call_seed)
            training_ledger.append({
                "seed": call_seed,
                "distribution": {"kind": distribution.kind,
                                 "left": distribution.left.tolist(),
                                 "right": distribution.right.tolist()},
                "policy": trained.policy.tolist(),
                "simulation_steps": trained.simulation_steps,
                "candidates": list(search.last_ledger),
            })
            write_progress()
            return trained

        def simulated_target(target_seed):
            rng = np.random.default_rng(target_seed)
            return MobileContactMechanismSpec(
                servo_gain=float(rng.uniform(.55, 1.15)),
                payload_mass=float(rng.uniform(.3, 1.0)),
                delay_steps=int(rng.integers(0, 4)),
                body_servo_gain=float(rng.uniform(.85, 1.15)),
            )

        def evaluate(policy, mechanism, call_seed, ledger):
            check_time()
            result, episodes = task.evaluate_target(
                policy, mechanism, call_seed,
                simulator=ledger is simulated_ledger,
            )
            ledger.append({
                "seed": call_seed, "reward": result.reward,
                "control_steps": result.target_control_steps,
                "features": result.features.tolist(), "episodes": list(episodes),
            })
            write_progress()
            return result

        meta_cfg = config["meta_training"]
        meta = AdaptSimMetaTraining(
            solver, train, simulated_target,
            lambda policy, mechanism, call_seed: evaluate(
                policy, mechanism, call_seed, simulated_ledger,
            ), seed=seed,
        ).run_population(
            particles, iterations=meta_cfg["outer_iterations"],
            inner_episodes=meta_cfg["inner_episodes"],
            adaptation_steps=meta_cfg["adaptation_steps"],
            kill_frequency=meta_cfg["kill_frequency"],
            spawn_reward_threshold=meta_cfg["spawn_reward_threshold"],
            meta_spawn_perturbation=meta_cfg["meta_spawn_perturbation"],
            inner_spawn_perturbation=meta_cfg["inner_spawn_perturbation"],
            batch_size=meta_cfg["batch_size"],
            update_every=meta_cfg["update_every"],
            gradient_steps=meta_cfg["gradient_steps"],
            exploration_probability=meta_cfg["exploration_probability"],
        )
        target = MobileContactMechanismSpec(**config["deployment_target"]["mechanism"])
        adaptation = AdaptSimAdaptation(
            solver, train,
            lambda policy, call_seed: evaluate(
                policy, target, call_seed, deployment_ledger,
            ), seed=seed + 500000,
        )
        selection = adaptation.select_particle(list(meta.final_particles))
        deployment = adaptation.run(
            selection.distribution, updates=config["deployment_target"]["updates"],
            initial_policy=selection.policy,
        )
        candidate_count = sum(
            len(candidate["episodes"]) for training in training_ledger
            for candidate in training["candidates"]
        )
        if (len(training_ledger) != 21 or len(simulated_ledger) != 16
                or len(deployment_ledger) != 5 or candidate_count != 840
                or candidate_count + 2 * (len(simulated_ledger) + len(deployment_ledger)) != 882
                or sum(row["simulation_steps"] for row in training_ledger) !=
                meta.simulation_steps + selection.simulation_steps + deployment["simulation_steps"]):
            raise RuntimeError("T4 AdaptSim rollout ledger differs from frozen budget")
        check_time()
        checkpoint = output / "inference_and_policy.pt"
        fd, temporary = tempfile.mkstemp(dir=output, prefix="checkpoint-", suffix=".pt")
        os.close(fd)
        try:
            torch.save({
                "version": TRAINING_VERSION,
                "q_network": {key: value.detach().cpu()
                              for key, value in solver.backend.network.state_dict().items()},
                "final_policy": np.asarray(deployment["final_policy"]).tolist(),
            }, temporary)
            os.replace(temporary, checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report.update({
            "status": "complete_development",
            "meta_training": {
                "records": list(meta.records), "iterations": meta.iterations,
                "inner_episodes": meta.inner_episodes,
                "inference_updates": meta.inference_updates,
                "simulation_steps": meta.simulation_steps,
                "simulated_target_rollouts": meta.simulated_target_rollouts,
                "simulated_target_control_steps": meta.simulated_target_control_steps,
                "final_particles": [{"kind": item.kind,
                                     "left": item.left.tolist(),
                                     "right": item.right.tolist()}
                                    for item in meta.final_particles],
            },
            "deployment": {
                "selection": {"selected_index": selection.selected_index,
                              "records": list(selection.records),
                              "simulation_steps": selection.simulation_steps,
                              "target_control_steps": selection.target_control_steps},
                "records": deployment["records"],
                "simulation_steps": deployment["simulation_steps"],
                "target_control_steps": deployment["target_control_steps"],
                "final_policy": np.asarray(deployment["final_policy"]).tolist(),
            },
            "checkpoint": {"path": str(checkpoint.relative_to(root)),
                           "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            "wall_seconds": time.monotonic() - started,
            "formal_performance_evidence": False,
        })
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
                       "traceback": traceback.format_exc(),
                       "wall_seconds": time.monotonic() - started})
        raise
    finally:
        report["policy_training_ledger"] = training_ledger
        report["simulated_target_ledger"] = simulated_ledger
        report["deployment_target_ledger"] = deployment_ledger
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "version": TRAINING_VERSION, "status": report["status"],
            "wall_seconds": report.get("wall_seconds", time.monotonic() - started),
            "policy_training_calls": len(training_ledger),
            "candidate_physics_episodes": sum(
                len(candidate["episodes"]) for training in training_ledger
                for candidate in training["candidates"]
            ),
            "simulated_target_rollouts": len(simulated_ledger),
            "deployment_target_rollouts": len(deployment_ledger),
        })
    return report
