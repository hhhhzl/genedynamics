"""Task-owned T3 binding for simulator-trained closed-loop mating feedback.

AdaptSim may search the declared public-feedback parameters under sampled
construction mechanisms. The actor observes only public robot and contact
history. This task-policy optimizer is an explicit long-horizon adaptation of
the authors' environment-specific inner learner, not their native NAF policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.core.artifacts import atomic_json
from genedynamics.solvers.single.adaptsim import (
    AdaptSimSolver, ParameterDistribution, ParameterSpec,
)
from genedynamics.solvers.single.adaptsim.learning import (
    AdaptationEvaluation, AdaptSimAdaptation, AdaptSimMetaTraining,
    CEMTaskPolicySearch, PolicySearchEvaluation,
)
from genedynamics.tasks.transfer.contact import ContactMechanismSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies import (
    PublicBiasFeedbackSpec,
)
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicLivenessMatingFeedbackPolicy,
)


POLICY_FIELDS = (
    "recovery_spatial_stiffness", "position_gain", "compensation_gain",
    "lead_xy_m", "normal_admittance_gain", "insertion_force_scale",
    "guard_prediction_seconds",
)
POLICY_LOWER = np.asarray([-.5, 2., .2, .001, .001, .5, 0.], dtype=np.float64)
POLICY_UPPER = np.asarray([.8, 10., 2., .01, .01, 2., .2], dtype=np.float64)


@dataclass(frozen=True)
class MatingSearchTask:
    mechanism: ContactMechanismSpec
    mode_seeds: tuple[int, int]


class MatingPolicySearchTask:
    """Evaluate one seven-parameter public feedback policy in both T3 modes."""

    def __init__(self, *, robot: str, feedback_base: dict[str, Any],
                 shared_execution: dict[str, Any],
                 config_overrides: dict[str, Any] | None = None,
                 contact_time_constant: float = .02) -> None:
        if robot not in {"panda", "xarm7"}:
            raise ValueError("T3 mating policy search requires a fixed arm")
        if not np.isfinite(contact_time_constant) or contact_time_constant <= 0:
            raise ValueError("contact time constant must be positive")
        self.robot = robot
        self.feedback_base = dict(feedback_base)
        self.shared_execution = dict(shared_execution)
        self.config_overrides = dict(config_overrides or {})
        self.contact_time_constant = float(contact_time_constant)
        if self.shared_execution.get("retreat_release_version") != "safe_settled_v1":
            raise ValueError("T3 task-policy search needs the shared safe-settled release")
        if self.feedback_base.get("insertion_lead_z_m") != self.shared_execution.get("insertion_lead_z_m"):
            raise ValueError("feedback insertion lead disagrees with shared execution")
        PublicBiasFeedbackSpec(**self.feedback_base)

    @staticmethod
    def _vector(policy: Any) -> np.ndarray:
        vector = np.asarray(policy, dtype=np.float64)
        if (vector.shape != (len(POLICY_FIELDS),)
                or not np.isfinite(vector).all()
                or np.any(vector < POLICY_LOWER)
                or np.any(vector > POLICY_UPPER)):
            raise ValueError("T3 feedback policy differs from declared public support")
        return vector

    def feedback(self, policy: Any) -> PublicBiasFeedbackSpec:
        vector = self._vector(policy)
        values = {**self.feedback_base,
                  **dict(zip(POLICY_FIELDS, vector.tolist(), strict=True))}
        return PublicBiasFeedbackSpec(**values)

    def sample_tasks(self, distribution: ParameterDistribution,
                     rng: np.random.Generator, count: int
                     ) -> tuple[MatingSearchTask, ...]:
        if distribution.kind not in {"gaussian", "uniform"} or len(distribution.left) != 3:
            raise ValueError("T3 AdaptSim distribution needs gain, mass and delay")
        tasks = []
        for gain, mass, delay in distribution.sample(rng, count):
            mechanism = ContactMechanismSpec(
                servo_gain=float(np.clip(gain, .4, 1.4)),
                payload_mass=float(np.clip(mass, .2, 1.2)),
                delay_steps=int(np.clip(np.rint(delay), 0, 3)),
                contact_time_constant=self.contact_time_constant,
            )
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=2))
            tasks.append(MatingSearchTask(mechanism, seeds))
        return tuple(tasks)

    def _episode(self, policy: Any, mechanism: ContactMechanismSpec,
                 mode: int, seed: int
                 ) -> tuple[float, np.ndarray, int, dict[str, Any]]:
        feedback = self.feedback(policy)
        task = instantiate_task(
            "T3", robot=self.robot, mechanism_spec=mechanism, seed=seed,
            config_overrides=self.config_overrides,
            shared_execution=self.shared_execution,
        )
        observation = task.reset(
            (0,), seed=seed, initial="jammed", modes=np.asarray([mode]),
        )
        actor = PublicLivenessMatingFeedbackPolicy(
            task.spec, feedback=feedback, modes=(mode,), seed=seed,
        )
        for _ in range(task.config.timeout_steps):
            transition = task.step(actor.act(observation))
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

    def evaluate_policy(self, policy: Any, tasks: tuple[MatingSearchTask, ...],
                        seed: int) -> PolicySearchEvaluation:
        del seed
        vector = self._vector(policy)
        if not tasks:
            raise ValueError("at least one simulated mating task required")
        scores, controls, episodes = [], 0, []
        for task in tasks:
            for mode in (0, 1):
                reward, _, steps, summary = self._episode(
                    vector, task.mechanism, mode, task.mode_seeds[mode],
                )
                scores.append(reward)
                controls += steps
                episodes.append({
                    "mechanism": {
                        "servo_gain": task.mechanism.servo_gain,
                        "payload_mass": task.mechanism.payload_mass,
                        "delay_steps": task.mechanism.delay_steps,
                        "contact_time_constant": task.mechanism.contact_time_constant,
                    },
                    **summary,
                })
        return PolicySearchEvaluation(float(np.mean(scores)), controls, tuple(episodes))

    def evaluate_target(self, policy: Any, mechanism: ContactMechanismSpec,
                        seed: int
                        ) -> tuple[AdaptationEvaluation, tuple[dict[str, Any], ...]]:
        vector = self._vector(policy)
        scores, features, summaries = [], [], []
        controls = 0
        for mode in (0, 1):
            reward, feature, steps, summary = self._episode(
                vector, mechanism, mode, seed + mode,
            )
            scores.append(reward)
            features.append(feature)
            controls += steps
            summaries.append(summary)
        return (AdaptationEvaluation(float(np.mean(scores)),
                                     np.concatenate(features), controls),
                tuple(summaries))


__all__ = ["MatingPolicySearchTask", "MatingSearchTask", "POLICY_FIELDS"]


DEVELOPMENT_VERSION = "adaptsim-mating-feedback-development-v1"
POPULATION_DEVELOPMENT_VERSION = "adaptsim-mating-feedback-population-development-v2"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"


def development_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Check the frozen T3 integration budget before any simulator reset."""
    version = config.get("version")
    if version not in {DEVELOPMENT_VERSION, POPULATION_DEVELOPMENT_VERSION}:
        raise ValueError("unknown T3 AdaptSim development version")
    population = version == POPULATION_DEVELOPMENT_VERSION
    maximum_wall_seconds = 5400 if population else 900
    if (config.get("task") != "T3"
            or config.get("robot") != "xarm7"
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": maximum_wall_seconds,
                "maximum_gpu_memory_gib": 2,
            }):
        raise ValueError("T3 AdaptSim development contract differs")
    expected_meta = (
        {"outer_iterations": 2, "inner_episodes": 2,
         "adaptation_steps": 2, "kill_frequency": 1,
         "spawn_reward_threshold": .65, "meta_spawn_perturbation": .1,
         "inner_spawn_perturbation": .05,
         "batch_size": 2, "update_every": 1, "gradient_steps": 1,
         "exploration_probability": .5}
        if population else
        {"episodes": 1, "adaptation_steps": 1, "batch_size": 1,
         "update_every": 1, "gradient_steps": 1,
         "exploration_probability": .5}
    )
    expected_updates = 2 if population else 1
    if (config.get("meta_training") != expected_meta
            or config.get("deployment_target", {}).get("updates") != expected_updates
            or config["deployment_target"].get("select_particles") is not True):
        raise ValueError("T3 AdaptSim meta/deployment budget differs")
    search = config["task_policy_search"]
    expected_search = (
        {"population": 5, "elite": 2, "iterations": 2, "tasks_per_candidate": 2}
        if population else
        {"population": 5, "elite": 1, "iterations": 1, "tasks_per_candidate": 1}
    )
    if {key: search[key] for key in (
        "population", "elite", "iterations", "tasks_per_candidate",
    )} != expected_search:
        raise ValueError("T3 policy training budget differs")
    if (config.get("policy_fields") != list(POLICY_FIELDS)
            or len(search["anchor"]) != len(POLICY_FIELDS)
            or not (np.asarray(search["lower"]) <= np.asarray(search["anchor"])).all()
            or not (np.asarray(search["anchor"]) <= np.asarray(search["upper"])).all()):
        raise ValueError("T3 public feedback policy bounds differ")
    task = MatingPolicySearchTask(
        robot=config["robot"], feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
    )
    task.feedback(search["anchor"])
    CEMTaskPolicySearch(
        search["lower"], search["upper"], search["anchor"],
        task.sample_tasks, task.evaluate_policy,
        **expected_search,
    )
    specs = [ParameterSpec(**row) for row in config["parameters"]]
    if [row.name for row in specs] != ["servo_gain", "payload_mass", "delay_steps"]:
        raise ValueError("T3 AdaptSim parameter order differs")
    if len(config["particles"]) != 2:
        raise ValueError("T3 development requires two initial particles")
    for row in config["particles"]:
        dist = ParameterDistribution(np.asarray(row["left"]),
                                     np.asarray(row["right"]), row["kind"])
        if dist.kind != "gaussian" or len(dist.left) != 3:
            raise ValueError("T3 particles must be three-dimensional Gaussian distributions")
    ContactMechanismSpec(**config["deployment_target"]["mechanism"])
    required = {
        "genedynamics/solvers/single/adaptsim/adaptsim.py",
        "genedynamics/solvers/single/adaptsim/backends/branching_q_torch.py",
        "genedynamics/solvers/single/adaptsim/learning/meta_training.py",
        "genedynamics/solvers/single/adaptsim/learning/adaptation.py",
        "genedynamics/solvers/single/adaptsim/learning/task_policy_search.py",
        "genedynamics/tasks/transfer/mating_policy_search.py",
        "genedynamics/tasks/transfer/controllers/mating_feedback.py",
        "scripts/paper/cross_ebdm/train_adaptsim_development.py",
    }
    if population:
        required |= {
            "genedynamics/envs/adapters/transfer_task.py",
            "genedynamics/tasks/transfer/contact.py",
            "genedynamics/tasks/transfer/episode.py",
            "genedynamics/tasks/transfer/evaluation.py",
            "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
            "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
            "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
            "genedynamics/core/control/transfer.py",
        }
    bindings = config.get("source_bindings")
    if not isinstance(bindings, list) or {row.get("path") for row in bindings} != required:
        raise ValueError("T3 development requires exact source bindings")
    for row in bindings:
        if set(row) != {"path", "sha256"}:
            raise ValueError("source binding requires path and SHA256")
        path = root / row["path"]
        if (not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError(f"T3 source binding changed: {path}")
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"T3 AdaptSim output exists: {output}")
    particle_count = len(config["particles"])
    if population:
        outer = expected_meta["outer_iterations"]
        inner = expected_meta["inner_episodes"]
        steps = expected_meta["adaptation_steps"]
        meta_policy_calls = outer * particle_count + outer * inner * (steps + 1)
    else:
        meta_policy_calls = expected_meta["episodes"] * (expected_meta["adaptation_steps"] + 1)
    policy_calls = meta_policy_calls + particle_count + expected_updates + 1
    simulated_target_rollouts = meta_policy_calls
    deployment_target_rollouts = particle_count + expected_updates + 1
    candidate_episodes = (policy_calls * expected_search["population"]
                          * expected_search["iterations"]
                          * expected_search["tasks_per_candidate"] * 2)
    maximum_physics_episodes = candidate_episodes + 2 * (
        simulated_target_rollouts + deployment_target_rollouts
    )
    if population and (policy_calls, simulated_target_rollouts,
                       deployment_target_rollouts, maximum_physics_episodes) != (21, 16, 5, 882):
        raise RuntimeError("T3 population budget arithmetic differs")
    return {
        "version": version, "status": "planned_unexecuted",
        "scope": "single-instance T3 AdaptSim method integration; task-adapted CEM inner learner; not main-table evidence",
        "source_bindings": bindings,
        "policy_training_calls": policy_calls,
        "simulated_target_rollouts": simulated_target_rollouts,
        "deployment_target_rollouts": deployment_target_rollouts,
        "candidate_physics_episodes": candidate_episodes,
        "maximum_physics_episodes": maximum_physics_episodes,
        "maximum_wall_seconds": maximum_wall_seconds,
        "output": config["output"],
        "all_failures_retained": True,
        "formal_performance_evidence": False,
    }


def run_development(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run one bounded, fully charged T3 meta/deployment integration pilot."""
    import torch

    plan = development_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T3 AdaptSim development requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T3 AdaptSim development saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(
        min(2 * 1024**3 / gpu.total_memory, 1.), 0,
    )
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    for row in config["source_bindings"]:
        source = root / row["path"]
        snapshot = output / "source_snapshot" / row["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("T3 source snapshot differs")
    started = time.monotonic()
    training_ledger: list[dict[str, Any]] = []
    simulated_target_ledger: list[dict[str, Any]] = []
    deployment_target_ledger: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": config["version"], "status": "running"}

    def check_time() -> None:
        if time.monotonic() - started > plan["maximum_wall_seconds"]:
            raise TimeoutError("T3 AdaptSim development wall ceiling exceeded")

    def write_progress() -> None:
        atomic_json(output / "progress.json", {
            "version": config["version"], "status": "running",
            "wall_seconds": time.monotonic() - started,
            "policy_training_calls": len(training_ledger),
            "candidate_physics_episodes": sum(
                len(candidate["episodes"])
                for training in training_ledger
                for candidate in training["candidates"]
            ),
            "simulated_target_rollouts": len(simulated_target_ledger),
            "deployment_target_rollouts": len(deployment_target_ledger),
        })

    try:
        seed = int(config["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        task = MatingPolicySearchTask(
            robot=config["robot"], feedback_base=config["feedback_base"],
            shared_execution=config["shared_execution"],
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
            feature_dim=8, hidden=(32, 32), gamma=.9, tau=.01,
            device="cuda:0",
        )
        particles = [ParameterDistribution(np.asarray(row["left"]),
                                           np.asarray(row["right"]), row["kind"])
                     for row in config["particles"]]

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
            return ContactMechanismSpec(
                servo_gain=float(rng.uniform(.55, 1.15)),
                payload_mass=float(rng.uniform(.3, 1.0)),
                delay_steps=int(rng.integers(0, 4)),
            )

        def evaluate_simulated(policy, mechanism, call_seed):
            check_time()
            evaluation, episodes = task.evaluate_target(policy, mechanism, call_seed)
            simulated_target_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(), "episodes": list(episodes),
            })
            write_progress()
            return evaluation

        meta_cfg = config["meta_training"]
        meta_trainer = AdaptSimMetaTraining(
            solver, train, simulated_target, evaluate_simulated, seed=seed,
        )
        if config["version"] == POPULATION_DEVELOPMENT_VERSION:
            meta = meta_trainer.run_population(
                particles,
                iterations=meta_cfg["outer_iterations"],
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
            particles = list(meta.final_particles)
        else:
            meta = meta_trainer.run(
                particles, episodes=meta_cfg["episodes"],
                adaptation_steps=meta_cfg["adaptation_steps"],
                batch_size=meta_cfg["batch_size"],
                update_every=meta_cfg["update_every"],
                gradient_steps=meta_cfg["gradient_steps"],
                exploration_probability=meta_cfg["exploration_probability"],
            )
        target_mechanism = ContactMechanismSpec(**config["deployment_target"]["mechanism"])

        def evaluate_deployment(policy, call_seed):
            check_time()
            evaluation, episodes = task.evaluate_target(policy, target_mechanism, call_seed)
            deployment_target_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(), "episodes": list(episodes),
            })
            write_progress()
            return evaluation

        adaptation = AdaptSimAdaptation(
            solver, train, evaluate_deployment, seed=seed + 500000,
        )
        selection = adaptation.select_particle(particles)
        deployment = adaptation.run(
            selection.distribution,
            updates=config["deployment_target"]["updates"],
            initial_policy=selection.policy,
        )
        candidate_episodes = sum(
            len(candidate["episodes"])
            for training in training_ledger
            for candidate in training["candidates"]
        )
        if (len(training_ledger) != plan["policy_training_calls"]
                or len(simulated_target_ledger) != plan["simulated_target_rollouts"]
                or len(deployment_target_ledger) != plan["deployment_target_rollouts"]
                or candidate_episodes != plan["candidate_physics_episodes"]
                or candidate_episodes + 2 * (
                    len(simulated_target_ledger) + len(deployment_target_ledger)
                ) != plan["maximum_physics_episodes"]
                or sum(row["simulation_steps"] for row in training_ledger) !=
                meta.simulation_steps + selection.simulation_steps + deployment["simulation_steps"]):
            raise RuntimeError("T3 AdaptSim rollout ledger differs from frozen budget")
        check_time()
        checkpoint = output / "inference_and_policy.pt"
        fd, temporary = tempfile.mkstemp(dir=output, prefix="checkpoint-", suffix=".pt")
        os.close(fd)
        try:
            torch.save({
                "version": config["version"],
                "q_network": {key: value.detach().cpu() for key, value in solver.backend.network.state_dict().items()},
                "final_policy": np.asarray(deployment["final_policy"]).tolist(),
            }, temporary)
            os.replace(temporary, checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        meta_report = {"records": list(meta.records),
                       "inference_updates": meta.inference_updates,
                       "simulation_steps": meta.simulation_steps,
                       "simulated_target_control_steps": meta.simulated_target_control_steps}
        if config["version"] == POPULATION_DEVELOPMENT_VERSION:
            meta_report.update({
                "iterations": meta.iterations,
                "inner_episodes": meta.inner_episodes,
                "simulated_target_rollouts": meta.simulated_target_rollouts,
                "final_particles": [{"kind": particle.kind,
                                     "left": particle.left.tolist(),
                                     "right": particle.right.tolist()}
                                    for particle in meta.final_particles],
            })
        report.update({
            "status": "complete_development",
            "meta_training": meta_report,
            "deployment": {"selection": {"selected_index": selection.selected_index,
                                          "records": list(selection.records),
                                          "simulation_steps": selection.simulation_steps,
                                          "target_control_steps": selection.target_control_steps},
                           "records": deployment["records"],
                           "simulation_steps": deployment["simulation_steps"],
                           "target_control_steps": deployment["target_control_steps"],
                           "final_policy": np.asarray(deployment["final_policy"]).tolist()},
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
        report["simulated_target_ledger"] = simulated_target_ledger
        report["deployment_target_ledger"] = deployment_target_ledger
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "version": config["version"], "status": report["status"],
            "wall_seconds": report.get("wall_seconds", time.monotonic() - started),
            "policy_training_calls": len(training_ledger),
            "candidate_physics_episodes": sum(
                len(candidate["episodes"])
                for training in training_ledger
                for candidate in training["candidates"]
            ),
            "simulated_target_rollouts": len(simulated_target_ledger),
            "deployment_target_rollouts": len(deployment_target_ledger),
        })
    return report


__all__ += ["DEVELOPMENT_VERSION", "POPULATION_DEVELOPMENT_VERSION",
            "development_plan", "run_development"]
