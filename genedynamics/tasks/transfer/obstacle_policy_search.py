"""T1 task binding for simulator-trained obstacle-path policies.

This module owns MuJoCo rollouts, public task reward/features, and the shared
13D execution chart.  AdaptSim owns simulator-distribution and policy-search
algorithms in its own solver package.  The construction mechanism is passed
only to the simulator, never to the policy or inference features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile
import time
import traceback

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.adaptsim import ParameterDistribution
from genedynamics.solvers.single.adaptsim.learning import (
    AdaptSimAdaptation,
    AdaptSimMetaTraining,
    CEMTaskPolicySearch,
    AdaptationEvaluation,
    PolicySearchEvaluation,
)
from genedynamics.solvers.single.adaptsim import AdaptSimSolver, ParameterSpec
from genedynamics.core.artifacts import atomic_json
from genedynamics.tasks.transfer.controllers.obstacle_feedback import OBSTACLE_BOUNDS
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from genedynamics.tasks.transfer.episode import episode_evidence
from genedynamics.tasks.transfer.runtime import MechanismSpec


@dataclass(frozen=True)
class ObstacleSearchTask:
    mechanism: MechanismSpec
    mode_seeds: tuple[int, int]


class ObstaclePolicySearchTask:
    """Evaluate a six-parameter path/velocity policy on both T1 modes."""

    def __init__(self, *, robot: str = "xarm7", delay_steps: int = 0) -> None:
        if robot not in {"xarm7", "ur5e"} or delay_steps < 0:
            raise ValueError("T1 path search supports xArm7/UR5e and nonnegative delay")
        self.robot = robot
        self.delay_steps = int(delay_steps)

    @staticmethod
    def _vector(policy: Any) -> np.ndarray:
        vector = np.asarray(policy, dtype=np.float64)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("T1 task policy needs four path values and two mode speed scales")
        if np.any(vector[4:] <= 0):
            raise ValueError("T1 mode speed scales must be positive")
        return vector

    def sample_tasks(
        self, distribution: ParameterDistribution, rng: np.random.Generator,
        count: int,
    ) -> tuple[ObstacleSearchTask, ...]:
        if distribution.kind not in {"gaussian", "uniform"} or len(distribution.left) != 2:
            raise ValueError("T1 AdaptSim distribution must contain gain and payload mass")
        parameters = distribution.sample(rng, count)
        tasks = []
        for gain, mass in parameters:
            # The task declares positive continuous mechanism bounds.  The
            # sampler clips Gaussian tails before constructing a simulator.
            mechanism = MechanismSpec(
                servo_gain=float(np.clip(gain, 0.4, 1.4)),
                payload_mass=float(np.clip(mass, 0.2, 1.2)),
                delay_steps=self.delay_steps,
            )
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=2))
            tasks.append(ObstacleSearchTask(mechanism, seeds))
        return tuple(tasks)

    def _episode(
        self, policy: np.ndarray, mechanism: MechanismSpec, mode: int, seed: int,
        *, evidence_path: Path | None = None,
    ) -> tuple[float, np.ndarray, int, dict[str, Any]]:
        task = instantiate_task("T1", robot=self.robot, mechanism_spec=mechanism, seed=seed)
        observation = task.reset((0,), seed=seed, modes=np.array([mode]))
        actor = PublicObstaclePathPolicy(task.spec, modes=[mode], path=policy[:4])
        for _ in range(task.config.timeout_steps):
            action = actor.act(observation)
            action[:, :3] = np.clip(
                action[:, :3] * policy[4 + mode],
                -OBSTACLE_BOUNDS[:3], OBSTACLE_BOUNDS[:3],
            )
            transition = task.step(action)
            observation = transition.observation
            if bool(transition.done[0]):
                break
        evaluation = task.evaluate()
        episode = evaluation["episodes"][-1]
        controls = int(episode["physical_control_steps"])
        error = float(episode["final_position_error_m"])
        if bool(episode["success"]):
            reward = 1.0 - 0.4 * controls / task.config.timeout_steps
        elif bool(episode["constraint_violation"]):
            reward = -1.0 - min(error, 1.0)
        else:
            reward = -min(error, 1.0) - 0.2 * controls / task.config.timeout_steps
        public_error = (
            np.asarray(observation["goal_position"][0], dtype=np.float64)
            - np.asarray(observation["tcp_position"][0], dtype=np.float64)
        )
        features = np.r_[np.clip(public_error / 0.25, -2., 2.),
                         controls / task.config.timeout_steps].astype(np.float32)
        summary = {
            "mode": mode, "reset_seed": seed,
            "success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "outcome": episode["outcome"],
            "control_steps": controls,
            "final_position_error_m": error,
            "reward": reward,
        }
        if evidence_path is not None:
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            if evidence_path.exists():
                raise FileExistsError(f"episode evidence exists: {evidence_path}")
            content = gzip.compress(
                json.dumps(episode_evidence(task.records, task.spec, evaluation),
                           sort_keys=True, allow_nan=False).encode(),
                mtime=0,
            )
            fd, temporary = tempfile.mkstemp(
                dir=evidence_path.parent, prefix="episode-", suffix=".json.gz",
            )
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                os.replace(temporary, evidence_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            summary["evidence"] = {
                "path": str(evidence_path),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        return reward, features, controls, summary

    def evaluate_policy(
        self, policy: np.ndarray, tasks: tuple[ObstacleSearchTask, ...], seed: int,
    ) -> PolicySearchEvaluation:
        vector = self._vector(policy)
        if not tasks:
            raise ValueError("at least one sampled simulator task required")
        scores, controls, episodes = [], 0, []
        for task in tasks:
            for mode in (0, 1):
                reward, _, steps, summary = self._episode(
                    vector, task.mechanism, mode, task.mode_seeds[mode]
                )
                scores.append(reward)
                controls += steps
                episodes.append({
                    "mechanism": {
                        "servo_gain": task.mechanism.servo_gain,
                        "payload_mass": task.mechanism.payload_mass,
                        "delay_steps": task.mechanism.delay_steps,
                    },
                    **summary,
                })
        return PolicySearchEvaluation(float(np.mean(scores)), controls, tuple(episodes))

    def evaluate_target(
        self, policy: np.ndarray, mechanism: MechanismSpec, seed: int,
    ) -> tuple[AdaptationEvaluation, tuple[dict[str, Any], ...]]:
        vector = self._vector(policy)
        scores, features, summaries = [], [], []
        controls = 0
        for mode in (0, 1):
            reward, feature, steps, summary = self._episode(
                vector, mechanism, mode, seed + mode
            )
            scores.append(reward)
            features.append(feature)
            controls += steps
            summaries.append(summary)
        return (
            AdaptationEvaluation(float(np.mean(scores)),
                                 np.concatenate(features), controls),
            tuple(summaries),
        )


__all__ = ["ObstaclePolicySearchTask", "ObstacleSearchTask"]


DEVELOPMENT_VERSION = "adaptsim-obstacle-development-v1"
META_DEVELOPMENT_VERSION = "adaptsim-obstacle-meta-development-v2"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"


def development_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate a T1-only, non-paper AdaptSim pilot before any physics call."""
    version = config.get("version")
    if version not in {DEVELOPMENT_VERSION, META_DEVELOPMENT_VERSION} or config.get("task") != "T1":
        raise ValueError("expected T1 AdaptSim development config")
    if config.get("robot") != "xarm7" or config.get("delay_steps") != 0:
        raise ValueError("first AdaptSim task binding is xArm7 with fixed zero delay")
    execution = config.get("execution", {})
    if (execution.get("cuda_visible_devices") != "1"
            or execution.get("physical_uuid") != GPU1_UUID
            or execution.get("maximum_gpu_memory_gib") != 2
            or execution.get("maximum_wall_seconds") not in (
                {180, 240} if version == DEVELOPMENT_VERSION else {1500}
            )
            or set(execution) != {"cuda_visible_devices", "physical_uuid", "maximum_wall_seconds", "maximum_gpu_memory_gib"}):
        raise ValueError("development pilot must retain its GPU1/time/memory limits")
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"pilot output exists: {output}")
    search = config["task_policy_search"]
    CEMTaskPolicySearch(
        search["lower"], search["upper"], search["anchor"],
        lambda distribution, rng, count: None,
        lambda policy, tasks, seed: PolicySearchEvaluation(0., 0),
        population=search["population"], elite=search["elite"],
        iterations=search["iterations"],
        tasks_per_candidate=search["tasks_per_candidate"],
        coordinate_probes=search.get("coordinate_probes", ()),
        probe_fraction=search.get("probe_fraction", 0.1),
    )
    if len(search["anchor"]) != 6:
        raise ValueError("T1 task policy vector must have six components")
    specs = [ParameterSpec(**row) for row in config["parameters"]]
    if [row.name for row in specs] != ["servo_gain", "payload_mass"]:
        raise ValueError("T1 mechanism parameter order is gain, then payload")
    meta = config["meta_training"]
    target = config["deployment_target"]
    select_particles = bool(target.get("select_particles", False))
    outer_iterations = int(meta.get("outer_iterations", 0))
    if version == DEVELOPMENT_VERSION:
        if int(meta["adaptation_steps"]) != 1 or int(target["updates"]) != 1:
            raise ValueError("development pilot is frozen to one inner and one target update")
        if outer_iterations:
            if (int(meta.get("inner_episodes_per_iteration", 0)) != 1
                    or outer_iterations != 2 or int(meta.get("kill_frequency", 0)) != 1
                    or not 0 <= float(meta.get("inner_spawn_perturbation", -1)) <= 0.1
                    or not 0 <= float(meta.get("meta_spawn_perturbation", -1)) <= 0.2
                    or not np.isfinite(float(meta.get("spawn_reward_threshold", float("nan"))))
                    or execution["maximum_wall_seconds"] != 240):
                raise ValueError("population development pilot exceeds its frozen schedule")
        elif int(meta["episodes"]) != 1 or execution["maximum_wall_seconds"] != 180:
            raise ValueError("first development pilot is frozen to one meta episode")
        if int(meta["batch_size"]) != 1 or int(meta["update_every"]) != 1:
            raise ValueError("first development pilot trains Q after its one meta transition")
        if len(config["particles"]) != 2:
            raise ValueError("first development pilot requires two initial particles")
    else:
        if (len(config["particles"]) != 3
                or outer_iterations != 4
                or {key: meta.get(key) for key in (
                    "inner_episodes_per_iteration", "adaptation_steps", "kill_frequency",
                    "spawn_reward_threshold", "meta_spawn_perturbation",
                    "inner_spawn_perturbation", "batch_size", "update_every",
                    "gradient_steps", "exploration_probability",
                )} != {
                    "inner_episodes_per_iteration": 2, "adaptation_steps": 2,
                    "kill_frequency": 2, "spawn_reward_threshold": .5,
                    "meta_spawn_perturbation": .1, "inner_spawn_perturbation": .05,
                    "batch_size": 4, "update_every": 1,
                    "gradient_steps": 4, "exploration_probability": .5,
                }
                or target.get("updates") != 2
                or select_particles is not True
                or {key: search.get(key) for key in (
                    "population", "elite", "iterations", "tasks_per_candidate",
                    "coordinate_probes", "probe_fraction",
                )} != {
                    "population": 7, "elite": 1, "iterations": 1,
                    "tasks_per_candidate": 1, "coordinate_probes": [4, 5],
                    "probe_fraction": .1,
                }):
            raise ValueError("AdaptSim meta-development schedule changed")
    for row in config["particles"]:
        distribution = ParameterDistribution(np.asarray(row["left"]), np.asarray(row["right"]), row["kind"])
        if distribution.kind != "gaussian" or len(distribution.left) != 2:
            raise ValueError("initial T1 particles must be two-dimensional Gaussian distributions")
    MechanismSpec(**target["mechanism"])
    meta_train_calls = (
        outer_iterations * (len(config["particles"]) + int(meta["inner_episodes_per_iteration"]) * (1 + int(meta["adaptation_steps"])))
        if outer_iterations else int(meta["episodes"]) * (1 + int(meta["adaptation_steps"]))
    )
    train_calls = (meta_train_calls
                   + 1 + target["updates"]
                   + (len(config["particles"]) if select_particles else 0))
    simulated_target_calls = meta_train_calls
    deployment_target_calls = 1 + target["updates"] + (len(config["particles"]) if select_particles else 0)
    return {
        "version": version,
        "status": "planned_unexecuted",
        "scope": (
            "T1 method integration smoke; no nominal or main-table qualification"
            if version == DEVELOPMENT_VERSION else
            "T1 multi-particle meta-development; held-out nominal qualification remains separate"
        ),
        "output": config["output"],
        "physical_gpu_uuid": GPU1_UUID,
        "policy_training_calls": train_calls,
        "initial_particle_selection_rollouts": len(config["particles"]) if select_particles else 0,
        "simulated_target_rollouts": simulated_target_calls,
        "outer_meta_iterations": outer_iterations,
        "deployment_target_rollouts": deployment_target_calls,
        "maximum_physics_episodes": (
            train_calls * search["population"] * search["iterations"]
            * search["tasks_per_candidate"] * 2
            + (simulated_target_calls + deployment_target_calls) * 2
        ),
        "maximum_wall_seconds": execution["maximum_wall_seconds"],
        "all_failures_retained": True,
    }


def run_adaptsim_development(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run and retain one complete simulator-trained T1 AdaptSim smoke."""
    import torch

    plan = development_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
        raise RuntimeError("this development pilot requires CUDA_VISIBLE_DEVICES=1")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("pilot must see exactly one physical GPU")
    gpu = torch.cuda.get_device_properties(0)
    if str(getattr(gpu, "uuid", "")).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("visible CUDA device is not the frozen physical GPU1")
    torch.cuda.set_per_process_memory_fraction(
        min(2 * 1024**3 / gpu.total_memory, 1.0), 0,
    )
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    started = time.monotonic()
    training_ledger: list[dict[str, Any]] = []
    simulated_target_ledger: list[dict[str, Any]] = []
    deployment_target_ledger: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": plan["version"], "status": "running", "plan": plan}

    def check_time() -> None:
        if time.monotonic() - started > config["execution"]["maximum_wall_seconds"]:
            raise TimeoutError("frozen AdaptSim pilot wall ceiling exceeded")

    try:
        seed = int(config["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        task = ObstaclePolicySearchTask(robot="xarm7", delay_steps=0)
        search_cfg = config["task_policy_search"]
        search = CEMTaskPolicySearch(
            search_cfg["lower"], search_cfg["upper"], search_cfg["anchor"],
            task.sample_tasks, task.evaluate_policy,
            population=int(search_cfg["population"]),
            elite=int(search_cfg["elite"]),
            iterations=int(search_cfg["iterations"]),
            tasks_per_candidate=int(search_cfg["tasks_per_candidate"]),
            coordinate_probes=search_cfg.get("coordinate_probes", ()),
            probe_fraction=float(search_cfg.get("probe_fraction", 0.1)),
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
                "distribution": {
                    "kind": distribution.kind,
                    "left": distribution.left.tolist(),
                    "right": distribution.right.tolist(),
                },
                "policy": trained.policy.tolist(),
                "simulation_steps": trained.simulation_steps,
                "candidates": list(search.last_ledger),
            })
            return trained

        def simulated_target(target_seed):
            rng = np.random.default_rng(target_seed)
            return MechanismSpec(
                servo_gain=float(rng.uniform(.55, 1.15)),
                payload_mass=float(rng.uniform(.3, 1.0)),
                delay_steps=0,
            )

        def evaluate_simulated(policy, mechanism, call_seed):
            check_time()
            evaluation, episodes = task.evaluate_target(policy, mechanism, call_seed)
            simulated_target_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(),
                "episodes": list(episodes),
            })
            return evaluation

        meta_cfg = config["meta_training"]
        meta_agent = AdaptSimMetaTraining(
            solver, train, simulated_target, evaluate_simulated, seed=seed,
        )
        if int(meta_cfg.get("outer_iterations", 0)):
            meta = meta_agent.run_population(
                particles, iterations=int(meta_cfg["outer_iterations"]),
                inner_episodes=int(meta_cfg["inner_episodes_per_iteration"]),
                adaptation_steps=int(meta_cfg["adaptation_steps"]),
                kill_frequency=int(meta_cfg["kill_frequency"]),
                spawn_reward_threshold=float(meta_cfg["spawn_reward_threshold"]),
                meta_spawn_perturbation=float(meta_cfg["meta_spawn_perturbation"]),
                inner_spawn_perturbation=float(meta_cfg["inner_spawn_perturbation"]),
                batch_size=int(meta_cfg["batch_size"]),
                update_every=int(meta_cfg["update_every"]),
                gradient_steps=int(meta_cfg["gradient_steps"]),
                exploration_probability=float(meta_cfg["exploration_probability"]),
            )
            particles = list(meta.final_particles)
        else:
            meta = meta_agent.run(
                particles, episodes=int(meta_cfg["episodes"]),
                adaptation_steps=int(meta_cfg["adaptation_steps"]),
                batch_size=int(meta_cfg["batch_size"]),
                update_every=int(meta_cfg["update_every"]),
                gradient_steps=int(meta_cfg["gradient_steps"]),
                exploration_probability=float(meta_cfg["exploration_probability"]),
            )
        target_mechanism = MechanismSpec(**config["deployment_target"]["mechanism"])

        def evaluate_deployment(policy, call_seed):
            check_time()
            evaluation, episodes = task.evaluate_target(policy, target_mechanism, call_seed)
            deployment_target_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(),
                "episodes": list(episodes),
            })
            return evaluation

        adaptation = AdaptSimAdaptation(
            solver, train, evaluate_deployment, seed=seed + 500000,
        )
        selection = None
        if config["deployment_target"].get("select_particles", False):
            selection = adaptation.select_particle(particles)
        deployment = adaptation.run(
            particles[0] if selection is None else selection.distribution,
            updates=int(config["deployment_target"]["updates"]),
            initial_policy=None if selection is None else selection.policy,
        )
        selection_simulation_steps = 0 if selection is None else selection.simulation_steps
        selection_target_steps = 0 if selection is None else selection.target_control_steps
        if (sum(row["simulation_steps"] for row in training_ledger)
                != meta.simulation_steps + selection_simulation_steps + deployment["simulation_steps"]):
            raise RuntimeError("policy-training simulation ledger differs")
        if (sum(row["control_steps"] for row in simulated_target_ledger)
                != meta.simulated_target_control_steps):
            raise RuntimeError("simulated target ledger differs")
        if (sum(row["control_steps"] for row in deployment_target_ledger)
                != selection_target_steps + deployment["target_control_steps"]):
            raise RuntimeError("deployment target ledger differs")
        checkpoint = output / "inference_and_policy.pt"
        fd, temp = tempfile.mkstemp(dir=output, prefix="checkpoint-", suffix=".pt")
        os.close(fd)
        try:
            torch.save({
                "version": plan["version"],
                "q_network": {key: value.detach().cpu() for key, value in solver.backend.network.state_dict().items()},
                "final_policy": np.asarray(deployment["final_policy"]).tolist(),
            }, temp)
            os.replace(temp, checkpoint)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        report.update({
            "status": (
                "complete_development_smoke" if plan["version"] == DEVELOPMENT_VERSION
                else "complete_meta_development"
            ),
            "meta_training": {
                "episodes": meta.inner_episodes if hasattr(meta, "inner_episodes") else meta.episodes,
                "outer_iterations": int(meta_cfg.get("outer_iterations", 0)),
                "transitions": (meta.inner_episodes if hasattr(meta, "inner_episodes") else meta.episodes) * int(meta_cfg["adaptation_steps"]),
                "inference_updates": meta.inference_updates,
                "simulation_steps": meta.simulation_steps,
                "simulated_target_rollouts": meta.simulated_target_rollouts,
                "simulated_target_control_steps": meta.simulated_target_control_steps,
                "records": list(meta.records),
            },
            "deployment": {
                "selection": None if selection is None else {
                    "selected_index": selection.selected_index,
                    "records": list(selection.records),
                    "simulation_steps": selection.simulation_steps,
                    "target_control_steps": selection.target_control_steps,
                    "target_rollouts": len(selection.records),
                },
                "records": deployment["records"],
                "simulation_steps": selection_simulation_steps + deployment["simulation_steps"],
                "target_control_steps": selection_target_steps + deployment["target_control_steps"],
                "target_rollouts": (0 if selection is None else len(selection.records)) + deployment["target_rollouts"],
            },
            "checkpoint": {
                "path": str(checkpoint.relative_to(root)),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
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
    return report


__all__ += ["DEVELOPMENT_VERSION", "META_DEVELOPMENT_VERSION", "development_plan", "run_adaptsim_development"]


NOMINAL_DIAGNOSIS_VERSION = "adaptsim-obstacle-nominal-diagnosis-v1"
NOMINAL_EVIDENCE_VERSION = "adaptsim-obstacle-nominal-evidence-v2"
SOURCE_NOMINAL_EVIDENCE_VERSION = "adaptsim-obstacle-source-nominal-evidence-v3"


def nominal_diagnosis_plan(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Bind an underbudget AdaptSim policy to disjoint public T1 resets."""
    import torch

    version = config.get("version")
    if (version not in {NOMINAL_DIAGNOSIS_VERSION, NOMINAL_EVIDENCE_VERSION,
                        SOURCE_NOMINAL_EVIDENCE_VERSION}
            or config.get("task") != "T1" or config.get("robot") != "xarm7"
            or config.get("delay_steps") != 0):
        raise ValueError("nominal diagnosis requires the existing T1 task binding")
    execution = config.get("execution", {})
    if execution != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 240, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("nominal diagnosis changed its GPU1 resource ceiling")
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"nominal diagnosis output exists: {output}")
    if version in {NOMINAL_EVIDENCE_VERSION, SOURCE_NOMINAL_EVIDENCE_VERSION}:
        bindings = config.get("source_bindings")
        required = {
            "genedynamics/tasks/transfer/obstacle_policy_search.py",
            "scripts/paper/cross_ebdm/train_adaptsim_development.py",
        }
        if (not isinstance(bindings, list)
                or {row.get("path") for row in bindings} != required):
            raise ValueError("evidence diagnosis needs exact task and entrypoint source bindings")
        for row in bindings:
            if set(row) != {"path", "sha256"}:
                raise ValueError("source binding requires path and SHA256")
            path = root / row["path"]
            if (not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
                raise ValueError(f"nominal source binding changed: {path}")

    pointers = {}
    for name in ("training_report", "checkpoint"):
        pointer = config[name]
        if set(pointer) != {"path", "sha256"}:
            raise ValueError(f"{name} must be SHA-bound")
        path = root / pointer["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != pointer["sha256"]:
            raise ValueError(f"{name} changed")
        pointers[name] = path
    report = json.loads(pointers["training_report"].read_text())
    if (report.get("version") not in {DEVELOPMENT_VERSION, META_DEVELOPMENT_VERSION}
            or report.get("status") != (
                "complete_development_smoke" if report.get("version") == DEVELOPMENT_VERSION
                else "complete_meta_development"
            )
            or report.get("checkpoint") != config["checkpoint"]
            or report.get("formal_performance_evidence") is not False):
        raise ValueError("nominal diagnosis requires a complete, non-paper AdaptSim pilot")
    recipe = pointers["training_report"].with_name("recipe.json")
    if not recipe.is_file():
        raise ValueError("AdaptSim source recipe is missing")
    training_recipe = json.loads(recipe.read_text())
    anchor = ObstaclePolicySearchTask._vector(config["anchor"])
    if anchor.tolist() != training_recipe["task_policy_search"]["anchor"]:
        raise ValueError("nominal anchor differs from training recipe")
    checkpoint = torch.load(pointers["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("version") != report["version"]:
        raise ValueError("AdaptSim checkpoint version changed")
    adapted = ObstaclePolicySearchTask._vector(checkpoint["final_policy"])

    if (config.get("requested_modes") != [0, 1]
            or config.get("reset_slots_per_cell") != 4
            or len(config.get("scene_cells", [])) !=
            (3 if version == SOURCE_NOMINAL_EVIDENCE_VERSION else 2)):
        raise ValueError("nominal diagnosis cell roster or modes changed")
    expected = {
        "deployment_nominal": {"servo_gain": 1.0, "payload_mass": .5, "delay_steps": 0},
        "target_gain_shift": {"servo_gain": .7, "payload_mass": .5, "delay_steps": 0},
    }
    if version == SOURCE_NOMINAL_EVIDENCE_VERSION:
        expected["source"] = {"servo_gain": .85, "payload_mass": .65, "delay_steps": 0}
    if {row["id"]: row["mechanism"] for row in config["scene_cells"]} != expected:
        raise ValueError("nominal diagnosis mechanisms changed")
    seed_anchor = config.get("seed_anchor")
    if not isinstance(seed_anchor, int) or not 2**31 <= seed_anchor < 2**32 - 10000:
        raise ValueError("nominal reset seeds must lie outside training sampler range")
    requests = []
    for cell_index, cell in enumerate(config["scene_cells"]):
        for slot in range(4):
            for mode in (0, 1):
                requests.append({
                    "scene_cell": cell["id"], "mechanism": cell["mechanism"],
                    "physical_initial_slot": slot, "requested_mode": mode,
                    "reset_seed": seed_anchor + cell_index * 1000 + slot * 2 + mode,
                })
    prior_seeds = {
        int(row["seed"]) for name in (
            "policy_training_ledger", "simulated_target_ledger", "deployment_target_ledger"
        ) for row in report[name]
    }
    prior_seeds.update(
        int(episode["reset_seed"])
        for name in ("simulated_target_ledger", "deployment_target_ledger")
        for row in report[name] for episode in row["episodes"]
    )
    if prior_seeds & {row["reset_seed"] for row in requests}:
        raise ValueError("nominal diagnosis reused an observed training reset")
    return {
        "version": version,
        "status": "planned_unexecuted", "scope": "seed-disjoint T1 opportunity diagnosis, not a qualified four-task baseline",
        "output": config["output"], "physical_gpu_uuid": GPU1_UUID,
        "training_report": config["training_report"], "checkpoint": config["checkpoint"],
        "training_recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "source_bindings": config.get("source_bindings", []),
        "policies": {"adaptsim_pilot": adapted.tolist(), "public_anchor": anchor.tolist()},
        "requests": requests, "planned_physics_episodes": len(requests) * 2,
        "maximum_wall_seconds": execution["maximum_wall_seconds"],
        "all_failures_retained": True, "formal_performance_evidence": False,
    }


def run_adaptsim_nominal_diagnosis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Evaluate the frozen pilot and anchor on exactly matched held-out resets."""
    import torch

    plan = nominal_diagnosis_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("nominal diagnosis requires only physical GPU1 visible")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).removeprefix("GPU-").lower() != GPU1_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("nominal diagnosis saw the wrong GPU")
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / gpu.total_memory, 0)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    task = ObstaclePolicySearchTask(robot="xarm7", delay_steps=0)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": plan["version"], "status": "running"}
    try:
        for request_index, request in enumerate(plan["requests"]):
            mechanism = MechanismSpec(**request["mechanism"])
            for method, policy in plan["policies"].items():
                if time.monotonic() - started > plan["maximum_wall_seconds"]:
                    raise TimeoutError("AdaptSim nominal diagnosis exceeded its wall ceiling")
                reward, _, controls, summary = task._episode(
                    np.asarray(policy), mechanism, request["requested_mode"],
                    request["reset_seed"],
                    evidence_path=(
                        output / "evidence" / f"{request_index:03d}_{method}.json.gz"
                        if plan["version"] in {NOMINAL_EVIDENCE_VERSION,
                                               SOURCE_NOMINAL_EVIDENCE_VERSION} else None
                    ),
                )
                rows.append({
                    "method": method, "scene_cell": request["scene_cell"],
                    "physical_initial_slot": request["physical_initial_slot"],
                    **summary,
                })
                atomic_json(output / "progress.json", rows)
        paired = []
        for index in range(0, len(rows), 2):
            adapted, anchor = rows[index:index + 2]
            if (adapted["method"] != "adaptsim_pilot" or anchor["method"] != "public_anchor"
                    or any(adapted[key] != anchor[key] for key in (
                        "scene_cell", "physical_initial_slot", "mode", "reset_seed"
                    ))):
                raise RuntimeError("nominal comparison lost its matched reset")
            paired.append({
                "scene_cell": adapted["scene_cell"], "mode": adapted["mode"],
                "reset_seed": adapted["reset_seed"],
                "adaptsim_success": adapted["success"],
                "anchor_success": anchor["success"],
                "reward_gain": adapted["reward"] - anchor["reward"],
                "control_step_reduction": anchor["control_steps"] - adapted["control_steps"],
            })
        report.update({
            "status": "complete_development_diagnosis", "planned_episodes": len(rows),
            "completed_episodes": len(rows), "physical_control_steps": sum(row["control_steps"] for row in rows),
            "paired": paired, "rows": rows, "wall_seconds": time.monotonic() - started,
            "formal_performance_evidence": False,
        })
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
                       "traceback": traceback.format_exc(), "rows": rows,
                       "wall_seconds": time.monotonic() - started})
        raise
    finally:
        atomic_json(output / "report.json", report)
    return report


__all__ += ["NOMINAL_DIAGNOSIS_VERSION", "NOMINAL_EVIDENCE_VERSION",
            "SOURCE_NOMINAL_EVIDENCE_VERSION",
            "nominal_diagnosis_plan", "run_adaptsim_nominal_diagnosis"]
