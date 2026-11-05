"""Task-owned T2 public-feedback screen for AdaptSim policy adaptation."""

from __future__ import annotations

import gzip
import hashlib
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
from genedynamics.experiments.framework import ExperimentConfig
from genedynamics.tasks.transfer.contact import ContactMechanismSpec
from genedynamics.tasks.transfer.controllers.adaptsim import (
    GUIDE_POLICY_FIELDS, GUIDE_POLICY_LOWER, GUIDE_POLICY_UPPER,
    public_guide_actor,
)
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.specs import task_spec, validate_shared_execution
from genedynamics.solvers.single.adaptsim import (
    AdaptSimSolver, ParameterDistribution, ParameterSpec,
)
from genedynamics.solvers.single.adaptsim.learning import (
    AdaptationEvaluation, AdaptSimAdaptation, AdaptSimMetaTraining,
    CEMTaskPolicySearch, PolicySearchEvaluation,
)


VERSION = "adaptsim-guide-policy-feasibility-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"


def _task_config(config: dict, root: Path):
    pointer = config["task_config"]
    if set(pointer) != {"path", "sha256"}:
        raise ValueError("T2 task config pointer requires path and SHA256")
    path = root / pointer["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != pointer["sha256"]:
        raise ValueError("T2 task config SHA differs")
    experiment = ExperimentConfig.from_yaml(path)
    if experiment.env_name != "guided_contact_alignment":
        raise ValueError("T2 task environment differs")
    values = dict(experiment.env_params)
    for key in ("robot", "history_len", "mode"):
        values.pop(key, None)
    shared = values.pop("shared_execution")
    mechanism = values.pop("mechanism", {})
    if mechanism != {"contact_time_constant": .04}:
        raise ValueError("T2 contact time constant differs")
    spec = task_spec("T2", robot="xarm7", config_overrides=values)
    validate_shared_execution(spec, ContactMechanismSpec(contact_time_constant=.04), shared)
    return values, shared


def feasibility_plan(config: dict, root: Path) -> dict:
    if config.get("version") != VERSION or config.get("task") != "T2" or config.get("robot") != "xarm7":
        raise ValueError("expected T2 AdaptSim guide feasibility recipe")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 600, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("T2 guide feasibility budget differs")
    _task_config(config, root)
    policies = config.get("policies", [])
    if len(policies) != 3 or {p.get("name") for p in policies} != {
        "anchor", "conservative", "aggressive",
    }:
        raise ValueError("expected three named T2 public-feedback probes")
    if config.get("policy_fields") != list(GUIDE_POLICY_FIELDS):
        raise ValueError("T2 policy field order differs")
    for policy in policies:
        if set(policy) != {"name", "values"}:
            raise ValueError("T2 policy row differs")
        values = np.asarray(policy["values"], dtype=np.float64)
        if (values.shape != (3,) or not np.isfinite(values).all()
                or np.any(values < GUIDE_POLICY_LOWER)
                or np.any(values > GUIDE_POLICY_UPPER)):
            raise ValueError("T2 policy outside declared public support")
    requests = config.get("requests", [])
    if len(requests) != 4 or {
        (r.get("scene_cell"), r.get("mode")) for r in requests
    } != {("source", 0), ("source", 1), ("nominal", 0), ("nominal", 1)}:
        raise ValueError("T2 source/nominal modes must each occur once")
    for row in requests:
        if set(row) != {"scene_cell", "mode", "seed", "mechanism"}:
            raise ValueError("T2 request row differs")
        ContactMechanismSpec(**row["mechanism"])
        if row["mechanism"].get("contact_time_constant") != .04:
            raise ValueError("T2 contact time constant differs")
    if (config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T2 feasibility must retain failures and remain development-only")
    output = config["output"]
    if output.split("/")[:4] != ["results", "arm", "guided_contact_alignment", "baseline"]:
        raise ValueError("T2 feasibility output must be task-local")
    if (root / output).exists():
        raise FileExistsError("T2 feasibility output already exists")
    return {
        "version": VERSION, "status": "planned_unexecuted", "task": "T2",
        "maximum_physics_episodes": 12, "maximum_wall_seconds": 600,
        "failure_replacement": False, "formal_performance_evidence": False,
        "scope": "AdaptSim T2 public policy sensitivity; no solver training or target adaptation",
        "output": output,
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
    overrides, shared = _task_config(config, root)
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
            for request in config["requests"]:
                if time.monotonic() - started > plan["maximum_wall_seconds"]:
                    raise TimeoutError("T2 feasibility wall ceiling exceeded")
                task = instantiate_task(
                    "T2", robot="xarm7", seed=request["seed"],
                    mechanism_spec=ContactMechanismSpec(**request["mechanism"]),
                    config_overrides=overrides, shared_execution=shared,
                )
                public = task.reset(
                    (0,), seed=request["seed"], modes=np.asarray([request["mode"]]),
                )
                actor = public_guide_actor(task, policy["values"], mode=request["mode"])
                for _ in range(task.config.timeout_steps):
                    transition = task.step(actor.act(public))
                    public = transition.observation
                    if bool(transition.done[0]):
                        break
                evaluation = task.evaluate()
                evaluation["accounting"] = dict(task.accounting)
                episode = evaluation["episodes"][-1]
                evidence_dir = output / "episodes" / (
                    f"{policy['name']}_{request['scene_cell']}_m{request['mode']}"
                )
                evidence_dir.mkdir(parents=True, exist_ok=False)
                evidence_path = evidence_dir / "evidence.json.gz"
                evidence_path.write_bytes(gzip.compress(
                    json.dumps(episode_evidence(task.records, task.spec, evaluation),
                               sort_keys=True, separators=(",", ":")).encode(),
                    compresslevel=6, mtime=0,
                ))
                audit = audit_episode_evidence(evidence_path)
                if not audit["passed"]:
                    raise RuntimeError("T2 evidence failed CPU reevaluation")
                row = {
                    "policy": policy["name"], **request,
                    "success": bool(episode["success"]),
                    "outcome": episode["outcome"],
                    "constraint_violation": bool(episode["constraint_violation"]),
                    "physical_control_steps": int(episode["physical_control_steps"]),
                    "authority_controls": int(actor.authority_controls),
                    "progress_deficit_controls": int(actor.progress_deficit_controls),
                    "evidence": {"path": str(evidence_path.relative_to(root)),
                                 "sha256": audit["sha256"], "cpu_reevaluation_passed": True},
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
    """Reevaluate all T2 pilot evidence without simulator or GPU calls."""
    if config.get("version") != VERSION:
        raise ValueError("unknown T2 feasibility recipe")
    output = root / config["output"]
    recipe_path = output / "recipe.json"
    report_path = output / "report.json"
    if json.loads(recipe_path.read_text()) != config:
        raise ValueError("T2 run recipe differs from audit recipe")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "complete_feasibility"
            or report.get("recipe_sha256") != hashlib.sha256(recipe_path.read_bytes()).hexdigest()
            or len(report.get("episodes", [])) != 12):
        raise ValueError("T2 completed report or recipe differs")
    expected = {
        (policy["name"], request["scene_cell"], request["mode"], request["seed"])
        for policy in config["policies"] for request in config["requests"]
    }
    observed = set()
    audited = []
    for row in report["episodes"]:
        key = (row["policy"], row["scene_cell"], row["mode"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError("T2 evidence request roster differs")
        observed.add(key)
        pointer = row["evidence"]
        evidence_path = root / pointer["path"]
        if not evidence_path.is_relative_to(output):
            raise ValueError("T2 evidence outside task-local run")
        result = audit_episode_evidence(evidence_path)
        episode = result["evaluation"]["episodes"][-1]
        if (not result["passed"] or result["sha256"] != pointer["sha256"]
                or bool(episode["success"]) != row["success"]
                or episode["outcome"] != row["outcome"]
                or int(episode["physical_control_steps"]) != row["physical_control_steps"]):
            raise ValueError(f"T2 evidence reevaluation differs: {key}")
        audited.append({"request": key, "evidence_sha256": result["sha256"],
                        "reset_physical_hash": result["reset_physical_hashes"][-1],
                        "outcome": episode["outcome"]})
    if observed != expected:
        raise ValueError("T2 evidence request coverage differs")
    audit = {
        "version": VERSION, "status": "passed", "physical_calls": 0,
        "gpu_calls": 0, "episodes": audited,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    atomic_json(output / "independent_audit.json", audit)
    return audit


TRAINING_VERSION = "adaptsim-guide-policy-population-development-v1"


class GuidePolicySearchTask:
    """Charge both T2 modes for one public-feedback policy candidate."""

    def __init__(self, *, config_overrides: dict, shared_execution: dict,
                 contact_time_constant: float = .04,
                 deadline: float | None = None):
        self.config_overrides = dict(config_overrides)
        self.shared_execution = dict(shared_execution)
        self.contact_time_constant = float(contact_time_constant)
        self.deadline = deadline
        spec = task_spec("T2", robot="xarm7", config_overrides=self.config_overrides)
        validate_shared_execution(
            spec, ContactMechanismSpec(contact_time_constant=self.contact_time_constant),
            self.shared_execution,
        )

    def sample_tasks(self, distribution: ParameterDistribution,
                     rng: np.random.Generator, count: int):
        if distribution.kind != "gaussian" or len(distribution.left) != 3:
            raise ValueError("T2 AdaptSim needs gain, mass, delay Gaussian particles")
        tasks = []
        for gain, mass, delay in distribution.sample(rng, count):
            mechanism = ContactMechanismSpec(
                servo_gain=float(np.clip(gain, .4, 1.4)),
                payload_mass=float(np.clip(mass, .2, 1.2)),
                delay_steps=int(np.clip(np.rint(delay), 0, 3)),
                contact_time_constant=self.contact_time_constant,
            )
            seeds = tuple(int(v) for v in rng.integers(0, 2**31, size=2))
            tasks.append((mechanism, seeds))
        return tuple(tasks)

    @staticmethod
    def _policy(policy):
        vector = np.asarray(policy, dtype=np.float64)
        if (vector.shape != (3,) or not np.isfinite(vector).all()
                or np.any(vector < GUIDE_POLICY_LOWER)
                or np.any(vector > GUIDE_POLICY_UPPER)):
            raise ValueError("T2 candidate policy outside public support")
        return vector

    def _episode(self, policy, mechanism: ContactMechanismSpec, mode: int,
                 seed: int):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("T2 AdaptSim development wall ceiling exceeded")
        vector = self._policy(policy)
        task = instantiate_task(
            "T2", robot="xarm7", mechanism_spec=mechanism, seed=seed,
            config_overrides=self.config_overrides,
            shared_execution=self.shared_execution,
        )
        public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
        actor = public_guide_actor(task, vector, mode=mode)
        for _ in range(task.config.timeout_steps):
            transition = task.step(actor.act(public))
            public = transition.observation
            if bool(transition.done[0]):
                break
        episode = task.evaluate()["episodes"][-1]
        controls = int(episode["physical_control_steps"])
        error = float(episode["final_position_error_m"])
        if bool(episode["success"]):
            reward = 1. - .3 * controls / task.config.timeout_steps
        elif bool(episode["constraint_violation"]):
            reward = -1. - min(error, 1.)
        else:
            reward = -min(error, 1.) - .2 * controls / task.config.timeout_steps
        public_error = (
            np.asarray(public["goal_position"][0], dtype=np.float64)
            - np.asarray(public["tcp_position"][0], dtype=np.float64)
        )
        wrench = np.asarray(public["measured_wrench"][0, :3], dtype=np.float64)
        features = np.asarray([
            np.clip(public_error[0] / .25, -2., 2.),
            np.clip(public_error[1] / .25, -2., 2.),
            np.clip(np.linalg.norm(wrench) / task.config.force_limit, 0., 2.),
            controls / task.config.timeout_steps,
        ], dtype=np.float32)
        summary = {
            "mode": mode, "reset_seed": seed,
            "success": bool(episode["success"]),
            "constraint_violation": bool(episode["constraint_violation"]),
            "outcome": episode["outcome"], "control_steps": controls,
            "final_position_error_m": error, "reward": reward,
        }
        return reward, features, controls, summary

    def evaluate_policy(self, policy, tasks, seed: int) -> PolicySearchEvaluation:
        del seed
        vector = self._policy(policy)
        if not tasks:
            raise ValueError("at least one simulated T2 task required")
        scores, controls, episodes = [], 0, []
        for mechanism, mode_seeds in tasks:
            for mode in (0, 1):
                reward, _, steps, summary = self._episode(
                    vector, mechanism, mode, mode_seeds[mode],
                )
                scores.append(reward)
                controls += steps
                episodes.append({
                    "mechanism": {
                        "servo_gain": mechanism.servo_gain,
                        "payload_mass": mechanism.payload_mass,
                        "delay_steps": mechanism.delay_steps,
                        "contact_time_constant": mechanism.contact_time_constant,
                    }, **summary,
                })
        return PolicySearchEvaluation(float(np.mean(scores)), controls, tuple(episodes))

    def evaluate_target(self, policy, mechanism: ContactMechanismSpec,
                        seed: int):
        vector = self._policy(policy)
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
        return (
            AdaptationEvaluation(
                float(np.mean(scores)), np.concatenate(features), controls,
            ), tuple(summaries),
        )


TRAINING_SOURCES = {
    "genedynamics/core/control/transfer.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/envs/domains/manipulation/transfer_workcell/contact_runtime.py",
    "genedynamics/solvers/single/adaptsim/adaptsim.py",
    "genedynamics/solvers/single/adaptsim/backends/branching_q_torch.py",
    "genedynamics/solvers/single/adaptsim/learning/adaptation.py",
    "genedynamics/solvers/single/adaptsim/learning/meta_training.py",
    "genedynamics/solvers/single/adaptsim/learning/task_policy_search.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/contact_reference.py",
    "genedynamics/tasks/transfer/runtime.py",
    "genedynamics/tasks/transfer/specs.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
    "genedynamics/tasks/transfer/guide_evaluation.py",
    "genedynamics/tasks/transfer/routes.py",
    "genedynamics/tasks/transfer/guide_policy_search.py",
    "genedynamics/tasks/transfer/controllers/adaptsim.py",
    "genedynamics/tasks/transfer/controllers/guide_contact.py",
    "genedynamics/tasks/transfer/controllers/guide_feedback.py",
    "genedynamics/tasks/transfer/controllers/guide_lift_route.py",
    "genedynamics/tasks/transfer/controllers/guide_pose_feedback.py",
    "genedynamics/tasks/transfer/controllers/guide_wrench_projection.py",
    "scripts/paper/cross_ebdm/train_adaptsim_development.py",
}


def training_plan(config: dict, root: Path) -> dict:
    """Zero-physics budget/source check for one complete T2 AdaptSim population run."""
    if (config.get("version") != TRAINING_VERSION or config.get("task") != "T2"
            or config.get("robot") != "xarm7"):
        raise ValueError("expected T2 AdaptSim population training recipe")
    if config.get("execution") != {
        "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
        "maximum_wall_seconds": 16200, "maximum_gpu_memory_gib": 2,
    }:
        raise ValueError("T2 AdaptSim population execution budget differs")
    overrides, shared = _task_config(config, root)
    GuidePolicySearchTask(config_overrides=overrides, shared_execution=shared)
    if config.get("policy_fields") != list(GUIDE_POLICY_FIELDS):
        raise ValueError("T2 training policy chart differs")
    search = config["task_policy_search"]
    expected_search = {
        "population": 5, "elite": 2, "iterations": 2,
        "tasks_per_candidate": 2,
    }
    if (set(search) != {"lower", "upper", "anchor", *expected_search}
            or {key: search[key] for key in expected_search} != expected_search
            or not np.array_equal(np.asarray(search["lower"]), GUIDE_POLICY_LOWER)
            or not np.array_equal(np.asarray(search["upper"]), GUIDE_POLICY_UPPER)
            or not np.array_equal(np.asarray(search["anchor"]), [.02, .008, .03])):
        raise ValueError("T2 CEM public-policy support or budget differs")
    CEMTaskPolicySearch(
        search["lower"], search["upper"], search["anchor"],
        lambda *_: (), lambda *_: PolicySearchEvaluation(0., 0),
        **expected_search,
    )
    expected_meta = {
        "outer_iterations": 2, "inner_episodes": 2,
        "adaptation_steps": 2, "kill_frequency": 1,
        "spawn_reward_threshold": .65, "meta_spawn_perturbation": .1,
        "inner_spawn_perturbation": .05, "batch_size": 2,
        "update_every": 1, "gradient_steps": 1,
        "exploration_probability": .5,
    }
    if (config.get("meta_training") != expected_meta
            or config.get("deployment_target", {}).get("updates") != 2
            or config["deployment_target"].get("select_particles") is not True):
        raise ValueError("T2 AdaptSim meta/deployment budget differs")
    specs = [ParameterSpec(**row) for row in config["parameters"]]
    if [row.name for row in specs] != ["servo_gain", "payload_mass", "delay_steps"]:
        raise ValueError("T2 AdaptSim parameter order differs")
    if len(config.get("particles", [])) != 2:
        raise ValueError("T2 AdaptSim requires two initial particles")
    solver = AdaptSimSolver(specs, feature_dim=8, hidden=(32, 32), device="cpu")
    for row in config["particles"]:
        distribution = ParameterDistribution(
            np.asarray(row["left"]), np.asarray(row["right"]), row["kind"],
        )
        solver._check_distribution(distribution)
    target = ContactMechanismSpec(**config["deployment_target"]["mechanism"])
    if target.contact_time_constant != .04:
        raise ValueError("T2 deployment contact time constant differs")
    bindings = config.get("source_bindings")
    if not isinstance(bindings, list) or {row.get("path") for row in bindings} != TRAINING_SOURCES:
        raise ValueError("T2 AdaptSim exact source manifest differs")
    if len(bindings) != len(TRAINING_SOURCES):
        raise ValueError("T2 AdaptSim duplicate source binding")
    for row in bindings:
        if set(row) != {"path", "sha256"}:
            raise ValueError("T2 source binding requires path and SHA256")
        path = root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"T2 source binding changed: {path}")
    output = config["output"]
    if output.split("/")[:5] != [
        "results", "arm", "guided_contact_alignment", "baseline", "adaptsim",
    ] or (root / output).exists():
        raise ValueError("T2 training output must be fresh and task-local")
    if (config.get("formal_performance_evidence") is not False
            or config.get("failure_replacement") is not False):
        raise ValueError("T2 development training is not formal performance evidence")
    policy_calls = 21
    simulated_target_rollouts = 16
    deployment_target_rollouts = 5
    candidate_episodes = (
        policy_calls * expected_search["population"]
        * expected_search["iterations"]
        * expected_search["tasks_per_candidate"] * 2
    )
    maximum_physics_episodes = candidate_episodes + 2 * (
        simulated_target_rollouts + deployment_target_rollouts
    )
    if maximum_physics_episodes != 882:
        raise RuntimeError("T2 AdaptSim charged budget arithmetic differs")
    return {
        "version": TRAINING_VERSION, "status": "planned_unexecuted", "task": "T2",
        "source_bindings": bindings,
        "policy_training_calls": policy_calls,
        "simulated_target_rollouts": simulated_target_rollouts,
        "deployment_target_rollouts": deployment_target_rollouts,
        "candidate_physics_episodes": candidate_episodes,
        "maximum_physics_episodes": maximum_physics_episodes,
        "maximum_wall_seconds": 16200,
        "output": output, "failure_replacement": False,
        "formal_performance_evidence": False,
        "method_identity": "AdaptSim_task_adapted_CEM",
        "scope": "one true T2 training target; population Q and task-specific CEM; not native NAF or formal main-table evidence",
    }


def run_training(config: dict, root: Path) -> dict:
    """Execute the charged T2 AdaptSim population and one real development target."""
    import torch

    plan = training_plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T2 AdaptSim training requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T2 AdaptSim training saw wrong physical GPU")
    torch.cuda.set_per_process_memory_fraction(min(2 * 1024**3 / gpu.total_memory, 1.), 0)
    torch.set_num_threads(1)
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", plan)
    for binding in config["source_bindings"]:
        source = root / binding["path"]
        snapshot = output / "source_snapshot" / binding["path"]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("T2 AdaptSim source snapshot differs")
    started = time.monotonic()
    training_ledger = []
    simulated_ledger = []
    deployment_ledger = []
    report = {"version": TRAINING_VERSION, "status": "running", "task": "T2"}

    def write_progress():
        atomic_json(output / "progress.json", {
            "version": TRAINING_VERSION, "status": report["status"],
            "wall_seconds": time.monotonic() - started,
            "policy_training_calls": len(training_ledger),
            "candidate_physics_episodes": sum(
                len(candidate["episodes"])
                for training in training_ledger for candidate in training["candidates"]
            ),
            "simulated_target_rollouts": len(simulated_ledger),
            "deployment_target_rollouts": len(deployment_ledger),
        })

    try:
        seed = int(config["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        overrides, shared = _task_config(config, root)
        task = GuidePolicySearchTask(
            config_overrides=overrides, shared_execution=shared,
            deadline=started + plan["maximum_wall_seconds"],
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
        particles = [ParameterDistribution(
            np.asarray(row["left"]), np.asarray(row["right"]), row["kind"],
        ) for row in config["particles"]]

        def train(distribution, warm_start, call_seed):
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
                contact_time_constant=.04,
            )

        def evaluate_simulated(policy, mechanism, call_seed):
            evaluation, episodes = task.evaluate_target(policy, mechanism, call_seed)
            simulated_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(), "episodes": list(episodes),
            })
            write_progress()
            return evaluation

        meta_cfg = config["meta_training"]
        meta = AdaptSimMetaTraining(
            solver, train, simulated_target, evaluate_simulated, seed=seed,
        ).run_population(
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
        target = ContactMechanismSpec(**config["deployment_target"]["mechanism"])

        def evaluate_deployment(policy, call_seed):
            evaluation, episodes = task.evaluate_target(policy, target, call_seed)
            deployment_ledger.append({
                "seed": call_seed, "reward": evaluation.reward,
                "control_steps": evaluation.target_control_steps,
                "features": evaluation.features.tolist(), "episodes": list(episodes),
            })
            write_progress()
            return evaluation

        adaptation = AdaptSimAdaptation(
            solver, train, evaluate_deployment, seed=seed + 500000,
        )
        selection = adaptation.select_particle(meta.final_particles)
        deployment = adaptation.run(
            selection.distribution,
            updates=config["deployment_target"]["updates"],
            initial_policy=selection.policy,
        )
        candidate_episodes = sum(
            len(candidate["episodes"])
            for training in training_ledger for candidate in training["candidates"]
        )
        if (len(training_ledger) != plan["policy_training_calls"]
                or len(simulated_ledger) != plan["simulated_target_rollouts"]
                or len(deployment_ledger) != plan["deployment_target_rollouts"]
                or candidate_episodes != plan["candidate_physics_episodes"]
                or candidate_episodes + 2 * (
                    len(simulated_ledger) + len(deployment_ledger)
                ) != plan["maximum_physics_episodes"]
                or sum(row["simulation_steps"] for row in training_ledger) !=
                meta.simulation_steps + selection.simulation_steps + deployment["simulation_steps"]):
            raise RuntimeError("T2 AdaptSim rollout ledger differs from frozen budget")
        if time.monotonic() - started > plan["maximum_wall_seconds"]:
            raise TimeoutError("T2 AdaptSim development wall ceiling exceeded")
        checkpoint = output / "inference_and_policy.pt"
        fd, temporary = tempfile.mkstemp(dir=output, prefix="checkpoint-", suffix=".pt")
        os.close(fd)
        try:
            torch.save({
                "version": TRAINING_VERSION,
                "q_network": {key: value.detach().cpu() for key, value in solver.backend.network.state_dict().items()},
                "final_policy": np.asarray(deployment["final_policy"]).tolist(),
            }, temporary)
            os.replace(temporary, checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report.update({
            "status": "complete_development",
            "meta_training": {
                "records": list(meta.records),
                "iterations": meta.iterations,
                "inner_episodes": meta.inner_episodes,
                "inference_updates": meta.inference_updates,
                "simulation_steps": meta.simulation_steps,
                "simulated_target_rollouts": meta.simulated_target_rollouts,
                "simulated_target_control_steps": meta.simulated_target_control_steps,
                "final_particles": [
                    {"kind": particle.kind, "left": particle.left.tolist(),
                     "right": particle.right.tolist()}
                    for particle in meta.final_particles
                ],
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
        write_progress()
    return report
