"""Task-owned T4 ASID information probe under the shared mobile execution."""

from __future__ import annotations

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
from genedynamics.tasks.transfer.contact import MobileContactMechanismSpec
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.episode import audit_episode_evidence, episode_evidence
from genedynamics.tasks.transfer.mobile_mating_policy_search import (
    MOBILE_CONFIG, MobileMatingPolicySearchTask,
)
from genedynamics.tasks.transfer.mating_policy_search import POLICY_FIELDS


VERSION = "asid-t4-prejoint-limit-information-v1"
GPU1_UUID = "GPU-58b54684-3577-11d3-f039-0e8469190739"
HORIZON = 32
FEATURES = (
    "tcp_x_mm", "tcp_y_mm", "tcp_z_mm",
    "body_x_mm", "body_y_mm", "body_z_mm",
    "wrench_fx_N", "wrench_fy_N", "wrench_fz_N",
)
POLICIES = (
    {"name": "anchor", "values": [.25, 6., 2., .008, .004, 1., 0.]},
    {"name": "conservative_recovery", "values": [.4, 5., 1.2, .006, .003, .9, .04]},
    {"name": "fast", "values": [.1, 8., 2., .01, .006, 1.2, 0.]},
)
SOURCES = {
    "genedynamics/core/control/transfer.py",
    "genedynamics/envs/adapters/transfer_task.py",
    "genedynamics/solvers/single/asid/asid.py",
    "genedynamics/solvers/single/asid/backends/asid_numpy.py",
    "genedynamics/tasks/transfer/contact.py",
    "genedynamics/tasks/transfer/controllers/mating_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/bias_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/governor_feedback.py",
    "genedynamics/tasks/transfer/controllers/cross_ebdm/policies/liveness.py",
    "genedynamics/tasks/transfer/episode.py",
    "genedynamics/tasks/transfer/evaluation.py",
    "genedynamics/tasks/transfer/mobile_mating_policy_search.py",
    "genedynamics/tasks/transfer/mobile_mating_asid.py",
    "scripts/paper/cross_ebdm/diagnose_asid_mobile_information.py",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(config: dict, root: Path) -> dict:
    expected_source = {
        "servo_gain": .85, "payload_mass": .65,
        "delay_steps": 0, "body_servo_gain": 1.,
    }
    if (config.get("version") != VERSION or config.get("task") != "T4"
            or config.get("robot") != "go2_z1"
            or config.get("source_mechanism") != expected_source
            or config.get("delta") != .1
            or config.get("horizon_controls") != HORIZON
            or config.get("feature_fields") != list(FEATURES)
            or config.get("policy_fields") != list(POLICY_FIELDS)
            or config.get("policies") != list(POLICIES)
            or config.get("modes") != [0, 1]
            or config.get("seeds") != [3302700062, 3302700063]
            or config.get("maximum_physics_episodes") != 42
            or config.get("maximum_physical_control_steps") != 1344
            or config.get("execution") != {
                "cuda_visible_devices": "1", "physical_uuid": GPU1_UUID,
                "maximum_wall_seconds": 1200, "maximum_gpu_memory_gib": 2,
            }
            or config.get("failure_replacement") is not False
            or config.get("formal_performance_evidence") is not False):
        raise ValueError("T4 ASID information probe contract differs")
    task_pointer = config.get("task_config", {})
    basis_pointer = config.get("public_policy_basis", {})
    reset_pointer = config.get("reset_seed_audit", {})
    for pointer in (task_pointer, basis_pointer, reset_pointer):
        if (set(pointer) != {"path", "sha256"}
                or not (root / pointer["path"]).is_file()
                or _sha(root / pointer["path"]) != pointer["sha256"]):
            raise ValueError("T4 ASID task, policy basis or reset audit binding differs")
    from genedynamics.experiments.framework import ExperimentConfig

    task = ExperimentConfig.from_yaml(root / task_pointer["path"])
    basis = json.loads((root / basis_pointer["path"]).read_text())
    resets = json.loads((root / reset_pointer["path"]).read_text())
    if (task.env_name != "mobile_keyed_insertion_recovery"
            or task.env_params["shared_execution"] != config.get("shared_execution")
            or basis.get("version") != "adaptsim-mobile-mating-population-development-v2"
            or basis["feedback_base"] != config.get("feedback_base")
            or basis["shared_execution"] != config["shared_execution"]
            or basis["policy_fields"] != config["policy_fields"]
            or resets.get("status") != "passed"
            or not set(config["seeds"]) <= set(resets["safe_seed_pool"])):
        raise ValueError("T4 ASID common policy or safe reset basis differs")
    search = MobileMatingPolicySearchTask(
        feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
    )
    for policy in POLICIES:
        search.feedback(policy["values"])
    bindings = config.get("source_bindings", [])
    if len(bindings) != len(SOURCES) or {row.get("path") for row in bindings} != SOURCES:
        raise ValueError("T4 ASID source roster differs")
    for row in bindings:
        if set(row) != {"path", "sha256"} or _sha(root / row["path"]) != row["sha256"]:
            raise ValueError(f"T4 ASID source SHA differs: {row.get('path')}")
    output = config.get("output", "")
    if (output.split("/")[:5] != [
        "results", "quadruped", "mobile_keyed_insertion_recovery",
        "baseline", "asid",
    ] or (root / output).exists()):
        raise ValueError("T4 ASID diagnostic output must be fresh and task-local")
    return {
        "version": VERSION, "status": "planned_unexecuted", "task": "T4",
        "output": output, "maximum_physics_episodes": 42,
        "maximum_physical_control_steps": 1344,
        "maximum_wall_seconds": 1200, "source_bindings": bindings,
        "failure_replacement": False, "formal_performance_evidence": False,
        "scope": "T4 public pre-joint-limit Fisher information only; no target interaction or task training",
    }


def _feature(public: dict) -> np.ndarray:
    return np.r_[
        np.asarray(public["tcp_position"][0], dtype=np.float64) * 1000.,
        np.asarray(public["body_position"][0], dtype=np.float64) * 1000.,
        np.asarray(public["measured_wrench"][0, :3], dtype=np.float64),
    ]


def _trajectory(config: dict, root: Path, policy: np.ndarray | None,
                mechanism: MobileContactMechanismSpec, mode: int, seed: int,
                evidence_path: Path, *, fixed_actions: np.ndarray | None = None,
                fixed_body_intents: np.ndarray | None = None):
    task = instantiate_task(
        "T4", robot="go2_z1", mechanism_spec=mechanism, seed=seed,
        shared_execution=config["shared_execution"], config_overrides=MOBILE_CONFIG,
    )
    public = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([mode]))
    actor = None
    if fixed_actions is None:
        search = MobileMatingPolicySearchTask(
            feedback_base=config["feedback_base"],
            shared_execution=config["shared_execution"],
        )
        actor = PublicLivenessMatingFeedbackPolicy(
            task.spec, feedback=search.feedback(policy), modes=(mode,), seed=seed,
        )
    values, actions, body_intents = [], [], []
    for tick in range(HORIZON):
        if bool(np.asarray(task.state.status)[0]):
            break
        action = (np.asarray(actor.act(public), dtype=np.float32)
                  if fixed_actions is None else
                  np.asarray(fixed_actions[tick], dtype=np.float32)[None])
        requested_body = (task.public_body_intent(public)
                          if fixed_body_intents is None else fixed_body_intents[tick])
        if requested_body is None:
            raise RuntimeError("T4 public body intent missing")
        body_intent = np.asarray(requested_body, dtype=np.float32)
        if body_intent.ndim == 1:
            body_intent = body_intent[None]
        actions.append(action[0].copy())
        body_intents.append(body_intent[0].copy())
        public = task.step_contact(action, body_twist=body_intent).observation
        values.append(_feature(public))
    if not values:
        raise ValueError("T4 ASID trajectory has no physical transitions")
    evaluation = task.evaluate()
    evidence_path.parent.mkdir(parents=True, exist_ok=False)
    evidence_path.write_bytes(gzip.compress(
        json.dumps(episode_evidence(task.records, task.spec, evaluation),
                   sort_keys=True, allow_nan=False).encode(), mtime=0,
    ))
    evidence_audit = audit_episode_evidence(evidence_path)
    if not evidence_audit["passed"]:
        raise ValueError("T4 ASID physical evidence failed CPU reevaluation")
    episode = evaluation["episodes"][-1]
    response = np.asarray(values, dtype=np.float64)
    if len(response) < HORIZON:
        response = np.concatenate((response, np.repeat(
            response[-1:], HORIZON - len(response), axis=0,
        )))
    row = {
        "mode": mode, "reset_seed": seed,
        "mechanism": {
            "servo_gain": mechanism.servo_gain,
            "payload_mass": mechanism.payload_mass,
            "delay_steps": mechanism.delay_steps,
            "body_servo_gain": mechanism.body_servo_gain,
        },
        "physical_control_steps": len(values),
        "outcome": episode["outcome"],
        "constraint_violation": bool(episode["constraint_violation"]),
        "evidence": {"path": str(evidence_path.relative_to(root)),
                     "sha256": _sha(evidence_path),
                     "reset_physical_hash": evidence_audit["reset_physical_hashes"][-1]},
    }
    return response, np.asarray(actions, dtype=np.float32), np.asarray(body_intents, dtype=np.float32), row


def run(config: dict, root: Path) -> dict:
    import torch

    frozen = plan(config, root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or torch.cuda.device_count() != 1:
        raise RuntimeError("T4 ASID probe requires physical GPU1 only")
    gpu = torch.cuda.get_device_properties(0)
    if str(gpu.uuid).lower().removeprefix("gpu-") != GPU1_UUID.lower().removeprefix("gpu-"):
        raise RuntimeError("T4 ASID probe saw wrong physical GPU")
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
            raise ValueError("T4 ASID source snapshot differs")
    started = time.monotonic()
    source = MobileContactMechanismSpec(**config["source_mechanism"])
    report = {**frozen, "status": "running", "rows": []}
    episodes = controls = 0

    def charged(policy, mechanism, mode, seed, path, *, fixed_actions=None,
                fixed_body_intents=None):
        nonlocal episodes, controls
        if time.monotonic() - started > 1200 or episodes >= 42:
            raise TimeoutError("T4 ASID physical or wall budget reached")
        episodes += 1
        response, actions, body, row = _trajectory(
            config, root, policy, mechanism, mode, seed, path,
            fixed_actions=fixed_actions, fixed_body_intents=fixed_body_intents,
        )
        controls += row["physical_control_steps"]
        if controls > 1344:
            raise ValueError("T4 ASID control budget exceeded")
        atomic_json(output / "progress.json", {
            "status": "running", "episodes": episodes,
            "physical_control_steps": controls,
            "completed_policy_mode_cells": len(report["rows"]),
        })
        return response, actions, body, row

    try:
        for policy in POLICIES:
            for mode, seed in zip(config["modes"], config["seeds"], strict=True):
                base = output / "episodes" / f"{policy['name']}_m{mode}"
                _, actions, body, source_row = charged(
                    np.asarray(policy["values"], dtype=np.float64), source,
                    mode, seed, base / "source" / "evidence.json.gz",
                )
                perturbations = []
                if (source_row["physical_control_steps"] == HORIZON
                        and not source_row["constraint_violation"]):
                    def transition(state, action_stream, theta):
                        del state
                        mechanism = MobileContactMechanismSpec(
                            servo_gain=float(theta[0]),
                            payload_mass=float(theta[1]),
                            delay_steps=0,
                            body_servo_gain=float(theta[2]),
                        )
                        trace, _, _, row = charged(
                            None, mechanism, mode, seed,
                            base / f"perturb_{len(perturbations):02d}" / "evidence.json.gz",
                            fixed_actions=action_stream, fixed_body_intents=body,
                        )
                        perturbations.append(row)
                        return trace.reshape(-1)

                    jacobian = ASIDSolver(transition, delta=.1).observation_jacobian(
                        None, actions,
                        np.asarray([source.servo_gain, source.payload_mass,
                                    source.body_servo_gain]),
                    ).reshape(HORIZON, len(FEATURES), 3)
                    valid = all(row["physical_control_steps"] == HORIZON
                                and not row["constraint_violation"] for row in perturbations)
                    singular = np.linalg.svd(jacobian.reshape(-1, 3), compute_uv=False).tolist()
                    position_norms = np.linalg.norm(
                        jacobian[:, :6].reshape(-1, 3), axis=0,
                    ).tolist()
                    score = float(np.sum(jacobian * jacobian))
                else:
                    valid = False
                    singular = position_norms = score = None
                report["rows"].append({
                    "policy": policy, "mode": mode, "reset_seed": seed,
                    "source": source_row, "perturbations": perturbations,
                    "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                    "source_body_intent_sha256": hashlib.sha256(body.tobytes()).hexdigest(),
                    "valid_full_horizon_and_no_violation": valid,
                    "information_score": score,
                    "full_singular_values": singular,
                    "position_column_norms": position_norms,
                })
        report["status"] = "complete_development_probe"
    except BaseException as exc:
        report.update({"status": "failed_retained",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
    finally:
        report.update({
            "physical_episodes_charged": episodes,
            "physical_control_steps_charged": controls,
            "wall_seconds": time.monotonic() - started,
            "recipe_sha256": _sha(output / "recipe.json"),
        })
        atomic_json(output / "report.json", report)
        atomic_json(output / "progress.json", {
            "status": report["status"], "episodes": episodes,
            "physical_control_steps": controls,
            "completed_policy_mode_cells": len(report["rows"]),
        })
    return report


def audit(config: dict, root: Path) -> dict:
    """Reevaluate all saved physical evidence and the frozen source/ledger."""
    if config.get("version") != VERSION:
        raise ValueError("unknown T4 ASID information audit recipe")
    output = root / config["output"]
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    if (json.loads((output / "recipe.json").read_text()) != config
            or report.get("status") != "complete_development_probe"
            or report.get("recipe_sha256") != _sha(output / "recipe.json")
            or len(report.get("rows", [])) != 6
            or report.get("physical_episodes_charged") != 42
            or report.get("physical_control_steps_charged") != 1344):
        raise ValueError("T4 ASID information report or budget differs")
    for binding in config["source_bindings"]:
        if _sha(output / "source_snapshot" / binding["path"]) != binding["sha256"]:
            raise ValueError("T4 ASID source snapshot differs")
    evidence = []
    for row in report["rows"]:
        if (row["policy"] not in POLICIES or row["mode"] not in (0, 1)
                or not row["valid_full_horizon_and_no_violation"]
                or len(row["perturbations"]) != 6
                or len(row["full_singular_values"]) != 3
                or len(row["position_column_norms"]) != 3):
            raise ValueError("T4 ASID information cell incomplete")
        for episode in [row["source"], *row["perturbations"]]:
            pointer = episode["evidence"]
            path = root / pointer["path"]
            result = audit_episode_evidence(path)
            if (not result["passed"] or result["sha256"] != pointer["sha256"]
                    or result["reset_physical_hashes"][-1] != pointer["reset_physical_hash"]
                    or result["evaluation"]["episodes"][-1]["outcome"] != episode["outcome"]
                    or episode["physical_control_steps"] != HORIZON):
                raise ValueError("T4 ASID physical episode re-score differs")
            evidence.append(pointer["sha256"])
    result = {
        "version": VERSION, "status": "passed_development_only",
        "task": "T4", "report_sha256": _sha(report_path),
        "source_snapshots_verified": len(config["source_bindings"]),
        "physical_evidence_verified": len(evidence),
        "physical_control_steps_verified": 1344,
        "body_intent_replay_independently_verified": False,
        "full_wrench_fisher_independently_rescored": False,
        "gpu_calls": 0, "physical_calls": 0,
        "formal_performance_evidence": False,
    }
    atomic_json(output / "independent_audit.json", result)
    return result
