#!/usr/bin/env python3
"""Validate or train task-local transport with a public terminal-progress target."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from genedynamics.learning.datasets.transfer.mobile_target import (
    PublicMobileTargetTrajectory,
)
from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION, VERSION, PublicTargetTrajectory,
)
from genedynamics.solvers.single.cross_ebdm.learning.gated_transport import (
    PublicAnchoringGateSpec,
    PublicCompletionProgressSpec,
    PublicFiniteTimeProgressSpec,
    PublicDirectionalResponseSpec,
    PublicEnsembleRobustTaskSpec,
    PublicPairedProgressSpec,
    PublicRelativeResponseSpec,
    PublicRecoveryGatedTransportTrainer,
    public_completion_progress,
    public_finite_time_progress,
    recenter_saturated_transport_output,
    recovery_response_reference,
    select_completion_compatible_checkpoint,
    select_finite_time_compatible_checkpoint,
)
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import (
    PublicIssuedResponseEnsemble,
)
from genedynamics.solvers.single.cross_ebdm.learning.local_transport import (
    PublicLocalResidualTransport,
)
from genedynamics.solvers.single.cross_ebdm.learning.recovery_transport import (
    recovery_policy_commands,
    recovery_task_proxy,
    recovery_task_proxy_members,
)
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.ablation import (
    treatment_metadata, treatment_objective_kwargs,
)
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.tasks.transfer.controllers.cross_ebdm.dense import (
    PublicDenseTransportActor,
)
from genedynamics.tasks.transfer.controllers.cross_ebdm.mobile_dense import (
    PublicLivenessMobileDenseTransportActor,
)
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import (
    PublicBiasFeedbackSpec,
)
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicLivenessMatingFeedbackPolicy,
)
from genedynamics.tasks.transfer.controllers.recovery_execution import PublicRecoveryGate
from genedynamics.tasks.transfer.specs import task_spec


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha(path),
    }


def checked(value: dict[str, str], label: str) -> Path:
    path = Path(value["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError(f"{label} changed")
    return path


def _merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge a small, SHA-bound task-local experiment variant."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load a full config or a compact variant of one frozen task config."""
    raw = json.loads(config_path.read_text())
    parent = raw.get("base_training_config")
    if parent is None:
        return raw
    if set(raw) != {"base_training_config", "overrides"}:
        raise ValueError(
            "compact training config requires only base_training_config and overrides"
        )
    parent_path = checked(parent, "base training config")
    parent_raw = json.loads(parent_path.read_text())
    parent_config = (
        load_training_config(parent_path)
        if parent_raw.get("base_training_config") is not None
        else parent_raw
    )
    overrides = raw["overrides"]
    allowed_overrides = {
        "version", "training", "preflight_output", "output", "preflight_report",
        "actor_manifest", "data", "selection", "limits",
    }
    projected_variant = (
        parent_config.get("version")
        == "t4-completion-projected-execution-preflight-v1"
    )
    if projected_variant:
        allowed_overrides.update({"instances", "projection_preflight_output"})
    if set(overrides) - allowed_overrides:
        raise ValueError("compact training config changes a frozen training input")
    limit_overrides = overrides.get("limits", {})
    if set(limit_overrides) - {"wall_seconds"}:
        raise ValueError("compact training may only extend the wall-time ceiling")
    if "wall_seconds" in limit_overrides:
        parent_wall = float(parent_config["limits"]["wall_seconds"])
        requested_wall = float(limit_overrides["wall_seconds"])
        if requested_wall < parent_wall or requested_wall > 7200.0:
            raise ValueError("wall-time extension must preserve the parent and stay <=7200s")
    if projected_variant and "instances" in overrides:
        instance_overrides = overrides["instances"]
        if (
            set(instance_overrides) != set(parent_config["instances"])
            or any(set(row) != {"candidate_checkpoint"}
                   for row in instance_overrides.values())
        ):
            raise ValueError(
                "projected preflight may only replace every candidate checkpoint"
            )
    training_overrides = overrides.get("training", {})
    if set(training_overrides) - {
        "variant", "initialization", "control_checkpoint_policy",
        "response_normalization", "shared_start_report", "paired_progress",
        "checkpoint_selection", "retain_validation_checkpoints", "ensemble_task",
        "target_sampling",
        "clean_treatment", "pilot_instances", "response_checkpoint_overrides",
    }:
        raise ValueError(
            "compact training config may only select a variant and its initialization"
        )
    initialization = training_overrides.get("initialization")
    if initialization is not None and initialization != {
        "kind": "recentered_current_checkpoint_fresh_optimizer",
        "calibration": "fit_plus_weak_log_public_states",
        "maximum_median_absolute_logit": 3.0,
    }:
        raise ValueError("compact training initialization changed the matched-control contract")
    paired_progress = training_overrides.get("paired_progress")
    expected_paired_progress = {
        "linear_margin": 1e-5,
        "angular_margin": 1e-4,
        "weight": 1.0,
    }
    version = overrides.get("version")
    if version in {
        "t3-task-first-mode-tail-recorded-completion-training-v8",
        "t3-task-first-mode-specific-recorded-completion-training-v9",
    }:
        expected_paired_progress.update({
            "tail_fraction": 0.1,
            "tail_weight": 4.0,
        })
    mode_specific = None
    if version == "t3-task-first-mode-specific-recorded-completion-training-v9":
        mode_specific = (
            paired_progress.get("requested_mode")
            if isinstance(paired_progress, dict) else None
        )
        if (
            isinstance(mode_specific, bool)
            or not isinstance(mode_specific, int)
            or mode_specific not in {0, 1}
        ):
            raise ValueError("mode-specific training requires public mode 0 or 1")
        expected_paired_progress["requested_mode"] = mode_specific
    if paired_progress is not None and paired_progress != expected_paired_progress:
        raise ValueError("compact training changed the paired-progress contract")
    ensemble_task = training_overrides.get("ensemble_task")
    if ensemble_task is not None and ensemble_task != {
        "reduction": "worst_paired_member",
    }:
        raise ValueError("compact training changed the ensemble-task contract")
    checkpoint_selection = training_overrides.get("checkpoint_selection")
    if checkpoint_selection is not None and checkpoint_selection != {
        "rule": "best_validation_paired_progress",
        "interval_updates": 100,
        "minimum_update": 100,
    }:
        raise ValueError("compact training changed the paired checkpoint selector")
    target_sampling = training_overrides.get("target_sampling")
    expected_target_sampling = (
        {"kind": "mode_specific", "modes": [mode_specific]}
        if mode_specific is not None
        else {"kind": "mode_balanced", "modes": [0, 1]}
    )
    if target_sampling not in (None, expected_target_sampling):
        raise ValueError("unknown target-state sampling contract")
    data_overrides = overrides.get("data", {})
    if set(data_overrides) - {
        "weak_log_split", "weak_log_remaining_controls_target",
        "additional_weak_log_manifests",
    }:
        raise ValueError(
            "compact training config may only change the weak-log split or "
            "remaining-control target"
        )
    split = data_overrides.get("weak_log_split")
    if split is not None and split != {
        "kind": "entry_index_modulo",
        "modulus": 4,
        "validation_residue": 0,
    }:
        raise ValueError("weak-log split changed the frozen mode-stratified contract")
    remaining_target = data_overrides.get("weak_log_remaining_controls_target")
    if remaining_target not in (None, "recorded_episode_end"):
        raise ValueError("unknown weak-log remaining-control target")
    additional_logs = data_overrides.get("additional_weak_log_manifests")
    if additional_logs is not None and (
        not isinstance(additional_logs, list)
        or not additional_logs
        or any(
            not isinstance(value, dict) or set(value) != {"path", "sha256"}
            for value in additional_logs
        )
    ):
        raise ValueError("additional weak-log manifests must be SHA pointers")
    selection_overrides = overrides.get("selection", {})
    if set(selection_overrides) - {"required_modes"}:
        raise ValueError("compact training config may only require validation modes")
    expected_required_modes = (
        [mode_specific] if mode_specific is not None else [0, 1]
    )
    if selection_overrides.get("required_modes") not in (
        None, expected_required_modes,
    ):
        raise ValueError("mode-stratified selection differs from the training modes")
    config = _merge_config(parent_config, overrides)
    selected_instances = config["training"].get("pilot_instances")
    if selected_instances is not None:
        if (
            not isinstance(selected_instances, list) or not selected_instances
            or len(selected_instances) != len(set(selected_instances))
            or not set(selected_instances) <= set(config["instances"])
        ):
            raise ValueError("pilot_instances must select unique declared instances")
        config["instances"] = {
            name: config["instances"][name] for name in selected_instances
        }
    response_overrides = config["training"].get("response_checkpoint_overrides")
    clean_name = config["training"].get("clean_treatment")
    clean_metadata = (
        treatment_metadata(clean_name) if clean_name is not None else None
    )
    if response_overrides is not None:
        if (
            clean_name != "no_calibration"
            or not isinstance(response_overrides, dict)
            or set(response_overrides) != set(config["instances"])
        ):
            raise ValueError(
                "response checkpoint overrides require every no-calibration instance"
            )
        for name, value in response_overrides.items():
            path = checked(value, f"{name} nominal response checkpoint")
            if "/budget_0/" not in path.as_posix():
                raise ValueError("no-calibration response must be the budget-0 model")
            config["instances"][name]["response_checkpoint"] = pointer(path)
    elif clean_metadata is not None and clean_name == "no_calibration":
        raise ValueError("no-calibration requires nominal response checkpoints")
    if clean_metadata is not None and clean_name != "no_calibration":
        if any(
            "/budget_60/" not in row["response_checkpoint"]["path"]
            and "budget_60" not in row["response_checkpoint"]["path"]
            for row in config["instances"].values()
        ):
            raise ValueError("calibrated treatments require target-calibrated response")
    config["clean_ablation"] = clean_metadata
    config["resolved_base_training_config"] = pointer(parent_path)
    shared = config["training"].get("shared_start_report")
    if shared is not None:
        shared_path = checked(shared, "shared task-only start report")
        shared_report = json.loads(shared_path.read_text())
        instances = shared_report.get("instances", {})
        if (
            shared_report.get("status") != "complete"
            or shared_report.get("passed") is not True
            or not set(config["instances"]) <= set(instances)
            or any(
                not row.get("selection", {}).get("accepted", False)
                for row in instances.values()
            )
        ):
            raise ValueError("shared task-only start report is not fully selected")
        for name, row in config["instances"].items():
            row["task_only_checkpoint"] = deepcopy(
                instances[name]["selected_checkpoint"]
            )
        config["resolved_shared_start_report"] = pointer(shared_path)
    return config


def training_variant(config: dict[str, Any]) -> dict[str, Any]:
    name = str(config["training"].get("variant", "transport"))
    clean_name = config["training"].get("clean_treatment")
    clean_kwargs = (
        treatment_objective_kwargs(clean_name)
        if clean_name is not None else None
    )
    def result(value):
        if clean_kwargs is None:
            return value
        return {
            **value,
            "response_anchoring_weight": clean_kwargs["response_weight"],
            "apply_anchoring_gate": clean_kwargs["apply_anchoring_gate"],
            "clean_ablation": treatment_metadata(clean_name),
        }
    if name == "transport":
        return result({
            "name": name,
            "initialization_policy": "current_transport",
            "response_anchoring_weight": 1.0,
        })
    if name == "matched_task_only":
        initialization = config["training"].get("initialization", {})
        if initialization.get("kind") != "recentered_current_checkpoint_fresh_optimizer":
            raise ValueError(
                "matched task-only requires recentered current-style fresh initialization"
            )
        return result({
            "name": name,
            "initialization_policy": "task_only",
            "response_anchoring_weight": 0.0,
        })
    if name in {
        "same_start_anchored", "same_start_task_only",
        "same_start_relative_anchored", "same_start_relative_task_only",
        "task_first_directional_anchored", "task_first_directional_task_only",
        "task_first_directional_same_state_anchored",
        "task_first_margin_anchored", "task_first_margin_task_only",
        "task_first_stable_margin_anchored",
        "task_first_physical_selector_anchored",
        "task_first_ensemble_robust_anchored",
    }:
        initialization = config["training"].get("initialization", {})
        if initialization.get("kind") != "recentered_current_checkpoint_fresh_optimizer":
            raise ValueError(
                "same-start variants require recentered fresh initialization"
            )
        return result({
            "name": name,
            "initialization_policy": "task_only",
            "response_anchoring_weight": 1.0 if name in {
                "same_start_anchored", "same_start_relative_anchored",
                "task_first_directional_anchored",
                "task_first_directional_same_state_anchored",
                "task_first_margin_anchored",
                "task_first_stable_margin_anchored",
                "task_first_physical_selector_anchored",
                "task_first_ensemble_robust_anchored",
            } else 0.0,
        })
    raise ValueError("unknown terminal-progress training variant")


def checked_parent_attempt(config: dict[str, Any]) -> dict[str, str] | None:
    value = config.get("parent_attempt")
    if value is None:
        return None
    path = checked(value, "parent training attempt")
    report = json.loads(path.read_text())
    if report.get("status") != "selection_failed" or report.get("passed") is not False:
        raise ValueError("tail-aligned training requires a rejected parent attempt")
    if report.get("new_physics_calls") != 0 or report.get("target_reward_or_truth_opened") is not False:
        raise ValueError("parent attempt exceeded public offline permissions")
    return pointer(path)


def checked_development_evidence(
    config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    values = config.get("development_evidence", {})
    if not isinstance(values, dict):
        raise ValueError("development_evidence must be a pointer mapping")
    resolved = {
        name: pointer(checked(value, f"development evidence {name}"))
        for name, value in sorted(values.items())
    }
    rejected = resolved.get("rejected_qualification")
    if rejected is not None:
        report = json.loads(checked(rejected, "rejected qualification").read_text())
        if (
            report.get("schema_version")
            != "four-task-paired-development-qualification-v2"
            or report.get("status")
            != "complete_recorded_development_qualification"
            or report.get("passed") is not False
            or report.get("confirmation_states_opened") != 0
            or report.get("performance_evidence") is not False
        ):
            raise ValueError("development parent is not a rejected paired qualification")
    return resolved


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def annotate_checkpoint(value: dict[str, Any], metadata: dict[str, Any] | None):
    result = deepcopy(value)
    if metadata is not None:
        result["clean_ablation"] = deepcopy(metadata)
    return result


def concatenate(rows: list[Any]) -> Any:
    if not rows:
        raise ValueError("nonempty public-state collection required")
    if isinstance(rows[0], dict):
        keys = set(rows[0])
        if any(set(row) != keys for row in rows):
            raise ValueError("public-state dictionaries differ")
        return {key: concatenate([row[key] for row in rows]) for key in rows[0]}
    return np.concatenate(rows)


def tensorize(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: tensorize(child, device) for key, child in value.items()}
    tensor = torch.as_tensor(value, device=device)
    return tensor.float() if tensor.is_floating_point() else tensor


def select(value: Any, indices: Any) -> Any:
    if isinstance(value, dict):
        return {key: select(child, indices) for key, child in value.items()}
    return value[indices]


def combine(left: Any, right: Any) -> Any:
    if isinstance(left, dict):
        return {key: combine(left[key], right[key]) for key in left}
    return torch.cat((left, right))


def progress_spec(config: dict[str, Any]):
    completion = config.get("completion_progress")
    finite_time = config.get("finite_time_progress")
    if (completion is None) == (finite_time is None):
        raise ValueError("exactly one terminal-progress spec required")
    if completion is not None:
        return "completion", PublicCompletionProgressSpec(**completion)
    return "finite_time", PublicFiniteTimeProgressSpec(**finite_time)


def progress_values(commands, batch, spec, feedback):
    if isinstance(spec, PublicCompletionProgressSpec):
        return public_completion_progress(commands, batch, spec)
    if isinstance(spec, PublicFiniteTimeProgressSpec):
        return public_finite_time_progress(commands, batch, spec, feedback)
    raise ValueError("unknown terminal-progress spec")


def progress_objective_kwargs(spec):
    if isinstance(spec, PublicCompletionProgressSpec):
        return {"completion_spec": spec}
    if isinstance(spec, PublicFiniteTimeProgressSpec):
        return {"finite_time_spec": spec}
    raise ValueError("unknown terminal-progress spec")


def response_normalization_spec(config):
    value = config["training"].get("response_normalization")
    if value is None:
        return None
    if "minimum_source_task_advantage" in value:
        return PublicDirectionalResponseSpec(**value)
    return PublicRelativeResponseSpec(**value)


def paired_progress_spec(config):
    value = config["training"].get("paired_progress")
    return None if value is None else PublicPairedProgressSpec(**value)


def ensemble_task_spec(config):
    value = config["training"].get("ensemble_task")
    return None if value is None else PublicEnsembleRobustTaskSpec(**value)


def select_progress_checkpoint(*, spec, **kwargs):
    if isinstance(spec, PublicCompletionProgressSpec):
        return select_completion_compatible_checkpoint(**kwargs)
    if isinstance(spec, PublicFiniteTimeProgressSpec):
        return select_finite_time_compatible_checkpoint(**kwargs)
    raise ValueError("unknown terminal-progress spec")


def validation_checkpoint_selection(config):
    value = config["training"].get("checkpoint_selection")
    if value is None:
        return None
    if value.get("rule") not in {
        "first_validation_compatible", "best_validation_progress_margin",
        "best_validation_paired_progress",
    }:
        raise ValueError("unknown terminal-progress checkpoint selection rule")
    interval = value.get("interval_updates")
    minimum = value.get("minimum_update", interval)
    updates = int(config["training"]["updates"])
    log_interval = int(config["training"]["log_interval"])
    if (
        isinstance(interval, bool) or not isinstance(interval, int) or interval < 1
        or isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < interval
        or minimum > updates or minimum % interval or interval % log_interval
    ):
        raise ValueError("checkpoint selection requires aligned positive update counts")
    return {
        "rule": value["rule"],
        "interval_updates": interval,
        "minimum_update": minimum,
    }


def checkpoint_progress_score(metrics, spec) -> float:
    """Lower is a larger worst-axis public progress margin."""
    if not isinstance(spec, PublicFiniteTimeProgressSpec):
        raise ValueError("best progress-margin selection is T3 finite-time only")
    return max(
        float(metrics["linear_deficit_p95"]) / spec.linear_progress_scale,
        float(metrics["angular_deficit_p95"]) / spec.angular_progress_scale,
    )


def checkpoint_paired_progress_score(metrics, metrics_by_mode) -> float:
    """Lower is a larger held-out fifth-percentile paired progress gain."""
    rows = (metrics, *metrics_by_mode.values())
    return -min(float(row["normalized_gain_p05"]) for row in rows)


def validate_checkpoint_selection_scope(selection, spec, paired_spec=None) -> None:
    if (
        selection is not None
        and selection["rule"] in {
            "best_validation_progress_margin",
            "best_validation_paired_progress",
        }
        and not isinstance(spec, PublicFiniteTimeProgressSpec)
    ):
        raise ValueError("best progress selection is T3 finite-time only")
    if (
        selection is not None
        and selection["rule"] == "best_validation_paired_progress"
        and not isinstance(paired_spec, PublicPairedProgressSpec)
    ):
        raise ValueError("paired checkpoint selection requires paired progress")


def _public_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key[7:]: archive[key].copy()
            for key in archive.files if key.startswith("public.")
        }


def _task_reference(actor) -> np.ndarray:
    reference = teacher_reference(
        actor.base.observer.memory, goal_z=actor.spec.config.goal[2],
    )
    cosine, sine = np.cos(reference[:, 3]), np.sin(reference[:, 3])
    rotation = np.zeros((len(reference), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = sine
    rotation[:, 1, 0] = sine
    rotation[:, 1, 1] = -cosine
    rotation[:, 2, 2] = -1.
    return np.concatenate(
        (reference[:, :3], rotation.reshape(-1, 9), reference[:, 4:5]), axis=1,
    ).astype(np.float32)


def prepare_episode(public, *, task, body, mode, seed, dummy, bank, feedback,
                    prior, stride, remaining_controls_target="task_timeout"):
    if task == "T3" and any(key.startswith("fixture_") for key in public):
        raise ValueError("static T3 completion features do not accept fixture observations")
    spec = task_spec(task, robot=body)
    if task == "T4":
        actor = PublicLivenessMobileDenseTransportActor(
            spec, dummy, bank, feedback, prior_path=prior, mode=mode, seed=seed,
        )
    elif task == "T3":
        actor = PublicDenseTransportActor(
            spec, dummy, bank, feedback, prior_path=prior, mode=mode, seed=seed,
            retreat_release_version="safe_settled_v1",
        )
    else:
        raise ValueError("terminal-progress training admits only T3 or T4 states")
    common = PublicLivenessMatingFeedbackPolicy.load(
        spec, None, feedback=feedback, modes=[mode], seed=seed,
    )
    gate = PublicRecoveryGate()
    rows = []
    enabled = 0
    for tick in range(len(public["tcp_position"])):
        observation = {key: value[tick:tick + 1] for key, value in public.items()}
        recovery_enabled = gate.observe(observation)
        common_action = common.act(observation)
        batch, _ = actor.prepare(observation)
        if tick % stride:
            continue
        batch["task_reference"] = _task_reference(actor)
        batch["completion_reference"] = np.concatenate((
            np.asarray(observation["goal_position"], np.float32),
            np.asarray(observation["goal_rotation"], np.float32).reshape(1, 9),
        ), axis=1)
        batch["recovery_enabled"] = np.asarray([recovery_enabled])
        batch["requested_mode"] = np.asarray([mode], dtype=np.int64)
        if remaining_controls_target == "task_timeout":
            remaining_controls = int(spec.config.timeout_steps) - tick
        elif remaining_controls_target == "recorded_episode_end":
            # One request produces one public transition, so a complete public
            # trajectory with N observations has N - 1 controls.  This target
            # is available from the B_log trajectory itself and does not open
            # reward, private state, or a success-filtered subset.  A timeout
            # remains a censored episode end rather than being relabeled as a
            # successful terminal state.
            remaining_controls = len(public["tcp_position"]) - 1 - tick
        else:
            raise ValueError("unknown remaining-control target")
        batch["remaining_controls"] = np.asarray([
            max(0, remaining_controls)
        ], dtype=np.int64)
        batch["common_action"] = (
            np.asarray(common_action, np.float32)
            / np.asarray([.08] * 3 + [.4] * 3 + [1.] * 6 + [25.], np.float32)
        )
        for key in ("source_current", "source_query_index", "source_query_distance"):
            batch.pop(key)
        rows.append(batch)
        enabled += int(recovery_enabled)
    return concatenate(rows), {"states": len(rows), "recovery_enabled_states": enabled}


def prepare_states(config, response_spec, bank, feedback):
    frozen_path = checked(config["data"]["frozen_public_inputs"], "frozen public inputs")
    frozen = json.loads(frozen_path.read_text())
    prior = ROOT / config["data"]["source_prior"]["path"]
    for name, expected in config["data"]["source_prior"]["sha256"].items():
        path = prior / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError("source prior changed")
    dummy = PublicLocalResidualTransport(response_spec)
    groups: dict[str, list[Any]] = {"fit": [], "validation": []}
    provenance = {"episodes": {}, "counts": {}}
    stride = int(config["data"]["state_stride"])
    run_task = config.get("task", "T4")
    allowed_tasks = set(config["data"].get("episode_tasks", ("T3", "T4")))
    validation_bodies = set(config["data"].get("validation_bodies", ()))
    for relative, metadata in frozen["training_validation_episodes"].items():
        if metadata.get("deduplicated_within_split", False):
            continue
        episode_task = metadata.get(
            "task", config["data"].get("frozen_episode_task", run_task),
        )
        if episode_task not in allowed_tasks:
            continue
        if (
            metadata["purpose"] == "validation" and validation_bodies
            and metadata["body"] not in validation_bodies
        ):
            continue
        path = ROOT / relative
        actual = sha(path)
        if frozen["sha256"].get(relative) != actual:
            raise ValueError(f"frozen public episode changed: {relative}")
        values, counts = prepare_episode(
            _public_episode(path), task=episode_task, body=metadata["body"],
            mode=int(metadata.get("mode", config["data"].get("frozen_mode", 0))),
            seed=int(metadata.get("seed", path.parent.name.removeprefix("seed_"))),
            dummy=dummy,
            bank=bank, feedback=feedback, prior=prior, stride=stride,
        )
        groups[metadata["purpose"]].append(values)
        provenance["episodes"][relative] = actual
        provenance["counts"][relative] = counts

    log_paths = [checked(config["data"]["weak_log_manifest"], "weak-log manifest")]
    log_paths.extend(
        checked(value, f"additional weak-log manifest {index}")
        for index, value in enumerate(
            config["data"].get("additional_weak_log_manifests", ()), start=1,
        )
    )
    logs = [json.loads(path.read_text()) for path in log_paths]
    if any(
        log["reward"] or log["success_filter"] or log["permission"] != "B_log"
        for log in logs
    ):
        raise ValueError("public weak-log permission changed")
    log_rows = []
    log_validation_rows = []
    weak_log_split = config["data"].get("weak_log_split")
    maximum_log_remaining = config["data"].get(
        "weak_log_remaining_controls_max"
    )
    if maximum_log_remaining is not None and (
        isinstance(maximum_log_remaining, bool)
        or not isinstance(maximum_log_remaining, int)
        or maximum_log_remaining < 1
    ):
        raise ValueError("weak-log remaining-control limit must be a positive integer")
    if run_task == "T3":
        trajectory_type = PublicTargetTrajectory
        target_body = "xarm7"
    elif run_task == "T4":
        trajectory_type = PublicMobileTargetTrajectory
        target_body = "go2_z1"
    else:
        raise ValueError("terminal-progress training admits only T3 or T4")
    for log_index, log in enumerate(logs):
        controls = 0
        for entry_index, entry in enumerate(log["entries"]):
            parent = trajectory_type.load(entry["path"])
            if run_task == "T3" and (
                    parent.metadata["version"] not in {VERSION, FOUR_TASK_VERSION}
                    or parent.metadata.get("environment", "keyed_insertion_recovery") != "keyed_insertion_recovery"):
                raise ValueError("static T3 public trajectory required; fixture data needs its own training entry")
            row = parent.prefix(int(entry["controls"]))
            if (
                parent.digest != entry["parent_sha256"]
                or row.digest != entry["prefix_sha256"]
            ):
                raise ValueError("public weak-log prefix changed")
            controls += len(row.requests)
            if getattr(row, "task", run_task) != run_task:
                raise ValueError("weak-log task differs from training task")
            values, counts = prepare_episode(
                row.public, task=run_task, body=target_body,
                mode=int(row.metadata["mode"]),
                seed=int(row.metadata["seed"]), dummy=dummy, bank=bank,
                feedback=feedback, prior=prior, stride=stride,
                remaining_controls_target=config["data"].get(
                    "weak_log_remaining_controls_target", "task_timeout",
                ),
            )
            count_key = (
                f"B_log/{log_index:02d}/{entry_index:04d}/"
                f"seed_{row.metadata['seed']}"
            )
            if maximum_log_remaining is not None:
                tail = (
                    values["remaining_controls"].reshape(-1)
                    <= maximum_log_remaining
                )
                counts = {
                    **counts,
                    "deadline_tail_states": int(tail.sum()),
                    "remaining_controls_max": maximum_log_remaining,
                }
                if bool(tail.any()):
                    values = select(values, tail)
                else:
                    provenance["counts"][count_key] = counts
                    continue
            validation_entry = bool(
                weak_log_split is not None
                and entry_index % int(weak_log_split["modulus"])
                == int(weak_log_split["validation_residue"])
            )
            (log_validation_rows if validation_entry else log_rows).append(values)
            provenance["counts"][count_key] = {
                **counts,
                "split": "validation" if validation_entry else "fit",
            }
        if controls != int(log["controls"]):
            raise ValueError("complete public weak-log denominator required")
    provenance.update({
        "frozen_public_inputs": pointer(frozen_path),
        "weak_log_manifest": pointer(log_paths[0]),
        "weak_log_manifests": [pointer(path) for path in log_paths],
        "source_bank_sha256": bank.digest,
        "target_reward_or_truth_opened": False,
        "stride": stride,
        "weak_log_remaining_controls_max": maximum_log_remaining,
        "weak_log_remaining_controls_target": config["data"].get(
            "weak_log_remaining_controls_target", "task_timeout",
        ),
        "weak_log_split": deepcopy(weak_log_split),
    })
    validation_rows = groups["validation"] + log_validation_rows
    return (
        concatenate(groups["fit"]), concatenate(validation_rows),
        concatenate(log_rows), provenance,
    )


def policy_metrics(policy, response, feedback, batch, terminal_spec):
    with torch.no_grad():
        _, commands = recovery_policy_commands(policy, feedback, batch)
        task = recovery_task_proxy(policy, response, feedback, batch)
        progress = progress_values(commands, batch, terminal_spec, feedback)
    return task, progress


def validation_metrics(policy, response, feedback, validation, completion_spec):
    tasks = []
    values: dict[str, list[torch.Tensor]] = {
        "linear_deficit": [], "angular_deficit": [], "useful": [],
        "admissible": [], "enabled": [], "terminal": [], "candidate_delta": [],
    }
    if isinstance(completion_spec, PublicFiniteTimeProgressSpec):
        values.update({
            "linear_rate_feasible": [], "angular_rate_feasible": [],
            "remaining_controls": [],
        })
    length = len(validation["target_features"])
    for start in range(0, length, 256):
        batch = select(validation, slice(start, min(start + 256, length)))
        task, completion = policy_metrics(
            policy, response, feedback, batch, completion_spec,
        )
        tasks.append(task.cpu())
        for key in values:
            values[key].append(completion[key].cpu())
    merged = {key: torch.cat(rows) for key, rows in values.items()}
    eligible = merged["enabled"] & ~merged["terminal"]
    if not bool(eligible.any()):
        raise ValueError("validation has no recovery-enabled nonterminal state")
    linear = merged["linear_deficit"][eligible]
    angular = merged["angular_deficit"][eligible]
    result = {
        "task_proxy": float(torch.cat(tasks).mean()),
        "eligible_states": int(eligible.sum()),
        "terminal_states": int((merged["enabled"] & merged["terminal"]).sum()),
        "linear_deficit_p50": float(torch.quantile(linear, .5)),
        "linear_deficit_p95": float(torch.quantile(linear, .95)),
        "linear_deficit_max": float(linear.max()),
        "angular_deficit_p50": float(torch.quantile(angular, .5)),
        "angular_deficit_p95": float(torch.quantile(angular, .95)),
        "angular_deficit_max": float(angular.max()),
        "admissible_fraction": float(merged["admissible"][eligible].float().mean()),
        "useful_fraction": float(merged["useful"][eligible].float().mean()),
        "candidate_delta_mean": float(merged["candidate_delta"][eligible].mean()),
    }
    if isinstance(completion_spec, PublicFiniteTimeProgressSpec):
        result.update({
            "linear_rate_feasible_fraction": float(
                merged["linear_rate_feasible"][eligible].float().mean()
            ),
            "angular_rate_feasible_fraction": float(
                merged["angular_rate_feasible"][eligible].float().mean()
            ),
            "remaining_controls_p50": float(torch.quantile(
                merged["remaining_controls"][eligible].float(), .5,
            )),
            "remaining_controls_min": int(
                merged["remaining_controls"][eligible].min()
            ),
        })
    return result


def required_validation_modes(config):
    modes = config["selection"].get("required_modes")
    return tuple(modes) if modes is not None else ()


def progress_validation_states(config, validation, paired_spec):
    """Restrict a declared mode-specific candidate to its public validation mode."""
    if paired_spec is None or paired_spec.requested_mode is None:
        return validation
    required = required_validation_modes(config)
    if required != (paired_spec.requested_mode,):
        raise ValueError("mode-specific validation mode differs from paired progress")
    requested = validation.get("requested_mode")
    if requested is None or requested.ndim != 1:
        raise ValueError("mode-specific validation requires one requested mode per state")
    mask = requested == paired_spec.requested_mode
    if not bool(mask.any()):
        raise ValueError("mode-specific validation has no requested state")
    return select(validation, mask)


def validation_metrics_by_mode(
    policy, response, feedback, validation, completion_spec, modes,
):
    if not modes:
        return {}
    requested = validation.get("requested_mode")
    if requested is None or requested.ndim != 1:
        raise ValueError("mode-stratified validation requires one requested mode per state")
    result = {}
    for mode in modes:
        mask = requested == int(mode)
        if not bool(mask.any()):
            raise ValueError(f"validation has no held-out states for mode {mode}")
        result[str(mode)] = validation_metrics(
            policy, response, feedback, select(validation, mask), completion_spec,
        )
    return result


def paired_validation_metrics(
    candidate, task_only, feedback, validation, finite_time_spec,
):
    """Measure held-out candidate progress against its paired task-only policy."""
    if not isinstance(finite_time_spec, PublicFiniteTimeProgressSpec):
        raise ValueError("paired validation is T3 finite-time only")
    values: dict[str, list[torch.Tensor]] = {
        "linear_gain": [], "angular_gain": [], "enabled": [], "terminal": [],
    }
    length = len(validation["target_features"])
    for start in range(0, length, 256):
        batch = select(validation, slice(start, min(start + 256, length)))
        with torch.no_grad():
            candidate_commands = recovery_policy_commands(
                candidate, feedback, batch,
            )[1]
            task_only_commands = recovery_policy_commands(
                task_only, feedback, batch,
            )[1]
            current = public_finite_time_progress(
                candidate_commands, batch, finite_time_spec, feedback,
            )
            reference = public_finite_time_progress(
                task_only_commands, batch, finite_time_spec, feedback,
            )
        values["linear_gain"].append((
            current["linear_progress_candidate"]
            - reference["linear_progress_candidate"]
        ).cpu())
        values["angular_gain"].append((
            current["angular_progress_candidate"]
            - reference["angular_progress_candidate"]
        ).cpu())
        values["enabled"].append(current["enabled"].cpu())
        values["terminal"].append(current["terminal"].cpu())
    merged = {key: torch.cat(rows) for key, rows in values.items()}
    eligible = merged["enabled"] & ~merged["terminal"]
    if not bool(eligible.any()):
        raise ValueError("paired validation has no eligible state")
    linear = merged["linear_gain"][eligible]
    angular = merged["angular_gain"][eligible]
    normalized = (
        linear / finite_time_spec.linear_progress_scale
        + angular / finite_time_spec.angular_progress_scale
    )
    return {
        "eligible_states": int(eligible.sum()),
        "linear_gain_mean": float(linear.mean()),
        "linear_gain_p05": float(torch.quantile(linear, .05)),
        "linear_gain_p50": float(torch.quantile(linear, .5)),
        "angular_gain_mean": float(angular.mean()),
        "angular_gain_p05": float(torch.quantile(angular, .05)),
        "angular_gain_p50": float(torch.quantile(angular, .5)),
        "normalized_gain_mean": float(normalized.mean()),
        "normalized_gain_p05": float(torch.quantile(normalized, .05)),
        "normalized_gain_p50": float(torch.quantile(normalized, .5)),
        "both_axes_nonnegative_fraction": float(
            ((linear >= 0.) & (angular >= 0.)).float().mean()
        ),
        "normalized_gain_positive_fraction": float(
            (normalized > 0.).float().mean()
        ),
    }


def paired_validation_metrics_by_mode(
    candidate, task_only, feedback, validation, finite_time_spec, modes,
):
    if not modes:
        return {}
    requested = validation.get("requested_mode")
    if requested is None or requested.ndim != 1:
        raise ValueError("paired mode validation requires requested modes")
    result = {}
    for mode in modes:
        mask = requested == int(mode)
        if not bool(mask.any()):
            raise ValueError(f"paired validation has no states for mode {mode}")
        result[str(mode)] = paired_validation_metrics(
            candidate, task_only, feedback, select(validation, mask),
            finite_time_spec,
        )
    return result


def progress_selection_decision(
    config, spec, candidate_metrics, task_metrics,
    candidate_by_mode, task_by_mode, *, candidate, task_only,
):
    kwargs = dict(
        candidate_proxy=candidate_metrics["task_proxy"],
        task_only_proxy=task_metrics["task_proxy"],
        task_tolerance=float(config["selection"]["task_tolerance"]),
        linear_deficit_p95=candidate_metrics["linear_deficit_p95"],
        angular_deficit_p95=candidate_metrics["angular_deficit_p95"],
        progress_tolerance=float(config["selection"]["progress_tolerance"]),
        useful_fraction=candidate_metrics["useful_fraction"],
        minimum_useful_fraction=float(
            config["selection"]["minimum_useful_fraction"]
        ),
        candidate=candidate, task_only=task_only,
    )
    decision = select_progress_checkpoint(spec=spec, **kwargs)
    per_mode = {}
    for mode in required_validation_modes(config):
        current = candidate_by_mode[str(mode)]
        reference = task_by_mode[str(mode)]
        per_mode[str(mode)] = select_progress_checkpoint(
            spec=spec,
            candidate_proxy=current["task_proxy"],
            task_only_proxy=reference["task_proxy"],
            task_tolerance=kwargs["task_tolerance"],
            linear_deficit_p95=current["linear_deficit_p95"],
            angular_deficit_p95=current["angular_deficit_p95"],
            progress_tolerance=kwargs["progress_tolerance"],
            useful_fraction=current["useful_fraction"],
            minimum_useful_fraction=kwargs["minimum_useful_fraction"],
            candidate=candidate, task_only=task_only,
        )
    if per_mode:
        decision["per_mode"] = per_mode
        decision["checks"].update({
            f"mode_{mode}_{name}": value
            for mode, row in per_mode.items()
            for name, value in row["checks"].items()
        })
        decision["accepted"] = bool(
            decision["accepted"]
            and all(row["accepted"] for row in per_mode.values())
        )
        decision["selected"] = candidate if decision["accepted"] else task_only
    return decision


def projected_validation_metrics(
    policy, response, feedback, validation, completion_spec,
):
    tasks = []
    deficits = {"linear": [], "angular": [], "delta": [], "eligible": []}
    length = len(validation["target_features"])
    for start in range(0, length, 256):
        batch = select(validation, slice(start, min(start + 256, length)))
        task, completion = policy_metrics(
            policy, response, feedback, batch, completion_spec,
        )
        tasks.append(task.cpu())
        deficits["linear"].append(completion["linear_deficit"].cpu())
        deficits["angular"].append(completion["angular_deficit"].cpu())
        deficits["delta"].append(completion["candidate_delta"].cpu())
        deficits["eligible"].append(
            (completion["enabled"] & ~completion["terminal"]).cpu()
        )
    values = {key: torch.cat(rows) for key, rows in deficits.items()}
    eligible = values["eligible"]
    if not bool(eligible.any()):
        raise ValueError("projection validation has no eligible state")
    linear = values["linear"][eligible]
    angular = values["angular"][eligible]
    delta = values["delta"][eligible]
    selected = torch.zeros_like(linear)
    unresolved = torch.ones_like(linear, dtype=torch.bool)
    tolerance = completion_spec.progress_tolerance
    counts = {}
    for fraction in (1., .5, .25):
        accepted = unresolved & (fraction * linear <= tolerance) & (
            fraction * angular <= tolerance
        )
        selected[accepted] = fraction
        counts[str(fraction)] = int(accepted.sum())
        unresolved &= ~accepted
    counts["0.0"] = int(unresolved.sum())
    projected_linear = selected * linear
    projected_angular = selected * angular
    positive = selected > 0
    useful = positive & (
        selected * delta >= completion_spec.minimum_candidate_delta
    )
    return {
        "task_proxy": float(torch.cat(tasks).mean()),
        "eligible_states": int(eligible.sum()),
        "blend_counts": counts,
        "positive_admission_fraction": float(positive.float().mean()),
        "full_admission_fraction": float((selected == 1).float().mean()),
        "projected_useful_fraction": float(useful.float().mean()),
        "projected_candidate_delta_mean": float((selected * delta).mean()),
        "linear_deficit_p95": float(torch.quantile(projected_linear, .95)),
        "linear_deficit_max": float(projected_linear.max()),
        "angular_deficit_p95": float(torch.quantile(projected_angular, .95)),
        "angular_deficit_max": float(projected_angular.max()),
    }


def load_context(config, device):
    manifest_path = checked(config["actor_manifest"], "actor manifest")
    manifest = json.loads(manifest_path.read_text())
    source_path = checked(config["data"]["source_manifest"], "source manifest")
    source = json.loads(source_path.read_text())
    first = config["instances"][sorted(config["instances"])[0]]
    first_response_path = checked(first["response_checkpoint"], "first response checkpoint")
    first_response = PublicIssuedResponseEnsemble.from_checkpoint(
        torch.load(first_response_path, map_location="cpu", weights_only=True), "cpu",
    )
    entries = []
    for row in source["data_manifest"]["offline"]:
        value = dict(row)
        path = Path(value["path"])
        if not path.is_absolute():
            path = ROOT / path
        value["path"] = str(path)
        entries.append(value)
    bank = PublicSkillBank(entries, first_response.spec, task="T3")
    feedback = PublicBiasFeedbackSpec(**config["feedback"])
    raw_fit, raw_validation, raw_logs, data = prepare_states(
        config, first_response.spec, bank, feedback,
    )
    return (
        tensorize(raw_fit, device), tensorize(raw_validation, device),
        tensorize(raw_logs, device), data, feedback, first_response.spec,
        pointer(manifest_path),
    )


def load_instance(name, row, device, feedback):
    response_path = checked(row["response_checkpoint"], f"{name} response checkpoint")
    warm_path = checked(row["warm_continuation"], f"{name} warm continuation")
    task_path = checked(row["task_only_checkpoint"], f"{name} task-only checkpoint")
    current_path = checked(row["current_checkpoint"], f"{name} current checkpoint")
    gate_path = checked(row["gate_sidecar"], f"{name} gate sidecar")
    response = PublicIssuedResponseEnsemble.from_checkpoint(
        torch.load(response_path, map_location="cpu", weights_only=True), device,
    ).eval()
    for parameter in response.parameters():
        parameter.requires_grad_(False)
    task_only = PublicLocalResidualTransport.from_checkpoint(
        torch.load(task_path, map_location="cpu", weights_only=True), device,
    ).eval()
    current = PublicLocalResidualTransport.from_checkpoint(
        torch.load(current_path, map_location="cpu", weights_only=True), device,
    ).eval()
    warm = torch.load(warm_path, map_location="cpu", weights_only=True)
    gate = json.loads(gate_path.read_text())["gate_spec"]
    return response, task_only, current, warm, PublicAnchoringGateSpec(**gate), {
        "response_checkpoint": pointer(response_path),
        "warm_continuation": pointer(warm_path),
        "task_only_checkpoint": pointer(task_path),
        "current_checkpoint": pointer(current_path),
        "gate_sidecar": pointer(gate_path),
    }


def _checkpoint_matches_policy(checkpoint, policy):
    current = policy.checkpoint()
    if any(
        checkpoint.get(key) != current.get(key)
        for key in ("version", "response_spec", "transport_spec")
    ):
        return False
    expected = checkpoint.get("state_dict", {})
    actual = current["state_dict"]
    return set(expected) == set(actual) and all(
        torch.equal(expected[key].detach().cpu(), actual[key].detach().cpu())
        for key in expected
    )


def uses_task_only_start(variant_name: str) -> bool:
    """Return whether a paired variant must share the frozen task-only start."""
    return variant_name in {
        "matched_task_only", "same_start_anchored", "same_start_task_only",
        "same_start_relative_anchored", "same_start_relative_task_only",
        "task_first_directional_anchored", "task_first_directional_task_only",
        "task_first_directional_same_state_anchored",
        "task_first_margin_anchored", "task_first_margin_task_only",
        "task_first_stable_margin_anchored",
        "task_first_physical_selector_anchored",
        "task_first_ensemble_robust_anchored",
    }


def initialize_candidate(
    config, task_only, current, warm, calibration, response, feedback, device,
):
    variant = training_variant(config)
    initialization = config["training"].get("initialization")
    if initialization is None:
        if variant["name"] != "transport":
            raise ValueError("matched task-only does not admit warm optimizer continuation")
        candidate = PublicLocalResidualTransport.from_checkpoint(
            warm["checkpoint"], device,
        )
        trainer = PublicRecoveryGatedTransportTrainer(
            candidate, response, feedback,
        )
        continuation = deepcopy(warm)
        if device.type == "cpu":
            continuation["rng"]["torch_cuda"] = []
        trainer.restore(continuation)
        return candidate, trainer, {
            "kind": "warm_continuation",
            "optimizer_restored": True,
            "steps_restored": int(warm["steps"]),
        }
    initialization_kind = initialization.get("kind")
    if initialization_kind not in {
        "recentered_checkpoint_fresh_optimizer",
        "recentered_current_checkpoint_fresh_optimizer",
    }:
        raise ValueError("unknown terminal-progress initialization")
    if initialization.get("calibration") != "fit_plus_weak_log_public_states":
        raise ValueError("recentered initialization requires declared public calibration")
    task_only_start = uses_task_only_start(variant["name"])
    base_policy = task_only if task_only_start else current
    if initialization_kind == "recentered_checkpoint_fresh_optimizer":
        if variant["name"] != "transport":
            raise ValueError("matched task-only cannot initialize from transport continuation")
        if not _checkpoint_matches_policy(warm["checkpoint"], current):
            raise ValueError("warm continuation checkpoint differs from current actor")
        initial_checkpoint = warm["checkpoint"]
        initialization_source = "warm_continuation_checkpoint"
    else:
        initial_checkpoint = base_policy.checkpoint()
        initialization_source = (
            "task_only_checkpoint"
            if task_only_start
            else "current_selected_checkpoint"
        )
    candidate = PublicLocalResidualTransport.from_checkpoint(
        initial_checkpoint, device,
    )
    recentering = recenter_saturated_transport_output(
        candidate, calibration, feedback,
        maximum_median_absolute_logit=float(
            initialization["maximum_median_absolute_logit"]
        ),
    )
    trainer = PublicRecoveryGatedTransportTrainer(candidate, response, feedback)
    return candidate, trainer, {
        "kind": initialization_kind,
        "source": initialization_source,
        "calibration": initialization["calibration"],
        "initial_checkpoint_matches_selected_base": _checkpoint_matches_policy(
            initial_checkpoint, base_policy,
        ),
        "training_variant": variant,
        "optimizer_restored": False,
        "steps_restored": 0,
        "recentered_output": recentering,
    }


def run_preflight(config, config_path):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight must not initialize CUDA")
    device = torch.device("cpu")
    parent_attempt = checked_parent_attempt(config)
    development_evidence = checked_development_evidence(config)
    fit, validation, logs, data, feedback, _, actor_manifest = load_context(config, device)
    calibration = combine(fit, logs)
    progress_kind, terminal_spec = progress_spec(config)
    relative_response_spec = response_normalization_spec(config)
    paired_spec = paired_progress_spec(config)
    validation = progress_validation_states(config, validation, paired_spec)
    robust_task_spec = ensemble_task_spec(config)
    variant = training_variant(config)
    modes = required_validation_modes(config)
    checkpoint_selection = validation_checkpoint_selection(config)
    retain_validation_checkpoints = bool(
        config["training"].get("retain_validation_checkpoints", False)
    )
    if retain_validation_checkpoints and checkpoint_selection is None:
        raise ValueError(
            "retained validation checkpoints require a frozen checkpoint interval"
        )
    validate_checkpoint_selection_scope(
        checkpoint_selection, terminal_spec, paired_spec,
    )
    target_sampling = config["training"].get("target_sampling")
    target_sampling_counts = None
    if target_sampling is not None:
        recorded_modes = logs.get("requested_mode")
        if recorded_modes is None:
            raise ValueError("mode-balanced target sampling requires public modes")
        recorded_modes = recorded_modes.detach().cpu().numpy().reshape(-1)
        target_sampling_counts = {
            str(mode): int((recorded_modes == int(mode)).sum())
            for mode in target_sampling["modes"]
        }
        if any(count == 0 for count in target_sampling_counts.values()):
            raise ValueError("mode-balanced target sampling found an empty mode")
    reports = {}
    for name, row in sorted(config["instances"].items()):
        response, task_only, current, warm, gate, inputs = load_instance(
            name, row, device, feedback,
        )
        candidate, trainer, initialization = initialize_candidate(
            config, task_only, current, warm, calibration, response, feedback, device,
        )
        current_metrics = validation_metrics(
            current, response, feedback, validation, terminal_spec,
        )
        task_metrics = validation_metrics(
            task_only, response, feedback, validation, terminal_spec,
        )
        initialized_metrics = validation_metrics(
            candidate, response, feedback, validation, terminal_spec,
        )
        initialized_paired = (
            paired_validation_metrics(
                candidate, task_only, feedback, validation, terminal_spec,
            )
            if paired_spec is not None else None
        )
        initialized_paired_by_mode = (
            paired_validation_metrics_by_mode(
                candidate, task_only, feedback, validation, terminal_spec, modes,
            )
            if paired_spec is not None else {}
        )
        reports[name] = {
            "current": current_metrics,
            "current_by_mode": validation_metrics_by_mode(
                current, response, feedback, validation, terminal_spec, modes,
            ),
            "task_only": task_metrics,
            "task_only_by_mode": validation_metrics_by_mode(
                task_only, response, feedback, validation, terminal_spec, modes,
            ),
            "initialized": initialized_metrics,
            "initialized_by_mode": validation_metrics_by_mode(
                candidate, response, feedback, validation, terminal_spec, modes,
            ),
            "initialized_paired": initialized_paired,
            "initialized_paired_by_mode": initialized_paired_by_mode,
            "initialization": initialization,
            "inputs": inputs,
            "gate_spec": gate.__dict__,
        }
        del response, task_only, current, candidate, trainer, warm

    # One exact restore/gradient proof covers the shared trainer implementation.
    name = sorted(config["instances"])[0]
    response, task_only, current, warm, gate, _ = load_instance(
        name, config["instances"][name], device, feedback,
    )
    candidate, trainer, initialization = initialize_candidate(
        config, task_only, current, warm, calibration, response, feedback, device,
    )
    count = min(int(config["training"]["batch_size"]), len(fit["target_features"]))
    if paired_spec is not None and paired_spec.requested_mode is not None:
        target_count = round(
            count * float(config["training"]["target_state_fraction"])
        )
        requested = logs.get("requested_mode")
        if requested is None:
            raise ValueError("mode-specific preflight requires requested modes")
        target_ids = torch.nonzero(
            requested.reshape(-1) == paired_spec.requested_mode,
            as_tuple=False,
        ).reshape(-1)
        if len(target_ids) < target_count:
            raise ValueError("mode-specific preflight has too few requested states")
        batch = combine(
            select(logs, target_ids[:target_count]),
            select(fit, torch.arange(count - target_count)),
        )
    else:
        batch = select(fit, torch.arange(count))
    with torch.no_grad():
        reference = (
            recovery_task_proxy_members(task_only, response, feedback, batch)
            if robust_task_spec is not None
            else recovery_task_proxy(task_only, response, feedback, batch)
        )
        response_reference = (
            recovery_response_reference(task_only, response, feedback, batch)
            if relative_response_spec is not None else None
        )
        task_only_commands = (
            recovery_policy_commands(task_only, feedback, batch)[1]
            if paired_spec is not None else None
        )
    before = deepcopy(trainer.state())
    first = trainer.step(
        batch, response_weight=variant["response_anchoring_weight"],
        task_only_reference=reference,
        task_only_response_reference=response_reference,
        response_spec=relative_response_spec,
        task_only_commands=task_only_commands,
        paired_progress_spec=paired_spec,
        ensemble_task_spec=robust_task_spec,
        apply_anchoring_gate=variant.get("apply_anchoring_gate", True),
        gate_spec=gate, **progress_objective_kwargs(terminal_spec),
    )
    after = deepcopy(trainer.state())
    trainer.restore(before)
    second = trainer.step(
        batch, response_weight=variant["response_anchoring_weight"],
        task_only_reference=reference,
        task_only_response_reference=response_reference,
        response_spec=relative_response_spec,
        task_only_commands=task_only_commands,
        paired_progress_spec=paired_spec,
        ensemble_task_spec=robust_task_spec,
        apply_anchoring_gate=variant.get("apply_anchoring_gate", True),
        gate_spec=gate, **progress_objective_kwargs(terminal_spec),
    )
    exact = first == second and all(
        torch.equal(after["checkpoint"]["state_dict"][key], trainer.state()["checkpoint"]["state_dict"][key])
        for key in after["checkpoint"]["state_dict"]
    )
    finite = all(np.isfinite(value) for value in first.values())
    output = ROOT / config["preflight_output"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": config["version"].replace("training", "preflight"),
        "status": "complete" if exact and finite else "failed",
        "passed": bool(exact and finite),
        "device": "cpu",
        "gpu_calls": 0,
        "confirmation_states_opened": 0,
        "instances": reports,
        "trainer_proof": {
            "finite": finite, "restore_exact": exact, "metrics": first,
            "progress_kind": progress_kind,
            "initialization": initialization,
            "checkpoint_selection": checkpoint_selection,
            "legacy_objective_is_default_when_progress_spec_absent": True,
            "training_variant": variant,
            "target_sampling": deepcopy(target_sampling),
            "target_sampling_counts": target_sampling_counts,
            "response_normalization": (
                relative_response_spec.__dict__
                if relative_response_spec is not None else None
            ),
            "paired_progress": (
                paired_spec.__dict__ if paired_spec is not None else None
            ),
            "ensemble_task": (
                robust_task_spec.__dict__
                if robust_task_spec is not None else None
            ),
        },
        "data": data,
        "actor_manifest": actor_manifest,
        "clean_ablation": deepcopy(config.get("clean_ablation")),
        "required_validation_modes": list(modes),
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "parent_attempt": parent_attempt,
            "development_evidence": development_evidence,
        },
        "new_physics_calls": 0,
        "target_reward_or_truth_opened": False,
        **(
            {"resolved_shared_start_report": config["resolved_shared_start_report"]}
            if "resolved_shared_start_report" in config else {}
        ),
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not report["passed"]:
        raise AssertionError("terminal-progress training preflight failed")


def run_projection_preflight(config, config_path):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU projection preflight must not initialize CUDA")
    device = torch.device("cpu")
    parent_attempt = checked_parent_attempt(config)
    _, validation, _, data, feedback, _, actor_manifest = load_context(config, device)
    completion_spec = PublicCompletionProgressSpec(**config["completion_progress"])
    reports = {}
    passed = True
    for name, row in sorted(config["instances"].items()):
        response, task_only, _, _, gate, inputs = load_instance(
            name, row, device, feedback,
        )
        candidate_path = checked(
            row["candidate_checkpoint"], f"{name} projected candidate checkpoint",
        )
        candidate = PublicLocalResidualTransport.from_checkpoint(
            torch.load(candidate_path, map_location="cpu", weights_only=True), device,
        ).eval()
        candidate_metrics = projected_validation_metrics(
            candidate, response, feedback, validation, completion_spec,
        )
        task_metrics = validation_metrics(
            task_only, response, feedback, validation, completion_spec,
        )
        checks = {
            "task_compatible": candidate_metrics["task_proxy"]
            <= task_metrics["task_proxy"] + float(config["selection"]["task_tolerance"]),
            "linear_progress_compatible": candidate_metrics["linear_deficit_max"]
            <= float(config["selection"]["progress_tolerance"]),
            "angular_progress_compatible": candidate_metrics["angular_deficit_max"]
            <= float(config["selection"]["progress_tolerance"]),
            "candidate_utility": candidate_metrics["projected_useful_fraction"]
            >= float(config["selection"]["minimum_useful_fraction"]),
        }
        accepted = all(checks.values())
        passed &= accepted
        reports[name] = {
            "accepted": accepted,
            "checks": checks,
            "candidate_validation": candidate_metrics,
            "task_only_validation": task_metrics,
            "candidate_checkpoint": pointer(candidate_path),
            "inputs": inputs,
            "gate_spec": gate.__dict__,
        }
    output = ROOT / config["projection_preflight_output"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": config["version"],
        "status": "complete" if passed else "rejected",
        "passed": bool(passed),
        "device": "cpu",
        "gpu_calls": 0,
        "new_physics_calls": 0,
        "confirmation_states_opened": 0,
        "target_reward_or_truth_opened": False,
        "projection": {
            "version": "public_completion_dominance_v1",
            "fractions": [1., .5, .25, 0.],
            "progress_tolerance": completion_spec.progress_tolerance,
        },
        "instances": reports,
        "data": data,
        "actor_manifest": actor_manifest,
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "parent_attempt": parent_attempt,
        },
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "passed": report["passed"],
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not report["passed"]:
        raise AssertionError("T4 projected execution preflight failed")


def run_training(config, config_path):
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("terminal-progress training requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("terminal-progress training requires exactly one logical CUDA device")
    device = torch.device(expected["logical_device"])
    identity = torch.cuda.get_device_properties(0)
    if str(identity.uuid) != expected["physical_uuid"]:
        raise RuntimeError("visible CUDA device is not the admitted physical GPU1")
    parent_attempt = checked_parent_attempt(config)
    development_evidence = checked_development_evidence(config)
    preflight_path = checked(config["preflight_report"], "CPU preflight")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("gpu_calls") != 0:
        raise ValueError("passed CPU-only terminal-progress preflight required")
    preflight_runner = checked(
        preflight["artifacts"]["runner"], "CPU preflight runner",
    )
    if preflight_runner.resolve() != Path(__file__).resolve():
        raise ValueError("CPU preflight used a different training entrypoint")
    expected_kind, _ = progress_spec(config)
    if preflight["trainer_proof"].get("progress_kind") != expected_kind:
        raise ValueError("CPU preflight used a different progress objective")
    if preflight.get("clean_ablation") != config.get("clean_ablation"):
        raise ValueError("CPU preflight used a different clean ablation treatment")

    fit, validation, logs, data, feedback, _, actor_manifest = load_context(config, device)
    calibration = combine(fit, logs)
    progress_kind, terminal_spec = progress_spec(config)
    relative_response_spec = response_normalization_spec(config)
    paired_spec = paired_progress_spec(config)
    validation = progress_validation_states(config, validation, paired_spec)
    robust_task_spec = ensemble_task_spec(config)
    variant = training_variant(config)
    modes = required_validation_modes(config)
    checkpoint_selection = validation_checkpoint_selection(config)
    retain_validation_checkpoints = bool(
        config["training"].get("retain_validation_checkpoints", False)
    )
    if retain_validation_checkpoints and checkpoint_selection is None:
        raise ValueError(
            "retained validation checkpoints require a frozen checkpoint interval"
        )
    control_checkpoint_policy = config["training"].get(
        "control_checkpoint_policy", "frozen_selector",
    )
    if control_checkpoint_policy not in {"frozen_selector", "terminal_update"}:
        raise ValueError("unknown training control checkpoint policy")
    validate_checkpoint_selection_scope(
        checkpoint_selection, terminal_spec, paired_spec,
    )
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", config)
    begun = time.monotonic()
    reports = {}
    try:
        for name, row in sorted(config["instances"].items()):
            response, task_only, current, warm, gate, inputs = load_instance(
                name, row, device, feedback,
            )
            candidate, trainer, initialization = initialize_candidate(
                config, task_only, current, warm, calibration, response, feedback, device,
            )
            seed = int(row["seed"])
            generator = np.random.default_rng(seed)
            batch_size = int(config["training"]["batch_size"])
            target_count = round(batch_size * float(config["training"]["target_state_fraction"]))
            target_sampling = config["training"].get("target_sampling")
            target_mode_ids = None
            if target_sampling is not None:
                modes_for_sampling = tuple(int(value) for value in target_sampling["modes"])
                if target_count % len(modes_for_sampling):
                    raise ValueError("mode-balanced target batch must divide evenly")
                recorded_modes = logs.get("requested_mode")
                if recorded_modes is None:
                    raise ValueError("mode-balanced target sampling requires public modes")
                recorded_modes = recorded_modes.detach().cpu().numpy().reshape(-1)
                target_mode_ids = {
                    mode: np.flatnonzero(recorded_modes == mode)
                    for mode in modes_for_sampling
                }
                if any(len(indices) == 0 for indices in target_mode_ids.values()):
                    raise ValueError("mode-balanced target sampling found an empty mode")
            curves = []
            stream = hashlib.sha256()
            selected_state = None
            selected_update = None
            selected_metrics = None
            selected_metrics_by_mode = None
            selected_paired_metrics = None
            selected_paired_metrics_by_mode = None
            selected_score = None
            retained_states = {}
            task_metrics = validation_metrics(
                task_only, response, feedback, validation, terminal_spec,
            )
            task_metrics_by_mode = validation_metrics_by_mode(
                task_only, response, feedback, validation, terminal_spec, modes,
            )
            for update in range(1, int(config["training"]["updates"]) + 1):
                if time.monotonic() - begun > float(config["limits"]["wall_seconds"]):
                    raise TimeoutError("terminal-progress training wall ceiling")
                if target_mode_ids is None:
                    log_ids = generator.integers(
                        len(logs["target_features"]), size=target_count,
                    )
                else:
                    per_mode = target_count // len(target_mode_ids)
                    log_ids = np.concatenate([
                        generator.choice(indices, size=per_mode, replace=True)
                        for indices in target_mode_ids.values()
                    ])
                    generator.shuffle(log_ids)
                fit_ids = generator.integers(
                    len(fit["target_features"]), size=batch_size - target_count,
                )
                stream.update(np.asarray(log_ids, np.int64).tobytes())
                stream.update(np.asarray(fit_ids, np.int64).tobytes())
                batch = combine(
                    select(logs, torch.as_tensor(log_ids, device=device)),
                    select(fit, torch.as_tensor(fit_ids, device=device)),
                )
                with torch.no_grad():
                    reference = (
                        recovery_task_proxy_members(
                            task_only, response, feedback, batch,
                        )
                        if robust_task_spec is not None
                        else recovery_task_proxy(
                            task_only, response, feedback, batch,
                        )
                    )
                    response_reference = (
                        recovery_response_reference(
                            task_only, response, feedback, batch,
                        )
                        if relative_response_spec is not None else None
                    )
                    task_only_commands = (
                        recovery_policy_commands(
                            task_only, feedback, batch,
                        )[1]
                        if paired_spec is not None else None
                    )
                metrics = trainer.step(
                    batch, response_weight=variant["response_anchoring_weight"],
                    task_only_reference=reference,
                    task_only_response_reference=response_reference,
                    response_spec=relative_response_spec,
                    task_only_commands=task_only_commands,
                    paired_progress_spec=paired_spec,
                    ensemble_task_spec=robust_task_spec,
                    apply_anchoring_gate=variant.get(
                        "apply_anchoring_gate", True,
                    ),
                    gate_spec=gate, **progress_objective_kwargs(terminal_spec),
                )
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"{name} nonfinite terminal-progress objective")
                if update % int(config["training"]["log_interval"]) == 0:
                    curve = {"update": update, "metrics": metrics}
                    if (
                        checkpoint_selection is not None
                        and update % checkpoint_selection["interval_updates"] == 0
                    ):
                        checkpoint_metrics = validation_metrics(
                            candidate, response, feedback, validation, terminal_spec,
                        )
                        checkpoint_metrics_by_mode = validation_metrics_by_mode(
                            candidate, response, feedback, validation,
                            terminal_spec, modes,
                        )
                        checkpoint_paired = (
                            paired_validation_metrics(
                                candidate, task_only, feedback, validation,
                                terminal_spec,
                            )
                            if paired_spec is not None else None
                        )
                        checkpoint_paired_by_mode = (
                            paired_validation_metrics_by_mode(
                                candidate, task_only, feedback, validation,
                                terminal_spec, modes,
                            )
                            if paired_spec is not None else {}
                        )
                        checkpoint_decision = progress_selection_decision(
                            config, terminal_spec,
                            checkpoint_metrics, task_metrics,
                            checkpoint_metrics_by_mode, task_metrics_by_mode,
                            candidate="candidate", task_only="task-only",
                        )
                        curve.update({
                            "validation": checkpoint_metrics,
                            "validation_by_mode": checkpoint_metrics_by_mode,
                            "selection_checks": checkpoint_decision["checks"],
                            **(
                                {
                                    "paired_validation": checkpoint_paired,
                                    "paired_validation_by_mode": (
                                        checkpoint_paired_by_mode
                                    ),
                                }
                                if checkpoint_paired is not None else {}
                            ),
                        })
                        score = None
                        if (
                            checkpoint_selection["rule"]
                            == "best_validation_paired_progress"
                        ):
                            score = checkpoint_paired_progress_score(
                                checkpoint_paired,
                                checkpoint_paired_by_mode,
                            )
                        elif isinstance(terminal_spec, PublicFiniteTimeProgressSpec):
                            score = max(
                                checkpoint_progress_score(metrics, terminal_spec)
                                for metrics in (
                                    checkpoint_metrics,
                                    *checkpoint_metrics_by_mode.values(),
                                )
                            )
                        if score is not None:
                            curve["progress_margin_score"] = score
                        eligible = (
                            update >= checkpoint_selection["minimum_update"]
                            and checkpoint_decision["accepted"]
                        )
                        first = (
                            checkpoint_selection["rule"]
                            == "first_validation_compatible"
                            and selected_state is None
                        )
                        best = (
                            checkpoint_selection["rule"] in {
                                "best_validation_progress_margin",
                                "best_validation_paired_progress",
                            }
                            and score is not None
                            and (selected_score is None or score < selected_score)
                        )
                        if eligible and (first or best):
                            selected_state = annotate_checkpoint(
                                candidate.checkpoint(),
                                config.get("clean_ablation"),
                            )
                            selected_update = update
                            selected_metrics = checkpoint_metrics
                            selected_metrics_by_mode = checkpoint_metrics_by_mode
                            selected_paired_metrics = checkpoint_paired
                            selected_paired_metrics_by_mode = (
                                checkpoint_paired_by_mode
                            )
                            selected_score = score
                        if retain_validation_checkpoints:
                            retained_states[update] = annotate_checkpoint(
                                candidate.checkpoint(),
                                config.get("clean_ablation"),
                            )
                    curves.append(curve)
                    print(json.dumps({"instance": name, "update": update, "metrics": metrics}), flush=True)

            folder = output / name
            folder.mkdir()
            retained_checkpoints = {}
            for update, retained_state in sorted(retained_states.items()):
                retained_path = (
                    folder / "candidates" / f"update_{update:04d}.pt"
                )
                save_torch(retained_path, retained_state)
                retained_checkpoints[str(update)] = pointer(retained_path)
            terminal_state = annotate_checkpoint(
                candidate.checkpoint(), config.get("clean_ablation"),
            )
            if checkpoint_selection is None:
                selected_state = terminal_state
                selected_update = int(config["training"]["updates"])
            else:
                save_torch(folder / "terminal_checkpoint.pt", terminal_state)
            if selected_state is not None:
                save_torch(folder / "checkpoint.pt", selected_state)
            continuation = trainer.state()
            continuation["checkpoint"] = annotate_checkpoint(
                continuation["checkpoint"], config.get("clean_ablation"),
            )
            if config.get("clean_ablation") is not None:
                continuation["clean_ablation"] = deepcopy(
                    config["clean_ablation"]
                )
            save_torch(folder / "continuation.pt", continuation)
            final_metrics = validation_metrics(
                candidate, response, feedback, validation, terminal_spec,
            )
            final_metrics_by_mode = validation_metrics_by_mode(
                candidate, response, feedback, validation, terminal_spec, modes,
            )
            final_paired_metrics = (
                paired_validation_metrics(
                    candidate, task_only, feedback, validation, terminal_spec,
                )
                if paired_spec is not None else None
            )
            final_paired_metrics_by_mode = (
                paired_validation_metrics_by_mode(
                    candidate, task_only, feedback, validation,
                    terminal_spec, modes,
                )
                if paired_spec is not None else {}
            )
            candidate_metrics = selected_metrics or final_metrics
            candidate_metrics_by_mode = (
                selected_metrics_by_mode or final_metrics_by_mode
            )
            candidate_paired_metrics = (
                selected_paired_metrics or final_paired_metrics
            )
            candidate_paired_metrics_by_mode = (
                selected_paired_metrics_by_mode or final_paired_metrics_by_mode
            )
            candidate_path = (
                folder / "checkpoint.pt" if selected_state is not None
                else folder / "terminal_checkpoint.pt"
            )
            candidate_pointer = pointer(candidate_path)
            terminal_path = (
                folder / "terminal_checkpoint.pt"
                if checkpoint_selection is not None else folder / "checkpoint.pt"
            )
            selection = progress_selection_decision(
                config, terminal_spec, candidate_metrics, task_metrics,
                candidate_metrics_by_mode, task_metrics_by_mode,
                candidate=candidate_pointer,
                task_only=inputs["task_only_checkpoint"],
            )
            report = {
                "status": "complete", "instance": name, "seed": seed,
                "updates": int(config["training"]["updates"]),
                "sample_stream_sha256": stream.hexdigest(),
                "candidate_validation": candidate_metrics,
                "candidate_validation_by_mode": candidate_metrics_by_mode,
                "candidate_paired_validation": candidate_paired_metrics,
                "candidate_paired_validation_by_mode": (
                    candidate_paired_metrics_by_mode
                ),
                "task_only_validation": task_metrics,
                "task_only_validation_by_mode": task_metrics_by_mode,
                "selection": selection,
                "selected_checkpoint": selection["selected"],
                "control_checkpoint": (
                    pointer(terminal_path)
                    if control_checkpoint_policy == "terminal_update"
                    else selection["selected"]
                ),
                "control_checkpoint_policy": control_checkpoint_policy,
                "gate_spec": gate.__dict__,
                "progress_kind": progress_kind,
                "training_variant": variant,
                "clean_ablation": deepcopy(config.get("clean_ablation")),
                "target_sampling": deepcopy(
                    config["training"].get("target_sampling")
                ),
                "response_normalization": (
                    relative_response_spec.__dict__
                    if relative_response_spec is not None else None
                ),
                "paired_progress": (
                    paired_spec.__dict__ if paired_spec is not None else None
                ),
                "ensemble_task": (
                    robust_task_spec.__dict__
                    if robust_task_spec is not None else None
                ),
                "progress_spec": terminal_spec.__dict__,
                "initialization": initialization,
                "checkpoint_selection": checkpoint_selection,
                "selected_update": selected_update,
                "selected_progress_margin_score": selected_score,
                "retained_validation_checkpoints": retained_checkpoints,
                "final_validation": final_metrics,
                "final_validation_by_mode": final_metrics_by_mode,
                "final_paired_validation": final_paired_metrics,
                "final_paired_validation_by_mode": final_paired_metrics_by_mode,
                "curve": curves, "inputs": inputs,
            }
            write_json(folder / "report.json", report)
            reports[name] = report
            write_json(output / "progress.json", reports)
            del response, task_only, current, candidate, trainer
            torch.cuda.empty_cache()

        expected_instances = len(config["instances"])
        selection_gate_passed = len(reports) == expected_instances and all(
            row["selection"]["accepted"] for row in reports.values()
        )
        passed = bool(
            len(reports) == expected_instances
            and (
                selection_gate_passed
                or control_checkpoint_policy == "terminal_update"
            )
        )
        result = {
            "schema_version": config["version"],
            "status": "complete" if passed else "selection_failed",
            "passed": passed,
            "instances": reports,
            "device": {
                **expected, "logical_name": identity.name,
                "observed_uuid": str(identity.uuid),
            },
            "data": data,
            "actor_manifest": actor_manifest,
            "artifacts": {
                "config": pointer(config_path),
                "runner": pointer(Path(__file__).resolve()),
                "preflight": pointer(preflight_path),
                "parent_attempt": parent_attempt,
                "development_evidence": development_evidence,
            },
            "training_updates": expected_instances * int(config["training"]["updates"]),
            "training_variant": variant,
            "clean_ablation": deepcopy(config.get("clean_ablation")),
            "pilot_instance_count": expected_instances,
            "target_sampling": deepcopy(
                config["training"].get("target_sampling")
            ),
            "response_normalization": (
                relative_response_spec.__dict__
                if relative_response_spec is not None else None
            ),
            "paired_progress": (
                paired_spec.__dict__ if paired_spec is not None else None
            ),
            "ensemble_task": (
                robust_task_spec.__dict__
                if robust_task_spec is not None else None
            ),
            "control_checkpoint_policy": control_checkpoint_policy,
            "selection_gate_passed": selection_gate_passed,
            "required_validation_modes": list(modes),
            **(
                {
                    "resolved_base_training_config": config[
                        "resolved_base_training_config"
                    ]
                }
                if "resolved_base_training_config" in config else {}
            ),
            **(
                {"resolved_shared_start_report": config["resolved_shared_start_report"]}
                if "resolved_shared_start_report" in config else {}
            ),
            "new_physics_calls": 0,
            "confirmation_states_opened": 0,
            "target_reward_or_truth_opened": False,
            "wall_seconds": time.monotonic() - begun,
        }
    except BaseException as error:
        result = {
            "schema_version": config["version"],
            "status": "failed", "passed": False, "error": repr(error),
            "instances": reports, "wall_seconds": time.monotonic() - begun,
            "new_physics_calls": 0, "confirmation_states_opened": 0,
        }
        raise
    finally:
        write_json(output / "report.json", result)
    print(json.dumps({
        "status": result["status"], "passed": result["passed"],
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not result["passed"]:
        raise AssertionError("not all terminal-progress checkpoints passed selection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-projection", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_training_config(config_path)
    if config.get("version") not in {
        "t4-completion-aligned-transport-training-v1",
        "t4-completion-tail-aligned-transport-training-v2",
        "t4-completion-projected-execution-preflight-v1",
        "t3-finite-time-terminal-progress-training-v1",
        "t3-finite-time-terminal-progress-training-v2",
        "t3-finite-time-terminal-progress-training-v3",
        "t3-finite-time-terminal-progress-training-v4",
        "t3-finite-time-terminal-progress-training-v5",
        "t3-finite-time-terminal-progress-matched-task-only-training-v1",
        "t3-finite-time-same-start-anchored-training-v1",
        "t3-finite-time-same-start-task-only-training-v1",
        "t3-finite-time-same-start-relative-anchored-training-v1",
        "t3-finite-time-same-start-relative-task-only-training-v1",
        "t3-task-first-directional-anchored-training-v1",
        "t3-task-first-directional-same-state-anchored-training-v1",
        "t3-task-first-directional-same-state-task-only-training-v1",
        "t3-task-first-directional-task-only-training-v1",
        "t3-task-first-margin-anchored-training-v1",
        "t3-task-first-margin-task-only-training-v1",
        "t3-task-first-stable-margin-anchored-training-v2",
        "t3-task-first-physical-selector-anchored-training-v3",
        "t3-task-first-ensemble-robust-anchored-training-v4",
        "t3-task-first-ensemble-robust-recorded-completion-training-v5",
        "t3-task-first-ensemble-robust-recorded-completion-training-v6",
        "t3-task-first-ensemble-robust-recorded-completion-training-v7",
        "t3-task-first-mode-tail-recorded-completion-training-v8",
        "t3-task-first-mode-specific-recorded-completion-training-v9",
        "t4-completion-tail-aligned-matched-task-only-training-v1",
        "t4-completion-same-start-anchored-training-v1",
        "t4-completion-same-start-task-only-training-v1",
    }:
        raise ValueError("unknown terminal-progress training version")
    os.environ.setdefault(
        "CROSS_EBDM_NATIVE_LIBRARY",
        str(ROOT / ".runtime/cross_ebdm/native/libcross_ebdm.so"),
    )
    if args.validate_only and args.validate_projection:
        parser.error("choose one validation mode")
    if args.validate_projection:
        run_projection_preflight(config, config_path)
    elif args.validate_only:
        run_preflight(config, config_path)
    else:
        run_training(config, config_path)


if __name__ == "__main__":
    main()
