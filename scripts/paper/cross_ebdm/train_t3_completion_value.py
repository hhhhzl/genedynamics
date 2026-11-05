#!/usr/bin/env python3
"""Train a public T3 remaining-control value ensemble on GPU1."""

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

from genedynamics.learning.datasets.transfer.target import (
    FOUR_TASK_VERSION, VERSION, PublicTargetTrajectory,
)
from genedynamics.solvers.single.cross_ebdm.learning.completion_value import (
    PublicActionConditionalCompletionValueEnsemble,
    PublicCompletionValueEnsemble,
    PublicCompletionValueSpec,
    PublicContextCompletionValueEnsemble,
    PublicContextCompletionValueSpec,
    PublicEndpointAnchoredCompletionValueEnsemble,
    PublicPhaseCompletionValueEnsemble,
    PublicPhaseCompletionValueSpec,
    PublicQualifiedContextCompletionValueEnsemble,
    completion_value_features,
    context_completion_value_features,
    phase_completion_value_features,
)
from genedynamics.tasks.transfer.observation import PublicControllerMemory
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


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("version") not in {
        "t3-public-completion-value-training-v1",
        "t3-public-phase-completion-value-training-v2",
        "t3-public-context-completion-value-training-v3",
        "t3-public-action-conditional-completion-value-training-v4",
        "t3-public-endpoint-anchored-completion-value-training-v5",
        "t3-public-qualified-context-completion-value-training-v6",
    }:
        raise ValueError("unknown T3 completion-value training version")
    if config.get("status") != "frozen_unexecuted":
        raise ValueError("completion-value plan is not frozen unexecuted")
    split = config["data"]["episode_split"]
    if split != {
        "kind": "within_manifest_entry_modulo",
        "modulus": 4,
        "validation_residue": 0,
    }:
        raise ValueError("completion-value split contract changed")
    if len(config["data"]["public_log_manifests"]) != 2:
        raise ValueError("exactly two mode-complete public-log manifests required")
    action_conditional = config["version"] in {
        "t3-public-action-conditional-completion-value-training-v4",
        "t3-public-endpoint-anchored-completion-value-training-v5",
    }
    if bool(config["data"].get("hold_decision_context_for_pair_endpoint", False)) != action_conditional:
        raise ValueError("action-conditional endpoint context contract mismatch")
    endpoint_anchored = (
        config["version"]
        == "t3-public-endpoint-anchored-completion-value-training-v5"
    )
    if bool(config["data"].get("anchor_conditional_endpoint", False)) != endpoint_anchored:
        raise ValueError("conditional endpoint anchoring contract mismatch")
    if endpoint_anchored and "maximum_conditional_endpoint_mae_controls" not in config["acceptance"]:
        raise ValueError("endpoint-anchored acceptance gate required")
    return config


def phase_model(config: dict[str, Any]) -> bool:
    return config["version"] in {
        "t3-public-phase-completion-value-training-v2",
        "t3-public-context-completion-value-training-v3",
        "t3-public-action-conditional-completion-value-training-v4",
        "t3-public-endpoint-anchored-completion-value-training-v5",
        "t3-public-qualified-context-completion-value-training-v6",
    }


def context_model(config: dict[str, Any]) -> bool:
    return config["version"] in {
        "t3-public-context-completion-value-training-v3",
        "t3-public-action-conditional-completion-value-training-v4",
        "t3-public-endpoint-anchored-completion-value-training-v5",
        "t3-public-qualified-context-completion-value-training-v6",
    }


def action_conditional_model(config: dict[str, Any]) -> bool:
    return config["version"] in {
        "t3-public-action-conditional-completion-value-training-v4",
        "t3-public-endpoint-anchored-completion-value-training-v5",
    }


def endpoint_anchored_model(config: dict[str, Any]) -> bool:
    return (
        config["version"]
        == "t3-public-endpoint-anchored-completion-value-training-v5"
    )


def qualified_context_model(config: dict[str, Any]) -> bool:
    return (
        config["version"]
        == "t3-public-qualified-context-completion-value-training-v6"
    )


def model_types(config: dict[str, Any]):
    if qualified_context_model(config):
        return (
            PublicContextCompletionValueSpec,
            PublicQualifiedContextCompletionValueEnsemble,
        )
    if endpoint_anchored_model(config):
        return (
            PublicContextCompletionValueSpec,
            PublicEndpointAnchoredCompletionValueEnsemble,
        )
    if action_conditional_model(config):
        return (
            PublicContextCompletionValueSpec,
            PublicActionConditionalCompletionValueEnsemble,
        )
    if context_model(config):
        return PublicContextCompletionValueSpec, PublicContextCompletionValueEnsemble
    if phase_model(config):
        return PublicPhaseCompletionValueSpec, PublicPhaseCompletionValueEnsemble
    return PublicCompletionValueSpec, PublicCompletionValueEnsemble


def public_controller_trace(
    public: dict[str, np.ndarray], mode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observer = PublicControllerMemory(task_spec("T3", robot="xarm7"), [mode])
    contexts = []
    phases = []
    contact_counts = []
    for tick in range(len(public["tcp_position"])):
        context = observer.update({
            key: np.asarray(value)[tick:tick + 1]
            for key, value in public.items()
        })
        contexts.append(context[0])
        phases.append(int(observer.memory.phase[0]))
        contact_counts.append(int(observer.memory.contact_count[0]))
    return (
        np.asarray(contexts, np.float32),
        np.asarray(phases, np.int64),
        np.asarray(contact_counts, np.int64),
    )


def verify_device(config: dict[str, Any]) -> torch.device:
    expected = config["device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected["process_visible_devices"]:
        raise RuntimeError("completion-value training requires CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one logical CUDA device required")
    device = torch.device(expected["logical_device"])
    properties = torch.cuda.get_device_properties(0)
    if str(properties.uuid).lower() != expected["physical_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("visible device is not admitted GPU1")
    return device


def public_features(
    public: dict[str, np.ndarray], ticks: np.ndarray, mode: int,
    *, phases: np.ndarray | None = None,
    contexts: np.ndarray | None = None,
    contact_counts: np.ndarray | None = None,
    controller_ticks: np.ndarray | None = None,
) -> np.ndarray:
    if any(key.startswith("fixture_") for key in public):
        raise ValueError("static T3 completion features do not accept fixture observations")
    controller_ticks = ticks if controller_ticks is None else controller_ticks
    if controller_ticks.shape != ticks.shape:
        raise ValueError("physical and controller feature ticks must align")
    arguments = dict(
        position=torch.from_numpy(np.asarray(public["tcp_position"])[ticks]),
        rotation=torch.from_numpy(np.asarray(public["tcp_rotation"])[ticks]),
        twist=torch.from_numpy(np.asarray(public["tcp_twist"])[ticks]),
        wrench=torch.from_numpy(np.asarray(public["measured_wrench"])[ticks]),
        goal_position=torch.from_numpy(np.asarray(public["goal_position"])[ticks]),
        goal_rotation=torch.from_numpy(np.asarray(public["goal_rotation"])[ticks]),
        controller_feedback=torch.from_numpy(
            np.asarray(public["controller_feedback"])[ticks]
        ),
        mode=torch.full((len(ticks),), mode, dtype=torch.long),
    )
    if contexts is not None:
        if phases is None or contact_counts is None:
            raise ValueError("context completion features require full controller trace")
        values = context_completion_value_features(
            **arguments,
            controller_context=torch.from_numpy(contexts[controller_ticks]),
            contact_count=torch.from_numpy(contact_counts[controller_ticks]),
        )
    elif phases is not None:
        if contact_counts is not None:
            raise ValueError("contact counts are only admitted by the context model")
        values = phase_completion_value_features(
            **arguments, phase=torch.from_numpy(phases[ticks]),
        )
    else:
        if contact_counts is not None:
            raise ValueError("contact counts are only admitted by the context model")
        values = completion_value_features(**arguments)
    return values.numpy().astype(np.float32, copy=False)


def concatenate(rows: list[np.ndarray], *, width: int | None = None) -> np.ndarray:
    if not rows:
        shape = (0,) if width is None else (0, width)
        return np.empty(shape, np.float32)
    return np.concatenate(rows)


def load_dataset(config: dict[str, Any]) -> tuple[dict, dict]:
    maximum = int(config["model"]["maximum_remaining_controls"])
    horizon = int(config["data"]["temporal_horizon_controls"])
    stride = int(config["data"]["state_stride"])
    split_spec = config["data"]["episode_split"]
    spec_type, _ = model_types(config)
    feature_dim = spec_type(
        **{key: value for key, value in config["model"].items()}
    ).feature_dim
    groups = {
        name: {
            "features": [], "targets": [], "modes": [],
            "pair_current": [], "pair_future": [], "pair_targets": [],
            "pair_future_targets": [], "pair_modes": [],
            **({"phases": [], "pair_phases": []} if phase_model(config) else {}),
        }
        for name in ("fit", "validation")
    }
    provenance = {"manifests": [], "episodes": {}, "counts": {}}
    observed_modes: set[int] = set()
    for manifest_index, value in enumerate(config["data"]["public_log_manifests"]):
        manifest_path = checked(value, f"public-log manifest {manifest_index}")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["permission"] != "B_log"
            or manifest["reward"]
            or manifest["success_filter"]
        ):
            raise ValueError("unfiltered public B_log required")
        controls = 0
        for entry_index, entry in enumerate(manifest["entries"]):
            trajectory = PublicTargetTrajectory.load(entry["path"])
            if (trajectory.metadata["version"] not in {VERSION, FOUR_TASK_VERSION}
                    or trajectory.metadata.get("environment", "keyed_insertion_recovery") != "keyed_insertion_recovery"):
                raise ValueError("static T3 public trajectory required; fixture data needs its own training entry")
            if (
                trajectory.digest != entry["parent_sha256"]
                or trajectory.digest != entry["prefix_sha256"]
                or len(trajectory.requests) != int(entry["controls"])
                or trajectory.task != "T3"
            ):
                raise ValueError("public completion trajectory changed")
            controls += len(trajectory.requests)
            mode = int(trajectory.metadata["mode"])
            observed_modes.add(mode)
            if phase_model(config):
                contexts, phases, contact_counts = public_controller_trace(
                    trajectory.public, mode,
                )
                if not context_model(config):
                    contexts = contact_counts = None
            else:
                contexts = phases = contact_counts = None
            validation = (
                entry_index % int(split_spec["modulus"])
                == int(split_spec["validation_residue"])
            )
            split = "validation" if validation else "fit"
            end = len(trajectory.requests)
            first = max(0, end - maximum)
            ticks = np.arange(first, end + 1, stride, dtype=np.int64)
            if ticks[-1] != end:
                ticks = np.concatenate((ticks, np.asarray([end], np.int64)))
            features = public_features(
                trajectory.public, ticks, mode, phases=phases,
                contexts=contexts, contact_counts=contact_counts,
            )
            targets = ((end - ticks) / maximum).astype(np.float32)
            pair_ticks = np.arange(first, end - horizon + 1, stride, dtype=np.int64)
            pair_future = pair_ticks + horizon
            group = groups[split]
            group["features"].append(features)
            group["targets"].append(targets)
            group["modes"].append(np.full(len(ticks), mode, np.int64))
            group["pair_current"].append(
                public_features(
                    trajectory.public, pair_ticks, mode, phases=phases,
                    contexts=contexts, contact_counts=contact_counts,
                )
            )
            group["pair_future"].append(
                public_features(
                    trajectory.public, pair_future, mode, phases=phases,
                    contexts=contexts, contact_counts=contact_counts,
                    controller_ticks=(
                        pair_ticks if action_conditional_model(config) else None
                    ),
                )
            )
            group["pair_targets"].append(
                np.full(len(pair_ticks), horizon / maximum, np.float32)
            )
            group["pair_future_targets"].append(
                ((end - pair_future) / maximum).astype(np.float32)
            )
            group["pair_modes"].append(
                np.full(len(pair_ticks), mode, np.int64)
            )
            if phases is not None:
                group["phases"].append(phases[ticks])
                group["pair_phases"].append(phases[pair_ticks])
            key = f"manifest_{manifest_index:02d}/entry_{entry_index:04d}"
            provenance["episodes"][key] = {
                "digest": trajectory.digest,
                "seed": int(trajectory.metadata["seed"]),
                "mode": mode,
                "controls": end,
                "split": split,
                "states": len(ticks),
                "temporal_pairs": len(pair_ticks),
            }
        if controls != int(manifest["controls"]):
            raise ValueError("public-log control denominator changed")
        provenance["manifests"].append(pointer(manifest_path))
    if observed_modes != {0, 1}:
        raise ValueError("mode-complete public completion data required")
    data = {}
    for split, rows in groups.items():
        data[split] = {
            "features": concatenate(rows["features"], width=feature_dim),
            "targets": concatenate(rows["targets"]),
            "modes": concatenate(rows["modes"]).astype(np.int64),
            "pair_current": concatenate(rows["pair_current"], width=feature_dim),
            "pair_future": concatenate(rows["pair_future"], width=feature_dim),
            "pair_targets": concatenate(rows["pair_targets"]),
            "pair_future_targets": concatenate(rows["pair_future_targets"]),
            "pair_modes": concatenate(rows["pair_modes"]).astype(np.int64),
            **(
                {
                    "phases": concatenate(rows["phases"]).astype(np.int64),
                    "pair_phases": concatenate(rows["pair_phases"]).astype(np.int64),
                }
                if phase_model(config) else {}
            ),
        }
        provenance["counts"][split] = {
            "states": len(data[split]["features"]),
            "temporal_pairs": len(data[split]["pair_current"]),
            "states_by_mode": {
                str(mode): int((data[split]["modes"] == mode).sum())
                for mode in (0, 1)
            },
            "pairs_by_mode": {
                str(mode): int((data[split]["pair_modes"] == mode).sum())
                for mode in (0, 1)
            },
            **(
                {
                    "states_by_phase": {
                        str(phase): int((data[split]["phases"] == phase).sum())
                        for phase in range(4)
                    },
                    "pairs_by_phase": {
                        str(phase): int((data[split]["pair_phases"] == phase).sum())
                        for phase in range(4)
                    },
                }
                if phase_model(config) else {}
            ),
        }
    return data, provenance


def to_device(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in data.items()
    }


def balanced_indices(
    modes: torch.Tensor, count: int, generator: torch.Generator,
) -> torch.Tensor:
    if count % 2:
        raise ValueError("mode-balanced batch size must be even")
    rows = []
    for mode in (0, 1):
        available = torch.nonzero(modes == mode, as_tuple=False).reshape(-1)
        if len(available) == 0:
            raise ValueError("mode-balanced batch found an empty mode")
        selected = torch.randint(
            len(available), (count // 2,), generator=generator,
            device=modes.device,
        )
        rows.append(available[selected])
    result = torch.cat(rows)
    return result[torch.randperm(len(result), generator=generator, device=modes.device)]


def training_indices(
    data: dict[str, torch.Tensor], count: int, generator: torch.Generator,
    *, pairs: bool,
) -> torch.Tensor:
    modes = data["pair_modes" if pairs else "modes"]
    phase_key = "pair_phases" if pairs else "phases"
    if phase_key not in data:
        return balanced_indices(modes, count, generator)
    if count % 8:
        raise ValueError("phase-and-mode-balanced batch size must divide by eight")
    phases = data[phase_key]
    rows = []
    for mode in (0, 1):
        for phase in range(4):
            available = torch.nonzero(
                (modes == mode) & (phases == phase), as_tuple=False,
            ).reshape(-1)
            if len(available) == 0:
                raise ValueError("phase-and-mode-balanced batch found an empty cell")
            selected = torch.randint(
                len(available), (count // 8,), generator=generator,
                device=modes.device,
            )
            rows.append(available[selected])
    result = torch.cat(rows)
    return result[
        torch.randperm(len(result), generator=generator, device=modes.device)
    ]


def validation_metrics(
    model: PublicCompletionValueEnsemble,
    data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    maximum = model.spec.maximum_remaining_controls
    model.eval()
    with torch.no_grad():
        state_members = model(data["features"])
        pair_current = model(data["pair_current"])
        pair_future = model(data["pair_future"])
    state_mean = state_members.mean(0)
    pair_members = pair_current - pair_future
    pair_mean = pair_members.mean(0)
    grouped = "phases" in data
    result = {"by_cell" if grouped else "by_mode": {}}
    groups = (
        [(f"mode_{mode}_phase_{phase}", mode, phase)
         for mode in (0, 1) for phase in range(4)]
        if grouped else [(str(mode), mode, None) for mode in (0, 1)]
    )
    rows = result["by_cell" if grouped else "by_mode"]
    for key, mode, phase in groups:
        states = data["modes"] == mode
        pairs = data["pair_modes"] == mode
        if phase is not None:
            states = states & (data["phases"] == phase)
            pairs = pairs & (data["pair_phases"] == phase)
        if not bool(states.any()) or not bool(pairs.any()):
            raise ValueError(f"completion-value validation cell {key} is empty")
        state_error = (state_mean[states] - data["targets"][states]) * maximum
        pair_truth = data["pair_targets"][pairs] * maximum
        pair_prediction = pair_mean[pairs] * maximum
        pair_residual = pair_prediction - pair_truth
        endpoint_error = (
            pair_future.mean(0)[pairs] - data["pair_future_targets"][pairs]
        ) * maximum
        member_gain = pair_members[:, pairs] * maximum
        current = {
            "states": int(states.sum()),
            "temporal_pairs": int(pairs.sum()),
            "state_mae_controls": float(state_error.abs().mean()),
            "state_rmse_controls": float(state_error.square().mean().sqrt()),
            "pair_mae_controls": float(pair_residual.abs().mean()),
            "pair_rmse_controls": float(pair_residual.square().mean().sqrt()),
            "conditional_endpoint_mae_controls": float(
                endpoint_error.abs().mean()
            ),
            "pair_direction_accuracy": float((pair_prediction > 0).float().mean()),
            "all_members_positive_fraction": float(
                (member_gain.amin(0) > 0).float().mean()
            ),
            "pair_residual_abs_p95_controls": float(
                torch.quantile(pair_residual.abs(), .95)
            ),
            "conservative_optimism_p95_controls": float(
                torch.quantile(
                    (member_gain.amin(0) - pair_truth).clamp(min=0), .95,
                )
            ),
        }
        rows[key] = current
    result["score"] = max(
        row["pair_mae_controls"] + .1 * row["state_mae_controls"]
        for row in rows.values()
    )
    return result


def acceptance(config: dict, metrics: dict) -> dict:
    thresholds = config["acceptance"]
    checks = {}
    key = "by_cell" if "by_cell" in metrics else "by_mode"
    for group, row in metrics[key].items():
        prefix = group if key == "by_cell" else f"mode_{group}"
        checks[f"{prefix}_state_mae"] = (
            row["state_mae_controls"] <= thresholds["maximum_state_mae_controls"]
        )
        checks[f"{prefix}_pair_mae"] = (
            row["pair_mae_controls"] <= thresholds["maximum_pair_mae_controls"]
        )
        checks[f"{prefix}_direction"] = (
            row["pair_direction_accuracy"]
            >= thresholds["minimum_pair_direction_accuracy"]
        )
        checks[f"{prefix}_ensemble_direction"] = (
            row["all_members_positive_fraction"]
            >= thresholds["minimum_all_members_positive_fraction"]
        )
        if "maximum_conditional_endpoint_mae_controls" in thresholds:
            checks[f"{prefix}_conditional_endpoint_mae"] = (
                row["conditional_endpoint_mae_controls"]
                <= thresholds["maximum_conditional_endpoint_mae_controls"]
            )
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def selection_score(config: dict, metrics: dict) -> float:
    """Worst normalized frozen-gate ratio; below one passes every gate."""
    thresholds = config["acceptance"]
    key = "by_cell" if "by_cell" in metrics else "by_mode"
    values = []
    for row in metrics[key].values():
        values.extend((
            row["state_mae_controls"] / thresholds["maximum_state_mae_controls"],
            row["pair_mae_controls"] / thresholds["maximum_pair_mae_controls"],
            thresholds["minimum_pair_direction_accuracy"]
            / max(row["pair_direction_accuracy"], 1e-12),
            thresholds["minimum_all_members_positive_fraction"]
            / max(row["all_members_positive_fraction"], 1e-12),
        ))
        if "maximum_conditional_endpoint_mae_controls" in thresholds:
            values.append(
                row["conditional_endpoint_mae_controls"]
                / thresholds["maximum_conditional_endpoint_mae_controls"]
            )
    return float(max(values))


def run_preflight(config: dict, config_path: Path) -> None:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU completion-value preflight must not initialize CUDA")
    data, provenance = load_dataset(config)
    spec_type, model_type = model_types(config)
    spec = spec_type(**config["model"])
    torch.manual_seed(int(config["training"]["seed"]))
    model = model_type(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    fit = to_device(data["fit"], torch.device("cpu"))
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    state_ids = training_indices(fit, 32, generator, pairs=False)
    pair_ids = training_indices(fit, 32, generator, pairs=True)
    state_prediction = model(fit["features"][state_ids])
    pair_prediction = (
        model(fit["pair_current"][pair_ids])
        - model(fit["pair_future"][pair_ids])
    )
    loss = (
        torch.nn.functional.smooth_l1_loss(
            state_prediction, fit["targets"][state_ids].expand_as(state_prediction)
        )
        + torch.nn.functional.smooth_l1_loss(
            pair_prediction,
            fit["pair_targets"][pair_ids].expand_as(pair_prediction),
        )
    )
    if endpoint_anchored_model(config):
        loss = loss + torch.nn.functional.smooth_l1_loss(
            model(fit["pair_future"][pair_ids]),
            fit["pair_future_targets"][pair_ids].expand_as(pair_prediction),
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    output = ROOT / config["preflight_output"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "t3-public-completion-value-preflight-v1",
        "status": "complete" if finite else "failed",
        "passed": finite,
        "gpu_calls": 0,
        "loss": float(loss),
        "data": provenance,
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
        },
        "target_reward_or_truth_opened": False,
        "confirmation_states_opened": 0,
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "status": report["status"], "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not finite:
        raise AssertionError("completion-value preflight failed")


def run_training(config: dict, config_path: Path) -> None:
    device = verify_device(config)
    preflight_path = checked(config["preflight_report"], "completion-value preflight")
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("gpu_calls") != 0:
        raise ValueError("passed CPU completion-value preflight required")
    data, provenance = load_dataset(config)
    fit = to_device(data["fit"], device)
    validation = to_device(data["validation"], device)
    spec_type, model_type = model_types(config)
    spec = spec_type(**config["model"])
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = model_type(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    pair_batch_size = int(config["training"]["pair_batch_size"])
    updates = int(config["training"]["updates"])
    interval = int(config["training"]["validation_interval"])
    temporal_weight = float(config["training"]["temporal_weight"])
    endpoint_anchor_weight = float(
        config["training"].get("endpoint_anchor_weight", 0.)
    )
    if endpoint_anchored_model(config) != (endpoint_anchor_weight > 0.):
        raise ValueError("endpoint anchor training weight contract mismatch")
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", config)
    curves = []
    best_state = None
    best_metrics = None
    best_update = None
    best_score = None
    begun = time.monotonic()
    for update in range(1, updates + 1):
        model.train()
        state_ids = training_indices(fit, batch_size, generator, pairs=False)
        pair_ids = training_indices(fit, pair_batch_size, generator, pairs=True)
        state_truth = fit["targets"][state_ids]
        pair_truth = fit["pair_targets"][pair_ids]
        state_prediction = model(fit["features"][state_ids])
        pair_prediction = (
            model(fit["pair_current"][pair_ids])
            - model(fit["pair_future"][pair_ids])
        )
        absolute_loss = torch.nn.functional.smooth_l1_loss(
            state_prediction, state_truth.expand_as(state_prediction),
        )
        temporal_loss = torch.nn.functional.smooth_l1_loss(
            pair_prediction, pair_truth.expand_as(pair_prediction),
        )
        monotone_loss = (
            pair_truth.expand_as(pair_prediction) - pair_prediction
        ).clamp(min=0).mean()
        endpoint_anchor_loss = torch.nn.functional.smooth_l1_loss(
            model(fit["pair_future"][pair_ids]),
            fit["pair_future_targets"][pair_ids].expand_as(pair_prediction),
        )
        loss = (
            absolute_loss + temporal_weight * (temporal_loss + monotone_loss)
            + endpoint_anchor_weight * endpoint_anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update % interval == 0:
            metrics = validation_metrics(model, validation)
            score = (
                selection_score(config, metrics)
                if action_conditional_model(config) or qualified_context_model(config)
                else float(metrics["score"])
            )
            curve = {
                "update": update, "loss": float(loss.detach().cpu()),
                "absolute_loss": float(absolute_loss.detach().cpu()),
                "temporal_loss": float(temporal_loss.detach().cpu()),
                "monotone_loss": float(monotone_loss.detach().cpu()),
                "endpoint_anchor_loss": float(endpoint_anchor_loss.detach().cpu()),
                "validation": metrics, "selection_score": score,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
            if best_score is None or score < best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                best_metrics = metrics
                best_update = update
    if best_state is None or best_metrics is None:
        raise RuntimeError("completion-value selection produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    decision = acceptance(config, best_metrics)
    calibration = {
        "version": "public-T3-completion-value-temporal-calibration-v1",
        "validation_metrics": best_metrics,
        "minimum_admission_gain_controls": max(
            row["conservative_optimism_p95_controls"]
            for row in best_metrics[
                "by_cell" if "by_cell" in best_metrics else "by_mode"
            ].values()
        ),
        "episode_split": config["data"]["episode_split"],
    }
    checkpoint_path = output / "checkpoint.pt"
    save_torch(checkpoint_path, model.checkpoint(calibration=calibration))
    result = {
        "schema_version": config["version"],
        "status": "complete" if decision["passed"] else "qualification_failed",
        "passed": decision["passed"],
        "selection": {
            "update": best_update, "score": best_score,
            "metrics": best_metrics, "decision": decision,
        },
        "calibration": calibration,
        "data": provenance,
        "device": {
            **config["device"],
            "observed_uuid": str(torch.cuda.get_device_properties(0).uuid),
        },
        "artifacts": {
            "config": pointer(config_path),
            "runner": pointer(Path(__file__).resolve()),
            "preflight": pointer(preflight_path),
            "checkpoint": pointer(checkpoint_path),
        },
        "curve": curves,
        "new_physics_calls": 0,
        "target_reward_or_truth_opened": False,
        "confirmation_states_opened": 0,
        "wall_seconds": time.monotonic() - begun,
    }
    write_json(output / "report.json", result)
    print(json.dumps({
        "status": result["status"], "passed": result["passed"],
        "output": str(output / "report.json"),
        "sha256": sha(output / "report.json"),
    }, indent=2))
    if not result["passed"]:
        raise AssertionError("completion-value validation gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    if args.validate_only:
        run_preflight(config, config_path)
    else:
        run_training(config, config_path)


if __name__ == "__main__":
    main()
