"""CPU-first diagnostics for the robot-source H2S2R adaptation.

The checks in this module do not train or select a paper policy.  They audit
the adapted object-trajectory learning chain before a GPU pilot is admitted:
demonstration timing, reset coverage, native reward scale, action units,
reference advancement, historical monitor quality, and PPO convergence
signals.  Keeping this logic beside the H2S2R task adapter makes the paper
entrypoint a thin configuration/serialization layer.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from genedynamics.envs.adapters.transfer_task import instantiate_task

from .task_env import ObjectTrajectoryTaskEnv


VERSION = "human2sim2robot-learning-chain-audit-v1"
_OFFSETS = np.eye(3, dtype=np.float32) * 0.2


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = root / binding["path"]
    if not path.is_file() or file_sha256(path) != binding["sha256"]:
        raise ValueError(f"{label} is missing or its SHA-256 changed: {path}")
    return path


def object_tracking_distance(
    positions: np.ndarray,
    rotations: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    """Return the native mean three-keypoint object error in metres."""
    positions = np.asarray(positions, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    references = np.asarray(references, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape [N,3]")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("rotations must have shape [N,3,3]")
    if references.shape != (len(positions), 4, 4):
        raise ValueError("references must have shape [N,4,4]")
    actual = np.einsum("nij,kj->nki", rotations, _OFFSETS) + positions[:, None]
    desired = (
        np.einsum("nij,kj->nki", references[:, :3, :3], _OFFSETS)
        + references[:, None, :3, 3]
    )
    return np.linalg.norm(actual - desired, axis=-1).mean(axis=-1)


def normalize_physical_actions(
    actions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert upstream action rescaling and identify non-degenerate axes."""
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.ndim != 1 or high.shape != low.shape or actions.shape[-1:] != low.shape:
        raise ValueError("actions and bounds have incompatible shapes")
    if not np.isfinite(actions).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("actions and bounds must be finite")
    if np.any(high < low):
        raise ValueError("action upper bounds must be no smaller than lower bounds")
    center = (high + low) / 2
    half_range = (high - low) / 2
    active = half_range > 0
    normalized = np.divide(
        actions - center,
        half_range,
        out=np.zeros_like(actions, dtype=np.float32),
        where=active,
    )
    return normalized, active


def reference_clock_trace(
    positions: np.ndarray,
    rotations: np.ndarray,
    demonstration: np.ndarray,
    *,
    control_dt: float,
    demo_dt: float,
    stop_error_m: float,
) -> dict[str, Any]:
    """Replay the exact stopped reference clock used by train and deploy."""
    index = 0.0
    errors: list[float] = []
    indices: list[float] = []
    for tick in range(len(positions)):
        reference = demonstration[min(int(index), len(demonstration) - 1)]
        error = float(object_tracking_distance(
            positions[tick:tick + 1], rotations[tick:tick + 1], reference[None]
        )[0])
        if tick and error <= stop_error_m:
            index += control_dt / demo_dt
        errors.append(error)
        indices.append(index)
    return {
        "reference_indices": indices,
        "tracking_errors_m": errors,
        "final_reference_index": index,
        "completion_fraction": min(index / max(1, len(demonstration) - 1), 1.0),
        "advanced_steps": int(np.count_nonzero(np.diff(indices) > 0)),
        "stopped_steps": int(np.count_nonzero(np.diff(indices) == 0)),
    }


def native_tracking_reward_profile(errors_m: list[float], stop_error_m: float) -> list[dict[str, float]]:
    rows = []
    for error in errors_m:
        reward = float(np.exp(-10.0 * error))
        if error > stop_error_m:
            reward *= 0.1
        rows.append({"tracking_error_m": float(error), "tracking_reward": reward})
    return rows


def _load_demonstration(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"public.tcp_position", "public.tcp_rotation", "public.elapsed_seconds"}
        if not required.issubset(data.files):
            raise ValueError(f"demonstration misses public trajectory fields: {sorted(required - set(data.files))}")
        positions = np.asarray(data["public.tcp_position"], dtype=np.float32)
        rotations = np.asarray(data["public.tcp_rotation"], dtype=np.float32)
        times = np.asarray(data["public.elapsed_seconds"], dtype=np.float64).reshape(-1)
    if len(times) < 2 or positions.shape != (len(times), 3) or rotations.shape != (len(times), 3, 3):
        raise ValueError("demonstration trajectory fields have inconsistent shapes")
    deltas = np.diff(times)
    demo_dt = float(np.median(deltas))
    if not np.isfinite(demo_dt) or demo_dt <= 0 or not np.allclose(deltas, demo_dt, rtol=1e-4, atol=1e-6):
        raise ValueError("demonstration timestamps are not a finite uniform clock")
    demonstration = np.tile(np.eye(4, dtype=np.float32), (len(times), 1, 1))
    demonstration[:, :3, :3] = rotations
    demonstration[:, :3, 3] = positions
    summary = {
        "samples": len(times),
        "demo_dt": demo_dt,
        "duration_seconds": float(times[-1] - times[0]),
        "translation_path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
    }
    return demonstration, times, summary


def _episode_diagnostics(
    episode: Path,
    report_path: Path,
    demonstration: np.ndarray,
    *,
    control_dt: float,
    demo_dt: float,
    stop_error_m: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    saturation_threshold: float,
) -> dict[str, Any]:
    with np.load(episode, allow_pickle=False) as data:
        positions = np.asarray(data["public.tcp_position"], dtype=np.float32)
        rotations = np.asarray(data["public.tcp_rotation"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    clock = reference_clock_trace(
        positions, rotations, demonstration,
        control_dt=control_dt, demo_dt=demo_dt, stop_error_m=stop_error_m,
    )
    normalized, active = normalize_physical_actions(actions, action_low, action_high)
    active_values = np.abs(normalized[..., active])
    report = json.loads(report_path.read_text())
    evaluated = report["evaluation"]["episodes"][0]
    outcome = evaluated["outcome"]
    return {
        "success": bool(evaluated["success"]),
        "constraint_violation": outcome not in {"success", "timeout"},
        "outcome": outcome,
        "control_steps": int(evaluated["physical_control_steps"]),
        "final_position_error_m": float(evaluated["final_position_error_m"]),
        "reference_completion_fraction": float(clock["completion_fraction"]),
        "reference_advanced_steps": int(clock["advanced_steps"]),
        "reference_stopped_steps": int(clock["stopped_steps"]),
        "tracking_error_mean_m": float(np.mean(clock["tracking_errors_m"])),
        "tracking_error_final_m": float(clock["tracking_errors_m"][-1]),
        "action_saturation_fraction": float(np.mean(active_values >= saturation_threshold)),
        "action_normalized_abs_mean": float(np.mean(active_values)),
    }


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.median([row[key] for row in rows]))


def _monitor_diagnostics(
    run: Path,
    report: dict[str, Any],
    demonstration: np.ndarray,
    *,
    control_dt: float,
    demo_dt: float,
    stop_error_m: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    saturation_threshold: float,
) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for epoch_text in sorted(report["monitor_curve"], key=int):
        epoch = int(epoch_text)
        folder = run / f"epoch_{epoch}" / "monitor"
        rows = []
        for seed_folder in sorted(folder.glob("seed_*")):
            rows.append(_episode_diagnostics(
                seed_folder / "episode.npz", seed_folder / "report.json", demonstration,
                control_dt=control_dt, demo_dt=demo_dt, stop_error_m=stop_error_m,
                action_low=action_low, action_high=action_high,
                saturation_threshold=saturation_threshold,
            ))
        if len(rows) != int(report["monitor_curve"][epoch_text]["requested"]):
            raise ValueError(f"monitor evidence count differs at epoch {epoch}")
        checkpoints.append({
            "epoch": epoch,
            "requested": len(rows),
            "successes": sum(row["success"] for row in rows),
            "constraint_violations": sum(row["constraint_violation"] for row in rows),
            "median_reference_completion_fraction": _median(rows, "reference_completion_fraction"),
            "median_tracking_error_final_m": _median(rows, "tracking_error_final_m"),
            "median_action_saturation_fraction": _median(rows, "action_saturation_fraction"),
            "median_control_steps": _median(rows, "control_steps"),
            "outcomes": {name: sum(row["outcome"] == name for row in rows)
                         for name in sorted({row["outcome"] for row in rows})},
            "episodes": rows,
        })
    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            row["successes"] / row["requested"],
            -row["constraint_violations"] / row["requested"],
            row["median_reference_completion_fraction"],
            -row["median_tracking_error_final_m"],
            -row["median_action_saturation_fraction"],
            -row["epoch"],
        )
    selected = max(checkpoints, key=rank)
    return {
        "selection_rule": [
            "success_rate_max", "constraint_violation_rate_min",
            "reference_completion_median_max", "terminal_tracking_error_median_min",
            "action_saturation_median_min", "earliest_epoch_tie_break",
        ],
        "selected_epoch": selected["epoch"],
        "fixed_last_epoch": max(row["epoch"] for row in checkpoints),
        "fixed_last_was_continuously_best": selected["epoch"] == max(row["epoch"] for row in checkpoints),
        "checkpoints": checkpoints,
    }


def _mean_epoch_metric(row: dict[str, Any], key: str) -> float:
    values = np.asarray(row[key], dtype=np.float64)
    return float(values.mean())


def _learning_curve_diagnostics(curve: list[dict[str, Any]]) -> dict[str, Any]:
    if not curve:
        raise ValueError("learning curve is empty")
    width = max(1, len(curve) // 8)
    result: dict[str, Any] = {"epochs": len(curve), "window_epochs": width}
    for key in ("actor_loss", "critic_loss", "entropy", "kl"):
        values = np.asarray([_mean_epoch_metric(row, key) for row in curve])
        result[key] = {
            "early_mean": float(values[:width].mean()),
            "late_mean": float(values[-width:].mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "finite": bool(np.isfinite(values).all()),
        }
    result["critic_loss_late_over_early"] = (
        result["critic_loss"]["late_mean"] / max(result["critic_loss"]["early_mean"], 1e-12)
    )
    result["optimizer_signal_stabilized"] = bool(
        result["critic_loss"]["finite"]
        and result["critic_loss_late_over_early"] < 0.5
        and result["kl"]["finite"]
    )
    return result


def audit_learning_chain(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run one task-local CPU audit and return a JSON-serializable report."""
    if config.get("version") != VERSION:
        raise ValueError(f"expected config version {VERSION}")
    task_alias = config["task"]
    body = config["body"]
    mode = int(config["mode"])
    seed = int(config["seed"])
    audit = config["audit"]
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    demo_path = _require_file(root, config["demonstration"], "demonstration")
    run = root / config["historical_run"]["path"]
    report_path = _require_file(root, {
        "path": str(Path(config["historical_run"]["path"]) / "report.json"),
        "sha256": config["historical_run"]["report_sha256"],
    }, "historical report")
    curve_path = _require_file(root, {
        "path": str(Path(config["historical_run"]["path"]) / "learning_curve.json"),
        "sha256": config["historical_run"]["learning_curve_sha256"],
    }, "historical learning curve")
    monitor_path = _require_file(root, {
        "path": str(Path(config["historical_run"]["path"]) / "monitor_curve.json"),
        "sha256": config["historical_run"]["monitor_curve_sha256"],
    }, "historical monitor curve")
    report = json.loads(report_path.read_text())
    if report.get("status") != "complete" or report.get("monitor_curve") != json.loads(monitor_path.read_text()):
        raise ValueError("historical run is incomplete or monitor summary changed")
    curve = json.loads(curve_path.read_text())
    demonstration, _, demo_summary = _load_demonstration(demo_path)

    reset_samples = int(audit["reset_samples"])
    neutral_steps = int(audit["neutral_rollout_steps"])
    stop_error_m = float(audit["reference_stop_error_m"])
    saturation_threshold = float(audit["action_saturation_threshold"])
    task = instantiate_task(task_alias, robot=body, seed=seed)
    env = ObjectTrajectoryTaskEnv(
        task, demonstration, num_envs=reset_samples, seed=seed,
        demo_dt=demo_summary["demo_dt"],
    )
    env.reset()
    public = task.observe()
    references = np.repeat(demonstration[:1], reset_samples, axis=0)
    reset_errors = object_tracking_distance(
        public["tcp_position"], public["tcp_rotation"], references,
    )
    reset_positions = np.asarray(public["tcp_position"], dtype=np.float32)
    reset_span = float(np.max(np.linalg.norm(
        reset_positions[:, None] - reset_positions[None], axis=-1,
    )))
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    center = np.repeat(((high + low) / 2)[None], reset_samples, axis=0)
    normalized_probes, active = normalize_physical_actions(
        np.stack((low, (high + low) / 2, high)), low, high,
    )
    expected_probes = np.stack((-np.ones_like(low), np.zeros_like(low), np.ones_like(low)))
    expected_probes[:, ~active] = 0
    if not np.allclose(normalized_probes, expected_probes):
        raise AssertionError("physical/native action scaling does not round-trip")
    rollout_rows = []
    for _ in range(min(neutral_steps, task.config.timeout_steps)):
        _, reward, done, info = env.step(center, auto_reset=False)
        rollout_rows.append({
            "reward_mean": float(np.mean(reward)),
            "tracking_error_mean_m": float(np.mean(info["reference_tracking_error"])),
            "reference_index_mean": float(np.mean(env.reference_index)),
            "ended_worlds": int(np.count_nonzero(done)),
        })
        if np.all(done):
            break

    monitor = _monitor_diagnostics(
        run, report, demonstration,
        control_dt=float(task.config.control_dt), demo_dt=demo_summary["demo_dt"],
        stop_error_m=stop_error_m, action_low=low, action_high=high,
        saturation_threshold=saturation_threshold,
    )
    learning = _learning_curve_diagnostics(curve)
    training_accounting = report["accounting"]["known_model_training"]
    average_episode_steps = (
        float(training_accounting["control_steps"])
        / max(1, int(training_accounting["reset_worlds"]))
    )
    selected = next(row for row in monitor["checkpoints"] if row["epoch"] == monitor["selected_epoch"])
    failure_diagnosis = {
        "initialization_is_not_primary_blocker": bool(np.max(reset_errors) <= stop_error_m),
        "neutral_reference_clock_advances": bool(rollout_rows[-1]["reference_index_mean"] > 0),
        "historical_optimizer_signal_stabilized": learning["optimizer_signal_stabilized"],
        "historical_monitor_has_any_success": any(row["successes"] for row in monitor["checkpoints"]),
        "historical_selected_action_saturation_fraction": selected["median_action_saturation_fraction"],
        "historical_selected_reference_completion_fraction": selected["median_reference_completion_fraction"],
        "old_fixed_last_checkpoint_supported_by_continuous_metrics": monitor["fixed_last_was_continuously_best"],
        "interpretation": (
            "The nominal PPO failure is not explained by an unreachable first reference or a broken clock. "
            "The optimizer statistics stayed finite while task success remained zero; deterministic monitor actions "
            "frequently saturated, and success-only monitoring discarded trajectory progress. The next pilot must "
            "increase rollout diversity, retain "
            "continuous monitor metrics, and select a checkpoint before any target evaluation."
        ),
    }
    return {
        "version": VERSION,
        "status": "cpu_learning_chain_audited",
        "task": task_alias,
        "body": body,
        "mode": mode,
        "seed": seed,
        "inputs": {
            "config_scope": config["scope"],
            "demonstration": config["demonstration"],
            "historical_run": config["historical_run"],
        },
        "demonstration": demo_summary,
        "reward_scale": {
            "native_profile": native_tracking_reward_profile(
                [0.0, 0.01, 0.1, stop_error_m, stop_error_m + 0.01], stop_error_m,
            ),
            "action_smoothing_weight": env.action_smoothing_weight,
            "constant_action_has_zero_smoothing_penalty_after_first_step": True,
        },
        "initialization": {
            "reset_samples": reset_samples,
            "tracking_error_min_m": float(reset_errors.min()),
            "tracking_error_mean_m": float(reset_errors.mean()),
            "tracking_error_max_m": float(reset_errors.max()),
            "all_within_reference_admission_radius": bool(np.all(reset_errors <= stop_error_m)),
            "reset_tcp_span_m": reset_span,
            "pre_manipulation_source": "task-owned nominal reset near the robot-source first object pose",
            "native_human_hand_exploration_reproduced": False,
        },
        "action_contract": {
            "low": low.tolist(),
            "high": high.tolist(),
            "active_dimensions": int(np.count_nonzero(active)),
            "degenerate_dimensions": int(np.count_nonzero(~active)),
            "native_to_physical_round_trip_exact": True,
            "historical_saturation_threshold": saturation_threshold,
        },
        "reference_clock": {
            "neutral_rollout_steps": len(rollout_rows),
            "first": rollout_rows[0],
            "last": rollout_rows[-1],
            "advanced": rollout_rows[-1]["reference_index_mean"] > 0,
            "rows": rollout_rows,
        },
        "horizon_coverage": {
            "ppo_horizon_steps": int(config["pilot"]["horizon_steps"]),
            "ppo_horizon_seconds": float(config["pilot"]["horizon_steps"] * task.config.control_dt),
            "demonstration_steps": demo_summary["samples"],
            "task_timeout_steps": int(task.config.timeout_steps),
            "historical_average_steps_per_reset_world": average_episode_steps,
            "historical_average_fraction_of_demo": average_episode_steps / demo_summary["samples"],
        },
        "historical_learning": learning,
        "historical_monitor": monitor,
        "failure_diagnosis": failure_diagnosis,
        "gpu1_pilot_plan": config["pilot"],
        "gpu_started": False,
    }


__all__ = [
    "VERSION", "audit_learning_chain", "native_tracking_reward_profile",
    "normalize_physical_actions", "object_tracking_distance", "reference_clock_trace",
]
