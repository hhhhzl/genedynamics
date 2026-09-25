"""Fit/calibrate the CPU MGA reliability gate from existing metrics records.

Example:
  python scripts/tasks/robot/arm/train_mga_reliability.py \
    --metrics path/to/rigid/metrics.json path/to/soft/metrics.json \
    --calibration-seeds 11 --output path/to/reliability.json

Training and calibration use only the supplied seen-domain records.  Held-out
unseen records may be supplied with ``--test-metrics`` for reporting, but are
never used to fit or calibrate the checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from genedynamics.learning.reliability import (
    FEATURE_NAMES,
    LinearReliabilityModel,
    PEG_INSERT_FEATURE_NAMES,
    PEG_INSERT_RISK_NAMES,
    RISK_NAMES,
    samples_from_records,
)


def _metric_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("metrics.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"missing metric record source: {path}")
    return sorted(set(item.resolve() for item in files))


def _records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _metric_files(paths):
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON record list")
        records.extend(payload)
    return records


def _result_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("results.json")))
        elif path.name == "results.json" and path.is_file():
            files.append(path)
        else:
            raise ValueError(
                f"{path} must be a unified-runner results.json or result root"
            )
    return sorted(set(item.resolve() for item in files))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_record_from_result(path: Path) -> dict[str, Any]:
    """Replay a unified-runner Surface trajectory into the shared signals.

    The runner's executed actions are authoritative.  Replaying them through
    the resolved execution environment avoids depending on the deprecated
    solver-local experiment entry and gives reliability training the same
    signal contract as paper metrics.
    """
    import jax

    from genedynamics.core.types import Trajectory
    from genedynamics.experiments.plugins.environments._contact_task import (
        ContactTaskEnvironmentPlugin,
    )
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_signals,
    )

    with path.open() as handle:
        result = json.load(handle)
    snapshot = dict(result.get("config_snapshot") or {})
    task = str(snapshot.get("env_name", ""))
    if task != "manipulator_surface_scan":
        raise ValueError(f"{path} is not a Surface-scan result")
    seed = int(result["seed"])
    suite = str(result.get("suite", result.get("level")))
    suites = {
        str(item.get("name")): item for item in snapshot.get("suites", ())
    }
    level = str((suites.get(suite) or {}).get("level", suite))
    env_params = dict(snapshot.get("env_params") or {})
    env_params.setdefault("level", level)
    method_params = dict(snapshot.get("method_params") or {})
    controller = str(
        method_params.get("controller_method", snapshot.get("method", "mga"))
    )
    plugin = ContactTaskEnvironmentPlugin(task)
    model_env = plugin.create_env({
        **env_params,
        "_experiment_seed": seed,
        "_controller_method": controller,
        "_execution_env_params": dict(
            snapshot.get("execution_env_params") or {}
        ),
    })
    execution_env = plugin.execution_env(model_env)
    trajectory_path = path.parent / "trajectory" / "trajectory.json"
    with trajectory_path.open() as handle:
        trajectory_payload = json.load(handle)
    actions = trajectory_payload.get("actions") or []
    # Preserve the shared trajectory invariant while deliberately omitting
    # structured pipeline states: the extractor then takes its documented
    # action-replay fallback through the resolved execution environment.
    trajectory = Trajectory(
        states=[None] * (len(actions) + 1), actions=actions, info={}
    )
    x0 = execution_env.reset(jax.random.PRNGKey(seed))
    signals = arm_surface_scan_signals(
        trajectory, execution_env, None, None, x0=x0, seed=seed
    )

    def values(name: str):
        array = np.asarray(signals[name])
        return array.reshape(len(array), -1).squeeze().tolist()

    series = {key: values(key) for key in (
        "force", "force_cmd", "in_contact", "deformation", "normal_offset",
        "gate_path_error",
    )}
    series["actions"] = np.asarray(actions, np.float32).tolist()
    series.update({
        "f_min": float(signals["f_min"]),
        "f_max": float(signals["f_max"]),
        "f_target": float(execution_env._config.f_target),
        "deformation_safe": float(
            getattr(execution_env._config, "deformation_safe", 0.0)
        ),
        "deformation_scale": float(
            getattr(execution_env._config, "deformation_scale", 1.0)
        ),
        "path_scale": float(
            getattr(execution_env._config, "geometry_gate_path_scale", 0.02)
        ),
        "normal_scale": float(
            getattr(execution_env._config, "geometry_gate_normal_scale", 0.005)
        ),
        "scan_rate": float(getattr(execution_env._config, "scan_rate", 0.0)),
        "scan_span": float(getattr(execution_env._config, "scan_span", 1.0)),
    })
    checkpoint = (
        (result.get("provenance") or {}).get("checkpoints", {})
        .get("policy_ckpt", {})
    )
    return {
        "task": task,
        "suite": suite,
        "level": level,
        "seed": seed,
        "policy_training_seed": (
            (snapshot.get("metadata") or {}).get("policy_training_seed")
        ),
        "policy_checkpoint_sha256": checkpoint.get("sha256"),
        "source_result": str(path),
        "source_result_sha256": _sha256(path),
        "source_trajectory_sha256": _sha256(trajectory_path),
        "series": series,
    }


def _peg_insert_record_from_result(
    path: Path,
    result: dict[str, Any] | None = None,
    trajectory_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one unified-runner PegInsert rollout to the fit schema.

    PegInsert's persisted task signals are already measured in the hidden
    execution environment.  Consuming those signals directly is important:
    replaying the submitted actions in the nominal model would erase the pose,
    sensing, latency, and contact-parameter mismatch that reliability is meant
    to learn.
    """
    if result is None:
        with path.open() as handle:
            result = json.load(handle)
    snapshot = dict(result.get("config_snapshot") or {})
    task = str(snapshot.get("env_name", ""))
    if task != "manipulator_peg_insert":
        raise ValueError(f"{path} is not a PegInsert result")
    execution = dict(result.get("execution_status") or {})
    if execution and execution.get("state", "completed") != "completed":
        raise ValueError(f"incomplete PegInsert rollout cannot be fitted: {path}")

    trajectory_path = path.parent / "trajectory" / "trajectory.json"
    if trajectory_payload is None:
        with trajectory_path.open() as handle:
            trajectory_payload = json.load(handle)
    actions = np.asarray(trajectory_payload.get("actions") or (), np.float32)
    signals = dict(trajectory_payload.get("task_signals") or {})
    if actions.ndim != 2 or actions.shape[1:] != (13,):
        raise ValueError(
            f"invalid PegInsert action array in {trajectory_path}: "
            f"{actions.shape}"
        )
    if not len(actions) or not np.all(np.isfinite(actions)):
        raise ValueError(f"non-finite or empty PegInsert actions: {trajectory_path}")
    expected_steps = int(snapshot.get("n_steps", len(actions)))
    if len(actions) != expected_steps:
        raise ValueError(
            f"PegInsert fit requires the complete {expected_steps}-step "
            f"execution, got {len(actions)} in {trajectory_path}"
        )

    required = (
        "pose", "angle_vec", "measured_lateral_force",
        "measured_axial_force", "measured_bending_torque",
        "measured_wrench_delta", "contact_count", "stall_steps",
        "force_violation", "torque_violation", "jammed", "axial_force",
    )
    missing = [name for name in required if name not in signals]
    if missing:
        raise ValueError(
            f"PegInsert task signals missing in {trajectory_path}: "
            f"{', '.join(missing)}"
        )
    arrays = {name: np.asarray(signals[name]) for name in required}
    bad_lengths = {}
    for name, value in arrays.items():
        length = None if value.ndim == 0 else int(len(value))
        if length != len(actions):
            bad_lengths[name] = length
    if bad_lengths:
        raise ValueError(
            f"PegInsert task signals must match {len(actions)} actions: "
            f"{bad_lengths}"
        )
    nonfinite = [
        name for name, value in arrays.items()
        if not np.all(np.isfinite(value))
    ]
    if nonfinite:
        raise ValueError(
            f"non-finite PegInsert task signals in {trajectory_path}: "
            f"{', '.join(nonfinite)}"
        )
    if arrays["pose"].shape != (len(actions), 3):
        raise ValueError(
            f"PegInsert pose must have shape ({len(actions)}, 3), got "
            f"{arrays['pose'].shape}"
        )
    if arrays["angle_vec"].shape != (len(actions), 3):
        raise ValueError(
            f"PegInsert angle_vec must have shape ({len(actions)}, 3), got "
            f"{arrays['angle_vec'].shape}"
        )
    controls = signals.get("controls")
    if controls is not None:
        controls_array = np.asarray(controls, np.float32)
        if controls_array.shape != actions.shape or not np.allclose(
            controls_array, actions, rtol=0.0, atol=1.0e-7
        ):
            raise ValueError(
                f"PegInsert persisted controls disagree with actions: "
                f"{trajectory_path}"
            )

    # Resolve the execution-task parameters in the same order as the runner:
    # base env parameters first, then the suite's hidden execution overrides.
    # These are only normalization/latency constants; the physical labels
    # above always come from the persisted execution trajectory.
    params = dict(snapshot.get("env_params") or {})
    params.update(snapshot.get("execution_env_params") or {})
    defaults = {
        "socket_depth": 0.040,
        "lateral_force_limit": 20.0,
        "f_min": 0.0,
        "f_max": 30.0,
        "f_cmd_pad": 10.0,
        "bending_torque_limit": 1.5,
        "jam_dwell_steps": 4,
        "f_target": 10.0,
        "action_delay_steps": 0,
        "sensor_delay_steps": 0,
        "approach_gap": 0.008,
        "translation_step": (0.001, 0.001, 0.0015),
        "rotation_step": 0.015,
    }
    series = {
        "actions": actions.tolist(),
        **{name: arrays[name].tolist() for name in required},
        **{name: params.get(name, default) for name, default in defaults.items()},
    }
    checkpoint = (
        (result.get("provenance") or {}).get("checkpoints", {})
        .get("policy_ckpt", {})
    )
    return {
        "task": task,
        "suite": str(result.get("suite", result.get("level", ""))),
        "level": str(params.get("level", "wide")),
        "seed": int(result["seed"]),
        "policy_training_seed": (
            (snapshot.get("metadata") or {}).get("policy_training_seed")
        ),
        "policy_checkpoint_sha256": checkpoint.get("sha256"),
        "behavior_method": str(snapshot.get("method", "")),
        "behavior_name": str(snapshot.get("name", "")),
        "sample_source": "executed_commit_horizon",
        "source_result": str(path),
        "source_result_sha256": _sha256(path),
        "source_trajectory_sha256": _sha256(trajectory_path),
        "series": series,
    }


def _record_from_result(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        result = json.load(handle)
    task = str((result.get("config_snapshot") or {}).get("env_name", ""))
    if task == "manipulator_surface_scan":
        return _surface_record_from_result(path)
    if task == "manipulator_peg_insert":
        trajectory_path = path.parent / "trajectory" / "trajectory.json"
        with trajectory_path.open() as handle:
            trajectory_payload = json.load(handle)
        return _peg_insert_record_from_result(
            path, result=result, trajectory_payload=trajectory_payload
        )
    raise ValueError(f"unsupported reliability task in {path}: {task!r}")


def _result_records(paths: list[str]) -> list[dict[str, Any]]:
    return [_record_from_result(path) for path in _result_files(paths)]


def _assert_disjoint_protocol(
    train_provenance: list[dict[str, Any]],
    calibration_provenance: list[dict[str, Any]],
    evaluation_seeds: set[int],
) -> None:
    train_ids = {
        (str(row["suite"]), int(row["seed"])) for row in train_provenance
    }
    calibration_ids = {
        (str(row["suite"]), int(row["seed"]))
        for row in calibration_provenance
    }
    overlap = sorted(train_ids.intersection(calibration_ids))
    if overlap:
        raise ValueError(
            f"training/calibration trajectories overlap: {overlap}"
        )
    used_seeds = {seed for _, seed in train_ids | calibration_ids}
    leaked = sorted(used_seeds.intersection(evaluation_seeds))
    if leaked:
        raise ValueError(
            f"formal evaluation seeds entered reliability fit/calibration: {leaked}"
        )


def _report(model, x, y) -> dict[str, Any]:
    if not len(x):
        return {"rows": 0}
    prediction = np.asarray(model.predict(x))
    upper = np.asarray(model.predict_upper(x))
    support_score = np.asarray(model.support_score(x))
    in_support = support_score <= 1.0
    conditional_coverage = (
        np.mean(y[in_support] <= upper[in_support] + 1.0e-7, axis=0)
        if np.any(in_support) else np.full(len(model.risk_names), np.nan)
    )
    report = {
        "rows": int(len(x)),
        "mae": {
            name: float(value)
            for name, value in zip(model.risk_names, np.mean(np.abs(prediction - y), axis=0))
        },
        "upper_coverage": {
            name: float(value)
            for name, value in zip(model.risk_names, np.mean(y <= upper + 1.0e-7, axis=0))
        },
        "in_support_upper_coverage": {
            name: float(value)
            for name, value in zip(model.risk_names, conditional_coverage)
        },
        "mean_upper": {
            name: float(value)
            for name, value in zip(model.risk_names, np.mean(upper, axis=0))
        },
        "in_support_rate": float(np.mean(in_support)),
        "support_score_p95": float(np.quantile(support_score, 0.95)),
        "support_score_max": float(np.max(support_score)),
    }
    if model.classification_probabilities:
        count = int(model.probability_risk_count)
        class_counts = {}
        auroc = {}
        for index, name in enumerate(model.risk_names[:count]):
            positive = prediction[y[:, index] > 0.5, index]
            negative = prediction[y[:, index] <= 0.5, index]
            class_counts[name] = {
                "positive": int(len(positive)),
                "negative": int(len(negative)),
            }
            auroc[name] = (
                float(np.mean(
                    (positive[:, None] > negative[None, :])
                    + 0.5 * (positive[:, None] == negative[None, :])
                ))
                if len(positive) and len(negative) else None
            )
        report["class_counts"] = class_counts
        report["probability_auroc"] = auroc
        report["probability_brier"] = {
            name: float(value)
            for name, value in zip(
                model.risk_names[:count],
                np.mean((prediction[:, :count] - y[:, :count]) ** 2, axis=0),
            )
        }
        # Per-outcome coverage is not a meaningful calibration target for a
        # Bernoulli probability.  Keep the legacy field for compatibility but
        # label its scope explicitly in new reports.
        report["upper_coverage_semantics"] = {
            name: (
                "event_probability" if index < count
                else "one_sided_conformal_outcome"
            )
            for index, name in enumerate(model.risk_names)
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--calibration-metrics", nargs="*", default=[])
    parser.add_argument(
        "--training-results", nargs="*", default=[],
        help="unified-runner results.json files or roots used only for fitting",
    )
    parser.add_argument(
        "--calibration-results", nargs="*", default=[],
        help="disjoint unified-runner results.json files or roots for conformal calibration",
    )
    parser.add_argument("--test-metrics", nargs="*", default=[])
    parser.add_argument(
        "--test-results", nargs="*", default=[],
        help="held-out unified-runner results used only for reporting",
    )
    parser.add_argument("--output")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--evaluation-seeds", nargs="*", type=int, default=[])
    parser.add_argument(
        "--required-policy-sha256", default=None,
        help="require this policy artifact in both fit and calibration splits",
    )
    args = parser.parse_args(argv)

    if args.load_checkpoint:
        if not (args.test_metrics or args.test_results):
            raise ValueError(
                "--load-checkpoint requires --test-metrics/--test-results"
            )
        model = LinearReliabilityModel.load(args.load_checkpoint)
        test_records = [
            *_records(args.test_metrics), *_result_records(args.test_results),
        ]
        test_x, test_y, _ = samples_from_records(test_records)
        print(json.dumps({
            "checkpoint": args.load_checkpoint,
            "held_out_test": _report(model, test_x, test_y),
        }, indent=2))
        return 0
    if not (args.metrics or args.training_results) or not args.output:
        raise ValueError(
            "fitting requires --metrics/--training-results and --output"
        )

    records = _records(args.metrics)
    calibration_seeds = set(args.calibration_seeds)
    train_records = [r for r in records if int(r["seed"]) not in calibration_seeds]
    calibration_records = [r for r in records if int(r["seed"]) in calibration_seeds]
    calibration_records.extend(_records(args.calibration_metrics))
    train_records.extend(_result_records(args.training_results))
    calibration_records.extend(_result_records(args.calibration_results))
    if not train_records or not calibration_records:
        raise ValueError("seed split produced an empty training or calibration set")
    if args.required_policy_sha256:
        for label, split_records in (
            ("training", train_records),
            ("calibration", calibration_records),
        ):
            hashes = {
                str(record["policy_checkpoint_sha256"])
                for record in split_records
                if record.get("policy_checkpoint_sha256")
            }
            if hashes != {args.required_policy_sha256}:
                raise ValueError(
                    f"{label} split policy hashes {sorted(hashes)} do not "
                    f"exactly match required {args.required_policy_sha256}"
                )

    train_x, train_y, train_provenance = samples_from_records(train_records)
    cal_x, cal_y, cal_provenance = samples_from_records(calibration_records)
    _assert_disjoint_protocol(
        train_provenance,
        cal_provenance,
        set(args.evaluation_seeds),
    )
    peg_insert = train_x.shape[1] == len(PEG_INSERT_FEATURE_NAMES)
    if peg_insert:
        for label, values in (("training", train_y), ("calibration", cal_y)):
            for index, name in enumerate(PEG_INSERT_RISK_NAMES[:3]):
                positive = int(np.count_nonzero(values[:, index] > 0.5))
                negative = int(np.count_nonzero(values[:, index] <= 0.5))
                if not positive or not negative:
                    raise ValueError(
                        f"PegInsert {label} split lacks both classes for "
                        f"{name}: positive={positive}, negative={negative}; "
                        "collect stress trajectories instead of fitting a "
                        "degenerate reliability head"
                    )
    model = LinearReliabilityModel.fit(
        train_x,
        train_y,
        cal_x,
        cal_y,
        ridge=args.ridge,
        quantile=args.quantile,
        feature_names=(PEG_INSERT_FEATURE_NAMES if peg_insert else FEATURE_NAMES),
        risk_names=(PEG_INSERT_RISK_NAMES if peg_insert else RISK_NAMES),
        state_feature_count=(32 if peg_insert else 6),
        support_state_feature_count=(32 if peg_insert else 6),
        probability_risk_count=(3 if peg_insert else 2),
        classification_probabilities=peg_insert,
        # The phase-residual basis remains available for controlled ablations,
        # but held-out PegInsert rollouts rejected it as the canonical gate.
        # Keep the production trainer on the validated shared linear basis.
        basis_mode="linear",
    )
    metadata = {
        "training_records": len(train_records),
        "calibration_records": len(calibration_records),
        "training_rows": len(train_x),
        "calibration_rows": len(cal_x),
        "training_suites": dict(Counter(x["suite"] for x in train_provenance)),
        "calibration_suites": dict(Counter(x["suite"] for x in cal_provenance)),
        # Explicit calibration files may contain seeds that do not appear in
        # ``--calibration-seeds``.  Record the samples actually used so the
        # paper split is auditable instead of reporting only the CLI selector.
        "calibration_seeds": sorted({
            int(x["seed"]) for x in cal_provenance
        }),
        "training_seeds": sorted({
            int(x["seed"]) for x in train_provenance
        }),
        "evaluation_seeds_excluded": sorted(set(args.evaluation_seeds)),
        "split_disjoint": True,
        "required_policy_sha256": args.required_policy_sha256,
        "fit_scope": "development_fit_only",
        "performance_validated": False,
        "promotion_eligible": False,
        "promotion_blocking_reasons": [
            "independent_paired_validation_not_performed",
        ],
        "training_policy_checkpoint_sha256": sorted({
            str(record["policy_checkpoint_sha256"])
            for record in train_records
            if record.get("policy_checkpoint_sha256")
        }),
        "calibration_policy_checkpoint_sha256": sorted({
            str(record["policy_checkpoint_sha256"])
            for record in calibration_records
            if record.get("policy_checkpoint_sha256")
        }),
        "training_files": [
            *list(args.metrics), *list(args.training_results),
        ],
        "deployment_family_calibration_files": [
            *list(args.calibration_metrics), *list(args.calibration_results),
        ],
        "source_hashes": {
            str(path): _sha256(path)
            for path in [
                *_metric_files(args.metrics),
                *_metric_files(args.calibration_metrics),
                *_result_files(args.training_results),
                *_result_files(args.calibration_results),
            ]
        },
        "ridge": args.ridge,
        "quantile": args.quantile,
        "basis_mode": model.basis_mode,
        "architecture": (
            "standardized_ridge_logistic_split_conformal_backtracking"
            if peg_insert else "standardized_multioutput_ridge_split_conformal"
        ),
        "checkpoint_selection_rule": "single_frozen_fit_no_evaluation_selection",
        "protocol": args.protocol,
        "task": (
            "manipulator_peg_insert" if peg_insert
            else "manipulator_surface_scan"
        ),
    }
    if peg_insert:
        metadata.update({
            "training_sample_sources": dict(Counter(
                x.get("sample_source", "legacy_metric_record")
                for x in train_provenance
            )),
            "calibration_sample_sources": dict(Counter(
                x.get("sample_source", "legacy_metric_record")
                for x in cal_provenance
            )),
            "training_behavior_methods": dict(Counter(
                str(x.get("behavior_method", "unknown"))
                for x in train_provenance
            )),
            "calibration_behavior_methods": dict(Counter(
                str(x.get("behavior_method", "unknown"))
                for x in cal_provenance
            )),
            "training_behavior_names": dict(Counter(
                str(x.get("behavior_name", "unknown"))
                for x in train_provenance
            )),
            "calibration_behavior_names": dict(Counter(
                str(x.get("behavior_name", "unknown"))
                for x in cal_provenance
            )),
        })
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"reliability output already exists; choose a new path: {output}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(
            f"stale temporary reliability output: {temporary}"
        )
    model.save(temporary, metadata=metadata)
    temporary.replace(output)

    report = {
        "checkpoint": str(output),
        "train": _report(model, train_x, train_y),
        "calibration": _report(model, cal_x, cal_y),
    }
    if args.test_metrics or args.test_results:
        test_records = [
            *_records(args.test_metrics), *_result_records(args.test_results),
        ]
        test_x, test_y, _ = samples_from_records(test_records)
        report["held_out_test"] = _report(model, test_x, test_y)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
