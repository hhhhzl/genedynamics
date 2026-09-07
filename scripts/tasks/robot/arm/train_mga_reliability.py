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


def _result_records(paths: list[str]) -> list[dict[str, Any]]:
    return [_surface_record_from_result(path) for path in _result_files(paths)]


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
    parser.add_argument("--output")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--evaluation-seeds", nargs="*", type=int, default=[])
    args = parser.parse_args(argv)

    if args.load_checkpoint:
        if not args.test_metrics:
            raise ValueError("--load-checkpoint requires --test-metrics")
        model = LinearReliabilityModel.load(args.load_checkpoint)
        test_x, test_y, _ = samples_from_records(_records(args.test_metrics))
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

    train_x, train_y, train_provenance = samples_from_records(train_records)
    cal_x, cal_y, cal_provenance = samples_from_records(calibration_records)
    _assert_disjoint_protocol(
        train_provenance,
        cal_provenance,
        set(args.evaluation_seeds),
    )
    peg_insert = train_x.shape[1] == len(PEG_INSERT_FEATURE_NAMES)
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output, metadata=metadata)

    report = {
        "checkpoint": str(output),
        "train": _report(model, train_x, train_y),
        "calibration": _report(model, cal_x, cal_y),
    }
    if args.test_metrics:
        test_x, test_y, _ = samples_from_records(_records(args.test_metrics))
        report["held_out_test"] = _report(model, test_x, test_y)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
