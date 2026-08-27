"""Fit the H1 reliability bound from unified-runner task signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from genedynamics.learning.reliability import LinearReliabilityModel


FEATURE_NAMES = (
    "box_progress", "box_goal_distance", "box_lateral_offset", "box_yaw",
    "hand_force_ratio", "wall_force_ratio", "nonhand_force_ratio",
    "support_offset_ratio", "contact_acquired", "unjam_released",
    "force_integrator_ratio", "corridor_clearance_ratio",
    "translation_action_rms", "stiffness_action_rms", "mean_force_action",
    "max_force_action", "first_action_delta_rms", "action_roughness",
    "mean_semantic_action_0", "mean_semantic_action_3",
    "mean_semantic_action_4", "walk_action_rms", "walk_action_max",
    "is_walk",
)
RISK_NAMES = ("force_violation", "invalid_contact_or_fall", "balance", "force_mae")


def _result_files(roots):
    files = []
    for raw in roots:
        path = Path(raw)
        files.extend(path.rglob("results.json") if path.is_dir() else [path])
    return sorted(set(path.resolve() for path in files if path.is_file()))


def _rows(files, seeds, methods):
    xs, ys, provenance = [], [], []
    for path in files:
        result = json.loads(path.read_text())
        seed = int(result["seed"])
        if seed not in seeds:
            continue
        if (result.get("config_snapshot") or {}).get("env_name") != "humanoid_box_push":
            continue
        snapshot = result.get("config_snapshot") or {}
        if methods and str(snapshot.get("name", snapshot.get("method"))) not in methods:
            continue
        trajectory = path.parent / "trajectory" / "trajectory.json"
        payload = json.loads(trajectory.read_text())
        signals = payload.get("task_signals") or {}
        if not signals:
            # Backward-compatible replay for the frozen P7 development runs.
            # The executed action sequence remains authoritative; no solver is
            # rerun and no formal seed is introduced.
            import jax
            from genedynamics.core.types import Trajectory
            from genedynamics.experiments.plugins.environments._contact_task import (
                ContactTaskEnvironmentPlugin,
            )
            from genedynamics.experiments.plugins.metrics.extractors import (
                humanoid_box_push_signals,
            )

            snapshot = result["config_snapshot"]
            plugin = ContactTaskEnvironmentPlugin("humanoid_box_push")
            env = plugin.create_env({
                **dict(snapshot.get("env_params") or {}),
                "_experiment_seed": seed,
                "_controller_method": dict(
                    snapshot.get("method_params") or {}
                ).get("controller_method", snapshot.get("method")),
            })
            actions = payload.get("actions") or []
            replay = Trajectory(
                states=[None] * (len(actions) + 1), actions=actions, info={},
            )
            signals = humanoid_box_push_signals(
                replay, env, None, None,
                x0=env.reset(jax.random.PRNGKey(seed)), seed=seed,
            )
        x = np.asarray(signals.get("reliability_features"), np.float64)
        y = np.asarray(signals.get("reliability_risk"), np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"{trajectory}: invalid H1 reliability features {x.shape}")
        if y.shape != (len(x), len(RISK_NAMES)):
            raise ValueError(f"{trajectory}: invalid H1 reliability risks {y.shape}")
        xs.append(x)
        ys.append(y)
        provenance.append({
            "result": str(path), "suite": result.get("suite"), "seed": seed,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        print(
            json.dumps({
                "replayed": str(path), "rows": int(len(x)), "seed": seed,
            }),
            flush=True,
        )
    if not xs:
        raise ValueError(f"no H1 rows for seeds {sorted(seeds)}")
    return np.concatenate(xs), np.concatenate(ys), provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[5, 6])
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["model_based_only"],
        help="Development controllers used to collect executed actions.",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    training = set(args.training_seeds)
    calibration = set(args.calibration_seeds)
    evaluation = set(args.evaluation_seeds)
    if training & calibration or (training | calibration) & evaluation:
        raise ValueError("training, calibration, and formal evaluation seeds must be disjoint")
    files = _result_files(args.roots)
    methods = set(args.methods)
    train_x, train_y, train_provenance = _rows(files, training, methods)
    cal_x, cal_y, cal_provenance = _rows(files, calibration, methods)
    model = LinearReliabilityModel.fit(
        train_x, train_y, cal_x, cal_y,
        feature_names=FEATURE_NAMES, risk_names=RISK_NAMES,
        state_feature_count=12, support_state_feature_count=12,
        probability_risk_count=2, classification_probabilities=True,
        quantile=float(args.quantile),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": "humanoid_box_push_reliability_v1",
        "task": "humanoid_box_push",
        "training_seeds": sorted(training),
        "calibration_seeds": sorted(calibration),
        "evaluation_seeds_excluded": sorted(evaluation),
        "training_budget": {"unit": "environment_transitions", "value": len(train_x)},
        "training_provenance": train_provenance,
        "calibration_provenance": cal_provenance,
        "architecture": "standardized_ridge_logistic_split_conformal",
        "collection_methods": sorted(methods),
        "checkpoint_selection_rule": "single_frozen_fit_no_evaluation_selection",
    }
    model.save(output, metadata=metadata)
    print(json.dumps({
        "checkpoint": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "training_rows": len(train_x), "calibration_rows": len(cal_x),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
