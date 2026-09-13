"""Fit the H1 reliability bound from unified-runner task signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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


def _sequence_rows(payload, snapshot, *, horizon_steps=17):
    """Pair observable ``s_t, U[t:t+H]`` with the H post-transition risks.

    The metric extractor persists features at s_(t+1), including action
    statistics for just u_t.  Only its first 12, action-independent entries
    can be reused, shifted back one index.  Rebuild the remaining entries to
    match HumanoidBoxPushEnv.reliability_features_sequence exactly.

    A dense MGA candidate contains Hsample+1 controls.  A shortened window
    would train a different risk target, so discard right-censored windows.
    The unobserved reset row and starts after absorbing success are excluded;
    a window reaching success still includes its recorded absorbing risks,
    exactly as the deployment rollout does.  No physics replay is required.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    actions = np.asarray(payload.get("actions", ()), np.float64)
    signals = payload.get("task_signals") or {}
    post_features = np.asarray(signals.get("reliability_features"), np.float64)
    post_risks = np.asarray(signals.get("reliability_risk"), np.float64)
    success = np.asarray(signals.get("task_success"), np.float64).reshape(-1)
    if actions.ndim != 2:
        raise ValueError(f"invalid H1 actions {actions.shape}")
    n = len(actions)
    if post_features.shape != (n, len(FEATURE_NAMES)):
        raise ValueError(f"invalid H1 reliability features {post_features.shape}")
    if post_risks.shape != (n, len(RISK_NAMES)) or success.shape != (n,):
        raise ValueError("H1 risks/success must contain one row per transition")
    schema = ((signals.get("task_metadata") or {}).get("reliability_contract") or {}).get("schema") or {}
    scoped = int(schema.get("version", 0)) >= 3 or "mga_execution_mode" in signals
    unload = np.zeros(n, dtype=bool)
    instantaneous = None
    if scoped:
        mode = np.asarray(signals.get("mga_execution_mode", ())).reshape(-1)
        applicable = np.asarray(signals.get("reliability_execution_applicable", ())).reshape(-1)
        risk_valid = np.asarray(signals.get("reliability_risk_valid", ())).reshape(-1)
        instantaneous = np.asarray(signals.get("safety_margins", ()), np.float64)
        if (mode.shape != (n,) or applicable.shape != (n,) or risk_valid.shape != (n,)
                or instantaneous.shape != (n, 4)):
            raise ValueError("H1 normal-only labels require complete execution mode/mask/instantaneous margins")
        if not (np.isin(mode, [0, 1]).all() and np.isin(applicable, [0, 1]).all()
                and np.isin(risk_valid, [0, 1]).all()):
            raise ValueError("invalid H1 reliability execution flags")
        unload = mode == 1
        if not np.array_equal(applicable.astype(bool), ~unload):
            raise ValueError("H1 reliability applicability disagrees with committed execution mode")
        if not risk_valid[~unload].all() or risk_valid[unload].any():
            raise ValueError("unknown NORMAL coverage or mislabelled UNLOAD applicability")
        if not np.isnan(post_risks[unload]).all():
            raise ValueError("UNLOAD labels must be explicitly N/A, not normal-model risk observations")
        if not np.isfinite(instantaneous).all():
            raise ValueError("non-finite H1 instantaneous safety margins")
    if not all(np.all(np.isfinite(a)) for a in (
        actions, post_features, post_risks[~unload], success,
    )):
        raise ValueError("non-finite H1 reliability trajectory")
    params = snapshot.get("env_params") or {}
    is_walk = str(params.get("level", "push_to_line")).lower() == "push_walk"
    position_width = 8 if params.get("use_base", False) else 5
    semantic_width = position_width + 6 + 1
    if (actions.shape[1] < semantic_width
            or (is_walk and actions.shape[1] == semantic_width)
            or (not is_walk and actions.shape[1] != semantic_width)):
        raise ValueError("H1 action width disagrees with the resolved task layout")
    if not np.all(post_features[:, -1] == float(is_walk)):
        raise ValueError("H1 saved features disagree with the resolved walk level")
    states = payload.get("states") or []
    if states and len(states) != n + 1:
        raise ValueError("H1 compact states must have len(actions)+1 entries")
    xs, ys, starts = [], [], []
    terminal_starts = 0
    censored_starts = 0
    unload_starts = 0
    for t in range(1, n):
        if success[t - 1] > 0.5:
            terminal_starts += 1
            continue
        if t + horizon_steps > n:
            censored_starts += 1
            continue
        if unload[t:t + horizon_steps].any():
            unload_starts += 1
            continue
        candidate = actions[t:t + horizon_steps]
        semantic = candidate[:, :semantic_width]
        # Compact states preserve the controller memory at decision time.
        # Legacy action-only traces have the same u_(t-1) before success.
        compact = states[t] if states else None
        info = (compact.get("info") or {}) if isinstance(compact, dict) else {}
        previous = np.asarray(info.get("prev_action", actions[t - 1]), np.float64)
        if previous.shape != actions[t - 1].shape or not np.all(np.isfinite(previous)):
            raise ValueError(f"invalid H1 prev_action at decision {t}")
        delta0 = semantic[0] - previous[:semantic_width]
        force = semantic[:, position_width + 6:semantic_width]
        sequence = np.asarray([
            np.sqrt(np.mean(semantic[:, :position_width] ** 2)),
            np.sqrt(np.mean(semantic[:, position_width:position_width + 6] ** 2)),
            np.mean(force), np.max(np.abs(force)),
            np.sqrt(np.mean(delta0 ** 2)),
            np.sqrt(np.mean(np.diff(semantic, axis=0) ** 2))
            if horizon_steps > 1 else 0.0,
            np.mean(semantic[:, 0]), np.mean(semantic[:, 3]),
            np.mean(semantic[:, 4]),
            np.sqrt(np.mean(candidate[:, semantic_width:] ** 2)) if is_walk else 0.0,
            np.max(np.abs(candidate[:, semantic_width:])) if is_walk else 0.0,
            float(is_walk),
        ])
        risks = post_risks[t:t + horizon_steps]
        tail = max(1, (horizon_steps + 4) // 5)
        xs.append(np.concatenate([post_features[t - 1, :12], sequence]))
        label = np.asarray([
            np.max(risks[:, 0]), np.max(risks[:, 1]),
            np.mean(np.sort(risks[:, 2])[-tail:]), np.mean(risks[:, 3]),
        ])
        if instantaneous is not None:
            # s_t is post row t-1. This is its endpoint, not the preceding
            # interval's peak risk (which belongs to a different candidate).
            initial = instantaneous[t - 1]
            initial_heads = np.asarray([
                float(initial[0] > 0.), float(initial[1] > 0. or initial[3] > 0.),
                max(initial[2], 0.),
            ])
            label[:3] = np.maximum(label[:3], initial_heads)
        ys.append(label)
        starts.append(t)
    return (
        np.asarray(xs, np.float64).reshape(-1, len(FEATURE_NAMES)),
        np.asarray(ys, np.float64).reshape(-1, len(RISK_NAMES)),
        {"decision_steps": starts, "horizon_steps": horizon_steps,
         "source_transitions": n, "unobserved_reset_starts": int(n > 0),
         "absorbing_starts_excluded": terminal_starts,
         "right_censored_starts_excluded": censored_starts,
         "unload_crossing_starts_excluded": unload_starts,
         "unload_transitions_excluded_from_normal_labels": int(unload.sum())},
    )


def _label_coverage(y):
    return {
        name: {"positive": int(np.count_nonzero(y[:, i] > 0.0)),
               "zero": int(np.count_nonzero(y[:, i] == 0.0)),
               "positive_fraction": float(np.mean(y[:, i] > 0.0)) if len(y) else None}
        for i, name in enumerate(RISK_NAMES)
    }


def _model_diagnostics(model, x, y):
    """Empirical split diagnostics; calibration coverage is not a test claim."""
    prediction = np.asarray(model.predict(x))
    upper = np.asarray(model.predict_upper(x))
    state_support = np.asarray(model.support_score_state(x)) <= 1.0
    joint_support = np.asarray(model.support_score(x)) <= 1.0
    covered = y <= upper + 1.0e-7
    return {
        "rows": len(y),
        "labels": _label_coverage(y),
        "binary_brier": {
            name: float(np.mean((prediction[:, i] - y[:, i]) ** 2))
            for i, name in enumerate(RISK_NAMES[:2])
        },
        # Event probabilities are not upper bounds on an individual binary
        # observation.  Comparing y=1 to p<1 would report almost every event
        # as an uncovered sample even for a calibrated classifier.
        "binary_threshold_errors": {
            name: {
                "threshold": 0.5,
                "false_negative_rate": float(np.mean(upper[y[:, i] > 0.5, i] <= 0.5))
                if np.any(y[:, i] > 0.5) else None,
                "false_positive_rate": float(np.mean(upper[y[:, i] <= 0.5, i] > 0.5))
                if np.any(y[:, i] <= 0.5) else None,
            }
            for i, name in enumerate(RISK_NAMES[:2])
        },
        "mae": {
            name: float(np.mean(np.abs(prediction[:, i] - y[:, i])))
            for i, name in enumerate(RISK_NAMES)
        },
        "continuous_upper_coverage": {
            name: float(np.mean(covered[:, i]))
            for i, name in enumerate(RISK_NAMES) if i >= 2
        },
        "state_support_fraction": float(np.mean(state_support)),
        "joint_support_fraction": float(np.mean(joint_support)),
        "in_state_support_continuous_upper_coverage": {
            name: float(np.mean(covered[state_support, i])) if np.any(state_support) else None
            for i, name in enumerate(RISK_NAMES) if i >= 2
        },
    }


def _fit_audit(train_y, cal_y, cal_prediction, train_provenance, cal_provenance):
    """Describe development evidence without treating overlapping rows as trials."""
    source_counts = {}
    blocking = ["independent_candidate_and_performance_validation_not_performed"]
    for split, labels, sources in (
        ("training", train_y, train_provenance),
        ("calibration", cal_y, cal_provenance),
    ):
        hashes = sorted({row["feature_risk_sha256"] for row in sources})
        source_counts[split] = {
            "window_rows": int(len(labels)),
            "source_trajectory_count": len(sources),
            "unique_feature_risk_trajectory_count": len(hashes),
            "unique_feature_risk_sha256": hashes,
        }
        for i, name in enumerate(RISK_NAMES[:2]):
            if not np.any(labels[:, i] > 0.5):
                blocking.append(f"missing_positive_class:{split}:{name}")
            if not np.any(labels[:, i] <= 0.5):
                blocking.append(f"missing_negative_class:{split}:{name}")
    duplicates = [
        {"training_result": train["result"], "calibration_result": cal["result"]}
        for train in train_provenance for cal in cal_provenance
        if train["feature_risk_sha256"] == cal["feature_risk_sha256"]
    ]
    if duplicates:
        blocking.append("identical_training_calibration_feature_risk_trajectories")
    brier = {}
    for i, name in enumerate(RISK_NAMES[:2]):
        prevalence = float(np.mean(train_y[:, i]))
        constant = float(np.mean((cal_y[:, i] - prevalence) ** 2))
        model = float(np.mean((cal_y[:, i] - cal_prediction[:, i]) ** 2))
        brier[name] = {
            "training_prevalence": prevalence,
            "constant_calibration_brier": constant,
            "model_calibration_brier": model,
            "model_minus_constant": model - constant,
        }
    return {
        "fit_scope": "development_fit_only",
        "performance_validated": False,
        "promotion_eligible": False,
        "promotion_blocking_reasons": blocking,
        "trajectory_independence_validated": False,
        "source_counts": source_counts,
        "calibration_brier_vs_training_prevalence": brier,
        "identical_training_calibration_sources": duplicates,
    }


def _result_files(roots):
    files = []
    for raw in roots:
        path = Path(raw)
        files.extend(path.rglob("results.json") if path.is_dir() else [path])
    return sorted(set(path.resolve() for path in files if path.is_file()))


def _validate_splits(training, calibration, evaluation, formal):
    excluded = set(evaluation) | set(formal)
    if training & calibration or (training | calibration) & excluded:
        raise ValueError("training, calibration, and formal/development evaluation seeds must be disjoint")
    return excluded


def _collection_contract(signals, *, horizon_steps):
    """Validate persisted provenance without inventing labels using today's env.

    Historical signals remain readable through ``_sequence_rows`` for audits,
    but cannot enter a new fit without their collection-time semantic contract.
    Source changes require new collection, not silent replay or back-filling.
    """
    from genedynamics.envs.domains.humanoid.box_push_brax import (
        H1_RELIABILITY_SCHEMA, HumanoidBoxPushEnv,
    )

    metadata = signals.get("task_metadata") or {}
    contract = metadata.get("reliability_contract") or {}
    if (contract.get("schema") != H1_RELIABILITY_SCHEMA
            or contract.get("horizon_steps", "missing") is not None):
        raise ValueError("H1 collection contract missing or obsolete; historical data is audit-only")
    source = metadata.get("reliability_source_sha256")
    if source != HumanoidBoxPushEnv.reliability_source_sha256():
        raise ValueError("H1 collection source mismatch; recollect instead of replaying old data")
    required = {"dt", "timestep", "robot", "level", "action_layout", "parameters",
                "walk_parameters", "policy_interface"}
    if not required.issubset(contract):
        raise ValueError("incomplete H1 collection contract")
    if (contract["schema"]["feature_names"] != list(FEATURE_NAMES)
            or contract["schema"]["risk_names"] != list(RISK_NAMES)):
        raise ValueError("H1 fitter feature/risk schema mismatch")
    return {**contract, "horizon_steps": int(horizon_steps)}, source


def _fitted_contracts(training, calibration):
    """Every advertised task interface must occur in both disjoint splits."""
    def contracts(rows):
        return {json.dumps(row["reliability_contract"], sort_keys=True) for row in rows}

    train, cal = contracts(training), contracts(calibration)
    if not train or train != cal:
        raise ValueError("H1 training/calibration task contracts differ; do not advertise uncalibrated interfaces")
    return [json.loads(value) for value in sorted(train)]


def _rows(files, seeds, methods, *, horizon_steps=17):
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
        execution = result.get("execution_status") or {}
        if execution.get("state") == "aborted_unrecoverable":
            raise ValueError(
                f"aborted H1 collection cannot be fitted: {path}: {execution.get('reason', 'unspecified')}"
            )
        trajectory = path.parent / "trajectory" / "trajectory.json"
        payload = json.loads(trajectory.read_text())
        signals = payload.get("task_signals") or {}
        contract, source = _collection_contract(signals, horizon_steps=horizon_steps)
        resolved = snapshot.get("env_params") or {}
        if (contract["level"] != resolved.get("level", "push_to_line")
                or contract["action_layout"]["action_size"] != np.asarray(payload.get("actions")).shape[-1]):
            raise ValueError("H1 collection contract disagrees with saved task/actions")
        x, y, window_info = _sequence_rows(
            {**payload, "task_signals": signals}, snapshot,
            horizon_steps=horizon_steps,
        )
        if not len(x):
            continue
        xs.append(x)
        ys.append(y)
        provenance.append({
            "result": str(path), "suite": result.get("suite"), "seed": seed,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "feature_risk_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
            "window_sampling": window_info,
            "reliability_contract": contract,
            "collection_source_sha256": source,
            "collection_domain": {
                "env_params": snapshot.get("env_params") or {},
                "execution_env_params": snapshot.get("execution_env_params") or {},
            },
            "task_contract": {
                "level": (snapshot.get("env_params") or {}).get("level", "push_to_line"),
                "walk_success_mode": (snapshot.get("env_params") or {}).get(
                    "walk_success_mode", "legacy",
                ) if (snapshot.get("env_params") or {}).get("level") == "push_walk" else None,
            },
        })
        print(
            json.dumps({
                "source": str(path), "rows": int(len(x)), "seed": seed,
                "label_coverage": _label_coverage(y),
            }),
            flush=True,
        )
    if not xs:
        raise ValueError(f"no H1 rows for seeds {sorted(seeds)}")
    return np.concatenate(xs), np.concatenate(ys), provenance


def main() -> int:
    from genedynamics.experiments.framework.config import ExperimentConfig
    from genedynamics.learning.reliability import LinearReliabilityModel

    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[102])
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=[110, 111],
                        help="Additional excluded development seeds; canonical formal seeds are always excluded.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["model_based_only"],
        help="Development controllers used to collect executed actions.",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument(
        "--horizon-steps", type=int, default=17,
        help="Target MGA dense candidate length (Hsample + 1), not source policy length.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Reliability output already exists; choose a new --output: {output}")
    training = set(args.training_seeds)
    calibration = set(args.calibration_seeds)
    canonical_path = Path(__file__).resolve().parents[4] / "configs/humanoid/push_to_line/_base.yaml"
    canonical = ExperimentConfig.from_yaml(canonical_path)
    evaluation = _validate_splits(
        training, calibration, set(args.evaluation_seeds), set(canonical.seeds),
    )
    files = _result_files(args.roots)
    methods = set(args.methods)
    train_x, train_y, train_provenance = _rows(
        files, training, methods, horizon_steps=args.horizon_steps,
    )
    cal_x, cal_y, cal_provenance = _rows(
        files, calibration, methods, horizon_steps=args.horizon_steps,
    )
    contracts = _fitted_contracts(train_provenance, cal_provenance)
    model = LinearReliabilityModel.fit(
        train_x, train_y, cal_x, cal_y,
        feature_names=FEATURE_NAMES, risk_names=RISK_NAMES,
        state_feature_count=12, support_state_feature_count=12,
        probability_risk_count=2, classification_probabilities=True,
        quantile=float(args.quantile),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": "humanoid_box_push_reliability_sequence",
        "task": "humanoid_box_push",
        "reliability_contracts": contracts,
        "compatibility_scope": "semantic_interface_only_not_candidate_or_OOD_calibration_validation",
        **_fit_audit(
            train_y, cal_y, np.asarray(model.predict(cal_x)),
            train_provenance, cal_provenance,
        ),
        "training_seeds": sorted(training),
        "calibration_seeds": sorted(calibration),
        "evaluation_seeds_excluded": sorted(evaluation),
        "training_budget": {"unit": "committed_candidate_windows", "value": len(train_x)},
        "feature_time": "pre_transition_state_and_recorded_future_controls",
        "label_window": {"horizon_steps": args.horizon_steps,
                         "aggregation": "initial_instantaneous_max_event_top20pct_post_balance_mean_endpoint_force_error",
                         "initial_state": "include_observed_decision_endpoint_omit_unobserved_reset",
                         "execution_scope": "normal_only_exclude_any_unload_transition_in_window",
                         "right_censoring": "omit_incomplete_windows",
                         "absorbing_success": "exclude_starts_retain_future_risks"},
        "training_label_coverage": _label_coverage(train_y),
        "calibration_label_coverage": _label_coverage(cal_y),
        "training_diagnostics": _model_diagnostics(model, train_x, train_y),
        "calibration_diagnostics": _model_diagnostics(model, cal_x, cal_y),
        "training_domains": sorted({str(row["suite"]) for row in train_provenance}),
        "calibration_domains": sorted({str(row["suite"]) for row in cal_provenance}),
        "walking_collection_contracts": sorted({
            row["task_contract"]["walk_success_mode"]
            for row in [*train_provenance, *cal_provenance]
            if row["task_contract"]["walk_success_mode"] is not None
        }),
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
        "fit_scope": metadata["fit_scope"],
        "performance_validated": metadata["performance_validated"],
        "promotion_eligible": metadata["promotion_eligible"],
        "promotion_blocking_reasons": metadata["promotion_blocking_reasons"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
