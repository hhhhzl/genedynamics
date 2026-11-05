"""Shared experiment-result auditing, aggregation, and equivalence utilities.

The runtime metric plugins live in :mod:`genedynamics.experiments.plugins.metrics`.
This module handles the other half of the experiment contract: validating paired
configs and persisted result trees, aggregating seed-level metrics, and comparing
unified-runner outputs with frozen legacy records. It contains no task execution
or solver logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


BUDGET_KEYS = ("Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init")


def resolved_suite_budgets(config: Any) -> Dict[str, Dict[str, Any]]:
    """Resolve per-condition episode lengths and shared planning budgets."""
    conditions = config.suites or [
        {"name": str(level)} for level in config.obstacle_levels
    ]
    budgets = {}
    for suite in conditions:
        resolved = config.for_suite(suite)
        budgets[str(suite["name"])] = {
            "n_steps": resolved.n_steps,
            **{key: resolved.method_params.get(key) for key in BUDGET_KEYS},
        }
    return budgets


_CONTACT_TASKS = {
    "manipulator_surface_scan", "manipulator_peg_insert",
    "humanoid_box_push",
}

# Result trees produced before the public algorithm rename remain immutable.
# Canonicalize only while reading them so old evidence can still be summarized
# together with new MGA runs without rewriting provenance on disk.
_LEGACY_RESULT_ALIASES = {
    "full_mdac": "mga",
    "model_based_mdac": "model_based_only",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_checkpoint_lock(
    task: str,
    task_records: Sequence[Tuple[Path, Any]],
    errors: List[str],
    warnings: List[str],
    *,
    verify_artifacts: bool = True,
) -> None:
    """Validate the P8 frozen learned-component contract for one task."""
    if task not in _CONTACT_TASKS:
        return
    reference_path, reference = task_records[0]
    metadata = dict(reference.metadata or {})
    lock_status = str(metadata.get("learned_component_lock_status", "frozen"))
    if lock_status.startswith("pending"):
        errors.append(
            f"{task}: learned component lock is {lock_status}; "
            "fresh training/calibration must finish before formal execution"
        )
        return
    overlap_waiver = dict(metadata.get("formal_seed_overlap_waiver") or {})
    overlap_accepted = overlap_waiver.get("accepted") is True
    if overlap_accepted and not str(overlap_waiver.get("reason", "")).strip():
        errors.append(
            f"{reference_path}: accepted formal_seed_overlap_waiver needs a reason"
        )
    if task == "humanoid_box_push":
        if metadata.get("evidence_scope") != "full_eight_algorithm":
            errors.append(
                f"{reference_path}: H1 must declare "
                "evidence_scope=full_eight_algorithm"
            )

    lock = dict(metadata.get("learned_component_lock") or {})
    if not lock:
        errors.append(f"{reference_path}: missing P8 learned_component_lock")
        return
    formal_seeds = {int(seed) for seed in reference.seeds}
    required = {
        "path", "sha256", "protocol", "training_seeds", "training_budget",
        "training_domains", "architecture", "selection_rule",
    }
    for name, raw in lock.items():
        entry = dict(raw or {})
        missing = sorted(required.difference(entry))
        if missing:
            errors.append(
                f"{reference_path}: lock {name!r} misses {missing}"
            )
            continue
        if verify_artifacts:
            checkpoint = Path(str(entry["path"]))
            if not checkpoint.is_file():
                errors.append(f"{reference_path}: missing locked checkpoint {checkpoint}")
            else:
                actual = _file_sha256(checkpoint)
                if actual != str(entry["sha256"]):
                    errors.append(
                        f"{reference_path}: checkpoint hash mismatch for {name}: "
                        f"{actual} != {entry['sha256']}"
                    )
        used_seeds = {
            int(seed) for seed in (
                list(entry.get("training_seeds") or ())
                + list(entry.get("calibration_seeds") or ())
            )
        }
        leaked = sorted(formal_seeds.intersection(used_seeds))
        if leaked:
            message = (
                f"{task}: locked component {name} uses formal evaluation "
                f"seeds {leaked}"
            )
            if overlap_accepted:
                warnings.append(f"{message} (explicitly waived)")
            else:
                errors.append(message)
        budget = dict(entry.get("training_budget") or {})
        if not budget.get("unit") or int(budget.get("value", 0)) <= 0:
            errors.append(
                f"{reference_path}: {name} has invalid training_budget"
            )

    full = next((cfg for _, cfg in task_records if cfg.name == "mga"), None)
    standalone = next(
        (cfg for _, cfg in task_records if cfg.name == "standalone_rl"), None
    )
    if full is not None and standalone is not None:
        full_policy = (
            full.method_params.get("policy_ckpt_by_suite")
            or full.method_params.get("policy_ckpt")
        )
        standalone_policy = (
            standalone.method_params.get("policy_ckpt_by_suite")
            or standalone.method_params.get("policy_ckpt")
        )
        if full_policy != standalone_policy:
            errors.append(
                f"{task}: MGA and standalone RL do not share one checkpoint"
            )

    if task == "humanoid_box_push":
        lock_paths = {
            str(dict(entry or {}).get("path")): name
            for name, entry in lock.items()
        }
        required_names = {
            "mga", "model_based_only", "standalone_rl", "dial",
            "mppi", "pegasusflow", "issa", "atacom",
            "no_rl_prior", "no_learned_reliability",
            "no_controllability_geometry",
            "no_retraction", "no_stiffness",
        }
        actual_names = {cfg.name for _, cfg in task_records}
        missing_names = sorted(required_names.difference(actual_names))
        if missing_names:
            errors.append(f"{task}: missing formal configs {missing_names}")
        for path, cfg in task_records:
            for key in ("policy_ckpt", "atacom_policy_ckpt", "reliability_ckpt"):
                configured = cfg.method_params.get(key)
                if configured and str(configured) not in lock_paths:
                    errors.append(f"{path}: unlocked H1 checkpoint {configured}")
                mapping = cfg.method_params.get(f"{key}_by_suite") or {}
                for suite, checkpoint in mapping.items():
                    if str(checkpoint) not in lock_paths:
                        errors.append(
                            f"{path}: unlocked H1 checkpoint for {suite}: {checkpoint}"
                        )
        return

    for path, cfg in task_records:
        # Development/diagnostic configs may intentionally reference an
        # unpromoted candidate.  The learned-component lock governs canonical
        # paper runs only; development artifacts remain auditable through
        # their own provenance without becoming formal dependencies.
        if not (cfg.metadata or {}).get("formal_experiment", True):
            continue
        controller = str(
            cfg.method_params.get("controller_method", cfg.method)
        )
        bindings = {
            "atacom_policy_ckpt": "atacom_policy",
            "reliability_ckpt": "reliability",
        }
        if cfg.method_params.get("policy_ckpt"):
            bindings["policy_ckpt"] = (
                "atacom_policy" if controller == "atacom" else "policy"
            )
        for key, lock_name in bindings.items():
            configured = cfg.method_params.get(key)
            if not configured:
                continue
            entry = dict(lock.get(lock_name) or {})
            if str(configured) != str(entry.get("path")):
                errors.append(
                    f"{path}: {key}={configured!r} is not the locked "
                    f"{lock_name} checkpoint"
                )


def aggregate_receding_diagnostics(
    infos: Sequence[Any], task: str | None = None,
) -> Dict[str, float]:
    """Aggregate controller diagnostics persisted by a receding rollout."""
    rows = [item for item in infos if isinstance(item, dict)]

    def has(key: str) -> bool:
        return any(key in row for row in rows)

    def applicable(row: Dict[str, Any], flag: str | None) -> bool:
        if flag is None or flag not in row:
            return True  # Frozen tasks/archives did not publish applicability.
        value = np.asarray(row[flag])
        if value.shape != () or value.item() not in (0, 1):
            raise ValueError(f"invalid diagnostic applicability: {flag}")
        return bool(value.item())

    def eligible(key: str, flag: str | None = None):
        return [row for row in rows if key in row and applicable(row, flag)]

    def mean_scalar(key: str, flag: str | None = None) -> float:
        values = [np.asarray(row[key], dtype=float).reshape(-1)
                  for row in eligible(key, flag)]
        return float(np.mean(np.concatenate(values))) if values else float("nan")

    out: Dict[str, float] = {}
    if has("prior_accepted"):
        out["prior_acceptance_rate"] = mean_scalar("prior_accepted")
        comparison_flag = (
            "prior_comparison_applicable" if has("prior_comparison_applicable") else None
        )
        if comparison_flag is not None:
            out["prior_comparison_count"] = float(len(eligible(
                "prior_predicted_improvement", comparison_flag
            )))
        if comparison_flag is None or eligible("prior_predicted_improvement", comparison_flag):
            out["prior_predicted_improvement_mean"] = mean_scalar(
                "prior_predicted_improvement", comparison_flag
            )
        out.update({
            "prior_risk_ok_rate": mean_scalar("prior_risk_ok"),
            "prior_force_veto_rate": mean_scalar("prior_force_veto"),
        })
        learned_flag = (
            "reliability_incumbent_applicable"
            if has("reliability_incumbent_applicable") else None
        )
        if learned_flag is not None:
            out["reliability_incumbent_count"] = float(len(eligible(
                "reliability_risk_incumbent", learned_flag
            )))
        additive_flag = (
            "additive_prior_comparison_applicable"
            if has("additive_prior_comparison_applicable") else None
        )
        if additive_flag is not None:
            out["additive_prior_comparison_count"] = float(len(eligible(
                "additive_prior_improvement", additive_flag
            )))
        out["prior_fallback_rate"] = 1.0 - out["prior_acceptance_rate"]
        insertion = task == "manipulator_peg_insert"
        labels = (
            ("force_violation", "torque_violation", "jam", "force_mae")
            if insertion else
            ("force_violation", "invalid_contact", "balance", "force_mae")
            if task == "humanoid_box_push" else
            ("force_violation", "contact_loss", "deformation", "force_mae")
        )
        for source in (
            "prior_risk_rl", "prior_risk_incumbent", "prior_risk_refined",
            "prior_risk_atacom", "prior_risk_emergency",
            "reliability_risk_incumbent", "reliability_risk_refined",
        ):
            flag = learned_flag if source == "reliability_risk_incumbent" else None
            vectors = [np.asarray(row[source], dtype=float).reshape(-1)
                       for row in eligible(source, flag)]
            if vectors:
                mean = np.mean(np.stack(vectors), axis=0)
                out.update({
                    f"{source}_{label}": float(value)
                    for label, value in zip(labels, mean)
                })
        for key in (
            "incumbent_revalidated_safe", "refined_revalidated_safe",
            "emergency_selected", "emergency_revalidated_safe",
            "emergency_incumbent_active", "selected_revalidated_safe",
            "emergency_unrecoverable", "reliability_support_incumbent",
            "reliability_support_refined", "reliability_abstained",
            "proposal_stochastic_count",
            "proposal_atacom_count", "proposal_logp_mean",
            "proposal_gaussian_best_reward", "proposal_rl_best_reward",
            "proposal_atacom_best_reward", "proposal_gaussian_weight",
            "proposal_rl_weight", "proposal_atacom_weight",
            "atacom_incumbent_selected", "prior_score_atacom",
            "additive_prior_selected", "additive_prior_candidate_count",
            "additive_prior_any_safe", "additive_prior_improvement",
            "additive_prior_score", "additive_gaussian_score",
            "emergency_fallback_recovery", "emergency_validation_candidate_count",
        ):
            flag = (
                learned_flag if key == "reliability_support_incumbent"
                else additive_flag if key == "additive_prior_improvement" else None
            )
            if has(key) and (flag is None or eligible(key, flag)):
                out[key] = mean_scalar(key, flag)
    if has("failure"):
        out.update({
            "issa_projection_failure_rate": mean_scalar("failure"),
            "issa_intervention_mean": mean_scalar("intervention"),
            "issa_projected_margin_mean": mean_scalar("margin"),
        })
    if has("atacom_slack_norm"):
        out.update({
            "atacom_slack_norm_mean": mean_scalar("atacom_slack_norm"),
            "atacom_action_norm_mean": mean_scalar("atacom_action_norm"),
        })
    return out


def _config_paths(roots: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in roots:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(
                p for p in path.rglob("*.yaml") if not p.name.startswith("_")
            ))
        elif path.suffix in {".yaml", ".yml"}:
            paths.append(path)
    return sorted(set(p.resolve() for p in paths))


def audit_configs(
    roots: Sequence[str], *, verify_artifacts: bool = True
) -> Dict[str, Any]:
    """Check paired protocol fields and, optionally, locked artifacts.

    ``verify_artifacts=False`` is intended for source distributions that
    publish reproducibility configs while keeping large trained checkpoints
    outside the package.
    """
    from genedynamics.experiments.framework.config import ExperimentConfig

    records = []
    errors: List[str] = []
    warnings: List[str] = []
    for path in _config_paths(roots):
        try:
            cfg = ExperimentConfig.from_yaml(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        validation = cfg.validate()
        errors.extend(f"{path}: {error}" for error in validation)
        if validation:
            continue
        records.append((path, cfg))
    by_task: Dict[str, List[Tuple[Path, ExperimentConfig]]] = defaultdict(list)
    for record in records:
        by_task[record[1].env_name].append(record)
    run_count_by_task: Dict[str, int] = {}
    for task, task_records in by_task.items():
        formal_task_records = [
            (path, cfg) for path, cfg in task_records
            if (cfg.metadata or {}).get("formal_experiment", True)
        ]
        reference = (
            formal_task_records[0][1]
            if formal_task_records else task_records[0][1]
        )
        full_cfg = next(
            (cfg for _, cfg in task_records if cfg.name == "mga"), None
        )
        ref_protocol = {
            "seeds": reference.seeds,
            "n_steps": reference.n_steps,
            "suites": reference.suites,
            "budget": {key: reference.method_params.get(key) for key in BUDGET_KEYS},
            "resolved_suite_budgets": resolved_suite_budgets(reference),
        }
        names = set()
        for path, cfg in task_records:
            if cfg.name in names:
                errors.append(f"{task}: duplicate algorithm name {cfg.name!r}")
            names.add(cfg.name)
            is_formal = (cfg.metadata or {}).get("formal_experiment", True)
            if is_formal:
                current = {
                    "seeds": cfg.seeds,
                    "n_steps": cfg.n_steps,
                    "suites": cfg.suites,
                    "budget": {
                        key: cfg.method_params.get(key) for key in BUDGET_KEYS
                    },
                    "resolved_suite_budgets": resolved_suite_budgets(cfg),
                }
                if current != ref_protocol:
                    errors.append(
                        f"{path}: paired protocol differs from "
                        f"{formal_task_records[0][0]}"
                    )
            controller = cfg.method_params.get("controller_method", cfg.method)
            if cfg.method in {"mppi", "pegasusflow", "issa", "atacom", "standalone_rl"}:
                if controller.startswith("mga") or controller == "dial":
                    errors.append(f"{path}: baseline routes through {controller!r}")
            ablated = (cfg.metadata or {}).get("ablated_component")
            if ablated and full_cfg is not None and ablated not in {
                "learned_reliability",
            }:
                from genedynamics.solvers.single.mga.core.method_registry import (
                    METHOD_TABLE, diff_flags, resolve_method,
                )
                full_controller = str(full_cfg.method_params.get(
                    "controller_method", full_cfg.method,
                ))
                if controller in METHOD_TABLE and full_controller in METHOD_TABLE:
                    differences = diff_flags(
                        resolve_method(full_controller), resolve_method(controller)
                    )
                    if differences != (str(ablated),):
                        errors.append(
                            f"{path}: ablation {ablated!r} changes flags "
                            f"{differences!r}"
                        )
            available_suites = {
                str(item.get("name")) for item in cfg.suites
            }
            formal_suites = list(
                (cfg.metadata or {}).get("formal_suites")
                or sorted(available_suites)
            )
            unknown = sorted(set(formal_suites).difference(available_suites))
            if unknown:
                errors.append(f"{path}: unknown formal_suites {unknown}")
            # Diagnostic controllers remain available through the same runner
            # and config inheritance, but are not silently counted as paper
            # experiments.  The default is formal for every existing task, so
            # this opt-out does not alter legacy matrices.
            if is_formal:
                run_count_by_task[task] = run_count_by_task.get(task, 0) + (
                    len(cfg.seeds) * len(formal_suites)
                )
        _audit_checkpoint_lock(
            task,
            task_records,
            errors,
            warnings,
            verify_artifacts=verify_artifacts,
        )
        expected = (reference.metadata or {}).get("expected_formal_task_runs")
        if expected is not None and run_count_by_task[task] != int(expected):
            errors.append(
                f"{task}: formal matrix has {run_count_by_task[task]} runs, "
                f"expected {expected}"
            )
    return {
        "ok": not errors,
        "config_count": len(records),
        "task_count": len(by_task),
        "run_count_by_task": run_count_by_task,
        "formal_run_count": sum(run_count_by_task.values()),
        "warnings": warnings,
        "errors": errors,
    }


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[Tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(child, child_prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            yield prefix, number


def _bootstrap_mean(values: np.ndarray, seed: int = 20260824) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def _paired_sign_flip_p(delta: np.ndarray, seed: int = 20260824) -> float:
    """Two-sided paired randomization test under exchangeability."""
    if delta.size == 0:
        return float("nan")
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(10000, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def paired_block_bootstrap(
    values: np.ndarray, *, resamples: int = 10000, seed: int = 267100,
) -> np.ndarray:
    """Resample crossed instance/request units, retaining modes in each block.

    Axes: instance, fixed stratum, physical initial block, mode, metric.
    Request draws are shared across sampled instances; otherwise a common
    evaluation cohort would incorrectly become independent reset cohorts.
    Returned axes: bootstrap draw, stratum, mode, metric.  A deterministic
    comparator is broadcast before subtraction, never resampled as trained
    instances.  No physics or learned runtime is imported here.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 5 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("expected finite instance/stratum/initial/mode/metric tensor")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    instances, strata, initials, _, _ = values.shape
    rng = np.random.default_rng(seed)
    result = []
    # Bound temporary memory even for a larger experiment matrix.
    for start in range(0, resamples, 200):
        count = min(200, resamples - start)
        instance_draw = rng.integers(instances, size=(count, instances))
        initial_draw = rng.integers(initials, size=(count, strata, initials))
        sampled = values[
            instance_draw[:, :, None, None],
            np.arange(strata)[None, None, :, None],
            initial_draw[:, None, :, :],
        ]
        result.append(sampled.mean(axis=(1, 3)))
    return np.concatenate(result)


def exact_instance_sign_flip(delta: np.ndarray) -> float | None:
    """Supplemental two-sided symmetry test, conditional on evaluated requests.

    This is not a randomized treatment test or a replacement for the crossed
    bootstrap.  Each whole trained/calibrated instance gets one sign.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not np.isfinite(delta).all():
        raise ValueError("instance differences must be a finite vector")
    if len(delta) < 2:
        return None
    if len(delta) > 16:
        raise ValueError("exact sign flip is limited to 16 independent instances")
    masks = np.arange(2 ** len(delta))[:, None]
    signs = 2 * ((masks >> np.arange(len(delta))) & 1) - 1
    null = np.abs((signs * delta).mean(axis=1))
    return float(np.mean(null >= abs(delta.mean()) - 1e-12))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm adjusted p-values in the original order, including ties."""
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("invalid p-value")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    if len(values):
        adjusted[order] = np.minimum(
            1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
        )
    return adjusted.tolist()


_OUTCOME_METRICS = (
    "constrained_success", "constraint_violation",
    "capped_normalized_completion_time",
)
_TABLE_METRICS = (
    "constrained_success_rate", "constraint_violation_rate",
    "capped_normalized_completion_time", "worst_group_constrained_success",
)


def _outcome_tensor(rows: Sequence[dict]) -> tuple[np.ndarray, dict]:
    """Validate a balanced outcome cell and collapse non-training aliases."""
    roles = {row["instance_role"] for row in rows}
    if len(roles) != 1:
        raise ValueError("a cell must have one explicit instance role")
    learned = roles == {"independent_training_or_calibration_instance"}
    rejected = all(row["predeployment_rejection"] == 1 for row in rows)
    if not learned and not rejected and roles != {"deterministic_single_identity"}:
        raise ValueError("unrecognized independent-unit definition")
    instances = sorted({row["training_or_calibration_instance"] for row in rows})
    strata = sorted({(r["mechanism_group"], r["body_support_cell"]) for r in rows})
    initials = sorted({int(r["physical_initial_slot"]) for r in rows})
    modes = sorted({int(r["requested_mode"]) for r in rows})
    lookup = {}
    for row in rows:
        key = (row["training_or_calibration_instance"],
               (row["mechanism_group"], row["body_support_cell"]),
               int(row["physical_initial_slot"]), int(row["requested_mode"]))
        if key in lookup:
            raise ValueError("duplicate instance/request outcome")
        lookup[key] = [float(row[key]) for key in _OUTCOME_METRICS]
    if len(lookup) != len(instances) * len(strata) * len(initials) * len(modes):
        raise ValueError("unbalanced outcome cell; never silently drop pairs")
    tensor = np.asarray([
        [[[lookup[instance, stratum, initial, mode] for mode in modes]
          for initial in initials] for stratum in strata] for instance in instances
    ])
    if not learned:
        if not np.array_equal(tensor, np.broadcast_to(tensor[0], tensor.shape)):
            raise ValueError("non-training slots are not identical scoring aliases")
        tensor = tensor[:1]
    return tensor, {
        "independent_training_instances": len(instances) if learned else 0,
        "deterministic_identities": int(not learned and not rejected),
        "planned_instance_slots": len(instances),
        "all_predeployment_rejected": rejected,
        "instance_ids": instances if learned else [],
        "strata": strata, "initial_blocks_per_stratum": len(initials),
        "initial_ids": initials, "modes": modes,
    }


def _outcome_statistics(group_means: np.ndarray) -> np.ndarray:
    """Last axes are mechanism/support, mode, metric; worst is their joint min."""
    return np.concatenate((
        group_means.mean(axis=(-3, -2)),
        group_means[..., 0].min(axis=(-2, -1))[..., None],
    ), axis=-1)


def _interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def _select_strongest_baseline(points: dict[str, np.ndarray]) -> str:
    """Apply the frozen success/safety/speed/ascending-id ordering."""
    if not points:
        raise ValueError("at least one strong baseline is required")
    return min(
        points,
        key=lambda method: (
            -float(points[method][0]),
            float(points[method][1]),
            float(points[method][2]),
            method,
        ),
    )


def _alias_specs(aliases: dict[str, Any], tasks: Sequence[str]) -> list[dict[str, Any]]:
    """Normalize legacy all-task aliases and task-scoped alias declarations."""
    known_tasks = set(tasks)
    specs = []
    for main, declaration in aliases.items():
        if isinstance(declaration, str):
            spec = {
                "main": main, "ablation": declaration, "tasks": list(tasks),
                "tolerated_failure_categories": [],
            }
        elif isinstance(declaration, dict):
            allowed = {"method", "tasks", "tolerated_failure_categories"}
            unexpected = set(declaration) - allowed
            if unexpected:
                raise ValueError(f"unexpected alias fields: {sorted(unexpected)}")
            spec = {
                "main": main,
                "ablation": declaration["method"],
                "tasks": list(declaration.get("tasks", tasks)),
                "tolerated_failure_categories": list(
                    declaration.get("tolerated_failure_categories", ())
                ),
            }
        else:
            raise TypeError("alias declaration must be a method name or mapping")
        unknown = set(spec["tasks"]) - known_tasks
        if unknown:
            raise ValueError(f"alias names unknown tasks: {sorted(unknown)}")
        specs.append(spec)
    return specs


def summarize_confirmation(
    root: Path, source: Path, output: Path, *, references: dict[str, str],
    ablation_terms: dict[str, dict[str, float]], aliases: dict[str, str],
    permission_classes: dict[str, str],
    resamples: int = 10000, seed: int = 267100,
    contract_limitations: Sequence[str] = (),
    protocol_sources: Sequence[dict] = (),
) -> dict:
    """Independently audit saved outcomes and write new CPU-only statistics.

    Does not call the execution aggregator, change its SHA-bound artifacts, or
    certify physical predicates that were not persisted.  References and named
    scientific contrasts are supplied by the paper entrypoint.
    """
    root = root.resolve()
    source = (root / source).resolve()
    output = (root / output).resolve()
    if output.exists():
        raise FileExistsError(f"derived output must be a new directory: {output}")
    launch_path, audit_path = source / "launch.json", source / "audit.json"
    launch = json.loads(launch_path.read_text())
    audit = json.loads(audit_path.read_text())
    if launch["status"] != "complete" or not audit["passed"]:
        raise ValueError("input confirmation is not complete and execution-audited")
    inputs = {}

    def checked(label, pointer):
        path = (root / pointer["path"]).resolve()
        actual = _file_sha256(path)
        if actual != pointer["sha256"]:
            raise ValueError(f"input SHA256 mismatch: {label}")
        inputs[label] = {"path": str(path.relative_to(root)), "sha256": actual}
        return path

    for label, pointer in audit["artifacts"].items():
        checked(f"audit.{label}", pointer)
    paths = {
        label: checked(label, launch[label])
        for label in ("outcomes", "aggregates", "state_receipts", "package_report")
    }
    paths.update({
        label: checked(label, pointer)
        for label, pointer in launch["sealed_artifacts"].items()
    })
    package = json.loads(paths["package_report"].read_text())
    claim_contract = package.get("transport_claim_contract")
    if (
        package.get("version") == "four-task-transfer-confirmation-package-v11"
        and claim_contract is None
    ):
        raise ValueError("confirmation package has no frozen transport claim contract")
    if claim_contract is not None and (
        not isinstance(claim_contract, dict)
        or claim_contract.get("version")
        != "four-task-stable-transport-and-strong-baseline-v1"
    ):
        raise ValueError("confirmation package has an invalid transport claim contract")
    inputs["launch"] = {"path": str(launch_path.relative_to(root)),
                        "sha256": _file_sha256(launch_path)}
    inputs["execution_audit"] = {"path": str(audit_path.relative_to(root)),
                                 "sha256": _file_sha256(audit_path)}
    protocol_evidence = []
    for index, pointer in enumerate(protocol_sources):
        path = checked(f"protocol.{index}", pointer)
        evidence = dict(pointer)
        if pointer.get("member"):
            import zipfile

            with zipfile.ZipFile(path) as archive:
                data = archive.read(pointer["member"])
            if hashlib.sha256(data).hexdigest() != pointer["member_sha256"]:
                raise ValueError("archived protocol member SHA256 mismatch")
        protocol_evidence.append(evidence)

    def read_lines(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    rows = read_lines(paths["outcomes"])
    schedule = read_lines(paths["evaluation_schedule"])
    roster_rows = read_lines(paths["request_roster"])
    roster = {r["request_id"]: r for r in roster_rows}
    if len(roster) != len(roster_rows):
        raise ValueError("duplicate roster request")
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {r["schedule_id"] for r in schedule}:
        raise ValueError("outcomes do not uniquely cover the frozen schedule")
    if len(schedule) != len(rows) or len(rows) != launch["completed_records"]:
        raise ValueError("schedule/launch count mismatch")
    if sorted(r["ordinal"] for r in rows) != list(range(1, len(rows) + 1)):
        raise ValueError("outcome ordinals are not a complete unique sequence")
    for scheduled in schedule:
        row = by_id[scheduled["schedule_id"]]
        for key in ("family", "method", "task"):
            if row[key] != scheduled[key]:
                raise ValueError(f"schedule identity mismatch: {key}")
        if row["base_request_id"] != scheduled["request_id"]:
            raise ValueError("scheduled request identity mismatch")
        scheduled_instance = scheduled.get("instance") or scheduled.get("training_or_calibration_instance")
        if scheduled_instance and row["training_or_calibration_instance"] != scheduled_instance:
            raise ValueError("scheduled instance identity mismatch")
        request = roster[row["base_request_id"]]
        for key in ("task", "mechanism_group", "body_support_cell",
                    "requested_mode", "physical_initial_slot"):
            if row[key] != request[key]:
                raise ValueError(f"roster metadata mismatch: {key}")
        if row["confirmation_seed"] != request["reset_seed"]:
            raise ValueError("confirmation seed mismatch")
        scores = [row[m] for m in _OUTCOME_METRICS]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in scores):
            raise ValueError("invalid bounded metric")
        rejected = bool(row["predeployment_rejection"])
        if rejected and (scores != [0, 1, 1] or row["target_physical_execution"]):
            raise ValueError("rejection penalty/execution mismatch")
        if not rejected and not row["target_physical_execution"]:
            raise ValueError("non-rejected result has no physical execution")
        if row["constrained_success"] and (row["constraint_violation"] or rejected):
            raise ValueError("successful outcome conflicts with penalty")
    receipts = read_lines(paths["state_receipts"])
    physical_ids = {r["record_id"] for r in rows if r["target_physical_execution"]}
    if len(physical_ids) != launch["physical_executions"]:
        raise ValueError("launch physical count differs from outcomes")
    if len(receipts) != len(physical_ids) or {r["record_id"] for r in receipts} != physical_ids:
        raise ValueError("physical receipts do not uniquely cover execution")
    reset_hashes = defaultdict(set)
    for receipt in receipts:
        if receipt["written_before_first_actuation"] is not True:
            raise ValueError("receipt was not persisted before action")
        row = by_id[receipt["record_id"]]
        for key in ("base_request_id", "task", "family", "method", "instance_role",
                    "training_or_calibration_instance", "requested_mode", "body_support_cell",
                    "mechanism_group", "physical_initial_slot", "confirmation_seed"):
            if receipt[key] != row[key]:
                raise ValueError(f"receipt identity mismatch: {key}")
        reset_hashes[receipt["base_request_id"]].add(receipt["post_reset_state_sha256"])
    if any(len(hashes) != 1 for hashes in reset_hashes.values()):
        raise ValueError("paired requests have inconsistent reset states")

    cells = defaultdict(list)
    for row in rows:
        cells[row["family"], row["method"], row["task"]].append(row)
    old = {(r["family"], r["method"], r["task"]): r
           for r in json.loads(paths["aggregates"].read_text())}
    if set(cells) != set(old):
        raise ValueError("stored aggregate identities differ from raw outcomes")
    summaries, groups, corrections, tensors, draws, metadata = [], [], [], {}, {}, {}
    for key, items in sorted(cells.items()):
        tensor, meta = _outcome_tensor(items)
        tensors[key], metadata[key] = tensor, meta
        bootstrap = paired_block_bootstrap(tensor, resamples=resamples, seed=seed)
        draws[key] = _outcome_statistics(bootstrap)
        point = _outcome_statistics(tensor.mean(axis=(0, 2)))
        count = len(items)
        physical = [r for r in items if r["target_physical_execution"]]
        summary = dict(zip(("family", "method", "task"), key))
        summary.update(meta)
        summary.update({
            "requested_episodes": count, "unique_requests": len({r["base_request_id"] for r in items}),
            "target_physical_execution_count": len(physical),
            "successes": sum(r["constrained_success"] for r in items),
            "predeployment_rejection_rate": sum(r["predeployment_rejection"] for r in items) / count,
            "executed_constraint_violation_rate": (
                sum(r["constraint_violation"] for r in physical) / len(physical)
                if physical else None
            ),
            "outcomes": dict(Counter(r["failure_category"] for r in items)),
            "rejection_reasons": dict(Counter(r["rejection_reason"] for r in items
                                             if r["predeployment_rejection"])),
            "metric_scope": "recorded_evaluator_scores",
        })
        for index, metric in enumerate(_TABLE_METRICS):
            summary[metric] = float(point[index])
            summary[f"{metric}_ci95"] = (
                None if meta["all_predeployment_rejected"] else _interval(draws[key][:, index])
            )
            previous = old[key][metric]
            if not np.isclose(previous, point[index], atol=1e-12, rtol=0):
                if metric != "worst_group_constrained_success":
                    raise ValueError(f"unexpected aggregate discrepancy: {key}/{metric}")
                corrections.append({**dict(zip(("family", "method", "task"), key)),
                                    "metric": metric, "original": previous,
                                    "corrected": float(point[index]),
                                    "reason": "joint mechanism/support/requested-mode groups"})
        for metric in ("requested_episodes", "target_physical_execution_count",
                       "predeployment_rejection_rate", "outcomes"):
            if summary[metric] != old[key][metric]:
                raise ValueError(f"independent count audit failed: {key}/{metric}")
        summaries.append(summary)
        grouped = defaultdict(list)
        for row in items:
            grouped[row["mechanism_group"], row["body_support_cell"], row["requested_mode"]].append(row)
        for group, members in sorted(grouped.items()):
            physical_group = [r for r in members if r["target_physical_execution"]]
            group_row = {**dict(zip(("family", "method", "task"), key)),
                         **dict(zip(("mechanism_group", "body_support_cell", "requested_mode"), group)),
                         "requested_episodes": len(members),
                         "physical_executions": len(physical_group),
                         "successes": sum(r["constrained_success"] for r in members),
                         "outcomes": dict(Counter(r["failure_category"] for r in members))}
            for name in ("physical_control_steps", "final_position_error_m", "force_peak_N",
                         "continuous_guide_contact_ratio"):
                values = [r[name] for r in physical_group if r.get(name) is not None]
                group_row[f"{name}_min_median_max"] = (
                    [min(values), float(np.median(values)), max(values)] if values else None
                )
            groups.append(group_row)

    alias_checks = []
    compared_fields = (*_OUTCOME_METRICS, "predeployment_rejection", "action_stream_sha256",
                       "physical_control_steps", "failure_category")
    tasks = sorted({r["task"] for r in rows})
    for spec in _alias_specs(aliases, tasks):
        main, ablation = spec["main"], spec["ablation"]
        tolerated = set(spec["tolerated_failure_categories"])
        for task in spec["tasks"]:
            left = cells["main", main, task]
            right = {(r["training_or_calibration_instance"], r["base_request_id"]): r
                     for r in cells["ablation", ablation, task]}
            if len(left) != len(right):
                raise ValueError("main/ablation shared-method measurements disagree")
            tolerated_pairs = 0
            identical_pairs = 0
            for row in left:
                key = row["training_or_calibration_instance"], row["base_request_id"]
                if key not in right:
                    raise ValueError("main/ablation shared-method measurements disagree")
                paired = right[key]
                if row["failure_category"] in tolerated or paired["failure_category"] in tolerated:
                    tolerated_pairs += 1
                    continue
                if any(row.get(field) != paired.get(field) for field in compared_fields):
                    raise ValueError("main/ablation shared-method measurements disagree")
                identical_pairs += 1
            alias_checks.append({"main": main, "ablation": ablation, "task": task,
                                 "identical_measurements": identical_pairs,
                                 "tolerated_failure_mismatches": tolerated_pairs,
                                 "tolerated_failure_categories": sorted(tolerated),
                                 "independent_extra_evidence": False})

    contrasts = []
    terms_by_family = {
        "main": {f"{references['main']}_minus_{method}": {references["main"]: 1, method: -1}
                 for method in sorted({k[1] for k in cells if k[0] == "main"})
                 if method != references["main"]},
        "ablation": ablation_terms,
    }
    for family, named_terms in terms_by_family.items():
        for name, terms in named_terms.items():
            for task in tasks:
                keys = [(family, method, task) for method in terms]
                metas = [metadata[key] for key in keys]
                instance_sets = {tuple(m["instance_ids"]) for m in metas if m["instance_ids"]}
                if len(instance_sets) > 1 or len({str((m["strata"], m["modes"], m["initial_ids"])) for m in metas}) != 1:
                    raise ValueError("contrast lacks exactly matched instances/requests")
                count = max(tensors[key].shape[0] for key in keys)
                coefficients = list(terms.values())
                delta_tensor = sum(coef * tensors[key] for coef, key in zip(coefficients, keys))
                # Bootstrap linear metrics as paired differences, so fixed references
                # share the same request draws with every learned instance.
                paired_draws = paired_block_bootstrap(delta_tensor, resamples=resamples, seed=seed)
                linear_draws = paired_draws.mean(axis=(1, 2))
                estimates = sum(coef * _outcome_statistics(tensors[key].mean(axis=(0, 2)))
                                for coef, key in zip(coefficients, keys))
                # The worst-group contrast is difference of minima, NOT min difference.
                joint_draws = [paired_block_bootstrap(
                    np.broadcast_to(tensors[key], (count, *tensors[key].shape[1:])),
                    resamples=resamples, seed=seed,
                ) for key in keys]
                worst_draws = sum(coef * _outcome_statistics(draw)[:, 3]
                                  for coef, draw in zip(coefficients, joint_draws))
                is_learned = any(m["independent_training_instances"] for m in metas)
                all_rejected = all(m["all_predeployment_rejected"] for m in metas)
                permissions = {method: permission_classes[method] for method in terms}
                same_permission = len(set(permissions.values())) == 1
                for index, metric in enumerate(_TABLE_METRICS):
                    row = {"family": family, "contrast": name, "task": task,
                           "metric": metric, "terms": terms,
                           "estimate": float(estimates[index]),
                           "ci95": None if all_rejected else _interval(
                               linear_draws[:, index] if index < 3 else worst_draws),
                           "independent_training_instances": count if is_learned else 0,
                           "permission_classes": permissions,
                           "comparison_scope": "same_permission" if same_permission else "cross_permission_descriptive_only",
                           "fixed_reference_broadcast_is_not_replication": any(not m["instance_ids"] for m in metas),
                           "supplemental_instance_sign_flip_p": exact_instance_sign_flip(
                               delta_tensor[..., index].mean(axis=(1, 2, 3))
                           ) if index < 3 and is_learned and same_permission else None,
                           "inference_scope": "recorded_scores_fixed_mechanism_support_mixture"}
                    if index == 0 and len(terms) == 2 and sorted(coefficients) == [-1, 1]:
                        row["favorable_pairs"] = int(np.count_nonzero(delta_tensor[..., 0] > 0))
                        row["adverse_pairs"] = int(np.count_nonzero(delta_tensor[..., 0] < 0))
                    contrasts.append(row)
    for family in terms_by_family:
        valid = [row for row in contrasts if row["family"] == family
                 and row["supplemental_instance_sign_flip_p"] is not None]
        adjusted = holm_adjust([r["supplemental_instance_sign_flip_p"] for r in valid])
        for row, pvalue in zip(valid, adjusted):
            row["supplemental_holm_p"] = pvalue
            row["holm_family_size"] = len(valid)

    claim_evaluation = None
    if claim_contract is not None:
        claim_tasks = ["T1", "T2", "T3", "T4"]
        if sorted(tasks) != claim_tasks:
            raise ValueError("transport claim requires exactly T1 through T4")
        claim_reference = str(references["main"])
        claim_control = str(claim_contract["transport_control"])
        strong_baselines = list(claim_contract["strong_executable_baselines"])
        if (
            claim_reference != "cross_ebdm"
            or claim_control != "task_only"
            or strong_baselines != ["fada", "dial_transfer", "response_mpc"]
            or claim_contract.get("task_weighting")
            != "equal_macro_average_over_T1_T2_T3_T4"
        ):
            raise ValueError("transport claim identities or task weighting changed")

        def macro_method_point(method: str) -> np.ndarray:
            return np.mean([
                _outcome_statistics(
                    tensors["main", method, task].mean(axis=(0, 2))
                )
                for task in claim_tasks
            ], axis=0)

        def macro_pair(reference: str, comparator: str) -> dict[str, Any]:
            task_points = []
            task_draws = []
            success_pairs = {}
            for task_index, task in enumerate(claim_tasks):
                left_key = ("main", reference, task)
                right_key = ("main", comparator, task)
                left, right = tensors[left_key], tensors[right_key]
                left_meta, right_meta = metadata[left_key], metadata[right_key]
                learned_instance_sets = {
                    tuple(value) for value in (
                        left_meta["instance_ids"], right_meta["instance_ids"]
                    ) if value
                }
                if (
                    len(learned_instance_sets) > 1
                    or left_meta["strata"] != right_meta["strata"]
                    or left_meta["initial_ids"] != right_meta["initial_ids"]
                    or left_meta["modes"] != right_meta["modes"]
                ):
                    raise ValueError(
                        f"macro claim pair lacks matched units: {task}/{comparator}"
                    )
                count = max(left.shape[0], right.shape[0])
                left = np.broadcast_to(left, (count, *left.shape[1:]))
                right = np.broadcast_to(right, (count, *right.shape[1:]))
                left_success = left[..., 0]
                right_success = right[..., 0]
                success_pairs[task] = {
                    "favorable_pairs": int(np.count_nonzero(
                        (left_success > .5) & (right_success <= .5)
                    )),
                    "adverse_pairs": int(np.count_nonzero(
                        (left_success <= .5) & (right_success > .5)
                    )),
                    "tied_success_pairs": int(np.count_nonzero(
                        (left_success > .5) & (right_success > .5)
                    )),
                    "tied_failure_pairs": int(np.count_nonzero(
                        (left_success <= .5) & (right_success <= .5)
                    )),
                    "total_pairs": int(left_success.size),
                }
                task_points.append(
                    _outcome_statistics(left.mean(axis=(0, 2)))
                    - _outcome_statistics(right.mean(axis=(0, 2)))
                )
                task_seed = seed + 1009 * (task_index + 1)
                linear = paired_block_bootstrap(
                    left - right, resamples=resamples, seed=task_seed,
                ).mean(axis=(1, 2))
                left_draw = paired_block_bootstrap(
                    left, resamples=resamples, seed=task_seed,
                )
                right_draw = paired_block_bootstrap(
                    right, resamples=resamples, seed=task_seed,
                )
                worst = (
                    _outcome_statistics(left_draw)[:, 3]
                    - _outcome_statistics(right_draw)[:, 3]
                )
                task_draws.append(np.column_stack((linear, worst)))
            point = np.mean(task_points, axis=0)
            draws_macro = np.mean(task_draws, axis=0)
            return {
                "reference": reference,
                "comparator": comparator,
                "metrics": {
                    metric: {
                        "macro_delta": float(point[index]),
                        "paired_bootstrap_ci95": _interval(draws_macro[:, index]),
                    }
                    for index, metric in enumerate(_TABLE_METRICS)
                },
                "task_deltas": {
                    task: {
                        metric: float(task_points[task_index][metric_index])
                        for metric_index, metric in enumerate(_TABLE_METRICS)
                    }
                    for task_index, task in enumerate(claim_tasks)
                },
                "paired_success_outcomes": success_pairs,
            }

        transport_pair = macro_pair(claim_reference, claim_control)
        strong_pairs = {
            method: macro_pair(claim_reference, method)
            for method in strong_baselines
        }
        strong_points = {
            method: macro_method_point(method) for method in strong_baselines
        }
        selected_strongest = _select_strongest_baseline(strong_points)
        stable_gate = claim_contract["stable_transport_gate"]
        success_by_task = [
            transport_pair["task_deltas"][task]["constrained_success_rate"]
            for task in claim_tasks
        ]
        stable_checks = {
            "all_four_task_success_deltas_nonnegative": sum(
                value >= -1e-12 for value in success_by_task
            ) >= int(stable_gate["minimum_nonnegative_task_success_deltas"]),
            "at_least_one_task_success_delta_positive": sum(
                value > 1e-12 for value in success_by_task
            ) >= int(stable_gate["minimum_strictly_positive_task_success_deltas"]),
            "macro_success_delta_strictly_positive": (
                transport_pair["metrics"]["constrained_success_rate"]["macro_delta"]
                > float(stable_gate["minimum_macro_success_delta"]) + 1e-12
            ),
            "macro_constraint_delta_nonpositive": (
                transport_pair["metrics"]["constraint_violation_rate"]["macro_delta"]
                <= float(stable_gate["maximum_macro_constraint_violation_delta"]) + 1e-12
            ),
            "macro_capped_completion_delta_nonpositive": (
                transport_pair["metrics"]["capped_normalized_completion_time"]["macro_delta"]
                <= float(stable_gate["maximum_macro_capped_completion_time_delta"]) + 1e-12
            ),
            "transport_preserves_every_task_only_success": (
                not bool(stable_gate.get(
                    "require_transport_to_preserve_every_task_only_success", False,
                ))
                or all(
                    transport_pair["paired_success_outcomes"][task]["adverse_pairs"] == 0
                    for task in claim_tasks
                )
            ),
        }
        reference_permission = permission_classes[claim_reference]
        strong_checks = {
            "all_superiority_comparisons_same_permission": all(
                permission_classes[method] == reference_permission
                for method in strong_baselines
            ),
            "positive_macro_success_delta_against_every_strong_baseline": all(
                pair["metrics"]["constrained_success_rate"]["macro_delta"] > 1e-12
                for pair in strong_pairs.values()
            ),
            "nonpositive_macro_constraint_delta_against_every_strong_baseline": all(
                pair["metrics"]["constraint_violation_rate"]["macro_delta"] <= 1e-12
                for pair in strong_pairs.values()
            ),
            "selected_strongest_success_ci95_strictly_positive": (
                strong_pairs[selected_strongest]["metrics"][
                    "constrained_success_rate"
                ]["paired_bootstrap_ci95"][0] > 0.
            ),
        }
        claim_evaluation = {
            "version": claim_contract["version"],
            "contract": claim_contract,
            "task_weighting": claim_contract["task_weighting"],
            "transport_pair": transport_pair,
            "strong_baseline_pairs": strong_pairs,
            "strong_baseline_macro_points": {
                method: {
                    metric: float(values[index])
                    for index, metric in enumerate(_TABLE_METRICS)
                }
                for method, values in strong_points.items()
            },
            "selected_strongest_baseline": selected_strongest,
            "stable_transport_checks": stable_checks,
            "strong_baseline_checks": strong_checks,
            "stable_transport_supported": all(stable_checks.values()),
            "strong_baseline_superiority_supported": all(strong_checks.values()),
            "core_claim_supported": (
                all(stable_checks.values()) and all(strong_checks.values())
            ),
            "coverage_only_rows_excluded_from_superiority_claim": list(
                claim_contract["coverage_only_rows"]
            ),
        }

    report = {
        "version": "frozen-outcome-offline-statistics-v1",
        "status": "complete_recorded_score_analysis",
        "raw_integrity_passed": True,
        "full_task_contract_certified": False if contract_limitations else None,
        "contract_limitations": list(contract_limitations),
        "inputs": inputs,
        "protocol_evidence": protocol_evidence,
        "analysis_source": {"path": str(Path(__file__).resolve().relative_to(root)),
                            "sha256": _file_sha256(Path(__file__))},
        "analysis_recipe": {
            "references": references, "ablation_terms": ablation_terms,
            "aliases": aliases, "permission_classes": permission_classes,
            "numpy_version": np.__version__, "physics_calls": 0,
            "gpu_calls": 0,
        },
        "counts": {"records": len(rows), "physical_executions": len(physical_ids),
                   "predeployment_rejections": len(rows) - len(physical_ids),
                   "cells": len(summaries), "corrected_scalar_cells": len(corrections)},
        "statistics": {
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            "interval": "pointwise 95% empirical percentile, not simultaneous confidence intervals",
            "resampling": "instances crossed with shared initial-block draws within fixed mechanism/support strata; modes paired",
            "worst_group": ["mechanism_group", "body_support_cell", "requested_mode"],
            "primary_scientific_contrast": "gated full minus matched task-only",
            "holm": "supplemental exact instance sign-flip p; all three mean metrics, tasks and contrasts in each main/ablation family",
            "holm_family_membership": "conservative post-access clarification, not a newly preregistered list",
            "sign_flip_assumption": "independent symmetric instance differences conditional on evaluated requests; not a randomized treatment test",
            "rejected_slots": "included in scoring denominator, excluded from training-instance resampling",
            "saturated_interval": "a collapsed empirical interval is not certification of true zero/one probability or equivalence",
            "cross_permission_comparisons": "descriptive only; no causal or superiority claims",
            "wall_seconds": "contended episode time, not end-to-end per-control latency",
        },
        **({"transport_claim": claim_evaluation}
           if claim_evaluation is not None else {}),
        "main_ablation_alias_checks": alias_checks,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("summary", summaries),
                          ("main_table", [r for r in summaries if r["family"] == "main"]),
                          ("ablation_table", [r for r in summaries if r["family"] == "ablation"]),
                          ("contrasts", contrasts),
                          ("failure_groups", groups), ("aggregate_corrections", corrections)):
        path = output / f"{name}.json"
        path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if records:
            fields = sorted({key for record in records for key in record})
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                                  for key, value in row.items()} for row in records)
    if claim_evaluation is not None:
        (output / "transport_claim.json").write_text(
            json.dumps(claim_evaluation, indent=2, sort_keys=True) + "\n"
        )
    # Check again after analysis so even an accidental concurrent input rewrite fails.
    for pointer in inputs.values():
        if _file_sha256(root / pointer["path"]) != pointer["sha256"]:
            raise ValueError("an input changed during offline analysis")
    report["input_sha_recheck_passed"] = True
    report["outputs"] = {path.name: _file_sha256(path) for path in sorted(output.iterdir())}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def collect_results(roots: Sequence[str]) -> List[Dict[str, Any]]:
    records = []
    for root_text in roots:
        root = Path(root_text)
        for path in sorted(root.rglob("results.json")):
            with open(path) as handle:
                data = json.load(handle)
            parts = path.parts
            raw_algorithm = (
                path.parents[2].name
                if path.parents[1].name.startswith("level_")
                else path.parents[2].name
            )
            algorithm = _LEGACY_RESULT_ALIASES.get(raw_algorithm, raw_algorithm)
            record = {
                "path": str(path),
                "algorithm": algorithm,
                "raw_algorithm": raw_algorithm,
                "suite": str(data.get("suite", data.get("level"))),
                "seed": int(data["seed"]),
                "metrics": data.get("metrics") or {},
                **({"execution_status": data["execution_status"]}
                   if "execution_status" in data else {}),
            }
            records.append(record)
    return records


def summarize_results(roots: Sequence[str], output_dir: str) -> Dict[str, Any]:
    """Write scalar summaries and paired deltas from authoritative seed files."""
    records = collect_results(roots)
    attempts: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        attempts[(record["algorithm"], record["suite"])].append(record)
    has_aborts = any(
        (record.get("execution_status") or {}).get("state") == "aborted_unrecoverable"
        for record in records
    )
    groups: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = defaultdict(list)
    for record in records:
        for metric, value in _numeric_leaves(record["metrics"]):
            groups[(record["algorithm"], record["suite"], metric)].append(
                (record["seed"], value)
            )
    if has_aborts:
        for (algorithm, suite), items in attempts.items():
            groups[(algorithm, suite, "execution.abort_rate")] = [
                (record["seed"], float((record.get("execution_status") or {}).get("state")
                                      == "aborted_unrecoverable"))
                for record in items
            ]
    rows = []
    for (algorithm, suite, metric), pairs in sorted(groups.items()):
        values = np.asarray([value for _, value in pairs], dtype=float)
        low, high = _bootstrap_mean(values)
        rows.append({
            "algorithm": algorithm, "suite": suite, "metric": metric,
            "n": len(values), "mean": float(values.mean()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "ci95_low": low, "ci95_high": high,
            **({
                "n_attempted": len(attempts[(algorithm, suite)]),
                "n_completed": sum(
                    (record.get("execution_status") or {}).get("state", "completed") == "completed"
                    for record in attempts[(algorithm, suite)]
                ),
            } if has_aborts else {}),
        })
    paired = []
    lookup = {
        key: dict(pairs) for key, pairs in groups.items()
    }
    full_keys = [key for key in lookup if key[0] == "mga"]
    algorithms = sorted({key[0] for key in lookup if key[0] != "mga"})
    for _, suite, metric in full_keys:
        full = lookup[("mga", suite, metric)]
        for algorithm in algorithms:
            other = lookup.get((algorithm, suite, metric), {})
            seeds = sorted(set(full).intersection(other))
            if not seeds:
                continue
            delta = np.asarray([full[s] - other[s] for s in seeds], dtype=float)
            low, high = _bootstrap_mean(delta)
            paired.append({
                "reference": "mga", "algorithm": algorithm,
                "suite": suite, "metric": metric, "n_pairs": len(seeds),
                "mean_paired_delta": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "paired_sign_flip_p": _paired_sign_flip_p(delta),
            })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", rows), ("paired_deltas", paired)):
        with open(out / f"{name}.json", "w") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
        fields = list(data[0]) if data else []
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(data)
    representative = _representative_seeds(records)
    with open(out / "representative_seeds.json", "w") as handle:
        json.dump(representative, handle, indent=2, allow_nan=False)
    plot_files, plot_error = _write_plots(rows, out)
    return {
        "result_count": len(records), "summary_rows": len(rows),
        "paired_rows": len(paired), "plots": plot_files,
        "plot_error": plot_error, "representative_groups": len(representative),
    }


def _representative_seeds(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Locked qualitative rule: successful seed nearest metric-wise medians."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (record.get("execution_status") or {}).get("state", "completed") != "completed":
            continue  # Keep failed attempts in SSR, not as representative completed motion.
        groups[(record["algorithm"], record["suite"])].append(record)
    selections = []
    for (algorithm, suite), items in sorted(groups.items()):
        leaves = [dict(_numeric_leaves(item["metrics"])) for item in items]
        success_keys = sorted({
            key for row in leaves for key in row
            if key.endswith("safe_insertion_success") or key.endswith("safe_success")
        })
        eligible = list(range(len(items)))
        if success_keys:
            successful = [
                index for index, row in enumerate(leaves)
                if any(row.get(key, 0.0) > 0.5 for key in success_keys)
            ]
            if successful:
                eligible = successful
        common = sorted(set.intersection(*(
            set(leaves[index]) for index in eligible
        ))) if eligible else []
        # Exclude runtime and identifiers; use finite task metrics only.
        common = [key for key in common if "runtime" not in key]
        distances = []
        for index in eligible:
            distance = 0.0
            for key in common:
                values = np.asarray([leaves[i][key] for i in eligible], dtype=float)
                scale = max(float(np.max(values) - np.min(values)), 1e-12)
                distance += ((leaves[index][key] - float(np.median(values))) / scale) ** 2
            distances.append((distance, int(items[index]["seed"]), index))
        if not distances:
            continue
        distance, selected_seed, _ = min(distances)
        selections.append({
            "algorithm": algorithm, "suite": suite, "seed": selected_seed,
            "rule": "successful_seed_nearest_normalized_multimetric_median",
            "distance": float(distance), "metric_count": len(common),
        })
    return selections


def _certified_rejection_error(
    data: Dict[str, Any], expected_steps: int, trajectory_path: Path,
) -> str | None:
    """Validate an algorithmic pre-physics rejection as a complete attempt.

    A task-owned MGA certificate may intentionally refuse to execute when no
    revalidated normal/incumbent/emergency candidate exists.  That is a failed
    task attempt, not a missing experiment.  It is accepted only when the
    persisted prefix and rejection boundary agree exactly; simulator errors
    and arbitrary early exits remain incomplete.
    """
    status = data.get("execution_status") or {}
    if status.get("state", "completed") == "completed":
        return None
    if status.get("state") != "aborted_unrecoverable":
        return f"unknown execution state {status.get('state')!r}"
    if status.get("reason") != "no_revalidated_safe_candidate":
        return f"uncertified abort reason {status.get('reason')!r}"
    requested = status.get("requested_steps")
    executed = status.get("executed_steps")
    rejected = status.get("rejected_step")
    if requested != expected_steps:
        return f"rejection requested_steps={requested!r}, expected {expected_steps}"
    if (
        not isinstance(executed, int)
        or not 0 <= executed < expected_steps
        or rejected != executed
    ):
        return (
            "rejection boundary is inconsistent: "
            f"executed={executed!r}, rejected={rejected!r}"
        )
    details = status.get("details") or {}
    required = {"normal_refined_safe", "incumbent_safe", "emergency_safe"}
    if not required.issubset(details) or any(
        type(details[key]) is not bool for key in required
    ):
        return "rejection certificate details are missing or malformed"
    if any(details.values()):
        return "rejection claims a revalidated safe candidate existed"
    if status.get("metrics_scope") != "actual_execution_prefix_only":
        return "rejection metrics scope is not the actual execution prefix"
    if not isinstance(data.get("partial_metrics"), dict):
        return "rejection is missing prefix diagnostics"
    if not trajectory_path.is_file():
        return "rejection is missing its executed trajectory"
    try:
        with trajectory_path.open() as handle:
            trajectory = json.load(handle)
    except Exception as exc:
        return f"invalid rejected trajectory: {exc}"
    if len(trajectory.get("actions") or ()) != executed:
        return "rejected trajectory length does not match executed_steps"
    return None


def _plot_metric_rows(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """Keep endpoint force diagnostics separate from physical-substep safety."""
    if metric in {"force_peak", "physics_force_peak", "physics_force_normalized_cvar95"}:
        # A suffix match would put force_peak and physics_force_peak in the
        # same figure with indistinguishable algorithm/suite labels. Preserve
        # the old endpoint figure; H1's new primary has its own exact key/file.
        return [row for row in rows if row["metric"].rsplit(".", 1)[-1] == metric]
    return [row for row in rows if row["metric"].endswith(metric)]


def _write_plots(rows: List[Dict[str, Any]], output_dir: Path) -> Tuple[List[str], Any]:
    """Export compact paper diagnostics as both PNG and vector PDF."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    wanted = (
        "safe_insertion_success", "insertion_success",
        "trajectory_path_coverage", "safe_success", "force_peak",
        "physics_force_peak", "physics_force_normalized_cvar95",
    )
    files: List[str] = []
    for suffix in wanted:
        selected = _plot_metric_rows(rows, suffix)
        if not selected:
            continue
        labels = [f"{row['algorithm']}\n{row['suite']}" for row in selected]
        means = [row["mean"] for row in selected]
        lows = [row["mean"] - row["ci95_low"] for row in selected]
        highs = [row["ci95_high"] - row["mean"] for row in selected]
        width = max(7.0, 0.36 * len(selected))
        fig, ax = plt.subplots(figsize=(width, 4.2))
        x = np.arange(len(selected))
        ax.bar(x, means, color="#3568a8", alpha=0.85)
        ax.errorbar(x, means, yerr=[lows, highs], fmt="none", color="black", capsize=2)
        ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel(suffix)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = output_dir / f"{suffix}.{extension}"
            fig.savefig(path, dpi=240 if extension == "png" else None)
            files.append(str(path))
        plt.close(fig)
    # PegInsert success--force Pareto, one point per algorithm/suite.
    success = {
        (row["algorithm"], row["suite"]): row
        for row in rows if row["metric"].endswith("safe_insertion_success")
    }
    force = {
        (row["algorithm"], row["suite"]): row
        for row in rows if row["metric"].endswith("peak_lateral_force")
    }
    keys = sorted(set(success).intersection(force))
    if keys:
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        for key in keys:
            ax.scatter(force[key]["mean"], success[key]["mean"], s=32)
            ax.annotate(f"{key[0]} / {key[1]}",
                        (force[key]["mean"], success[key]["mean"]), fontsize=6)
        ax.set_xlabel("peak lateral force (lower is safer)")
        ax.set_ylabel("strict safe insertion success (higher is better)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = output_dir / f"peg_success_force_pareto.{extension}"
            fig.savefig(path, dpi=240 if extension == "png" else None)
            files.append(str(path))
        plt.close(fig)
    return files, None


def verify_results(
    config_roots: Sequence[str],
    require_visuals: bool = False,
    development_root: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    suites: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Verify formal configs or an isolated development subset.

    Development overrides mirror the runner CLI.  This makes a two-seed P7
    matrix verifiable without copying or editing canonical YAML files.
    """
    from genedynamics.experiments.framework.config import ExperimentConfig

    errors: List[str] = []
    checked = 0
    for path in _config_paths(config_roots):
        cfg = ExperimentConfig.from_yaml(path)
        if development_root is not None:
            try:
                cfg.use_development_output_root(Path(development_root))
            except ValueError as exc:
                errors.append(f"{path}: {exc}")
                continue
        if seeds is not None:
            cfg.seeds = list(dict.fromkeys(int(seed) for seed in seeds))
            if not cfg.seeds:
                errors.append(f"{path}: development seed selection is empty")
                continue
        if suites is not None:
            requested = list(dict.fromkeys(str(name) for name in suites))
            available = {str(suite.get("name")): suite for suite in cfg.suites}
            unknown = [name for name in requested if name not in available]
            if unknown:
                errors.append(f"{path}: unknown suites {unknown!r}")
                continue
            requested_set = set(requested)
            cfg.suites = [
                suite for suite in cfg.suites
                if str(suite.get("name")) in requested_set
            ]

        manifest = cfg.output_dir / "protocol_manifest.json"
        if not manifest.is_file():
            errors.append(f"missing {manifest}")
        else:
            try:
                with open(manifest) as handle:
                    manifest_data = json.load(handle)
                manifest_config = manifest_data.get("config") or {}
                if manifest_config.get("output_dir") != str(cfg.output_dir):
                    errors.append(f"output isolation mismatch {manifest}")
                if manifest_config.get("seeds") != cfg.seeds:
                    errors.append(f"seed protocol mismatch {manifest}")
                expected_suites = [str(item["name"]) for item in cfg.suites]
                actual_suites = [
                    str(item.get("name"))
                    for item in manifest_config.get("suites", [])
                ]
                if actual_suites != expected_suites:
                    errors.append(f"suite protocol mismatch {manifest}")
                expected_budgets = resolved_suite_budgets(cfg)
                saved_budgets = resolved_suite_budgets(ExperimentConfig(**manifest_config))
                if saved_budgets != expected_budgets:
                    errors.append(f"resolved suite budget mismatch {manifest}")
                # Older manifests encode the same budgets inside config only.
                if (
                    "resolved_suite_budgets" in manifest_data
                    and manifest_data["resolved_suite_budgets"] != expected_budgets
                ):
                    errors.append(f"resolved suite budget summary mismatch {manifest}")
                if development_root is not None:
                    manifest_metadata = manifest_config.get("metadata") or {}
                    if manifest_metadata.get("run_class") != "development":
                        errors.append(f"run class mismatch {manifest}")
                    if manifest_metadata.get("formal_seeds") is not False:
                        errors.append(f"formal/development mismatch {manifest}")
            except Exception as exc:
                errors.append(f"invalid JSON {manifest}: {exc}")
        for suite in cfg.suites or ({"name": level} for level in cfg.obstacle_levels):
            name = str(suite["name"])
            resolved_config = cfg.for_suite(suite)
            level_dir = cfg.output_dir / f"level_{name}"
            if not (level_dir / "summary.json").is_file():
                errors.append(f"missing {level_dir / 'summary.json'}")
            for seed in cfg.seeds:
                checked += 1
                seed_dir = level_dir / f"seed_{seed}"
                result_path = seed_dir / "results.json"
                if not result_path.is_file():
                    errors.append(f"missing {result_path}")
                    continue
                try:
                    with open(result_path) as handle:
                        data = json.load(handle)
                except Exception as exc:
                    errors.append(f"invalid JSON {result_path}: {exc}")
                    continue
                if data.get("seed") != seed or str(data.get("suite")) != name:
                    errors.append(f"identity mismatch {result_path}")
                trajectory_path = seed_dir / "trajectory" / "trajectory.json"
                rejection_error = _certified_rejection_error(
                    data, resolved_config.n_steps, trajectory_path
                )
                if rejection_error is not None:
                    errors.append(
                        f"execution aborted/incomplete {result_path}: "
                        f"{rejection_error}"
                    )
                snapshot = data.get("config_snapshot") or {}
                snapshot_budget = snapshot.get("method_params") or {}
                if snapshot.get("n_steps") != resolved_config.n_steps:
                    errors.append(f"budget mismatch n_steps in {result_path}")
                if snapshot.get("env_name") != cfg.env_name or snapshot.get("method") != cfg.method:
                    errors.append(f"resolved method/environment mismatch {result_path}")
                if snapshot.get("output_dir") != str(cfg.output_dir):
                    errors.append(f"resolved output mismatch {result_path}")
                snapshot_seeds = snapshot.get("seeds")
                if (
                    not isinstance(snapshot_seeds, list)
                    or seed not in snapshot_seeds
                ):
                    errors.append(f"resolved seed mismatch {result_path}")
                if development_root is not None:
                    snapshot_metadata = snapshot.get("metadata") or {}
                    if snapshot_metadata.get("run_class") != "development":
                        errors.append(f"development run class missing {result_path}")
                    if snapshot_metadata.get("formal_seeds") is not False:
                        errors.append(f"development seed label mismatch {result_path}")
                for key in BUDGET_KEYS:
                    if snapshot_budget.get(key) != resolved_config.method_params.get(key):
                        errors.append(f"budget mismatch {key} in {result_path}")
                if not trajectory_path.is_file():
                    errors.append(f"missing executed trajectory for {result_path}")
                if require_visuals:
                    artifacts = list((seed_dir / "trajectory").glob("*.png")) + list(
                        (seed_dir / "trajectory").glob("*.gif")
                    )
                    if not artifacts:
                        errors.append(f"missing visual artifact for {result_path}")
        overall_path = cfg.output_dir / "overall_summary.json"
        if not overall_path.is_file():
            errors.append(f"missing {overall_path}")
        else:
            try:
                with open(overall_path) as handle:
                    overall = json.load(handle)
                suite_count = len(cfg.suites) if cfg.suites else len(cfg.obstacle_levels)
                expected_runs = suite_count * len(cfg.seeds)
                if overall.get("total_experiments") != expected_runs:
                    errors.append(
                        f"run-count mismatch {overall_path}: "
                        f"expected {expected_runs}, got {overall.get('total_experiments')}"
                    )
            except Exception as exc:
                errors.append(f"invalid JSON {overall_path}: {exc}")
    return {"ok": not errors, "checked_runs": checked, "errors": errors}


def compare_runner_outputs(
    legacy_path: str,
    result_path: str,
    algorithm: str,
    suite: str,
    seed: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare a frozen legacy seed record with one unified-runner seed.

    Executed actions are compared strictly and scalar metrics tolerantly.
    Unmatched legacy metrics are reported rather than silently treated as equal.
    """
    with open(legacy_path) as handle:
        legacy_records = json.load(handle)
    if not isinstance(legacy_records, list):
        raise ValueError("legacy result file must contain a list of seed records")
    matches = [
        record for record in legacy_records
        if str(record.get("algorithm", record.get("variant"))) == algorithm
        and str(record.get("suite", record.get("level"))) == suite
        and int(record.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one legacy record, found {len(matches)}")
    legacy = matches[0]
    result_file = Path(result_path)
    with open(result_file) as handle:
        current = json.load(handle)
    trajectory_file = result_file.parent / "trajectory" / "trajectory.json"
    with open(trajectory_file) as handle:
        executed = json.load(handle)

    config_comparison = None
    legacy_manifest_path = Path(legacy_path).with_name("manifest.json")
    current_snapshot = current.get("config_snapshot") or {}
    if legacy_manifest_path.is_file() and current_snapshot:
        with open(legacy_manifest_path) as handle:
            legacy_manifest = json.load(handle)
        legacy_config = legacy_manifest.get("config") or {}
        legacy_env = dict(legacy_config.get("env_params") or {})
        if legacy_config.get("level") is not None:
            legacy_env.setdefault("level", legacy_config["level"])
        current_env = dict(current_snapshot.get("env_params") or {})
        legacy_method_params = dict(legacy_config.get("method_params") or {})
        current_method_params = dict(current_snapshot.get("method_params") or {})
        current_controller = current_method_params.pop(
            "controller_method", current_snapshot.get("method")
        )
        normalized_legacy = {
            "env_name": legacy_config.get("task"),
            "controller_method": legacy_config.get("method"),
            "n_steps": legacy_config.get("n_steps"),
            "env_params": legacy_env,
            "execution_env_params": legacy_config.get("execution_env_params") or {},
            "method_params": legacy_method_params,
        }
        normalized_current = {
            "env_name": current_snapshot.get("env_name"),
            "controller_method": current_controller,
            "n_steps": current_snapshot.get("n_steps"),
            "env_params": current_env,
            "execution_env_params": current_snapshot.get("execution_env_params") or {},
            "method_params": current_method_params,
        }
        config_comparison = {
            "equal": normalized_legacy == normalized_current,
            "legacy": normalized_legacy,
            "current": normalized_current,
            "legacy_git_sha": legacy_manifest.get("git_sha"),
            "current_git_sha": (current.get("provenance") or {}).get("git_sha"),
        }

    old_actions = np.asarray((legacy.get("series") or {}).get("actions", []), dtype=float)
    new_actions = np.asarray(executed.get("actions", []), dtype=float)
    actions_equal = bool(
        old_actions.shape == new_actions.shape
        and np.allclose(old_actions, new_actions, atol=atol, rtol=rtol)
    )
    ignored = {
        "series", "seed", "task", "method", "algorithm", "variant",
        "suite", "level", "policy_training_seed", "policy_training_steps",
    }
    legacy_metrics = dict(_numeric_leaves({
        key: value for key, value in legacy.items() if key not in ignored
    }))
    diagnostics = current.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = aggregate_receding_diagnostics(
            executed.get("infos") or (),
            task=(current.get("config_snapshot") or {}).get("env_name"),
        )
    current_metrics = dict(_numeric_leaves({
        "metrics": current.get("metrics") or {},
        "diagnostics": diagnostics,
    }))
    comparisons = []
    unmatched = []
    unmatched_runtime = []
    runtime_comparisons = []
    for old_key, old_value in sorted(legacy_metrics.items()):
        if old_key == "runtime" or old_key.endswith(".runtime"):
            candidates = [
                (key, value) for key, value in current_metrics.items()
                if key == old_key or key.endswith(f".{old_key}")
            ]
            if len(candidates) == 1:
                new_key, new_value = candidates[0]
                runtime_comparisons.append({
                    "legacy_metric": old_key,
                    "current_metric": new_key,
                    "legacy": old_value,
                    "current": new_value,
                    "ratio": new_value / old_value if old_value else None,
                    "synchronization_policy": "block_after_each_execution_step",
                })
            else:
                unmatched_runtime.append(old_key)
            continue
        candidates = [
            (key, value) for key, value in current_metrics.items()
            if key == old_key or key.endswith(f".{old_key}")
        ]
        if len(candidates) != 1:
            unmatched.append(old_key)
            continue
        new_key, new_value = candidates[0]
        comparisons.append({
            "legacy_metric": old_key,
            "current_metric": new_key,
            "legacy": old_value,
            "current": new_value,
            "equal": bool(np.isclose(old_value, new_value, atol=atol, rtol=rtol)),
        })
    return {
        "ok": actions_equal and bool(comparisons) and not unmatched
        and not unmatched_runtime
        and (config_comparison is None or config_comparison["equal"])
        and all(row["equal"] for row in comparisons),
        "config_comparison": config_comparison,
        "action_shape": list(new_actions.shape),
        "actions_equal": actions_equal,
        "matched_metrics": len(comparisons),
        "unmatched_legacy_metrics": unmatched,
        "unmatched_runtime_metrics": unmatched_runtime,
        "metric_comparisons": comparisons,
        "runtime_comparisons": runtime_comparisons,
    }


def _print_report(report: Dict[str, Any]) -> int:
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("ok", True) else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("roots", nargs="+")
    summary = sub.add_parser("summarize")
    summary.add_argument("roots", nargs="+")
    summary.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("roots", nargs="+")
    verify.add_argument("--require-visuals", action="store_true")
    verify.add_argument("--development-root")
    verify.add_argument("--seeds", type=int, nargs="+")
    verify.add_argument("--suites", nargs="+")
    compare = sub.add_parser("compare")
    compare.add_argument("--legacy", required=True)
    compare.add_argument("--result", required=True)
    compare.add_argument("--algorithm", required=True)
    compare.add_argument("--suite", required=True)
    compare.add_argument("--seed", type=int, required=True)
    compare.add_argument("--atol", type=float, default=1e-6)
    compare.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.command == "audit":
        return _print_report(audit_configs(args.roots))
    if args.command == "summarize":
        return _print_report(summarize_results(args.roots, args.output))
    if args.command == "verify":
        return _print_report(verify_results(
            args.roots,
            require_visuals=args.require_visuals,
            development_root=args.development_root,
            seeds=args.seeds,
            suites=args.suites,
        ))
    return _print_report(compare_runner_outputs(
        args.legacy, args.result, args.algorithm, args.suite, args.seed,
        args.atol, args.rtol,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
