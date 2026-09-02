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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


BUDGET_KEYS = ("Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init")
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
) -> None:
    """Validate the P8 frozen learned-component contract for one task."""
    if task not in _CONTACT_TASKS:
        return
    reference_path, reference = task_records[0]
    metadata = dict(reference.metadata or {})
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
            errors.append(
                f"{reference_path}: {name} uses formal evaluation seeds {leaked}"
            )
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
            "no_rl_prior", "no_learned_reliability", "no_tangent",
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

    def mean_scalar(key: str) -> float:
        values = [np.asarray(row[key], dtype=float).reshape(-1)
                  for row in rows if key in row]
        return float(np.mean(np.concatenate(values))) if values else float("nan")

    out: Dict[str, float] = {}
    if has("prior_accepted"):
        out.update({
            "prior_acceptance_rate": mean_scalar("prior_accepted"),
            "prior_predicted_improvement_mean": mean_scalar(
                "prior_predicted_improvement"
            ),
            "prior_risk_ok_rate": mean_scalar("prior_risk_ok"),
            "prior_force_veto_rate": mean_scalar("prior_force_veto"),
        })
        out["prior_fallback_rate"] = 1.0 - out["prior_acceptance_rate"]
        insertion = task == "manipulator_peg_insert"
        labels = (
            ("force_violation", "torque_violation", "jam", "force_mae")
            if insertion else
            ("force_violation", "contact_loss", "deformation", "force_mae")
        )
        for source in (
            "prior_risk_rl", "prior_risk_incumbent", "prior_risk_refined",
            "prior_risk_atacom", "prior_risk_emergency",
            "reliability_risk_incumbent", "reliability_risk_refined",
        ):
            vectors = [np.asarray(row[source], dtype=float).reshape(-1)
                       for row in rows if source in row]
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
        ):
            if has(key):
                out[key] = mean_scalar(key)
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


def audit_configs(roots: Sequence[str]) -> Dict[str, Any]:
    """Check paired protocol fields without opening any result files."""
    from genedynamics.experiments.framework.config import ExperimentConfig

    records = []
    errors: List[str] = []
    for path in _config_paths(roots):
        try:
            cfg = ExperimentConfig.from_yaml(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        validation = cfg.validate()
        errors.extend(f"{path}: {error}" for error in validation)
        records.append((path, cfg))
    by_task: Dict[str, List[Tuple[Path, ExperimentConfig]]] = defaultdict(list)
    for record in records:
        by_task[record[1].env_name].append(record)
    run_count_by_task: Dict[str, int] = {}
    for task, task_records in by_task.items():
        reference = task_records[0][1]
        full_cfg = next(
            (cfg for _, cfg in task_records if cfg.name == "mga"), None
        )
        ref_protocol = {
            "seeds": reference.seeds,
            "n_steps": reference.n_steps,
            "suites": reference.suites,
            "budget": {key: reference.method_params.get(key) for key in BUDGET_KEYS},
        }
        names = set()
        for path, cfg in task_records:
            if cfg.name in names:
                errors.append(f"{task}: duplicate algorithm name {cfg.name!r}")
            names.add(cfg.name)
            current = {
                "seeds": cfg.seeds,
                "n_steps": cfg.n_steps,
                "suites": cfg.suites,
                "budget": {key: cfg.method_params.get(key) for key in BUDGET_KEYS},
            }
            if current != ref_protocol:
                errors.append(f"{path}: paired protocol differs from {task_records[0][0]}")
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
            run_count_by_task[task] = run_count_by_task.get(task, 0) + (
                len(cfg.seeds) * len(formal_suites)
            )
        _audit_checkpoint_lock(task, task_records, errors)
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
            }
            records.append(record)
    return records


def summarize_results(roots: Sequence[str], output_dir: str) -> Dict[str, Any]:
    """Write scalar summaries and paired deltas from authoritative seed files."""
    records = collect_results(roots)
    groups: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = defaultdict(list)
    for record in records:
        for metric, value in _numeric_leaves(record["metrics"]):
            groups[(record["algorithm"], record["suite"], metric)].append(
                (record["seed"], value)
            )
    rows = []
    for (algorithm, suite, metric), pairs in sorted(groups.items()):
        values = np.asarray([value for _, value in pairs], dtype=float)
        low, high = _bootstrap_mean(values)
        rows.append({
            "algorithm": algorithm, "suite": suite, "metric": metric,
            "n": len(values), "mean": float(values.mean()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "ci95_low": low, "ci95_high": high,
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


def _write_plots(rows: List[Dict[str, Any]], output_dir: Path) -> Tuple[List[str], Any]:
    """Export compact paper diagnostics as both PNG and vector PDF."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    wanted = (
        "safe_insertion_success", "insertion_success",
        "trajectory_path_coverage", "safe_success", "force_peak",
    )
    files: List[str] = []
    for suffix in wanted:
        selected = [row for row in rows if row["metric"].endswith(suffix)]
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
                snapshot = data.get("config_snapshot") or {}
                snapshot_budget = snapshot.get("method_params") or {}
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
                    if snapshot_budget.get(key) != cfg.method_params.get(key):
                        errors.append(f"budget mismatch {key} in {result_path}")
                if not (seed_dir / "trajectory" / "trajectory.json").is_file():
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
