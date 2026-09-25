"""Audit and atomically promote a PegInsert MGA reliability checkpoint.

The fit artifact is deliberately non-authoritative.  Promotion requires an
independent, seed-paired development matrix against the otherwise identical
``no_learned_reliability`` controller.  This script never runs experiments or
selects a checkpoint; it only audits frozen results and, when explicitly
requested, writes the canonical promoted copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_IGNORED_METHOD_PARAMS = {
    "learned_reliability",
    "reliability_ckpt",
    "reliability_validation_authoritative",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("results.json")))
        elif path.name == "results.json" and path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"missing results.json input: {path}")
    return sorted(set(files))


def _paired_results(
    paths: list[str],
) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    rows: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path in _result_files(paths):
        result = _load(path)
        snapshot = result.get("config_snapshot") or {}
        if snapshot.get("env_name") != "manipulator_peg_insert":
            continue
        key = (
            str(result.get("suite", result.get("level", ""))),
            int(result["seed"]),
        )
        if key in rows:
            raise ValueError(f"duplicate PegInsert validation result: {key}")
        rows[key] = (path, result)
    return rows


def _checkpoint(result: dict[str, Any], name: str) -> str | None:
    return (
        (((result.get("provenance") or {}).get("checkpoints") or {}).get(name) or {})
        .get("sha256")
    )


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return ((result.get("metrics") or {}).get("peg_insert_metrics") or {})


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale temporary output: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _algorithm_params(result: dict[str, Any]) -> dict[str, Any]:
    params = dict(
        ((result.get("config_snapshot") or {}).get("method_params") or {})
    )
    for key in _IGNORED_METHOD_PARAMS:
        params.pop(key, None)
    return params


def _actions(path: Path) -> np.ndarray:
    trajectory = path.parent / "trajectory" / "trajectory.json"
    values = np.asarray(_load(trajectory).get("actions") or (), np.float32)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid validation trajectory actions: {trajectory}")
    return values


def audit(
    candidate: Path,
    validation_roots: list[str],
    control_roots: list[str],
    *,
    required_suites: tuple[str, ...],
    minimum_seeds: int,
    formal_seeds: set[int],
    expected_policy_sha256: str,
    max_depth_regression_m: float = 0.002,
    progress_depth_cap_m: float = 0.032,
) -> dict[str, Any]:
    if minimum_seeds < 2:
        raise ValueError("paired validation requires at least two seeds")
    payload = _load(candidate)
    metadata = dict(payload.get("metadata") or {})
    candidate_sha256 = _sha256(candidate)
    problems: list[str] = []
    if metadata.get("task") != "manipulator_peg_insert":
        problems.append("candidate_task_mismatch")
    if metadata.get("fit_scope") != "development_fit_only":
        problems.append("candidate_is_not_development_fit_only")
    if bool(metadata.get("performance_validated", False)) or bool(
        metadata.get("promotion_eligible", False)
    ):
        problems.append("candidate_is_already_promoted")
    for split in (
        "training_policy_checkpoint_sha256",
        "calibration_policy_checkpoint_sha256",
    ):
        if set(metadata.get(split) or ()) != {expected_policy_sha256}:
            problems.append(f"candidate_{split}_mismatch")

    validation = _paired_results(validation_roots)
    control = _paired_results(control_roots)
    common = set(validation).intersection(control)
    seeds_by_suite = {
        suite: {seed for name, seed in common if name == suite}
        for suite in required_suites
    }
    complete_seeds = (
        set.intersection(*seeds_by_suite.values()) if seeds_by_suite else set()
    )
    if len(complete_seeds) < minimum_seeds:
        problems.append(
            f"need_{minimum_seeds}_complete_paired_seeds_have_{len(complete_seeds)}"
        )
    fit_seeds = set(metadata.get("training_seeds") or ()) | set(
        metadata.get("calibration_seeds") or ()
    )
    leaked = sorted(complete_seeds.intersection(fit_seeds | formal_seeds))
    if leaked:
        problems.append(f"validation_seed_leakage:{leaked}")

    selected_keys = sorted(
        (suite, seed) for suite in required_suites for seed in complete_seeds
    )
    candidate_insertion = 0.0
    control_insertion = 0.0
    candidate_safe = 0.0
    control_safe = 0.0
    abstentions: list[float] = []
    changed_action_pairs = 0
    pair_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for key in selected_keys:
        learned_path, learned = validation[key]
        control_path, no_learned = control[key]
        source_hashes[str(learned_path)] = _sha256(learned_path)
        source_hashes[str(control_path)] = _sha256(control_path)
        if _checkpoint(learned, "reliability_ckpt") != candidate_sha256:
            problems.append(f"candidate_hash_mismatch:{key}")
        for label, result in (("candidate", learned), ("control", no_learned)):
            if _checkpoint(result, "policy_ckpt") != expected_policy_sha256:
                problems.append(f"{label}_policy_hash_mismatch:{key}")
        learned_contract = learned.get("component_contract") or {}
        control_contract = no_learned.get("component_contract") or {}
        if not bool(learned_contract.get("learned_reliability", False)):
            problems.append(f"candidate_did_not_load_reliability:{key}")
        if not bool(learned_contract.get("reliability_gate_authoritative", False)):
            problems.append(f"candidate_gate_not_authoritative:{key}")
        if not bool(
            learned_contract.get("reliability_validation_authoritative", False)
        ):
            problems.append(f"development_validation_override_missing:{key}")
        if bool(learned_contract.get("reliability_promotion_eligible", False)):
            problems.append(f"candidate_was_prematurely_promoted:{key}")
        if bool(control_contract.get("learned_reliability", False)):
            problems.append(f"control_loaded_reliability:{key}")
        if _algorithm_params(learned) != _algorithm_params(no_learned):
            problems.append(f"paired_algorithm_mismatch:{key}")
        for field in ("env_params", "execution_env_params"):
            if (learned.get("config_snapshot") or {}).get(field) != (
                no_learned.get("config_snapshot") or {}
            ).get(field):
                problems.append(f"paired_{field}_mismatch:{key}")

        learned_metrics = _metrics(learned)
        control_metrics = _metrics(no_learned)
        if not learned_metrics or not control_metrics:
            problems.append(f"missing_metrics:{key}")
            continue
        insertion = float(learned_metrics.get("insertion_success", 0.0))
        insertion_control = float(
            control_metrics.get("insertion_success", 0.0)
        )
        safe = float(learned_metrics.get("safe_insertion_success", 0.0))
        safe_control = float(
            control_metrics.get("safe_insertion_success", 0.0)
        )
        candidate_insertion += insertion
        control_insertion += insertion_control
        candidate_safe += safe
        control_safe += safe_control
        learned_depth = float(learned_metrics.get("max_insertion_depth", 0.0))
        control_depth = float(control_metrics.get("max_insertion_depth", 0.0))
        depth_regression = min(control_depth, progress_depth_cap_m) - min(
            learned_depth, progress_depth_cap_m
        )
        if depth_regression > max_depth_regression_m:
            problems.append(f"candidate_depth_regression:{key}")
        diagnostic = learned.get("diagnostics") or {}
        if "reliability_abstained" not in diagnostic:
            problems.append(f"missing_reliability_abstention:{key}")
            abstention = 1.0
        else:
            abstention = float(diagnostic["reliability_abstained"])
            abstentions.append(abstention)
        try:
            action_changed = not np.array_equal(
                _actions(learned_path), _actions(control_path)
            )
        except ValueError as error:
            problems.append(str(error))
            action_changed = False
        changed_action_pairs += int(action_changed)
        pair_rows.append({
            "suite": key[0],
            "seed": key[1],
            "candidate_insertion_success": insertion,
            "control_insertion_success": insertion_control,
            "candidate_safe_success": safe,
            "control_safe_success": safe_control,
            "candidate_max_depth_m": learned_depth,
            "control_max_depth_m": control_depth,
            "depth_regression_m": depth_regression,
            "reliability_abstained": abstention,
            "executed_actions_changed": action_changed,
        })

    if candidate_insertion < control_insertion:
        problems.append("candidate_insertion_success_below_control")
    if candidate_safe < control_safe:
        problems.append("candidate_safe_success_below_control")
    mean_abstention = float(np.mean(abstentions)) if abstentions else 1.0
    if mean_abstention >= 0.95:
        problems.append("candidate_gate_abstained_on_at_least_95_percent")
    if selected_keys and changed_action_pairs == 0:
        problems.append("candidate_gate_never_changed_executed_actions")

    problems = sorted(set(problems))
    return {
        "schema_version": 1,
        "candidate": str(candidate),
        "candidate_sha256": candidate_sha256,
        "expected_policy_sha256": expected_policy_sha256,
        "required_suites": list(required_suites),
        "minimum_complete_seeds": minimum_seeds,
        "complete_paired_seeds": sorted(complete_seeds),
        "pair_count": len(selected_keys),
        "candidate_insertion_success": candidate_insertion,
        "control_insertion_success": control_insertion,
        "candidate_safe_success": candidate_safe,
        "control_safe_success": control_safe,
        "mean_reliability_abstention": mean_abstention,
        "changed_action_pairs": changed_action_pairs,
        "pairs": pair_rows,
        "source_hashes": source_hashes,
        "passed": not problems,
        "blocking_reasons": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--validation-results", nargs="+", required=True)
    parser.add_argument("--control-results", nargs="+", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument(
        "--required-suites", nargs="+",
        default=["id_wide", "ood_pose", "ood_sensing"],
    )
    parser.add_argument("--minimum-seeds", type=int, default=2)
    parser.add_argument(
        "--formal-seeds", nargs="*", type=int, default=list(range(10))
    )
    parser.add_argument("--max-depth-regression-m", type=float, default=0.002)
    parser.add_argument("--progress-depth-cap-m", type=float, default=0.032)
    parser.add_argument("--report", required=True)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    candidate = Path(args.candidate).resolve()
    report = audit(
        candidate,
        args.validation_results,
        args.control_results,
        required_suites=tuple(args.required_suites),
        minimum_seeds=int(args.minimum_seeds),
        formal_seeds=set(args.formal_seeds),
        expected_policy_sha256=str(args.expected_policy_sha256),
        max_depth_regression_m=float(args.max_depth_regression_m),
        progress_depth_cap_m=float(args.progress_depth_cap_m),
    )
    report_path = Path(args.report)
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"promotion report already exists: {report_path}")
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    if not args.promote:
        return 0 if report["passed"] else 2
    if not report["passed"]:
        raise RuntimeError("reliability promotion blocked; inspect report")
    if not args.output:
        raise ValueError("--promote requires --output")
    target = Path(args.output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"promotion output already exists: {target}")

    payload = _load(candidate)
    metadata = dict(payload.get("metadata") or {})
    protocol = str(metadata.get("protocol", ""))
    suffix = "_development"
    metadata.update({
        "protocol": protocol[:-len(suffix)] if protocol.endswith(suffix) else protocol,
        "pre_promotion_protocol": protocol,
        "performance_validated": True,
        "promotion_eligible": True,
        "fit_scope": "promoted_after_independent_paired_validation",
        "promotion_blocking_reasons": [],
        "promotion_report": str(report_path),
        "promotion_report_sha256": _sha256(report_path),
        "promotion_validation_seeds": report["complete_paired_seeds"],
        "promotion_validation_suites": report["required_suites"],
        "pre_promotion_candidate_sha256": report["candidate_sha256"],
    })
    payload["metadata"] = metadata
    _atomic_json(target, payload)
    print(json.dumps({
        "promoted_checkpoint": str(target),
        "sha256": _sha256(target),
        "config_lock_update_required": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
