#!/usr/bin/env python3
"""Close out the frozen public-response calibration budget curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SEALED = (
    ROOT / "results/_development/p4_completion_20261015/results/arm/"
    "keyed_insertion_recovery/validation/p4_public_local_response/"
    "level_B_log/seed_38000/report.json"
)
SEALED_SHA256 = "bc561e81b869f293c0eb6c5dc4a9c7daa3a49670c4c704ded09d0442e1fbb0e4"
EXTENSION = (
    ROOT / "results/_development/p4_completion_20261015/results/arm/"
    "keyed_insertion_recovery/validation/p4_public_local_response_curve/"
    "level_B_log/seed_38000/report.json"
)
EXTENSION_SHA256 = "d55ebcc57bb3011d56364d814405c2fb6c156d76cfd9963fc450fa5d7d74c6d3"
DEFAULT_OUTPUT = (
    ROOT / "results/arm/keyed_insertion_recovery/analysis/"
    "response_budget/report.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def checked(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"frozen response-budget source changed: {path}")
    report = json.loads(path.read_text())
    if (
        report.get("status") != "complete"
        or report.get("new_target_calls") != 0
        or report.get("new_physics_calls") != 0
        or report.get("task_performance_evidence") is not False
    ):
        raise ValueError("response-budget source is not a sealed model diagnostic")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1"}:
        raise RuntimeError("response-budget closeout is CPU-only")
    sealed = checked(SEALED, SEALED_SHA256)
    extension = checked(EXTENSION, EXTENSION_SHA256)
    raw = {
        0: sealed["budgets"]["0"],
        30: extension["budgets"]["30"],
        60: extension["budgets"]["60"],
        120: sealed["budgets"]["120"],
    }
    inherited_zero = extension["budgets"]["0"]
    shared_zero_fields = {
        "updates", "extra_verification_update_executions", "resume_exact",
        "checkpoint_sha256", "held_out_source_nominal_validation",
        "B_log_in_sample_fit", "loss_curve",
    }
    if (
        any(inherited_zero.get(key) != raw[0].get(key) for key in shared_zero_fields)
        or inherited_zero.get("inherited") is not True
        or inherited_zero.get("new_update_executions_here") != 0
        or inherited_zero.get("parent_report_sha256") != SEALED_SHA256
    ):
        raise ValueError("the curve extension does not share the sealed budget-0 model")
    points = []
    for budget, row in raw.items():
        held_out = row["held_out_source_nominal_validation"]
        target_fit = row["B_log_in_sample_fit"]
        points.append({
            "target_observation_budget_seconds": budget,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "resume_exact": bool(row["resume_exact"]),
            "target_log_windows": 0 if target_fit is None else target_fit["windows"],
            "target_log_in_sample_normalized_mse": (
                None if target_fit is None else target_fit["normalized_MSE"]
            ),
            "held_out_source_nominal_normalized_mse": held_out["normalized_MSE"],
            "held_out_source_nominal_position_rmse_m": held_out[
                "public_position_RMSE_m"
            ],
            "held_out_source_nominal_rotation_rmse_rad": held_out[
                "rotation_increment_RMSE_rad"
            ],
        })
    source_errors = [
        point["held_out_source_nominal_normalized_mse"] for point in points
    ]
    report = {
        "schema_version": "keyed-insertion-response-budget-evidence-v1",
        "status": "complete_model_diagnostic",
        "task": "keyed_insertion_recovery",
        "budget_points": points,
        "evidence_scope": {
            "public_control_rate_history_only": True,
            "new_target_calls": 0,
            "new_physics_calls": 0,
            "task_performance_evidence": False,
            "same_budget_zero_checkpoint": True,
        },
        "observations": {
            "target_fit_available_only_for_positive_budget": True,
            "held_out_source_nominal_error_monotonic_with_budget": all(
                right <= left for left, right in zip(source_errors, source_errors[1:])
            ),
            "interpretation": (
                "Positive target-history budgets fit the observed target prefix, but "
                "held-out source/nominal response error is non-monotonic. These points "
                "establish a calibration-cost axis only; task-level benefit must come "
                "from matched physical evaluation of the frozen checkpoints."
            ),
        },
        "artifacts": {
            "sealed_0_120_report": pointer(SEALED),
            "curve_30_60_report": pointer(EXTENSION),
            "runner": pointer(Path(__file__).resolve()),
        },
        "gpu_calls": 0,
        "new_physics_calls": 0,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "sha256": sha256(output),
        "observations": report["observations"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
