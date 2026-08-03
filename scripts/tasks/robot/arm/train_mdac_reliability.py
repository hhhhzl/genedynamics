"""Fit/calibrate the CPU MDAC reliability gate from existing metrics records.

Example:
  python scripts/tasks/robot/arm/train_mdac_reliability.py \
    --metrics path/to/rigid/metrics.json path/to/soft/metrics.json \
    --calibration-seeds 11 --output path/to/reliability.json

Training and calibration use only the supplied seen-domain records.  Held-out
unseen records may be supplied with ``--test-metrics`` for reporting, but are
never used to fit or calibrate the checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from genedynamics.learning.reliability import (
    LinearReliabilityModel,
    RISK_NAMES,
    samples_from_records,
)


def _records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON record list")
        records.extend(payload)
    return records


def _report(model, x, y) -> dict[str, Any]:
    if not len(x):
        return {"rows": 0}
    prediction = np.asarray(model.predict(x))
    upper = np.asarray(model.predict_upper(x))
    support_score = np.asarray(model.support_score(x))
    in_support = support_score <= 1.0
    conditional_coverage = (
        np.mean(y[in_support] <= upper[in_support] + 1.0e-7, axis=0)
        if np.any(in_support) else np.full(len(RISK_NAMES), np.nan)
    )
    return {
        "rows": int(len(x)),
        "mae": {
            name: float(value)
            for name, value in zip(RISK_NAMES, np.mean(np.abs(prediction - y), axis=0))
        },
        "upper_coverage": {
            name: float(value)
            for name, value in zip(RISK_NAMES, np.mean(y <= upper + 1.0e-7, axis=0))
        },
        "in_support_upper_coverage": {
            name: float(value)
            for name, value in zip(RISK_NAMES, conditional_coverage)
        },
        "mean_upper": {
            name: float(value)
            for name, value in zip(RISK_NAMES, np.mean(upper, axis=0))
        },
        "in_support_rate": float(np.mean(in_support)),
        "support_score_p95": float(np.quantile(support_score, 0.95)),
        "support_score_max": float(np.max(support_score)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", default=[])
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--calibration-metrics", nargs="*", default=[])
    parser.add_argument("--test-metrics", nargs="*", default=[])
    parser.add_argument("--output")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--quantile", type=float, default=0.95)
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
    if not args.metrics or not args.output:
        raise ValueError("fitting requires --metrics and --output")

    records = _records(args.metrics)
    calibration_seeds = set(args.calibration_seeds)
    train_records = [r for r in records if int(r["seed"]) not in calibration_seeds]
    calibration_records = [r for r in records if int(r["seed"]) in calibration_seeds]
    calibration_records.extend(_records(args.calibration_metrics))
    if not train_records or not calibration_records:
        raise ValueError("seed split produced an empty training or calibration set")

    train_x, train_y, train_provenance = samples_from_records(train_records)
    cal_x, cal_y, cal_provenance = samples_from_records(calibration_records)
    model = LinearReliabilityModel.fit(
        train_x,
        train_y,
        cal_x,
        cal_y,
        ridge=args.ridge,
        quantile=args.quantile,
    )
    metadata = {
        "training_records": len(train_records),
        "calibration_records": len(calibration_records),
        "training_rows": len(train_x),
        "calibration_rows": len(cal_x),
        "training_suites": dict(Counter(x["suite"] for x in train_provenance)),
        "calibration_suites": dict(Counter(x["suite"] for x in cal_provenance)),
        "calibration_seeds": sorted(calibration_seeds),
        "deployment_family_calibration_files": list(args.calibration_metrics),
        "ridge": args.ridge,
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
