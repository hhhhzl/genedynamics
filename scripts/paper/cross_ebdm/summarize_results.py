#!/usr/bin/env python3
"""Summarize completed main/baseline/ablation results without running physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_ROOTS = (
    Path("results/arm/obstacle_transfer"),
    Path("results/arm/guided_contact_alignment"),
    Path("results/arm/keyed_insertion_recovery"),
    Path("results/quadruped/mobile_keyed_insertion_recovery"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--confirmation", type=Path,
                        help="Completed frozen confirmation directory (CPU analysis only).")
    parser.add_argument("--output", type=Path,
                        help="New derived statistics directory; original files are never replaced.")
    args = parser.parse_args()
    if args.confirmation:
        if args.roots or not args.output:
            parser.error("--confirmation requires --output and cannot be mixed with --root")
        from genedynamics.experiments.utils.metrics import summarize_confirmation

        report = summarize_confirmation(
            Path(__file__).resolve().parents[3], args.confirmation, args.output,
            references={"main": "cross_ebdm", "ablation": "gated_full"},
            ablation_terms={
                "continued_training": {"task_only": 1, "frozen_warm": -1},
                "matching_without_band": {"matching_no_band": 1, "task_only": -1},
                "band_without_matching": {"band_only": 1, "task_only": -1},
                "matching_with_band": {"ungated_full": 1, "band_only": -1},
                "band_with_matching": {"ungated_full": 1, "matching_no_band": -1},
                "matching_band_interaction": {
                    "ungated_full": 1, "band_only": -1,
                    "matching_no_band": -1, "task_only": 1,
                },
                "selective_gate": {"gated_full": 1, "ungated_full": -1},
            },
            # T1--T3 reuse the same frozen actor under two table roles.  Formal
            # runner interruptions remain scored failures but cannot be action-
            # stream equality observations.  T4 main uses the separately frozen
            # completion-projected actor, so its gated-full row is not an alias.
            aliases={"cross_ebdm": {
                "method": "gated_full", "tasks": ["T1", "T2", "T3"],
                "tolerated_failure_categories": ["runner_interruption_after_reset"],
            }},
            permission_classes={
                **{method: "B-log" for method in (
                    "cross_ebdm", "task_only", "fada",
                    "frozen_warm", "matching_no_band", "band_only",
                    "ungated_full", "gated_full",
                )},
                "acp": "C-supervised",
                "human2sim2robot": "C-supervised-model",
                "dial_transfer": "B-log-model",
                "response_mpc": "B-log-model",
                "public_geometry": "no-target-learning",
                "public_source_prior": "no-target-learning",
            },
            contract_limitations=[
                "Formal journals persist frozen online evaluator outputs, reset receipts, and action-stream hashes but not complete physical traces; offline statistics cannot independently recompute every physical predicate.",
                "T4 Cross-EBDM uses the completion-projected final actor; T4 gated-full is the earlier gated ablation actor and is analyzed as a distinct method rather than duplicate evidence.",
            ],
            protocol_sources=[],
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.output:
        parser.error("--output requires --confirmation")
    rows = []
    roots = args.roots or TASK_ROOTS
    for path in sorted(path for root in roots for path in root.glob("**/results.json")):
        parts = path.parts
        if not any(section in parts for section in ("main", "baseline", "ablation")):
            continue
        payload = json.loads(path.read_text())
        rows.append({
            "path": str(path),
            "level": payload.get("level"),
            "seed": payload.get("seed"),
            "metrics": payload.get("metrics", {}),
            "execution_status": payload.get("execution_status", {"state": "completed"}),
        })
    print(json.dumps({"runs": rows, "count": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
