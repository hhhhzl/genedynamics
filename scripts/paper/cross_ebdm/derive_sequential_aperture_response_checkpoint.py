#!/usr/bin/env python3
"""Bind the learned obstacle response objective to the sequential task phases."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.solvers.single.cross_ebdm.learning.semantic_phase import (
    validate_checkpoint,
)


SOURCE = ROOT / (
    "results/arm/obstacle_transfer/training/aperture_brake_response_transport/"
    "checkpoints/full_budget_120.json"
)
QUALIFICATION = ROOT / (
    "results/arm/obstacle_transfer/validation/"
    "sequential_aperture_reorientation/report.json"
)
OUTPUT = ROOT / (
    "results/arm/obstacle_transfer/training/"
    "sequential_aperture_response_transport"
)


def pointer(path):
    path = Path(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    qualification = json.loads(QUALIFICATION.read_text())
    if qualification.get("passed") is not True:
        raise ValueError("passed sequential-aperture source qualification required")
    checkpoint = deepcopy(json.loads(SOURCE.read_text()))
    checkpoint["task"] = "sequential_aperture_reorientation"
    selected = float(checkpoint["selection"]["selected_scale"])
    checkpoint["phase_scales"] = [1., 1., 1., 1., 1., 1., selected]
    # The full payload envelope crosses a solid frame during phases 2 and 5.
    # Both alignment/crossing pairs and the chamber reorientation stay
    # task-owned; the transferred response residual is admitted only after
    # both complete envelopes have cleared their frames.
    checkpoint["active_phases"] = [6]
    validate_checkpoint(checkpoint, task="sequential_aperture_reorientation")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    destination = OUTPUT / "full_budget_120.json"
    destination.write_text(json.dumps(
        checkpoint, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    manifest = {
        "version": "sequential-aperture-response-binding-v1",
        "status": "development_binding_complete",
        "source_checkpoint": pointer(SOURCE),
        "source_qualification": pointer(QUALIFICATION),
        "derived_checkpoint": pointer(destination),
        "binding_change": {
            "task": "sequential_aperture_reorientation",
            "active_public_phases": [6],
            "selected_scale_unchanged": selected,
            "response_weights_unchanged": True,
            "objective_rows_unchanged": True
        },
        "formal_performance_evidence": False
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True,
    ) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
