#!/usr/bin/env python3
"""CPU audit for the immutable T3 first-intervention risk-data run.

The frozen collector's audit path expects bare SHA pointers, while saved
evidence receipts also carry outcome metadata. Normalize only that pointer
shape, preserving the collector's original SHA-bound audit calculations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genedynamics.core.artifacts import atomic_json
from scripts.paper.cross_ebdm import collect_t3_first_intervention_force_risk as frozen


def audit(config_path: Path) -> dict:
    config, roster, _ = frozen.checked_config(config_path)
    original = frozen._checked

    def checked(pointer: dict, label: str) -> Path:
        if label == "T3 risk evidence":
            if not {"path", "sha256", "audit_passed", "reset_physical_hash", "outcome"} <= set(pointer):
                raise ValueError("saved T3 risk evidence receipt is incomplete")
            pointer = {"path": pointer["path"], "sha256": pointer["sha256"]}
        return original(pointer, label)

    frozen._checked = checked
    try:
        result = frozen.audit(config_path, config, roster)
    finally:
        frozen._checked = original
    runner = Path(__file__).resolve()
    result["independent_audit_entrypoint"] = {
        "path": str(runner.relative_to(ROOT)),
        "sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    result = audit(path)
    config = json.loads(path.read_text())
    output = ROOT / config["output"] / "independent_audit.json"
    atomic_json(output, result)
    print(json.dumps({"status": result["status"],
                      "request_count": result["request_count"],
                      "physical_evidence_verified": result["physical_evidence_verified"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
