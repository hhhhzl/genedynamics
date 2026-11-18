#!/usr/bin/env python3
"""Run the frozen CPU source qualification for reposition-then-dock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.reposition_dock_qualification import (
    qualify_reposition_dock,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/quadruped/mobile_keyed_insertion_recovery/validation/"
            "reposition_then_dock_cpu.json"
        ),
    )
    args = parser.parse_args()
    result = qualify_reposition_dock(
        args.config, progress=lambda value: print(json.dumps(value), flush=True),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
