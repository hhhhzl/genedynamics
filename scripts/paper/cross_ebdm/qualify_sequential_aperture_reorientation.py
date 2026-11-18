#!/usr/bin/env python3
"""Run CPU source qualification for sequential aperture reorientation."""
import argparse
from pathlib import Path

from genedynamics.tasks.transfer.aperture_qualification import qualify_aperture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/arm/obstacle_transfer/validation/"
            "sequential_aperture_reorientation_cpu.json"
        ),
    )
    args = parser.parse_args()
    result = qualify_aperture(
        args.config,
        progress=lambda row: print(row, flush=True),
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
