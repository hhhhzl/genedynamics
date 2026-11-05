"""Run the frozen CPU source qualification for semantic aperture braking."""
import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.aperture_qualification import qualify_aperture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/arm/obstacle_transfer/validation/aperture_brake_cpu.json"),
    )
    args = parser.parse_args()
    result = qualify_aperture(
        args.config,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
