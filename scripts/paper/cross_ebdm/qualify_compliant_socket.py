"""Run task-owned CPU qualification and public log collection for passive T3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.compliant_socket_qualification import qualify_compliant_socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=Path("configs/arm/compliant_keyed_insertion_recovery/validation/cpu_qualification.json"))
    args = parser.parse_args()
    result = qualify_compliant_socket(
        args.config, progress=lambda row: print(json.dumps(row), flush=True),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
