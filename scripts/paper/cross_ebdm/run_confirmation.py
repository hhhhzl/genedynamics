#!/usr/bin/env python3
"""Launch the single frozen four-task transfer confirmation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from genedynamics.tasks.transfer.confirmation import TransferConfirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--continue-claimed", action="store_true")
    parser.add_argument("--package-report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/_protocol/four_task_transfer/confirmation_v14/run"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    native_library = (root / ".runtime/cross_ebdm/native/libcross_ebdm.so").resolve()
    if not native_library.is_file():
        raise FileNotFoundError(native_library)
    configured = os.environ.get("CROSS_EBDM_NATIVE_LIBRARY")
    if configured is not None and Path(configured).expanduser().resolve() != native_library:
        raise RuntimeError("CROSS_EBDM_NATIVE_LIBRARY points to a different library")
    os.environ["CROSS_EBDM_NATIVE_LIBRARY"] = str(native_library)
    report = TransferConfirmation(
        root, output=args.output, device=args.device,
        package_report=args.package_report,
        continue_claimed=args.continue_claimed,
        workers=args.workers,
        devices=None if args.devices is None else tuple(args.devices),
    ).run()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
