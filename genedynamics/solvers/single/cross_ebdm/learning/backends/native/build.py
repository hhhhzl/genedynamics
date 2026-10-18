"""Build the optional Cross-EBDM CPU inference library explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("ensemble.cpp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c++17", "-O2", "-fPIC", "-shared"]
    subprocess.run([args.compiler, *flags, str(source), "-o", str(args.output)], check=True)
    manifest = {
        "abi_version": 1,
        "scope": "CPU inference-only",
        "platform": platform.platform(),
        "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
        "flags": flags,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "library": str(args.output.resolve()),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
