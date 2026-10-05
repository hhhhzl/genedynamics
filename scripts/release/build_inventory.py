#!/usr/bin/env python3
"""Build a deterministic inventory for the V1 source tree and wheel."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[2]
IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "dist", "site"}
FORBIDDEN_PARTS = {"3dgs", "mbd3d", "mrmfmbd", "soft_robot", "codesign", "morphology"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    files = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        path = ROOT / raw.decode()
        if path == ROOT / "release" / "v1_inventory.json":
            continue
        if path.is_file() and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts):
            files.append(path)
    return sorted(files)


def _source_inventory() -> dict:
    files = _release_files()
    by_root: Counter[str] = Counter()
    by_suffix: Counter[str] = Counter()
    large_files = []
    forbidden = []
    media = []
    total_bytes = 0
    for path in files:
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        total_bytes += size
        by_root[relative.parts[0]] += 1
        by_suffix[path.suffix.lower() or "<none>"] += 1
        lowered = {part.lower() for part in relative.parts}
        if lowered.intersection(FORBIDDEN_PARTS):
            forbidden.append(relative.as_posix())
        if size >= 10 * 1024 * 1024:
            large_files.append({"path": relative.as_posix(), "bytes": size})
        if relative.parts[:2] == ("docs", "assets") and path.suffix.lower() in {
            ".gif", ".jpg", ".jpeg", ".png", ".webp", ".mp4"
        }:
            media.append({
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            })
    return {
        "file_count": len(files),
        "bytes": total_bytes,
        "files_by_root": dict(sorted(by_root.items())),
        "files_by_suffix": dict(sorted(by_suffix.items())),
        "large_files_10mib": large_files,
        "forbidden_v1_paths": forbidden,
        "media": media,
    }


def _wheel_inventory(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = sorted(item.filename for item in archive.infolist() if not item.is_dir())
    forbidden = [
        name for name in names
        if set(part.lower() for part in Path(name).parts).intersection(FORBIDDEN_PARTS)
    ]
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "file_count": len(names),
        "forbidden_v1_paths": forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "release" / "v1_inventory.json"
    )
    args = parser.parse_args()
    inventory = {
        "schema_version": 1,
        "release": "0.1.0",
        "source": _source_inventory(),
    }
    if args.wheel:
        wheel = args.wheel.resolve()
        if not wheel.is_file():
            parser.error(f"wheel does not exist: {wheel}")
        inventory["wheel"] = _wheel_inventory(wheel)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
