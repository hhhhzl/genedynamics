#!/usr/bin/env python3
"""Fetch or verify pinned upstream baseline source checkouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements/baselines/sources.lock.json"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    ).strip()


def verify(spec: dict) -> dict:
    """Verify one checkout managed by this setup command."""
    root = ROOT / "baselines" / spec["directory"]
    if root.is_symlink():
        raise ValueError(f"source checkout cannot be a symlink: {root}")
    if not (root / ".git").exists():
        raise FileNotFoundError(f"source missing: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != spec["commit"]:
        raise ValueError(f"{root.name}: unexpected commit {commit}")
    if _git(root, "remote", "get-url", "origin") != spec["repository"]:
        raise ValueError(f"{root.name}: unexpected origin")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"{root.name}: tracked source modified")
    for name, digest in spec["licenses"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{root.name}: license mismatch: {name}")
    missing = [
        name for name in spec["required_source_files"] if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"{root.name}: sparse source missing: {missing}")
    return {"method": spec["method"], "commit": commit, "status": "source_verified"}


def fetch(spec: dict) -> dict:
    """Create one sparse, detached checkout under the repository baseline cache."""
    base = ROOT / "baselines"
    if base.is_symlink():
        raise ValueError("baselines must not be a symlink")
    base.mkdir(exist_ok=True)
    destination = base / spec["directory"]
    if destination.exists():
        return verify(spec)
    with tempfile.TemporaryDirectory(prefix=".source-", dir=base) as temporary:
        source = Path(temporary) / spec["directory"]
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "--no-checkout", spec["repository"], str(source),
            ],
            check=True,
            env=env,
        )
        if _git(source, "rev-parse", "HEAD") != spec["commit"]:
            subprocess.run(
                ["git", "-C", str(source), "fetch", "--depth", "1", "origin", spec["commit"]],
                check=True,
                env=env,
            )
        subprocess.run(
            ["git", "-C", str(source), "sparse-checkout", "set", "--cone", *spec["sparse_paths"]],
            check=True,
            env=env,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", spec["commit"]],
            check=True,
            env=env,
        )
        source.rename(destination)
    return verify(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "verify"))
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    specs = json.loads(LOCK.read_text())["sources"]
    by_name = {spec["method"]: spec for spec in specs}
    selected = args.methods or list(by_name)
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    operation = fetch if args.action == "fetch" else verify
    print(json.dumps([operation(by_name[name]) for name in selected], indent=2))


if __name__ == "__main__":
    main()
