"""Fetch or verify immutable, sparse official baseline source checkouts.

Run with system Python. Large assets, LFS weights and upstream environments are
not downloaded. Existing worktrees are verified, never reset or overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements/baselines/sources.lock.json"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    ).strip()


def verify(spec: dict) -> dict:
    root = ROOT / "baselines" / spec["directory"]
    if root.is_symlink():
        raise ValueError(f"source checkout cannot be a symlink: {root}")
    if not (root / ".git").exists():
        raise FileNotFoundError(f"source missing: {root}; run fetch")
    commit = git(root, "rev-parse", "HEAD")
    if commit != spec["commit"]:
        raise ValueError(f"{root.name}: unexpected commit {commit}")
    if git(root, "remote", "get-url", "origin") != spec["repository"]:
        raise ValueError(f"{root.name}: unexpected origin")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"{root.name}: tracked source modified")
    for name, digest in spec["licenses"].items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{root.name}: license mismatch: {name}")
    missing = [name for name in spec["required_source_files"] if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root.name}: sparse source missing: {missing}")
    return {"method": spec["method"], "commit": commit, "status": "source_verified",
            "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
            "path": str(root.relative_to(ROOT)),
            "scope": "source only; native assets, training and task qualification pending"}


def fetch(spec: dict) -> dict:
    base = ROOT / "baselines"
    if base.is_symlink():
        raise ValueError("baselines must not be a symlink")
    base.mkdir(exist_ok=True)
    destination = base / spec["directory"]
    if destination.exists():
        return verify(spec)
    # Publish only a fully checked out pinned tree; do not adopt existing dirs.
    with tempfile.TemporaryDirectory(prefix=".source-", dir=base) as temporary:
        root = Path(temporary) / spec["directory"]
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--no-checkout", spec["repository"], str(root)], check=True, env=env)
        if git(root, "rev-parse", "HEAD") != spec["commit"]:
            subprocess.run(["git", "-C", str(root), "fetch", "--depth", "1", "origin",
                            spec["commit"]], check=True, env=env)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "--cone",
                        *spec["sparse_paths"]], check=True, env=env)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", spec["commit"]],
                       check=True, env=env)
        root.rename(destination)
    return verify(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "verify"))
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    specs = json.loads(LOCK.read_text())["sources"]
    by_name = {spec["method"]: spec for spec in specs}
    selected = args.methods or list(by_name)
    unknown = set(selected) - set(by_name)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    operation = fetch if args.action == "fetch" else verify
    print(json.dumps([operation(by_name[name]) for name in selected], indent=2))


if __name__ == "__main__":
    main()
