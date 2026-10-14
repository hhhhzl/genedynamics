"""Fetch pinned public FADA oracle checkpoints with author LFS SHA256 verification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

REPOSITORY = "AngchenXie/fada-checkpoints"
REVISION = "8f08f077c32b6f9448587d8e34657dbb8088d76b"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(output, entry):
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid author checkpoint path")
    destination = output.joinpath(*relative.parts)
    if destination.is_symlink() or not destination.resolve().is_relative_to(output):
        raise ValueError("Checkpoint destination escapes output")
    expected = entry["lfs"]["oid"]
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{entry['path']}"
    if destination.exists():
        if destination.stat().st_size != entry["size"] or sha256(destination) != expected:
            raise ValueError(f"Existing checkpoint failed immutable author hash: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            while chunk := response.read(8 * 1024**2):
                digest.update(chunk)
                stream.write(chunk)
        if temporary.stat().st_size != entry["size"] or digest.hexdigest() != expected:
            raise ValueError(f"Downloaded checkpoint failed author LFS hash: {destination}")
        temporary.replace(destination)
    return {"path": entry["path"], "url": url, "bytes": entry["size"], "sha256": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "fetch"))
    parser.add_argument("--robot", choices=("g1", "t1"), default="g1")
    parser.add_argument("--selection", choices=("final", "history"), default="final")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-gib", type=float, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if "latex" in output.parts:
        parser.error("Native asset evidence belongs outside the paper draft")
    if not 1 <= args.workers <= 8 or args.max_gib <= 0:
        parser.error("workers must be in [1,8] and max-gib positive")
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=false"
    with urllib.request.urlopen(url, timeout=60) as response:
        inventory = json.load(response)
    selected = [entry for entry in inventory
                if re.fullmatch(rf"{args.robot}/model_\d+\.pt", entry.get("path", ""))
                and (args.selection == "history" or entry["path"].endswith("/model_24999.pt"))]
    if not selected:
        raise ValueError("No requested pinned native checkpoints found")
    total = sum(entry["size"] for entry in selected)
    if total > args.max_gib * 1024**3:
        raise ValueError(f"Native checkpoint selection exceeds budget: {total} bytes")
    report = {"repository": REPOSITORY, "revision": REVISION, "inventory_url": url,
              "robot": args.robot, "selection": args.selection, "selected_bytes": total,
              "checkpoint_count": len(selected), "contains_student_checkpoint": False,
              "qualification": "asset availability and integrity only", "members": []}
    (output / f"{args.robot}.inventory.json").write_text(json.dumps({**report, "entries": selected}, indent=2) + "\n")
    if args.action == "fetch":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(fetch, output, entry) for entry in selected]
            for result in as_completed(pending):
                report["members"].append(result.result())
                report["members"].sort(key=lambda item: item["path"])
                (output / f"{args.robot}.manifest.json").write_text(json.dumps(report, indent=2) + "\n")
                if len(report["members"]) % 20 == 0:
                    print(json.dumps({"downloaded": len(report["members"]), "total": len(selected)}), flush=True)
    print(json.dumps({key: value for key, value in report.items() if key != "members"}, indent=2))


if __name__ == "__main__":
    main()
