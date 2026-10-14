"""Fetch public, hash-pinned baseline assets without altering reference sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements/baselines/resources.lock.json"


def download_verified(spec: dict, output: Path, *, min_free_bytes=20 * 1024**3) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / spec["filename"]
    if destination.exists():
        digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
        if digest != spec["sha256"] or destination.stat().st_size != spec["bytes"]:
            raise ValueError(f"existing asset differs from pinned resource: {destination.name}")
    else:
        if shutil.disk_usage(output).free < min_free_bytes + spec["bytes"]:
            raise RuntimeError("insufficient free space for pinned asset")
        with tempfile.NamedTemporaryFile(dir=output, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            count = 0
            try:
                with urllib.request.urlopen(spec["url"], timeout=60) as response:
                    while chunk := response.read(1024**2):
                        count += len(chunk)
                        if count > spec["bytes"]:
                            raise ValueError("asset exceeds declared byte count")
                        stream.write(chunk)
                        digest.update(chunk)
                stream.flush()
                if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                    raise ValueError("asset size or SHA256 differs from pinned resource")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    receipt = {**spec, "local_path": str(destination.resolve()), "verified": True}
    (output / (spec["filename"] + ".receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources = {row["id"]: row for row in json.loads(LOCK.read_text())["resources"]}
    unknown = set(args.assets) - set(resources)
    if unknown:
        parser.error(f"unknown pinned assets: {sorted(unknown)}")
    for name in args.assets:
        receipt = download_verified(resources[name], args.output)
        print(json.dumps({key: receipt[key] for key in ("id", "local_path", "sha256", "bytes", "verified")}))


if __name__ == "__main__":
    main()
