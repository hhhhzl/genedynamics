"""Record actual checkouts and artifact bytes, not only package versions."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def digest_mapping(mapping):
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, default=str).encode()
    ).hexdigest()


def checkout_record(root):
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = "unversioned", True
    return {"path": str(root), "revision": revision, "dirty": dirty}


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(content, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_tree_digest(root):
    """Fingerprint Python source bytes independent of checkout/install location."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
