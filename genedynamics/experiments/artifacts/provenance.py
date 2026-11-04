"""Record actual checkouts and artifact bytes, not only package versions."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import os


def digest_mapping(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, default=str).encode()).hexdigest()


def checkout_record(root):
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
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
