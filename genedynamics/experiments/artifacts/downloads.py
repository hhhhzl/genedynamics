"""Atomic artifact downloads record bytes and source URLs for reproducibility."""

import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from genedynamics.learning.datasets.sequences.storage import file_digest


def download_artifact(url, destination, *, expected_sha256=None, timeout=60):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with (
            os.fdopen(descriptor, "wb") as target,
            urlopen(url, timeout=timeout) as response,
        ):
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
        checksum = file_digest(temporary)
        if expected_sha256 and checksum != expected_sha256:
            raise ValueError("Downloaded artifact checksum mismatch")
        os.replace(temporary, destination)
        return {"url": url, "sha256": checksum, "bytes": destination.stat().st_size}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
