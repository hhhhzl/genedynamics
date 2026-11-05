"""Atomic checkpoint writes and integrity checks; weights-only loading by default."""

import json
import os
import tempfile
from pathlib import Path

import torch

from genedynamics.experiments.artifacts.provenance import atomic_json
from genedynamics.learning.datasets.sequences.storage import file_digest


def save_bundle(path, payload, *, metadata, restore_level="learner"):
    if restore_level not in {"weights", "learner", "full"}:
        raise ValueError("Unknown checkpoint restore level")
    if (
        restore_level == "full"
        and not {"execution_state", "replay_cursor", "rng_state"} <= payload.keys()
    ):
        raise ValueError("Full restoration requires execution, replay and RNG state")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(descriptor)
    try:
        torch.save(dict(payload, schema_version=1), temporary)
        checksum = file_digest(temporary)
        os.replace(temporary, path)
        atomic_json(
            path.with_suffix(path.suffix + ".json"),
            {
                "schema_version": 1,
                "sha256": checksum,
                "restore_level": restore_level,
                "metadata": metadata,
            },
        )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return checksum


def load_bundle(path, *, device="cpu", expected_metadata=None):
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if manifest["schema_version"] != 1 or file_digest(path) != manifest["sha256"]:
        raise ValueError("Checkpoint schema or checksum mismatch")
    for key, value in (expected_metadata or {}).items():
        if manifest["metadata"].get(key) != value:
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    return torch.load(path, map_location=device, weights_only=True), manifest
