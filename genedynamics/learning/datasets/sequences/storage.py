"""Immutable, checksummed episode shards; split by original sequence."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

from .schema import Episode


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EpisodeStore:
    schema_version = 1

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, episode: Episode):
        key = hashlib.sha256(episode.identifier.encode()).hexdigest()
        data_path, record_path = self.root / f"{key}.npz", self.root / f"{key}.json"
        if record_path.exists() or data_path.exists():
            raise FileExistsError(f"Episode is immutable: {episode.identifier}")
        for record in self.records():
            if record["source_sequence"] == episode.source_sequence and record["split"] != episode.split:
                raise ValueError("Original sequence cannot occur in multiple dataset splits")
        descriptor, temporary = tempfile.mkstemp(dir=self.root, suffix=".npz")
        os.close(descriptor)
        try:
            np.savez_compressed(temporary, **episode.arrays)
            checksum = file_digest(temporary)
            os.replace(temporary, data_path)
            record = {"schema_version": self.schema_version, "identifier": episode.identifier,
                      "source_sequence": episode.source_sequence, "split": episode.split,
                      "transitions": len(episode), "file": data_path.name, "sha256": checksum,
                      "metadata": dict(episode.metadata)}
            record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return record_path

    def records(self, split=None):
        records = [json.loads(path.read_text()) for path in sorted(self.root.glob("*.json"))]
        return [record for record in records if split is None or record["split"] == split]

    def load(self, record):
        if record["schema_version"] != self.schema_version:
            raise ValueError("Unsupported episode schema")
        path = self.root / record["file"]
        if path.parent.resolve() != self.root.resolve() or file_digest(path) != record["sha256"]:
            raise ValueError("Episode path or checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return Episode(record["identifier"], record["source_sequence"], record["split"], arrays, record["metadata"])

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.records(), sort_keys=True).encode()).hexdigest()
