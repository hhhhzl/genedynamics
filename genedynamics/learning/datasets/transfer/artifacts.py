"""Serialization helpers for immutable public transfer datasets."""

import json

from genedynamics.core.artifacts import atomic_json


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


__all__ = ["atomic_json", "canonical_bytes"]

