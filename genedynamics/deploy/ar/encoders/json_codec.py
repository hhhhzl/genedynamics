"""JSON codec for ``WorldSnapshot`` messages (debug / browser / scene_server).

The single (de)serialization point for the JSON wire form. The P2 hot path will
add a ``flatbuffers_codec`` with the SAME message shape (`schema/world.fbs`),
sent as binary WebSocket frames over the existing ``scene_server`` — **no NATS**
(NATS is P3/edgecloud).
"""

from __future__ import annotations

import json
from typing import Any, Dict

_REQUIRED = ("schema_version", "frame", "units", "is_keyframe", "entities", "removed_ids")


def encode(snapshot: Dict[str, Any]) -> str:
    """Serialize a snapshot dict (from ``WorldState.snapshot``/``.delta``) to JSON."""
    return json.dumps(snapshot, separators=(",", ":"))


def decode(data: str | bytes) -> Dict[str, Any]:
    """Parse a JSON snapshot back into a dict."""
    return json.loads(data)


def validate(snapshot: Dict[str, Any]) -> None:
    """Raise ``ValueError`` if a snapshot is missing required fields / malformed."""
    missing = [k for k in _REQUIRED if k not in snapshot]
    if missing:
        raise ValueError(f"snapshot missing fields: {missing}")
    if not isinstance(snapshot["entities"], list):
        raise ValueError("entities must be a list")
    for e in snapshot["entities"]:
        if "id" not in e or "type" not in e or "pose" not in e:
            raise ValueError(f"malformed entity: {e!r}")
