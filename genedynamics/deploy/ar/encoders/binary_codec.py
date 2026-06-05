"""Binary codec for ``WorldSnapshot`` — gzip-compressed JSON (works today).

A drop-in binary wire form sent as WebSocket **binary** frames: same message
shape as ``json_codec``/`world.fbs`, just smaller on the wire. It needs no
codegen, so it's the interim "binary + compressed" P2 path. The zero-copy
**FlatBuffers** codec (``flatbuffers_codec``, needs ``flatc``) is the final hot
path; both are interchangeable behind the server's ``encode`` hook.
"""

from __future__ import annotations

import gzip
import json
from typing import Any, Dict

BINARY = True


def encode(snapshot: Dict[str, Any]) -> bytes:
    """Serialize a snapshot dict to gzip-compressed JSON bytes (binary frame)."""
    raw = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=6)


def decode(data: bytes) -> Dict[str, Any]:
    """Inflate gzip-JSON bytes back into a snapshot dict."""
    return json.loads(gzip.decompress(data))
