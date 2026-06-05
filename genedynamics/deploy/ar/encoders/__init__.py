"""Encoders: serialize a ``WorldSnapshot`` dict to the wire.

``json_codec`` is the debug / browser / today's-`scene_server` path. The P2
sibling ``flatbuffers_codec`` (generated from `schema/world.fbs`) is the
zero-copy hot path — same message shape, binary frames.
"""

from __future__ import annotations

from genedynamics.deploy.ar.encoders.json_codec import decode, encode

__all__ = ["encode", "decode"]
