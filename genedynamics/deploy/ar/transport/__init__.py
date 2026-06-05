"""Transport: serve a ``WorldState`` to AR clients.

- ``world_ws`` — generic WebSocket streamer: keyframe on connect + per-client
  deltas, codec-pluggable (JSON text / gzip-binary / FlatBuffers). No NATS.
- (P3) ``nats_pub`` — publish to the external edgecloud NATS backbone.

The existing ``scene_server`` (corridor JSON contract) stays as the M1 debug
path; ``world_ws`` is its generalized, entity-model successor.
"""

from __future__ import annotations

from genedynamics.deploy.ar.transport.world_ws import WorldStreamServer

__all__ = ["WorldStreamServer"]
