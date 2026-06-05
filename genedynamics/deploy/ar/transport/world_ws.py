"""Generic WorldState WebSocket streamer (P2 transport, no NATS).

Each client gets a **keyframe** on connect, then **deltas** (only entities whose
rev exceeds that client's cursor, plus removed ids). Wire form is codec-pluggable
via ``encode`` — ``json_codec.encode`` (text) by default, ``binary_codec.encode``
(gzip bytes) or the future ``flatbuffers_codec.encode`` for the hot path; str is
sent as a text frame, bytes as a binary frame.

This is the entity-model successor to ``scene_server`` (which stays as the M1
corridor-contract debug path). Embeddable (``await start()/stop()``) or
standalone (:meth:`run`). NATS only enters at P3 (edgecloud).
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, Optional

from websockets.asyncio.server import ServerConnection, serve

from genedynamics.deploy.ar.encoders import json_codec
from genedynamics.deploy.ar.world_state import WorldState

__all__ = ["WorldStreamServer"]


class WorldStreamServer:
    def __init__(
        self,
        world: WorldState,
        *,
        host: str = "0.0.0.0",
        port: int = 8766,
        hz: float = 30.0,
        encode: Callable[[dict], "str | bytes"] = json_codec.encode,
        on_tick: Optional[Callable[[WorldState], None]] = None,
        quiet: bool = False,
    ) -> None:
        self.world = world
        self.host = host
        self.port = int(port)
        self.hz = float(hz)
        self.encode = encode
        self.on_tick = on_tick
        self.quiet = quiet

        self._cursors: Dict[ServerConnection, int] = {}
        self._cm = None
        self._server = None
        self._bcast: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._stop = asyncio.Event()
        self._cm = serve(self._handler, self.host, self.port)
        self._server = await self._cm.__aenter__()
        self._bcast = asyncio.create_task(self._loop())
        if not self.quiet:
            print(f"[world_ws] serving ws://{self.host}:{self.port}/ @ {self.hz:g} Hz "
                  f"({len(self.world)} entities)", flush=True)

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._bcast is not None:
            try:
                await self._bcast
            except asyncio.CancelledError:
                pass
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = self._server = None

    def run(self) -> None:
        async def _main():
            await self.start()
            try:
                await self._stop.wait()  # type: ignore[union-attr]
            finally:
                await self.stop()
        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            if not self.quiet:
                print("\n[world_ws] stopped", flush=True)

    # ------------------------------------------------------------------
    async def _handler(self, conn: ServerConnection) -> None:
        # Keyframe on connect; cursor starts at the world's current rev.
        await conn.send(self.encode(self.world.snapshot()))
        self._cursors[conn] = self.world.rev
        try:
            async for _ in conn:  # ignore inbound (control msgs could go here)
                pass
        except Exception:  # noqa: BLE001 — client drop; websockets cleans up
            pass
        finally:
            self._cursors.pop(conn, None)

    async def _loop(self) -> None:
        assert self._stop is not None
        period = 1.0 / self.hz if self.hz > 0 else 0.1
        while not self._stop.is_set():
            if self.on_tick is not None:
                try:
                    self.on_tick(self.world)
                except Exception as exc:  # noqa: BLE001
                    if not self.quiet:
                        print(f"[world_ws] on_tick failed: {exc}", flush=True)
            for conn, cursor in list(self._cursors.items()):
                if cursor >= self.world.rev:
                    continue  # nothing changed for this client
                msg, new_cursor = self.world.delta(cursor)
                if msg["entities"] or msg["removed_ids"]:
                    try:
                        await conn.send(self.encode(msg))
                    except Exception:
                        self._cursors.pop(conn, None)
                        continue
                self._cursors[conn] = new_cursor
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass
