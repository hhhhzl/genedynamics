"""WebSocket broadcaster for the corridor scene contract (M1).

Streams :meth:`SceneSource.to_contract` to every connected client so AR
devices and the robot side share one obstacle definition in the Vicon world
frame. Built on the ``websockets`` library (the canonical, production-grade
Python WebSocket implementation). One process exposes:

* ``GET /``    — a tiny live HTML viewer (open it in a browser), and
* ``ws://…/ws`` — the raw WebSocket stream (point ``wscat`` here).

Each broadcast is the contract augmented with a monotonically increasing
``seq`` and ``server_time`` so the stream is visible even when the scene is
static; changes (preset swap, obstacle edit, watched-file reload, or a client
control message) appear on the next tick.

Run::

    python -m genedynamics.deploy.ar.scene_server --preset zone_d --port 8765
    # browser:  http://localhost:8765/
    # wscat:    wscat -c ws://localhost:8765/ws

Mutate the scene live via any of:

* editing a watched JSON spec — ``--watch scene.json`` (reloaded on mtime
  change),
* a control message over the socket, e.g.
  ``{"cmd": "set_preset", "preset": "zone_a"}``, or
* the in-process API (``source.set_preset(...)`` / ``set_obstacles(...)``)
  when the server is embedded in another program.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from http import HTTPStatus
from pathlib import Path
from typing import Optional

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from genedynamics.deploy.ar.scene_source import SceneSource

__all__ = ["SceneServer"]


_INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>corridor scene</title><style>
body{font-family:ui-monospace,Menlo,monospace;margin:1rem;background:#111;color:#ddd}
#hdr{font-size:14px;margin-bottom:.5rem}#hdr b{color:#6cf}
button{background:#222;color:#ddd;border:1px solid #444;padding:.3rem .6rem;margin-right:.3rem;cursor:pointer}
pre{background:#000;padding:.6rem;border:1px solid #333;max-height:75vh;overflow:auto;font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#3c3;margin-right:.4rem}
</style></head><body>
<div id=hdr><span class=dot id=dot></span><b id=preset>?</b>
 &nbsp; obstacles: <b id=nobs>?</b> &nbsp; rev: <b id=rev>?</b> &nbsp; seq: <b id=seq>?</b></div>
<div>switch:
 <button onclick=p('zone_a')>zone_a</button><button onclick=p('zone_b')>zone_b</button>
 <button onclick=p('zone_c')>zone_c</button><button onclick=p('zone_d')>zone_d</button>
 <button onclick=p('medium')>medium</button></div>
<pre id=out>connecting…</pre>
<script>
let ws;
function connect(){
 ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
 ws.onmessage=e=>{const c=JSON.parse(e.data);
  document.getElementById('preset').textContent=c.scene_preset||'(custom)';
  document.getElementById('nobs').textContent=(c.obstacles||[]).length;
  document.getElementById('rev').textContent=c.revision;
  document.getElementById('seq').textContent=c.seq;
  document.getElementById('out').textContent=JSON.stringify(c,null,2);
  document.getElementById('dot').style.background='#3c3';};
 ws.onclose=()=>{document.getElementById('dot').style.background='#c33';setTimeout(connect,800);};
}
function p(name){if(ws&&ws.readyState===1)ws.send(JSON.stringify({cmd:'set_preset',preset:name}));}
connect();
</script></body></html>"""


class SceneServer:
    """Broadcast a :class:`SceneSource` contract over WebSocket.

    Embeddable (``await start()`` / ``await stop()`` inside an existing event
    loop) or standalone (:meth:`run`).
    """

    def __init__(
        self,
        source: SceneSource,
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        hz: float = 10.0,
        watch_path: Optional[str | Path] = None,
        quiet: bool = False,
    ) -> None:
        self.source = source
        self.host = host
        self.port = int(port)
        self.hz = float(hz)
        self.watch_path = str(watch_path) if watch_path else None
        self.quiet = quiet

        self._server = None
        self._cm = None
        self._bcast_task: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None
        self._seq = 0
        self._watch_mtime: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Bind and begin broadcasting; returns once the socket is serving."""
        self._stop = asyncio.Event()
        self._seq = 0
        self._cm = serve(self._handler, self.host, self.port, process_request=self._process_request)
        self._server = await self._cm.__aenter__()
        self._bcast_task = asyncio.create_task(self._broadcast_loop())
        if not self.quiet:
            print(f"[scene_server] serving http://{self.host}:{self.port}/  "
                  f"(ws://{self.host}:{self.port}/ws) @ {self.hz:g} Hz"
                  + (f"  watching {self.watch_path}" if self.watch_path else ""),
                  flush=True)

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._bcast_task is not None:
            try:
                await self._bcast_task
            except asyncio.CancelledError:
                pass
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._server = None

    def run(self) -> None:
        """Blocking CLI entry: serve until Ctrl-C."""
        async def _main() -> None:
            await self.start()
            try:
                await self._stop.wait()  # type: ignore[union-attr]
            finally:
                await self.stop()
        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            if not self.quiet:
                print("\n[scene_server] stopped", flush=True)

    # ------------------------------------------------------------------
    # WS / HTTP handlers
    # ------------------------------------------------------------------
    def _process_request(self, connection: ServerConnection, request: Request) -> Optional[Response]:
        """Serve the HTML viewer for plain GETs; let ``/ws`` upgrade to WS."""
        path = request.path.split("?", 1)[0]
        if path in ("/ws", "/ws/"):
            return None  # proceed with the WebSocket handshake
        body = _INDEX_HTML.encode("utf-8")
        headers = Headers()
        headers["Content-Type"] = "text/html; charset=utf-8"
        headers["Content-Length"] = str(len(body))
        return Response(HTTPStatus.OK, "OK", headers, body)

    async def _handler(self, connection: ServerConnection) -> None:
        # Immediate snapshot so a new client never waits a full tick.
        await connection.send(self._payload())
        try:
            async for raw in connection:
                self._handle_control(raw if isinstance(raw, str) else raw.decode("utf-8", "ignore"))
        except Exception:  # noqa: BLE001 — client drop / protocol error; websockets cleans up
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _payload(self) -> str:
        contract = self.source.to_contract()
        return json.dumps({**contract, "seq": self._seq, "server_time": time.time()})

    def _handle_control(self, raw: str) -> None:
        try:
            cmd = json.loads(raw)
        except Exception:
            return
        kind = cmd.get("cmd")
        try:
            if kind == "set_preset" and cmd.get("preset"):
                self.source.set_preset(str(cmd["preset"]))
            elif kind == "set_obstacles":
                self.source.set_obstacles(cmd.get("obstacles") or [])
            elif kind == "set_robot_pose_world":
                self.source.set_robot_pose_world(cmd.get("pose"))
            elif kind == "set_plan_xy_scene":
                self.source.set_plan_xy_scene(cmd.get("plan"))
            elif kind == "set_t_world_scene":
                self.source.set_T_world_scene(cmd.get("T_world_scene") or cmd.get("value"))
        except Exception as exc:  # noqa: BLE001 — a bad control msg must not kill the loop
            if not self.quiet:
                print(f"[scene_server] control '{kind}' failed: {exc}", flush=True)

    async def _maybe_reload(self) -> None:
        if not self.watch_path:
            return
        try:
            mtime = os.path.getmtime(self.watch_path)
        except OSError:
            return
        if self._watch_mtime is None or mtime != self._watch_mtime:
            self._watch_mtime = mtime
            try:
                self.source.load_file(self.watch_path)
                if not self.quiet:
                    print(f"[scene_server] reloaded {self.watch_path} → rev {self.source.revision}", flush=True)
            except Exception as exc:  # noqa: BLE001
                if not self.quiet:
                    print(f"[scene_server] reload failed: {exc}", flush=True)

    async def _broadcast_loop(self) -> None:
        assert self._stop is not None
        period = 1.0 / self.hz if self.hz > 0 else 0.1
        while not self._stop.is_set():
            await self._maybe_reload()
            conns = self._server.connections if self._server is not None else ()
            if conns:
                broadcast(conns, self._payload())
            self._seq += 1
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_source(args: argparse.Namespace) -> SceneSource:
    t = {"x": args.t_world_scene[0], "y": args.t_world_scene[1], "yaw": args.t_world_scene[2]}
    # A watched file is the authoritative spec; otherwise --scene-file, else --preset.
    if args.watch:
        return SceneSource.from_file(args.watch, T_world_scene=t)
    if args.scene_file:
        return SceneSource.from_file(args.scene_file, T_world_scene=t)
    return SceneSource.from_preset(args.preset, T_world_scene=t)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="zone_d", help="Corridor scene preset (default: zone_d).")
    p.add_argument("--scene-file", default=None, help="JSON scene/contract spec to load instead of a preset.")
    p.add_argument("--watch", default=None, help="JSON spec to hot-reload on mtime change (live obstacle edits).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--hz", type=float, default=10.0, help="Broadcast rate (default 10).")
    p.add_argument("--t-world-scene", type=float, nargs=3, metavar=("X", "Y", "YAW"),
                   default=[0.0, 0.0, 0.0], help="SE(2) scene→world transform (Vicon-measured).")
    args = p.parse_args(argv)

    source = _build_source(args)
    SceneServer(source, host=args.host, port=args.port, hz=args.hz, watch_path=args.watch).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
