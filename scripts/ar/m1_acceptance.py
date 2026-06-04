#!/usr/bin/env python
"""M1 acceptance: scene_server streams the contract and reflects live edits.

Mirrors the manual acceptance (browser / wscat connect; edit scene_source →
client updates) deterministically:

1. start SceneServer(zone_d) in-process,
2. HTTP GET /            → the browser viewer HTML is served,
3. WS connect /ws        → receive a streaming contract (zone_d, 2 obstacles),
4. source.set_preset("zone_a")  (in-process edit) → client sees 1 obstacle,
5. client sends {"cmd":"set_preset","preset":"zone_c"} → client sees 6 obstacles.

Run (websockets only; no MuJoCo needed)::

    /opt/anaconda3/envs/fedguide/bin/python scripts/ar/m1_acceptance.py
"""

from __future__ import annotations

import asyncio
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from websockets.asyncio.client import connect  # noqa: E402

from genedynamics.deploy.ar.scene_server import SceneServer  # noqa: E402
from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402

HOST, PORT = "127.0.0.1", 8791
EXPECT = {"zone_d": 2, "zone_a": 1, "zone_c": 6}  # obstacle counts per preset


def _http_get(path: str) -> str:
    with urllib.request.urlopen(f"http://{HOST}:{PORT}{path}", timeout=3) as r:
        return r.read().decode("utf-8")


async def _recv_until(ws, preset: str, *, tries: int = 60):
    """Read stream frames until one carries `preset` (skip stale frames)."""
    import json
    for _ in range(tries):
        c = json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0))
        if c.get("scene_preset") == preset:
            return c
    raise AssertionError(f"never saw preset={preset}")


async def main() -> int:
    src = SceneSource.from_preset("zone_d", T_world_scene={"x": 1.2, "y": 0.4, "yaw": 0.0})
    srv = SceneServer(src, host=HOST, port=PORT, hz=25.0, quiet=True)
    await srv.start()
    failures = []
    try:
        # (2) browser path: GET / returns the viewer HTML
        html = await asyncio.get_event_loop().run_in_executor(None, _http_get, "/")
        ok_html = "<html" in html.lower() and "corridor scene" in html
        print(f"[2] GET /            -> {'PASS' if ok_html else 'FAIL'} (html {len(html)}B)")
        failures.append(ok_html)

        async with connect(f"ws://{HOST}:{PORT}/ws") as ws:
            # (3) initial stream frame: zone_d / 2 obstacles + transform present
            c0 = await _recv_until(ws, "zone_d")
            n0 = len(c0["obstacles"])
            ok0 = n0 == EXPECT["zone_d"] and c0["T_world_scene"] == {"x": 1.2, "y": 0.4, "yaw": 0.0}
            print(f"[3] WS connect       -> {'PASS' if ok0 else 'FAIL'} "
                  f"(preset={c0['scene_preset']} obstacles={n0} seq={c0['seq']} rev={c0['revision']})")
            failures.append(ok0)

            # (4) in-process edit reflected live
            src.set_preset("zone_a")
            c1 = await _recv_until(ws, "zone_a")
            ok1 = len(c1["obstacles"]) == EXPECT["zone_a"]
            print(f"[4] source.set_preset(zone_a) -> {'PASS' if ok1 else 'FAIL'} "
                  f"(obstacles={len(c1['obstacles'])} rev={c1['revision']})")
            failures.append(ok1)

            # (5) client control message reflected live
            import json
            await ws.send(json.dumps({"cmd": "set_preset", "preset": "zone_c"}))
            c2 = await _recv_until(ws, "zone_c")
            ok2 = len(c2["obstacles"]) == EXPECT["zone_c"]
            print(f"[5] client cmd set_preset(zone_c) -> {'PASS' if ok2 else 'FAIL'} "
                  f"(obstacles={len(c2['obstacles'])} rev={c2['revision']})")
            failures.append(ok2)
    finally:
        await srv.stop()

    passed = all(failures)
    print(f"\nM1 acceptance: {'ALL PASS ✅' if passed else 'FAILURES ❌'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
