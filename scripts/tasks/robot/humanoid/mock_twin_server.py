#!/usr/bin/env python
"""Dependency-light MOCK twin server for AR client bring-up (Unity / Swift / web).

Emits the SAME ``WorldSnapshot`` JSON contract as the real
``run_twin_server.py`` (schema/world.fbs shape; see schema/conventions.md), but
needs **only** ``websockets`` — no jax / mujoco / scene_source.

The room layout (obstacles, walls, markers, robot) is read from a JSON file
(``room.json`` next to this script, or ``--scene <path>``). EDIT THAT FILE AND
SAVE — the server reloads it LIVE and pushes a fresh keyframe, no restart needed.
If the file is missing/invalid it falls back to a built-in zone_d layout.

World frame: right-handed, meters, x=forward, y=left, z=up; quat = (w,x,y,z).

Run::

    python scripts/tasks/robot/humanoid/mock_twin_server.py --port 8766
    # AR clients connect to ws://<this-host>:8766/

This is a TEST DOUBLE. For real obstacle geometry + Vicon poses use
``run_twin_server.py`` on the twin-server machine.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import websockets

SCHEMA_VERSION = 2

# Default colors per role (0xRRGGBBAA). Override per-entity with "color": "RRGGBBAA".
_C = {
    "wall": 0x8899AACC,
    "marker": 0x33CC66DD,
    "obstacle": 0xCC5544DD,
    "robot": 0x4488CCBB,
    "occluder": 0x00000000,
}

_DEFAULT_LAYOUT = {
    "site": "zone_d",
    "robot": {"pos": [0.5, 0.0, 0.65], "yaw_deg": 0, "moving": True, "path_to_x": 3.5,
              "radius": 0.18, "height": 1.3},
    "markers": [
        {"id": "start", "pos": [0.5, 0.0, 0.05], "radius": 0.08},
        {"id": "goal", "pos": [3.5, 0.0, 0.05], "radius": 0.08},
    ],
    "obstacles": [
        {"id": "ball_L1", "shape": "sphere", "pos": [1.6, 0.25, 0.9], "radius": 0.12},
        {"id": "ball_R1", "shape": "sphere", "pos": [2.4, -0.25, 0.9], "radius": 0.12},
        {"id": "wall_lo", "shape": "box", "pos": [2.0, -0.8, 1.0], "half": [2.0, 0.02, 1.0]},
        {"id": "wall_hi", "shape": "box", "pos": [2.0, 0.8, 1.0], "half": [2.0, 0.02, 1.0]},
    ],
}


# ---- contract helpers -------------------------------------------------------
def _yaw_quat(deg):
    h = math.radians(deg) * 0.5
    return (math.cos(h), 0.0, 0.0, math.sin(h))  # (w, x, y, z) about world +z


def _color(spec, role):
    if isinstance(spec, str):
        try:
            return int(spec, 16) & 0xFFFFFFFF
        except ValueError:
            pass
    return _C[role]


def _geom(kind, half=(0.0, 0.0, 0.0), radius=0.0, height=0.0):
    return {"kind": kind, "half_extents": [float(half[0]), float(half[1]), float(half[2])],
            "radius": float(radius), "height": float(height),
            "asset_uri": "", "joints": [], "polyline": []}


def _entity(eid, etype, p, geom, color, rev, q=(1.0, 0.0, 0.0, 0.0), meta=""):
    return {"id": eid, "type": etype, "frame": "world",
            "pose": {"p": [float(p[0]), float(p[1]), float(p[2])],
                     "q": [float(q[0]), float(q[1]), float(q[2]), float(q[3])]},
            "geom": geom, "color_rgba": int(color) & 0xFFFFFFFF, "rev": int(rev), "meta": meta}


def _obstacle_entity(o, rev):
    shape = o.get("shape", "box")
    q = _yaw_quat(o.get("yaw_deg", 0.0))
    if shape == "sphere":
        g = _geom("sphere", radius=o.get("radius", 0.1))
    elif shape in ("cylinder", "capsule"):
        g = _geom(shape, radius=o.get("radius", 0.1), height=o.get("height", 1.0))
    else:  # box
        g = _geom("box", half=o.get("half", [0.1, 0.1, 0.1]))
    etype = o.get("type") or ("wall" if str(o.get("id", "")).startswith("wall") else "obstacle")
    return _entity(o["id"], etype, o["pos"], g, _color(o.get("color"), etype), rev, q=q,
                   meta=o.get("meta", ""))


def _static_entities(cfg, rev):
    ents = []
    for m in cfg.get("markers", []):
        ents.append(_entity(m["id"], "goal", m["pos"], _geom("sphere", radius=m.get("radius", 0.08)),
                            _color(m.get("color"), "marker"), rev, q=_yaw_quat(m.get("yaw_deg", 0.0))))
    for o in cfg.get("obstacles", []):
        ents.append(_obstacle_entity(o, rev))
    return ents


def _robot_entities(cfg, rev, t):
    r = cfg.get("robot", {})
    if not r:
        return []
    x0, y0, z0 = r.get("pos", [0.5, 0.0, 0.65])
    if r.get("moving"):
        x1 = r.get("path_to_x", x0)
        u = 0.5 * (1.0 - math.cos(2.0 * math.pi * (t / 16.0)))  # 0..1, period 16 s
        x = x0 + (x1 - x0) * u
        yaw = r.get("yaw_deg", 0.0) + (0.0 if math.sin(2.0 * math.pi * (t / 16.0)) >= 0 else 180.0)
    else:
        x, yaw = x0, r.get("yaw_deg", 0.0)
    q = _yaw_quat(yaw)
    rad, h = r.get("radius", 0.18), r.get("height", 1.3)
    base = _entity("g1/base", "robot", (x, y0, z0), _geom("cylinder", radius=rad, height=h),
                   _color(r.get("color"), "robot"), rev, q=q)
    ents = [base]
    # The occluder is an invisible depth-only proxy so a REAL robot can hide the
    # holograms behind it. In the mock (virtual moving robot) it just carves a
    # moving hole in the walls, so it's OFF by default — set robot.occluder=true
    # in the room json only when you actually have a real robot to occlude with.
    if r.get("occluder"):
        ents.append(_entity("g1/occluder", "occluder", (x, y0, z0 + 0.05),
                            _geom("cylinder", radius=rad + 0.05, height=h + 0.1), _C["occluder"], rev, q=q,
                            meta='{"render":"depth_only"}'))
    return ents


def _snapshot(cfg, entities, is_keyframe, rev, removed=None):
    return {"schema_version": SCHEMA_VERSION, "site": cfg.get("site", "my_room"),
            "stamp_ns": int(time.time() * 1e9), "frame": "world", "tracker": "mock",
            "units": "m_rad", "is_keyframe": bool(is_keyframe), "entities": entities,
            "removed_ids": list(removed or [])}


# ---- layout file (live-reloaded) -------------------------------------------
class Layout:
    def __init__(self, path: Path):
        self.path = path
        self.cfg = dict(_DEFAULT_LAYOUT)
        self._mtime = None
        self.reload(force=True)

    def reload(self, force=False) -> bool:
        """Return True if the layout changed (file edited)."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            if force:
                print(f"[mock_twin] {self.path.name} not found — using built-in zone_d", flush=True)
            return False
        if not force and mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            data = json.loads(self.path.read_text())
            self.cfg = data
            print(f"[mock_twin] loaded layout '{self.path.name}': "
                  f"{len(data.get('obstacles', []))} obstacles, {len(data.get('markers', []))} markers, "
                  f"robot={'yes' if data.get('robot') else 'no'}", flush=True)
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"[mock_twin] !! bad {self.path.name} ({e}); keeping previous layout", flush=True)
            return False


# ---- server -----------------------------------------------------------------
_rev = 0
_scene_version = 0   # bumped whenever ANY client switches the scene -> all clients re-keyframe


def _next_rev():
    global _rev
    _rev += 1
    return _rev


def _resolve_scene(name, here: Path):
    cand = here / f"room_{str(name).strip().lower()}.json"
    return cand if cand.exists() else None


async def _handler(ws, layout: Layout, hz: float, t0: float, here: Path):
    global _scene_version
    peer = getattr(ws, "remote_address", ("?", 0))
    print(f"[mock_twin] client connected {peer}", flush=True)

    def keyframe():
        cfg = layout.cfg
        ents = _static_entities(cfg, _next_rev()) + _robot_entities(cfg, _next_rev(), time.time() - t0)
        return _snapshot(cfg, ents, True, _rev)

    # --- receiver: handle scene-switch commands, e.g. {"scene":"zone_c"} ---
    async def receiver():
        global _scene_version
        try:
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                name = cmd.get("scene") or cmd.get("preset") or cmd.get("name")
                if not name:
                    continue
                p = _resolve_scene(name, here)
                if p is None:
                    print(f"[mock_twin] switch: room_{name}.json not found", flush=True)
                    continue
                layout.path = p
                layout.reload(force=True)
                _scene_version += 1   # all sender loops will re-keyframe
                print(f"[mock_twin] switched -> {layout.cfg.get('site')} (by {peer})", flush=True)
        except websockets.ConnectionClosed:
            pass

    recv_task = asyncio.ensure_future(receiver())

    await ws.send(json.dumps(keyframe()))
    print(f"[mock_twin] sent keyframe ({layout.cfg.get('site')})", flush=True)
    last_ver = _scene_version
    hb = 0
    hb_every = max(1, int(hz // 4))             # ~4 Hz heartbeat
    try:
        period = 1.0 / max(1.0, hz)
        while True:
            await asyncio.sleep(period)
            if _scene_version != last_ver:          # a client switched the scene
                last_ver = _scene_version
                await ws.send(json.dumps(keyframe()))
            elif layout.reload():                   # file edited live
                await ws.send(json.dumps(keyframe()))
            else:
                delta = _robot_entities(layout.cfg, _next_rev(), time.time() - t0)
                if delta:
                    await ws.send(json.dumps(_snapshot(layout.cfg, delta, False, _rev)))
                else:
                    hb += 1                          # heartbeat: wakes the client's receive loop
                    if hb >= hb_every:               # so it can flush queued commands (scene switch)
                        hb = 0
                        await ws.send(json.dumps(_snapshot(layout.cfg, [], False, _rev)))
    except websockets.ConnectionClosed:
        print(f"[mock_twin] client disconnected {peer}", flush=True)
    finally:
        recv_task.cancel()


async def _main(host, port, hz, scene_path, here):
    layout = Layout(scene_path)
    t0 = time.time()
    async with websockets.serve(lambda ws: _handler(ws, layout, hz, t0, here), host, port):
        print(f"[mock_twin] serving ws://{host}:{port}/ @ {hz:g} Hz  "
              f"(start={scene_path.name}; clients can switch with {{\"scene\":\"zone_x\"}})", flush=True)
        await asyncio.Future()


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--scene", type=Path, default=None,
                    help="explicit layout JSON (live-reloaded). Overrides --preset.")
    ap.add_argument("--preset", default=None,
                    help="scene name -> loads room_<preset>.json next to this script (e.g. zone_a).")
    ap.add_argument("--localization", default="mock", help="(accepted for parity)")
    args = ap.parse_args(argv)

    # Resolve which layout file to serve:
    #   --scene <path>      explicit (wins)
    #   --preset zone_a     -> room_zone_a.json (if it exists)
    #   (neither)           -> room.json
    if args.scene is not None:
        scene = args.scene
    elif args.preset:
        cand = here / f"room_{str(args.preset).strip().lower()}.json"
        scene = cand if cand.exists() else here / "room.json"
        if not cand.exists():
            print(f"[mock_twin] room_{args.preset}.json not found; falling back to room.json", flush=True)
    else:
        scene = here / "room.json"

    try:
        asyncio.run(_main(args.host, args.port, args.hz, scene, here))
    except KeyboardInterrupt:
        print("\n[mock_twin] stopped", flush=True)


if __name__ == "__main__":
    main()
