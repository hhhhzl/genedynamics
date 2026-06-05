#!/usr/bin/env python
"""Twin server: stream the shared world (obstacles + live robot) to AR clients.

Ties the AR pipeline together (no NATS, no robot SDK):
  * corridor producer → static obstacles/walls/goal (world frame via T_world_scene),
  * tracker producer  → robot base + occluder, refreshed every tick from a
    localization plugin (mock / Vicon shm / ROS2 — tracker-agnostic),
  * WorldStreamServer → keyframe-on-connect + per-client deltas over WebSocket.

Run (fedguide; needs websockets, NOT mujoco)::

    python scripts/tasks/robot/humanoid/run_twin_server.py --preset zone_d \
        --t-world-scene 1.2 0.4 0.0 --localization vicon --port 8766
    # AR clients connect to ws://<host>:8766/

Use ``--localization mock`` for a dry run with no Vicon (fixed robot pose).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.producers import populate_corridor, populate_tracker  # noqa: E402
from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402
from genedynamics.deploy.ar.transport import WorldStreamServer  # noqa: E402
from genedynamics.deploy.ar.world_state import WorldState  # noqa: E402


class _FixedPoseLocalization:
    """Minimal world-frame localization for dry runs (no Vicon)."""

    def __init__(self, qpos: Sequence[float]):
        self._qpos = list(qpos)

    def get_state(self):
        import numpy as np
        return np.asarray(self._qpos, dtype=float), np.zeros(6, dtype=float)

    def get_last_update_time(self):
        import time
        return time.time()

    def health(self):
        return "ok"


def make_localization(kind: str, config: Optional[dict] = None):
    """Build a WORLD-frame localization plugin for the robot pose."""
    config = config or {}
    if kind == "mock":
        return _FixedPoseLocalization(config.get("qpos", [1.7, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0]))
    if kind == "vicon":
        from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
        return ViconShmPlugin(config)
    if kind == "ros2":
        from genedynamics.deploy.localization.ros2_odometry_plugin import Ros2OdometryLocalizationPlugin
        return Ros2OdometryLocalizationPlugin(config)
    raise ValueError(f"unknown localization kind {kind!r}")


def build_twin_server(
    *,
    preset: str = "zone_d",
    t_world_scene: Any = None,
    localization: Any = None,
    robot_id: str = "g1",
    robot_urdf: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8766,
    hz: float = 30.0,
    quiet: bool = False,
) -> WorldStreamServer:
    """Compose scene + producers + streaming server (returns the server)."""
    src = SceneSource.from_preset(preset, T_world_scene=t_world_scene)
    world = WorldState(site=f"corridor_{preset}", tracker=getattr(localization, "__class__", type(localization)).__name__)
    populate_corridor(world, src)  # static obstacles (world frame)

    def on_tick(w: WorldState) -> None:
        if localization is not None:
            populate_tracker(w, localization, robot_id=robot_id, urdf=robot_urdf)

    return WorldStreamServer(world, host=host, port=port, hz=hz, on_tick=on_tick, quiet=quiet)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="zone_d")
    p.add_argument("--t-world-scene", type=float, nargs=3, metavar=("X", "Y", "YAW"), default=[0.0, 0.0, 0.0])
    p.add_argument("--localization", default="vicon", choices=["mock", "vicon", "ros2"])
    p.add_argument("--robot-id", default="g1")
    p.add_argument("--robot-urdf", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--hz", type=float, default=30.0)
    args = p.parse_args(argv)

    loc = make_localization(args.localization)
    server = build_twin_server(
        preset=args.preset, t_world_scene=tuple(args.t_world_scene), localization=loc,
        robot_id=args.robot_id, robot_urdf=args.robot_urdf, host=args.host, port=args.port, hz=args.hz,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
