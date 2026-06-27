#!/usr/bin/env python
"""Compute ``--t-world-scene`` so the robot's CURRENT mocap pose == the plan start.

Reads the live world-frame base pose (``ViconShmPlugin`` / ``mocap_state_shm`` — so
the NatNet writer must be running) and solves the SE(2) calibration ``T_world_scene``
(scene -> world) that maps the robot's current pose onto the corridor's start pose
``(start_x, start_y, yaw=0)``. Uses the SAME yaw convention as the runtime
(``Se2Transform`` / ``SceneFrameLocalization``), so the printed value drops straight
into ``run_real_g1.py --t-world-scene X Y YAW``.

    # NatNet writer must already be writing the shm:
    python scripts/tasks/robot/humanoid/register_start.py --preset zone_a

Place the robot where you want the corridor to BEGIN, facing where you want it to
walk (scene +x = the robot's forward), make sure there is clear floor ahead for the
corridor length, then run this. Re-run any time the robot is repositioned.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from genedynamics.deploy.ar.scene_source import SceneSource, Se2Transform
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="zone_a", help="Corridor preset (for its start_pos).")
    ap.add_argument("--scene-file", default=None, help="AR scene contract JSON (overrides --preset).")
    ap.add_argument("--shm-name", default=ViconShmPlugin.SHM_NAME)
    ap.add_argument("--samples", type=int, default=20, help="Mocap samples to average (robust to jitter).")
    args = ap.parse_args(argv)

    src = SceneSource.from_file(args.scene_file) if args.scene_file else SceneSource.from_preset(args.preset)
    sx, sy = float(src.scene.start_pos[0]), float(src.scene.start_pos[1])

    plugin = ViconShmPlugin({"shm_name": args.shm_name})
    Xs, Ys, sins, coss = [], [], [], []
    for _ in range(max(1, args.samples)):
        st = plugin.get_state()
        if st is not None:
            pose, _ = st
            Xs.append(float(pose[0])); Ys.append(float(pose[1]))
            psi = Se2Transform.yaw_from_quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
            sins.append(math.sin(psi)); coss.append(math.cos(psi))
        time.sleep(0.05)
    if not Xs:
        print("❌ no mocap samples — is the NatNet writer running and writing the shm?")
        return 1

    X0, Y0 = float(np.mean(Xs)), float(np.mean(Ys))
    Psi0 = math.atan2(float(np.mean(sins)), float(np.mean(coss)))  # circular mean

    # Solve T (scene->world) s.t. point_to_scene(X0,Y0)=(sx,sy) and yaw_to_scene(Psi0)=0:
    #   yaw = Psi0 ;  (x,y) = (X0,Y0) - R(Psi0)·(sx,sy)
    c, s = math.cos(Psi0), math.sin(Psi0)
    x_ts = X0 - (c * sx - s * sy)
    y_ts = Y0 - (s * sx + c * sy)
    yaw_ts = Psi0

    T = Se2Transform(x_ts, y_ts, yaw_ts)
    cx, cy = T.point_to_scene(X0, Y0)
    cyaw = T.yaw_to_scene(Psi0)

    print(f"robot world pose : X={X0:+.3f} Y={Y0:+.3f} yaw={math.degrees(Psi0):+.1f}°  "
          f"(n={len(Xs)} samples)")
    print(f"plan start (scene): ({sx:+.3f}, {sy:+.3f})  preset={args.preset if not args.scene_file else args.scene_file}")
    print(f"self-check        : robot → scene ({cx:+.3f}, {cy:+.3f}, {math.degrees(cyaw):+.1f}°)  "
          f"expect ≈ ({sx:+.3f}, {sy:+.3f}, +0.0°)  "
          f"{'✅' if abs(cx-sx)<1e-3 and abs(cy-sy)<1e-3 and abs(cyaw)<1e-3 else '❌'}")
    print()
    print(f"  --t-world-scene {x_ts:.4f} {y_ts:.4f} {yaw_ts:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
