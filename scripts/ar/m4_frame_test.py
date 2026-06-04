#!/usr/bin/env python
"""M4 (prep) test: Vicon WORLD <-> corridor SCENE frame transforms.

Pure math, no hardware. Verifies the ``Se2Transform`` (== T_world_scene) round
-trips and that a Vicon world-frame base estimate maps to the right scene-frame
base state the follower / governor consume.

    /opt/anaconda3/envs/fedguide/bin/python scripts/ar/m4_frame_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.scene_source import Se2Transform  # noqa: E402

EPS = 1e-9
checks = []


def approx(name, got, exp, eps=1e-6):
    ok = all(abs(g - e) <= eps for g, e in zip(got, exp)) if hasattr(got, "__iter__") else abs(got - exp) <= eps
    checks.append(ok)
    g = [round(v, 4) for v in got] if hasattr(got, "__iter__") else round(got, 4)
    e = [round(v, 4) for v in exp] if hasattr(exp, "__iter__") else round(exp, 4)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {g} exp {e}")


def main() -> int:
    T = Se2Transform(x=1.2, y=0.4, yaw=0.3)  # a non-trivial Vicon-measured transform

    # 1. point round-trip scene -> world -> scene
    for (x, y) in [(0.0, 0.0), (0.5, 0.0), (3.5, -0.25), (2.0, 0.25)]:
        X, Y = T.point_to_world(x, y)
        approx(f"point round-trip ({x},{y})", T.point_to_scene(X, Y), (x, y))

    # 2. scene origin maps to the transform's translation
    approx("scene origin -> world", T.point_to_world(0.0, 0.0), (1.2, 0.4))

    # 3. yaw round-trip + offset
    approx("scene psi=0 -> world yaw", T.yaw_to_world(0.0), 0.3)
    approx("world yaw round-trip", T.yaw_to_scene(T.yaw_to_world(0.7)), 0.7)

    # 4. velocity round-trip (rotation only, no translation)
    VX, VY = T.vel_to_world(0.3, 0.0)
    approx("velocity round-trip", T.vel_to_scene(VX, VY), (0.3, 0.0))
    approx("scene vx=0.3 -> world", (VX, VY), (0.3 * math.cos(0.3), 0.3 * math.sin(0.3)))

    # 5. quat -> yaw
    psi = 0.3
    qw, qz = math.cos(psi / 2), math.sin(psi / 2)
    approx("yaw_from_quat", T.yaw_from_quat(qw, 0.0, 0.0, qz), 0.3)

    # 6. scene_base_from_world: a Vicon estimate AT the world pose of scene
    #    (0.5, 0.0, psi=0) must come back as scene base (0.5, 0.0, 0.0).
    X0, Y0, Yaw0 = T.pose_to_world(0.5, 0.0, 0.0)
    qpos = [X0, Y0, 0.75, math.cos(Yaw0 / 2), 0.0, 0.0, math.sin(Yaw0 / 2)]
    qvel = [*T.vel_to_world(0.3, 0.0), 0.0, 0.0, 0.0, 0.1]  # scene vx=0.3, omega=0.1
    base = T.scene_base_from_world(qpos, qvel)
    approx("scene_base_from_world pos/psi", (base["x"], base["y"], base["psi"]), (0.5, 0.0, 0.0))
    approx("scene_base_from_world vel/omega", (base["vx"], base["vy"], base["omega"]), (0.3, 0.0, 0.1))

    # 7. identity transform is a no-op
    I = Se2Transform()
    approx("identity point", I.point_to_scene(*I.point_to_world(2.0, 0.25)), (2.0, 0.25))

    ok = all(checks)
    print(f"\nM4 frame transforms: {'ALL PASS ✅' if ok else 'FAIL ❌'} ({sum(checks)}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
