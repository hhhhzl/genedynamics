#!/usr/bin/env python
"""M3 acceptance: the runtime governor perceives & avoids AR obstacles.

The robot never replans here — it follows a FIXED zone_d plan. We inject the
shared AR scene into the governor via ``SceneSource.to_body_sdf_scene()`` and
show the governor (the runtime backstop) keeps the body clear of the AR
obstacles it "perceives":

  Part A (planned obstacles): governor OFF vs ON against the zone_d scene the
      plan was made for — ON holds more clearance.
  Part B (UNPLANNED AR obstacle): drop an extra AR sphere on the plan's path
      (the plan knows nothing about it). governor OFF drives into it; governor
      ON, perceiving it through the injected scene, steers clear.

Reuses the existing full plan (no replanning). Run in Docker::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \
        python scripts/ar/m3_acceptance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402
from genedynamics.envs.humanoid_corridor_2d import CorridorObstacle  # noqa: E402

PLAN = "results/ar/m2_zone_d_full/level_1/seed_0/trajectory/trajectory.json"
M_TRACK = 0.08  # the swept optimum for zone_d + Spark


def main() -> int:
    from scripts.tasks.robot.humanoid.sport_mode_corridor import diagnose

    plan = Path(PLAN)
    if not plan.exists():
        print(f"M3: plan missing ({plan}); run M2 first.", file=sys.stderr)
        return 2
    blob = json.loads(plan.read_text())
    best = int(blob.get("best_idx", 0))
    states = np.asarray(blob["candidate_states"], dtype=np.float64)[best]  # (T, 14)
    wp = states[int(0.70 * len(states))]  # a waypoint in the goal run-out
    ex_x, ex_y = float(wp[0]), float(wp[1]) + 0.15  # AR sphere just off the path

    def run(scene_dict, governor: bool):
        return diagnose(
            plan, control_hz=50.0,
            use_governor=governor, use_body_sdf_governor=governor,
            governor_cfg=({"v_max_lat": 0.55, "yaw_rate_max": 0.65, "m_track": M_TRACK}
                          if governor else None),
            gov_cmd_lpf=0.3 if governor else 1.0,
            body_sdf_activation_band=0.08, body_sdf_lookahead=0.15,
            body_sdf_scene=scene_dict, quiet=True,
        )

    # ---- Part A: planned obstacles (zone_d) ------------------------------
    scene_a = SceneSource.from_preset("zone_d").to_body_sdf_scene()
    a_off, a_on = run(scene_a, False), run(scene_a, True)

    # ---- Part B: an UNPLANNED AR obstacle dropped on the path ------------
    src_b = SceneSource.from_preset("zone_d")
    src_b.scene.obstacles.append(
        CorridorObstacle.sphere(ex_x, ex_y, 0.10, 0.0, 2.0, "ar_unplanned"))
    scene_b = src_b.to_body_sdf_scene()
    b_off, b_on = run(scene_b, False), run(scene_b, True)

    def sdf(r):
        return float(r.executed_min_body_sdf)

    print("\n========== M3 acceptance ==========")
    print(f"  Part A  zone_d (planned)         governor OFF: {sdf(a_off):+.4f}   ON: {sdf(a_on):+.4f}")
    print(f"  Part B  +AR sphere @({ex_x:.2f},{ex_y:.2f}) on path  OFF: {sdf(b_off):+.4f}   ON: {sdf(b_on):+.4f}")

    a_improved = sdf(a_on) > sdf(a_off) + 0.01
    b_off_hits = sdf(b_off) < 0.0
    b_governor_avoids = sdf(b_on) > sdf(b_off) + 0.05
    ok = a_improved and b_off_hits and b_governor_avoids

    print(f"\n  A: governor holds more clearance vs planned obstacles : {'PASS' if a_improved else 'FAIL'}")
    print(f"  B: blind follower drives into the unplanned AR sphere : {'PASS' if b_off_hits else 'FAIL'}")
    print(f"  B: governor PERCEIVES + AVOIDS the unplanned AR sphere: {'PASS' if b_governor_avoids else 'FAIL'}"
          f"  (Δ={sdf(b_on)-sdf(b_off):+.4f} m)")
    print(f"  RESULT: {'ALL PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
