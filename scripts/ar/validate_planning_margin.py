#!/usr/bin/env python
"""Validate that the real 2GO clearance knob (the ALM constraint-scheduler
``margin``) actually thickens the PLANNED body-SDF clearance.

Replans zone_d at a fattened ALM margin (full diffusion, no calibration) and
compares per-mode planned body-SDF clearance to a native reference plan
(ALM margin 0.07). Confirms the knob binds at full convergence (the earlier
``--fast`` test was unreliable).

Run in Docker::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \
        python scripts/ar/validate_planning_margin.py --margin 0.20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402
from genedynamics.envs.humanoid_corridor_2d import corridor_scene_to_dict  # noqa: E402

NATIVE = "results/ar/m2_zone_d_full/level_1/seed_0/trajectory/trajectory.json"


def planned_clears(plan_path: str | Path, scene: dict) -> list[float]:
    from scripts.tasks.robot.humanoid.governor_calibrate import _make_planned_clearance_fn
    blob = json.loads(Path(plan_path).read_text())
    fn = _make_planned_clearance_fn(scene)
    cands = np.asarray(blob["candidate_states"], dtype=np.float64)
    return [float(fn(c)) for c in cands]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--margin", type=float, default=0.20, help="ALM scheduler margin to test.")
    p.add_argument("--native", default=NATIVE, help="Native (margin 0.07) reference plan.")
    p.add_argument("--out", default="results/ar/margin_val")
    args = p.parse_args(argv)

    from scripts.ar.replan_from_scene import replan_from_scene

    scene = corridor_scene_to_dict(SceneSource.from_preset("zone_d").scene, scene_preset="ar_runtime")

    fat = replan_from_scene("zone_d", output_dir=args.out, collision_margin=args.margin, calibrate=False)
    fat_c = planned_clears(fat, scene)

    print("\n==== planning-margin validation (zone_d) ====")
    print(f"  FATTENED ALM margin={args.margin}: per-mode planned clearance = {[round(c, 4) for c in fat_c]}")
    print(f"    max = {max(fat_c):+.4f}   mean = {float(np.mean(fat_c)):+.4f}")

    if Path(args.native).exists():
        nat_c = planned_clears(args.native, scene)
        print(f"  NATIVE   ALM margin=0.07 : per-mode planned clearance = {[round(c, 4) for c in nat_c]}")
        print(f"    max = {max(nat_c):+.4f}   mean = {float(np.mean(nat_c)):+.4f}")
        gain = max(fat_c) - max(nat_c)
        binds = gain > 0.03
        print(f"\n  Δ max planned clearance = {gain:+.4f} m")
        print(f"  RESULT: ALM margin knob {'BINDS ✅ (thickens planned clearance)' if binds else 'does NOT bind ❌'}")
        return 0 if binds else 1
    else:
        print("  (no native reference plan found; reporting fattened only)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
