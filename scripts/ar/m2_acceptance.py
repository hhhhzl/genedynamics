#!/usr/bin/env python
"""M2 acceptance: replan from an AR scene, then certify the plan in MuJoCo.

1. replan_from_scene(<preset/scene>) → trajectory.json  (2GO, runtime preset),
2. diagnose(trajectory.json, governor on, body_sdf_scene=<AR scene>) in MuJoCo,
3. assert the run is ``certified_safe`` and the robot did not fall.

Needs jax + mujoco — run in Docker::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \
        python scripts/ar/m2_acceptance.py --preset zone_d --fast
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402
from genedynamics.envs.humanoid_corridor_2d import corridor_scene_to_dict  # noqa: E402

# Governor knobs mirror scripts/.../run_sport_mode_zones.py (the validated path).
GOVERNOR_CFG = {"v_max_lat": 0.55, "yaw_rate_max": 0.65}
GOV_CMD_LPF = 0.3
GOV_ACTIVATION_BAND = 0.08
GOV_LOOKAHEAD = 0.15
M_TRACK_FALLBACK = 0.02


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="zone_d")
    p.add_argument("--scene-file", default=None)
    p.add_argument("--out", default=None, help="Output root (default: a temp dir).")
    p.add_argument("--fast", action="store_true", help="Reduced modes/diffusion (quick check).")
    p.add_argument("--margin", type=float, default=None,
                   help="Planning collision margin; default None keeps the template's native value.")
    args = p.parse_args(argv)

    from scripts.ar.replan_from_scene import FAST_PLANNING, replan_from_scene
    from scripts.tasks.robot.humanoid.sport_mode_corridor import diagnose

    source = args.scene_file if args.scene_file else args.preset
    src = SceneSource.from_file(args.scene_file) if args.scene_file else SceneSource.from_preset(args.preset)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="m2_replan_"))

    # ----- 1. replan from the AR scene ------------------------------------
    traj = replan_from_scene(
        source, output_dir=out, collision_margin=args.margin,
        planning_overrides=FAST_PLANNING if args.fast else None,
    )

    # ----- 2. certify the replanned plan in MuJoCo (governor on) ----------
    # Read the governor knobs the replan baked in (best_idx is read by the
    # adapter when best_idx=None; m_track / gov_lookahead come off the plan).
    import json
    blob = json.loads(Path(traj).read_text())
    m_track = float(blob.get("m_track", M_TRACK_FALLBACK))
    lookahead = float(blob.get("gov_lookahead", GOV_LOOKAHEAD))
    scene_blob = corridor_scene_to_dict(src.scene, scene_preset="ar_runtime")
    print(f"\n[m2] diagnosing {traj} against {len(scene_blob['obstacles'])} obstacles "
          f"(baked best_idx={blob.get('best_idx')} m_track={m_track}) ...", flush=True)
    result = diagnose(
        traj,
        control_hz=50.0,
        use_governor=True,
        use_body_sdf_governor=True,
        governor_cfg={**GOVERNOR_CFG, "m_track": m_track},
        gov_cmd_lpf=GOV_CMD_LPF,
        body_sdf_activation_band=GOV_ACTIVATION_BAND,
        body_sdf_lookahead=lookahead,
        body_sdf_scene=scene_blob,
        quiet=True,
    )

    # ----- 3. verdict -----------------------------------------------------
    sdf = result.executed_min_body_sdf
    ok = bool(result.certified_safe) and not bool(result.fell_over)
    print("\n========== M2 acceptance ==========")
    print(f"  replanned plan : {traj}")
    print(f"  fell_over      : {result.fell_over}")
    print(f"  endpoint dist  : {result.endpoint_distance_m:.3f} m")
    print(f"  exec min body-SDF: {sdf:+.3f} m  ({'CERTIFIED' if result.certified_safe else 'VIOLATION'})")
    print(f"  RESULT         : {'ALL PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
