#!/usr/bin/env python
"""Sweep body-SDF governor knobs on one zone and report safety + smoothness.

Motivation: on the zone_d slalom (obstacle at +y then -y) the governor's
look-ahead anticipates the *next* obstacle and cuts the corner of the one just
passed, REDUCING executed clearance below the raw baseline. This sweeps the two
exposed knobs that control that anticipation / lag --

  * body_sdf_lookahead : how far ahead (m) the half-space anticipates obstacles.
                         Smaller => less early steering => less corner-cutting.
  * gov_cmd_lpf        : first-order LPF alpha on the governed output velocity
                         (1.0 = none; lower = smoother but more lag).

-- and reports, per config, the executed min body-SDF clearance (vs the
governor-off baseline) plus command / realized jerk and body sway, so we can
pick a config that recovers clearance WITHOUT hurting smoothness.

Run in docker (mujoco + torch)::

  docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \\
      python scripts/tasks/robot/humanoid/governor_sweep.py --zone twogo_zone_d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.tasks.robot.humanoid.sport_mode_corridor import diagnose, resolve_plan_path  # noqa: E402
from scripts.tasks.robot.humanoid.governor_compare import _metrics  # noqa: E402
from scripts.tasks.robot.humanoid.run_sport_mode_zones import GOVERNOR_CFG, _plan_m_track  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", default="twogo_zone_d")
    p.add_argument("--lookaheads", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.30])
    p.add_argument("--lpfs", nargs="+", type=float, default=[0.3, 1.0])
    p.add_argument("--activation-band", type=float, default=0.08)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--work-dir", default=str(_ROOT / "results" / "deploy" / "governor_sweep"))
    args = p.parse_args(argv)

    plan = resolve_plan_path(args.zone)
    m_track = _plan_m_track(plan)
    work = Path(args.work_dir) / args.zone

    rows = []  # (label, lookahead, lpf, metrics)

    # row 0: governor-off baseline (same Step-1 mode) for reference
    base = diagnose(plan, best_idx=None, use_governor=False, max_steps=args.max_steps,
                    out_dir=work / "baseline", quiet=True)
    rows.append(("baseline(off)", None, None, _metrics(work / "baseline")))

    for la in args.lookaheads:
        for lpf in args.lpfs:
            tag = f"la{la:.2f}_lpf{lpf:.1f}"
            diagnose(
                plan, best_idx=None,
                use_governor=True, use_body_sdf_governor=True,
                governor_cfg={**GOVERNOR_CFG, "m_track": m_track},
                gov_cmd_lpf=lpf,
                body_sdf_activation_band=args.activation_band,
                body_sdf_lookahead=la,
                max_steps=args.max_steps,
                out_dir=work / tag, quiet=True,
            )
            rows.append((tag, la, lpf, _metrics(work / tag)))

    base_sdf = rows[0][3]["min_body_sdf"]
    print("\n" + "=" * 92)
    print(f" GOVERNOR SWEEP  [{args.zone}]   (m_track={m_track}, activation_band={args.activation_band})")
    print(f" baseline(governor-off) min body-SDF = {base_sdf:+.4f} m  -- target: GOV >= this")
    print("=" * 92)
    hdr = (f" {'config':16s} {'min_sdf':>9s} {'vs base':>9s} {'cmd_jerk':>9s} "
           f"{'real_jerk':>10s} {'sway':>8s} {'max_lat':>8s}")
    print(hdr)
    print(" " + "-" * 90)
    best = None
    for label, la, lpf, m in rows:
        if m is None:
            print(f" {label:16s}   (no npz)")
            continue
        d = m["min_body_sdf"] - base_sdf
        flag = ""
        if la is not None:
            recovered = m["min_body_sdf"] >= base_sdf - 1e-4
            flag = " ✓recovered" if recovered else " ✗below-base"
            # rank: prefer recovered clearance, then lower cmd jerk
            score = (1 if recovered else 0, -m["cmd_lin_jerk_rms"])
            if best is None or score > best[0]:
                best = (score, label, la, lpf, m)
        print(f" {label:16s} {m['min_body_sdf']:+9.4f} {d:+9.4f} "
              f"{m['cmd_lin_jerk_rms']:9.2f} {m['realized_lin_jerk_rms']:10.2f} "
              f"{m['body_sway_std']:8.4f} {m['max_lateral_offset']:8.4f}{flag}")

    if best is not None:
        _, label, la, lpf, m = best
        print("\n" + "=" * 92)
        print(f" BEST (recovered clearance, then min cmd jerk): {label}")
        print(f"   body_sdf_lookahead={la}  gov_cmd_lpf={lpf}")
        print(f"   min body-SDF={m['min_body_sdf']:+.4f} m (base {base_sdf:+.4f}), "
              f"cmd_jerk={m['cmd_lin_jerk_rms']:.2f}, sway={m['body_sway_std']:.4f}")
        print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
