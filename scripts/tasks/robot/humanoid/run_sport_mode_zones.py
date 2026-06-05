#!/usr/bin/env python
"""Run :mod:`sport_mode_corridor` against a list of corridor plans.

Default list mirrors the per-zone planner outputs we care about:

* ``twogo_long``  — full combined Zone A+C+D run (the "real" plan).
* ``twogo_zone_a/b/c/d``  — single-zone runs, useful when a zone
  fails on its own and you want to isolate it.

Each run gets its own ``--out-dir`` under ``results/deploy/spark_rl/<alias>/``
so the npz / json / corridor_scene.json don't stomp each other.

Usage
-----
::

    docker run --rm -it -v "$PWD:/work" -w /work genedynamics/dev-cpu \\
        python scripts/tasks/robot/humanoid/run_sport_mode_zones.py

    # Just two zones, with plots:
    python scripts/tasks/robot/humanoid/run_sport_mode_zones.py \\
        --zones twogo_long twogo_zone_a --plot

    # Cap rollout length per plan (handy for the long combined run):
    python scripts/tasks/robot/humanoid/run_sport_mode_zones.py --max-steps 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Path: scripts/tasks/robot/humanoid/<file>.py — parents[4] is the project root.
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.tasks.robot.humanoid.sport_mode_corridor import (  # noqa: E402
    PLAN_ALIASES,
    diagnose,
    make_plot,
    print_report,
    resolve_plan_path,
)


DEFAULT_ZONES = ("twogo_long", "twogo_zone_a", "twogo_zone_b", "twogo_zone_c", "twogo_zone_d")

# Default = the execution-aware reference-governor pipeline (Steps 1-4).
# Step 1 (executed-safe mode selection) is baked into each plan's
# trajectory.json `best_idx` (with `planner_best_idx` preserving the planner's
# original cost-min pick), so it loads by default — no per-zone map needed here.
# Step 2/4: the body-SDF governor runs with the policy-trackable command
# envelope + the clearance-degradation tightening m_track, plus output command
# smoothing. See genedynamics/deploy/followers/governor/reference_selector.py.
GOVERNOR_CFG = {"v_max_lat": 0.55, "yaw_rate_max": 0.65}
GOV_CMD_LPF = 0.3
# Near-passthrough band: the body-SDF halfspace only binds when executed
# clearance < m_track + this. 0.08 (the BodySdfAdmissibleSet class default)
# keeps the governor a true backstop that engages only near obstacles, vs
# diagnose's own 0.30 default which is active almost everywhere.
GOV_ACTIVATION_BAND = 0.08
# Half-space anticipation distance. diagnose's 0.30 default OVER-anticipates the
# next obstacle in a slalom (opposite-side obstacles) and cuts the corner of the
# one just passed — on zone_d that REDUCED executed clearance below the raw
# baseline. governor_sweep.py shows 0.15 recovers it (+0.018→+0.053, better than
# baseline) AND lowers cmd jerk (15.5→11.0). 0.10–0.20 are equivalent; 0.15 keeps
# a little approach headroom.
GOV_LOOKAHEAD = 0.15
M_TRACK_FALLBACK = 0.02  # Step-4 tightening if the plan carries no derived m_track


def _plan_field(plan_path: Path, key: str, default: float) -> float:
    """Read a numeric governor knob baked into the plan's trajectory.json.

    Knobs are baked per-zone so the deploy is self-describing and DERIVED, not
    hand-set: ``m_track`` (Step-4 tightening, from governor_calibrate.py) and
    ``gov_lookahead`` (per-zone half-space anticipation, from governor_sweep.py
    — dense clusters like zone_c want ~0.30, slaloms like zone_d want ~0.15).
    Falls back to ``default`` when absent.
    """
    try:
        blob = json.loads(Path(plan_path).read_text())
        v = blob.get(key)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def _plan_m_track(plan_path: Path) -> float:
    return _plan_field(plan_path, "m_track", M_TRACK_FALLBACK)


def _plan_gov_lookahead(plan_path: Path) -> float:
    return _plan_field(plan_path, "gov_lookahead", GOV_LOOKAHEAD)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--zones",
        nargs="+",
        default=list(DEFAULT_ZONES),
        help=(
            "Plan aliases or paths to run sequentially. Aliases: "
            f"{', '.join(sorted(PLAN_ALIASES))}. Default: all five."
        ),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=_ROOT / "results" / "deploy" / "spark_rl",
        help="Parent directory for per-zone output dirs.",
    )
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--sim-dt", type=float, default=1.0 / 500.0)
    p.add_argument("--max-steps", type=int, default=None, help="Cap per-zone rollout length")
    p.add_argument("--plot", action="store_true", help="Save the 2x2 diagnose plot for each zone")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--continue-on-fall",
        action="store_true",
        help="Don't return non-zero on robot falls; useful when batching across hard zones.",
    )
    p.add_argument(
        "--no-governor",
        action="store_true",
        help="Disable the reference governor + Step-1 mode selection (raw open-loop baseline).",
    )
    args = p.parse_args(argv)

    args.out_root.mkdir(parents=True, exist_ok=True)
    overall_status = 0
    summary: List[str] = []

    for zone in args.zones:
        plan_path = resolve_plan_path(zone)
        if not plan_path.exists():
            print(f"[zones] SKIP {zone} — plan not found: {plan_path}", file=sys.stderr)
            overall_status = max(overall_status, 2)
            summary.append(f"{zone:20s}  SKIP (no plan)")
            continue

        # Use the alias as the dir name when given, else the seed_dir name.
        out_name = zone if zone in PLAN_ALIASES else plan_path.parent.parent.name
        out_dir = args.out_root / out_name

        if not args.quiet:
            print(f"\n[zones] === {zone} → {out_dir} ===")

        try:
            result = diagnose(
                plan_path,
                control_hz=args.control_hz,
                sim_dt=args.sim_dt,
                max_steps=args.max_steps,
                out_dir=out_dir,
                quiet=args.quiet,
                # Step 1: best_idx=None => adapter reads the plan's baked best_idx
                # (the executed-safe mode; see trajectory.json best_idx_source).
                use_governor=not args.no_governor,
                use_body_sdf_governor=not args.no_governor,
                governor_cfg=None if args.no_governor else {**GOVERNOR_CFG, "m_track": _plan_m_track(plan_path)},
                gov_cmd_lpf=1.0 if args.no_governor else GOV_CMD_LPF,
                body_sdf_activation_band=GOV_ACTIVATION_BAND,
                body_sdf_lookahead=_plan_gov_lookahead(plan_path),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[zones] FAIL {zone}: {exc}", file=sys.stderr)
            overall_status = max(overall_status, 3)
            summary.append(f"{zone:20s}  ERROR: {exc}")
            continue

        if not args.quiet:
            print_report(result)
        if args.plot:
            make_plot(result, out_dir / "sport_mode.png")

        verdict = "FELL" if result.fell_over else f"OK   ep={result.endpoint_distance_m:.2f}m"
        _sdf = result.executed_min_body_sdf
        cert = "" if _sdf != _sdf else f"  body_sdf={_sdf:+.3f}m {'CERT' if result.certified_safe else 'VIOL'}"
        summary.append(
            f"{zone:20s}  {verdict}  steps={result.n_steps:4d}  max_lat={result.max_lateral_offset_m:.2f}m{cert}"
        )
        if result.fell_over and not args.continue_on_fall:
            overall_status = max(overall_status, 1)

    print("\n" + "=" * 64)
    print(" Batch summary")
    print("=" * 64)
    for line in summary:
        print(" " + line)
    print()
    return overall_status


if __name__ == "__main__":
    sys.exit(main())
