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
        summary.append(
            f"{zone:20s}  {verdict}  steps={result.n_steps:4d}  max_lat={result.max_lateral_offset_m:.2f}m"
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
