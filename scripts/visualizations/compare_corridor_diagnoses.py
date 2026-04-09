#!/usr/bin/env python
"""Side-by-side comparison plot of the corridor controller diagnoses.

Loads the .npz files produced by:

* ``scripts/tasks/robot/diagnose_sport_mode_corridor.py`` (spark RL)
* ``scripts/tasks/robot/diagnose_wbc_corridor.py``        (WBC)

…and produces a single matplotlib figure: pelvis xy traces overlaid on
the plan, plus pelvis-z time-series so the fall events are visible.

Run inside dev-cpu after the diagnose scripts have populated the
artifact directories::

    docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:local \\
        python scripts/visualizations/compare_corridor_diagnoses.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Path: scripts/visualizations/<file>.py — parents[2] is the project root.
_ROOT = Path(__file__).resolve().parents[2]
RESULTS = _ROOT / "results" / "deploy"


def _load(npz_path: Path) -> Optional[dict]:
    if not npz_path.exists():
        print(f"[compare] missing: {npz_path}")
        return None
    return dict(np.load(npz_path, allow_pickle=True))


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot plot")
        return 1

    spark = _load(RESULTS / "diagnose_spark_rl" / "diagnose_sport_mode.npz")
    wbc = _load(RESULTS / "diagnose_wbc" / "diagnose_wbc.npz")

    if not any([spark, wbc]):
        print("[compare] no diagnose artifacts found — run the diagnose scripts first")
        return 1

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(14, 6))

    plan_xy = (spark or wbc)["plan_xy"]
    ax_xy.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", lw=2, label="plan")

    if wbc is not None:
        wbc_n = int(wbc["pelvis_xy"].shape[0])
        wbc_fell = bool(wbc["pelvis_z"].min() < 0.40)
        ax_xy.plot(wbc["pelvis_xy"][:, 0], wbc["pelvis_xy"][:, 1], "C1-", lw=1.5,
                   label=f"WBC ({wbc_n} steps, {'FELL' if wbc_fell else 'OK'})")
        ax_z.plot(wbc["pelvis_z"], "C1-", label="WBC", lw=1.5)
    if spark is not None:
        spark_n = int(spark["pelvis_xy"].shape[0])
        spark_fell = bool(spark["pelvis_z"].min() < 0.40)
        ax_xy.plot(spark["pelvis_xy"][:, 0], spark["pelvis_xy"][:, 1], "C0-", lw=2,
                   label=f"Spark RL ({spark_n} steps, {'FELL' if spark_fell else 'OK'})")
        ax_z.plot(spark["pelvis_z"], "C0-", label="Spark RL", lw=2)

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("Pelvis trace vs corridor plan")
    ax_xy.legend(loc="best", fontsize=9)
    ax_xy.grid(True, alpha=0.3)

    ax_z.axhline(0.40, color="gray", ls=":", label="fall threshold (0.40 m)")
    ax_z.set_xlabel("control step")
    ax_z.set_ylabel("pelvis z (m)")
    ax_z.set_title("Balance: pelvis height over time")
    ax_z.legend(loc="best", fontsize=9)
    ax_z.grid(True, alpha=0.3)

    fig.suptitle(
        "Corridor controller comparison — twogo_zone_a/level_1/seed_0",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = RESULTS / "diagnose_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"[compare] wrote {out_path}")

    # CSV-friendly summary
    print()
    print("controller,steps,fell,z_min,endpoint_m,mean_lat_m,max_lat_m")
    for label, data in [("wbc", wbc), ("spark_rl", spark)]:
        if data is None:
            continue
        n = int(data["pelvis_xy"].shape[0])
        z_min = float(data["pelvis_z"].min())
        fell = z_min < 0.40
        plan = data["plan_xy"]
        final_actual = data["pelvis_xy"][-1]
        final_plan = plan[min(n, plan.shape[0]) - 1]
        endpt = float(np.linalg.norm(final_actual - final_plan))
        lat = data["laterals"]
        print(
            f"{label},{n},{fell},{z_min:.3f},{endpt:.3f},"
            f"{float(lat.mean()):.3f},{float(lat.max()):.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
