#!/usr/bin/env python
"""GOV (reference governor on) vs BASELINE (governor off) metric comparison.

Loads the per-zone ``sport_mode.npz`` from two run roots (governed and baseline,
SAME Step-1 selected mode in both) and reports, per zone and on average, the
metrics that matter for the locked deploy:

  * safety       : executed minimum body-SDF clearance (higher = safer; <0 = hit)
  * cmd jerk     : RMS ||d^2/dt^2 (commanded base velocity)|| + yaw-rate jerk.
                   This is exactly what the governor's output low-pass filter
                   (gov_cmd_lpf) is meant to suppress.
  * realized jerk: RMS ||d^3/dt^3 (executed pelvis xy)|| (gait/body smoothness).
  * body sway    : std of the lateral tracking offset (lateral body wobble).
  * bobbing      : std of pelvis height (vertical bob).
  * pelvis roll  : std + max|roll| from the executed base quaternion.
  * tracking     : endpoint distance + mean/max lateral offset.

No simulation — pure post-hoc analysis (numpy only), so it runs on the host.

Usage::

  python scripts/tasks/robot/humanoid/governor_compare.py \\
      --gov-root   results/deploy/spark_rl_gov \\
      --base-root  results/deploy/spark_rl_baseline \\
      --zones twogo_zone_a twogo_zone_c twogo_zone_d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_CTRL_HZ = 50.0
_DT = 1.0 / _CTRL_HZ


def _roll_from_quat(quat: np.ndarray) -> np.ndarray:
    """Roll (rad) from (n,4) base quaternion columns (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def _jerk_rms(series: np.ndarray, dt: float, order: int) -> float:
    """RMS magnitude of the ``order``-th time derivative of ``series``.

    ``series`` is (n,) or (n, k); for k>1 we take the per-step Euclidean norm
    before the RMS over time. ``order=2`` on a velocity = jerk; ``order=3`` on a
    position = jerk.
    """
    s = np.asarray(series, dtype=np.float64)
    if s.shape[0] <= order + 1:
        return float("nan")
    d = np.diff(s, n=order, axis=0) / (dt ** order)
    mag = np.linalg.norm(d, axis=1) if d.ndim > 1 else np.abs(d)
    return float(np.sqrt(np.mean(mag ** 2)))


def _metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    npz = run_dir / "sport_mode.npz"
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    intent_lin = np.asarray(d["intent_lin_vel"], dtype=np.float64)   # (n,2) commanded base vel
    intent_yaw = np.asarray(d["intent_yaw_rate"], dtype=np.float64)  # (n,)
    pelvis_xy = np.asarray(d["pelvis_xy"], dtype=np.float64)         # (n,2)
    pelvis_z = np.asarray(d["pelvis_z"], dtype=np.float64)           # (n,)
    laterals = np.asarray(d["laterals"], dtype=np.float64)           # (n,)
    qpos = np.asarray(d["qpos"], dtype=np.float64)                   # (n, nq); [:,3:7] quat
    exec_sdf = (np.asarray(d["executed_body_sdf"], dtype=np.float64)
                if "executed_body_sdf" in d.files else np.array([np.nan]))
    roll = _roll_from_quat(qpos[:, 3:7]) if qpos.shape[1] >= 7 else np.array([np.nan])

    # safety + certificate (prefer the json scalar; fall back to npz array)
    min_sdf = float(np.nanmin(exec_sdf)) if exec_sdf.size else float("nan")
    sj = run_dir / "sport_mode.json"
    if sj.exists():
        blob = json.loads(sj.read_text())
        if blob.get("executed_min_body_sdf") is not None:
            min_sdf = float(blob["executed_min_body_sdf"])

    return {
        "n_steps": float(pelvis_xy.shape[0]),
        "min_body_sdf": min_sdf,
        "cmd_lin_jerk_rms": _jerk_rms(intent_lin, _DT, 2),
        "cmd_yaw_jerk_rms": _jerk_rms(intent_yaw, _DT, 2),
        "realized_lin_jerk_rms": _jerk_rms(pelvis_xy, _DT, 3),
        "body_sway_std": float(np.std(laterals)) if laterals.size else float("nan"),
        "bobbing_std": float(np.std(pelvis_z)) if pelvis_z.size else float("nan"),
        "pelvis_roll_std": float(np.std(roll)),
        "pelvis_roll_absmax": float(np.max(np.abs(roll))),
        "max_lateral_offset": float(np.max(laterals)) if laterals.size else float("nan"),
        "mean_lateral_offset": float(np.mean(laterals)) if laterals.size else float("nan"),
    }


# (label, key, "higher"|"lower" is better, unit)
_FIELDS = [
    ("safety: min body-SDF", "min_body_sdf", "higher", "m"),
    ("cmd lin jerk (RMS)", "cmd_lin_jerk_rms", "lower", "m/s^3"),
    ("cmd yaw jerk (RMS)", "cmd_yaw_jerk_rms", "lower", "rad/s^3"),
    ("realized lin jerk (RMS)", "realized_lin_jerk_rms", "lower", "m/s^3"),
    ("body sway (std lat)", "body_sway_std", "lower", "m"),
    ("bobbing (std z)", "bobbing_std", "lower", "m"),
    ("pelvis roll std", "pelvis_roll_std", "lower", "rad"),
    ("pelvis roll |max|", "pelvis_roll_absmax", "lower", "rad"),
    ("max lateral offset", "max_lateral_offset", "lower", "m"),
]


def _pct(base: float, gov: float, better: str) -> str:
    if base != base or gov != gov or abs(base) < 1e-12:
        return "  n/a"
    change = (gov - base) / abs(base) * 100.0
    improved = (gov > base) if better == "higher" else (gov < base)
    arrow = "↑" if change > 0 else "↓"
    tag = " ✓" if improved else (" ✗" if abs(change) > 1.0 else "  ")
    return f"{arrow}{abs(change):5.1f}%{tag}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gov-root", default=str(_ROOT / "results" / "deploy" / "spark_rl_gov"))
    p.add_argument("--base-root", default=str(_ROOT / "results" / "deploy" / "spark_rl_baseline"))
    p.add_argument("--zones", nargs="+",
                   default=["twogo_zone_a", "twogo_zone_c", "twogo_zone_d"])
    args = p.parse_args(argv)

    gov_root, base_root = Path(args.gov_root), Path(args.base_root)
    gov_all: List[Dict[str, float]] = []
    base_all: List[Dict[str, float]] = []

    for zone in args.zones:
        g = _metrics(gov_root / zone)
        b = _metrics(base_root / zone)
        if g is None or b is None:
            print(f"[compare] SKIP {zone}: gov={g is not None} base={b is not None} "
                  f"(missing sport_mode.npz)")
            continue
        gov_all.append(g)
        base_all.append(b)
        print("\n" + "=" * 78)
        print(f" {zone}    GOV vs BASELINE   (same Step-1 mode; GOV = governor + cmd-LPF)")
        print("=" * 78)
        print(f" {'metric':26s} {'baseline':>12s} {'GOV':>12s}   {'Δ (GOV vs base)':>16s}")
        print(" " + "-" * 76)
        for label, key, better, unit in _FIELDS:
            bv, gv = b[key], g[key]
            print(f" {label:26s} {bv:12.4f} {gv:12.4f}   {_pct(bv, gv, better):>16s}  ({unit})")

    if not gov_all:
        print("[compare] no comparable zones found.")
        return 1

    # --- average across zones ------------------------------------------------
    print("\n" + "=" * 78)
    print(f" AVERAGE across {len(gov_all)} zones")
    print("=" * 78)
    print(f" {'metric':26s} {'baseline':>12s} {'GOV':>12s}   {'Δ (GOV vs base)':>16s}")
    print(" " + "-" * 76)
    for label, key, better, unit in _FIELDS:
        bv = float(np.nanmean([m[key] for m in base_all]))
        gv = float(np.nanmean([m[key] for m in gov_all]))
        print(f" {label:26s} {bv:12.4f} {gv:12.4f}   {_pct(bv, gv, better):>16s}  ({unit})")
    print()
    print(" Legend: ✓ = GOV better on this metric, ✗ = GOV worse (>1% change),"
          " blank = ~tie. Safety higher is better; all others lower is better.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
