#!/usr/bin/env python
"""Reproducible reference-governor calibration (Steps 1, 3, 4) for the G1 corridor follower.

This is the LIVE, reproducible realisation of the 4-step framework: the deployed
``best_idx`` and ``m_track`` are DERIVED by running the policy, not hand-set. It
drives ``genedynamics.deploy.followers.governor.reference_selector`` end-to-end:

  * **Step 1** (:func:`select_mode`): load the planner's multimodal candidates,
    prune those outside the policy command envelope (a ``PolicyEnvelope`` proxy
    for the trackable set ``T_policy``), then ROLL OUT each survivor
    (``diagnose(use_governor=False)``) and pick the one with the best *executed*
    body-SDF clearance. This reproduces / validates the baked ``best_idx``
    (planned clearance alone is misleading — high-clearance modes are often
    untrackable).
  * **Step 3** (:func:`epsilon_from_clearance`): on the selected mode,
    ``eps = max(0, planned_clear - executed_clear)`` — the safety-directional
    clearance degradation, NOT the positional offset (which is dominated by
    harmless longitudinal lag).
  * **Step 4** (:func:`m_track_from_epsilon` + :func:`certify_safety`):
    ``m_track = L_g * eps * safety_factor`` (the body SDF is 1-Lipschitz so
    ``L_g = 1``) and the a-posteriori certificate ``executed g(x) <= 0``.

It writes the derived ``best_idx`` + ``m_track`` (+ provenance) back into the
plan's ``trajectory.json`` so the default deploy
(:mod:`run_sport_mode_zones`) loads them, and emits a calibration report.

Run inside docker (mujoco + the spark torch policy)::

  docker run --rm -v "$PWD:/work" -w /work genedynamics/dev-cpu:torch \\
      python scripts/tasks/robot/humanoid/governor_calibrate.py \\
          --zones twogo_zone_a twogo_zone_c twogo_zone_d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.tasks.robot.humanoid.sport_mode_corridor import (  # noqa: E402
    _infer_corridor_scene_meta,
    diagnose,
    resolve_plan_path,
)
from genedynamics.deploy.followers.governor import (  # noqa: E402
    BodyConfig,
    BodySdfAdmissibleSet,
)
from genedynamics.deploy.followers.governor.reference_selector import (  # noqa: E402
    PolicyEnvelope,
    certify_safety,
    epsilon_from_clearance,
    m_track_from_epsilon,
    select_mode,
)
from genedynamics.deploy.followers.common.plan_schema import (  # noqa: E402
    CorridorPlanSchema,
)

_SCHEMA = CorridorPlanSchema.corridor_14d()
_IX = _SCHEMA.index("x")
_IY = _SCHEMA.index("y")
_IPSI = _SCHEMA.index("psi")
_IH = _SCHEMA.index("h")
_IPSIT = _SCHEMA.index("psi_torso")
_IAL = _SCHEMA.index("a_left")
_IAR = _SCHEMA.index("a_right")


def _make_planned_clearance_fn(scene: dict):
    """Return ``states(T,14) -> min planned body-SDF clearance`` for ``scene``."""
    cset = BodySdfAdmissibleSet.from_scene_dict(scene, body=BodyConfig())
    cctx = cset._ctx

    def planned_clearance_fn(states: np.ndarray) -> float:
        S = np.asarray(states, dtype=np.float64)
        if S.ndim != 2 or S.shape[0] == 0:
            return float("inf")
        mn = float("inf")
        for s in S:
            cctx.body = BodyConfig(
                h=float(s[_IH]), psi_torso=float(s[_IPSIT]),
                a_left=float(s[_IAL]), a_right=float(s[_IAR]),
            )
            d = cset.body_min_sdf(float(s[_IX]), float(s[_IY]), float(s[_IPSI]), cctx)
            if d < mn:
                mn = d
        return float(mn)

    return planned_clearance_fn


def calibrate(
    zone: str,
    *,
    max_steps: Optional[int] = None,
    safety_factor: float = 1.3,
    envelope: Optional[PolicyEnvelope] = None,
    dt: float = 0.25,
    bake: bool = True,
    work_dir: Path = _ROOT / "results" / "deploy" / "governor_calib",
    quiet_diagnose: bool = True,
) -> dict:
    """Run live Step-1/3/4 calibration for one zone; optionally bake into the plan."""
    envelope = envelope or PolicyEnvelope()
    plan_path = resolve_plan_path(zone)
    blob = json.loads(plan_path.read_text())
    candidates = np.asarray(blob["candidate_states"], dtype=np.float64)  # (M, T, 14)
    M = candidates.shape[0]
    orig_best = int(blob.get("planner_best_idx", blob.get("best_idx", 0)))

    scene = _infer_corridor_scene_meta(plan_path) or {}
    planned_clearance_fn = _make_planned_clearance_fn(scene)

    print(f"[calib] zone={zone}  plan={plan_path}")
    print(f"[calib] {M} candidate modes; planner_best_idx={orig_best}; "
          f"envelope(v_lat={envelope.v_max_lat}, yaw={envelope.yaw_rate_max})")

    # Step 1 executed evaluator: roll out candidate idx with the RAW policy
    # (governor OFF) and read the executed body-SDF certificate quantity.
    def executed_eval(idx: int):
        run_dir = work_dir / zone / f"cand{idx}"
        r = diagnose(
            plan_path,
            best_idx=int(idx),
            use_governor=False,
            max_steps=max_steps,
            out_dir=run_dir,
            quiet=quiet_diagnose,
        )
        ec = float(r.executed_min_body_sdf)
        print(f"[calib]   cand{idx:>2d}: executed_min_body_sdf={ec:+.4f}  "
              f"fell={r.fell_over}  endpoint={r.endpoint_distance_m:.3f}m")
        return ec, bool(r.fell_over)

    # --- Step 1: feasible ∩ trackable selection by executed clearance --------
    best_idx, reports = select_mode(
        [candidates[i] for i in range(M)],
        planned_clearance_fn,
        dt,
        envelope,
        executed_eval=executed_eval,
        feasible_clear=0.0,
    )
    sel = reports[best_idx]

    # Ensure the selected mode has an executed clearance (it does when it came
    # from the rolled-out pool; if select_mode fell back to planned-only, run it).
    if sel.executed_clear is None:
        ec, fell = executed_eval(best_idx)
        sel.executed_clear, sel.fell = ec, fell

    planned_clear = float(sel.planned_clear)
    executed_clear = float(sel.executed_clear)

    # --- Step 3: epsilon = clearance degradation -----------------------------
    eps = epsilon_from_clearance(planned_clear, executed_clear)
    # --- Step 4: tighten + certify -------------------------------------------
    m_track = m_track_from_epsilon(eps, L_g=1.0, safety_factor=safety_factor)
    certified = certify_safety(executed_clear)

    report = {
        "zone": zone,
        "plan_path": str(plan_path),
        "n_modes": int(M),
        "planner_best_idx": orig_best,
        "selected_best_idx": int(best_idx),
        "reproduces_baked_best_idx": int(best_idx) == int(blob.get("best_idx", -1)),
        "safety_factor": float(safety_factor),
        "selected_mode": {
            "planned_clear": round(planned_clear, 4),
            "executed_clear": round(executed_clear, 4),
            "fell": bool(sel.fell) if sel.fell is not None else None,
        },
        "epsilon_track_clearance_degradation": round(float(eps), 4),
        "m_track_derived": round(float(m_track), 4),
        "certified_safe": bool(certified),
        "per_mode": [
            {
                "idx": r.idx,
                "planned_clear": round(float(r.planned_clear), 4),
                "demand_v_lat": round(float(r.demand_v_lat), 4),
                "demand_yaw_rate": round(float(r.demand_yaw_rate), 4),
                "trackable": bool(r.trackable),
                "executed_clear": (round(float(r.executed_clear), 4)
                                   if r.executed_clear is not None else None),
                "fell": (bool(r.fell) if r.fell is not None else None),
            }
            for r in reports
        ],
    }

    # --- Bake derived best_idx + m_track back into the plan ------------------
    if bake:
        if "planner_best_idx" not in blob:
            blob["planner_best_idx"] = int(blob.get("best_idx", orig_best))
        blob["best_idx"] = int(best_idx)
        blob["best_idx_source"] = "step1_executed_clearance"
        blob["m_track"] = round(float(m_track), 4)
        blob["m_track_source"] = "step3_clearance_degradation"
        plan_path.write_text(json.dumps(blob, indent=2))
        report["baked"] = True

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"{zone}.report.json").write_text(json.dumps(report, indent=2))

    # --- Readable report -----------------------------------------------------
    print("\n" + "=" * 64)
    print(f"REFERENCE-GOVERNOR CALIBRATION (live Step 1/3/4)  [{zone}]")
    print("=" * 64)
    print(f"  planner best_idx (cost-min)   : {orig_best}")
    print(f"  Step-1 selected best_idx      : {best_idx}"
          + ("  (== baked)" if report["reproduces_baked_best_idx"] else "  (DIFFERS from baked!)"))
    print(f"  selected planned clearance    : {planned_clear:+.4f} m")
    print(f"  selected executed clearance   : {executed_clear:+.4f} m  (fell={sel.fell})")
    print(f"  Step-3 epsilon (degradation)  : {eps:.4f} m")
    print(f"  Step-4 m_track = L_g*eps*sf   : {m_track:.4f} m  (sf={safety_factor})")
    print(f"  a-posteriori certificate      : "
          + ("CERTIFIED  g(x)<=0" if certified else "VIOLATED"))
    if bake:
        print(f"  baked best_idx+m_track into   : {plan_path}")
    print("=" * 64 + "\n")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zones", nargs="+", default=["twogo_zone_a", "twogo_zone_c", "twogo_zone_d"],
                   help="Plan aliases to calibrate (default: a, c, d).")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--safety-factor", type=float, default=1.3)
    p.add_argument("--dt", type=float, default=0.25, help="Plan step (s) for the kinematic envelope proxy.")
    p.add_argument("--v-max-lat", type=float, default=0.55)
    p.add_argument("--yaw-rate-max", type=float, default=0.65)
    p.add_argument("--no-bake", action="store_true", help="Compute + report but do NOT write back into the plan.")
    args = p.parse_args(argv)

    envelope = PolicyEnvelope(v_max_lat=args.v_max_lat, yaw_rate_max=args.yaw_rate_max)
    summary: List[str] = []
    for zone in args.zones:
        rep = calibrate(
            zone,
            max_steps=args.max_steps,
            safety_factor=args.safety_factor,
            envelope=envelope,
            dt=args.dt,
            bake=not args.no_bake,
        )
        repro = "==baked" if rep["reproduces_baked_best_idx"] else "DIFFERS"
        cert = "CERT" if rep["certified_safe"] else "VIOL"
        summary.append(
            f"{zone:16s} best_idx={rep['selected_best_idx']} ({repro})  "
            f"eps={rep['epsilon_track_clearance_degradation']:.3f}  "
            f"m_track={rep['m_track_derived']:.3f}  exec_clear="
            f"{rep['selected_mode']['executed_clear']:+.3f}  {cert}"
        )

    print("=" * 72)
    print(" CALIBRATION SUMMARY")
    print("=" * 72)
    for line in summary:
        print(" " + line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
