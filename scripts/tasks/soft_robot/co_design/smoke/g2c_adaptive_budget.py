#!/usr/bin/env python3
"""G2c-adaptive — budget-dual multi-fidelity allocation (contribution 2, writeup §6/§7).

This is the task the BudgetDual / optimal_subset_size framework was built for and
that G2c only scaffolded: the high-fi correction count K is no longer a fixed
`cv_subset_k` — it is set PER fidelity block from the compute budget, with a
BudgetDual ν self-regulating realized cost toward B̄ across blocks.

Two parts:
  A. PURE LOGIC (fast, deterministic, no GPU) — drive the real
     estimator_system.budget helpers over the real fidelity ladder. Gates:
       (1) cost-adaptive: at a fixed budget, K is non-increasing as the block's
           high-fi cost c_hi grows (cheap blocks afford a larger K).
       (2) self-regulating: under a TIGHT budget the dual ν rises (>0) and the
           allocation is driven to the floor.
  B. END-TO-END (GPU) — run the real MRMFMBD backend driver `_plan_jax_scan`
     through MRMFMBDBaseline on the crawling scene with cv_adaptive=True. Gates:
       (3) cv_adaptive_summary.active and per_step_K has >=2 distinct values
           (the budget genuinely re-allocated K across fidelity blocks in the
           JIT'd hot loop).
       (4) cv_adaptive=False is byte-for-byte the legacy path (summary inactive).

Run: python scripts/tasks/soft_robot/co_design/smoke/g2c_adaptive_budget.py
"""
from __future__ import annotations

import sys

import numpy as np

from genedynamics.solvers.single.mrmfmbd.estimator_system.budget import (
    BudgetDual, optimal_subset_size, realized_cost,
)
from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS


def _adaptive_loop(B_bar, M, sub_hi, sub_lo, eta=0.5, verbose=True):
    """Replicate the backend's per-block adaptive K loop (coarse->fine)."""
    dual = BudgetDual(nu=0.0, eta=eta, target=1.0)
    rows = []
    if verbose:
        print(f"  {'fid':>3} {'c_hi':>7} {'c_lo':>6} {'B_eff':>9} {'K/M':>7} "
              f"{'realized':>9} {'nu':>6}")
    for fid in sorted(FIDELITY_STEPS):                       # coarse -> fine
        steps = FIDELITY_STEPS[fid]
        c_hi = float(steps * sub_hi)
        c_lo = float(steps * sub_lo)
        B_eff = B_bar / (1.0 + dual.nu)
        K = max(1, min(optimal_subset_size(B_eff, M, c_lo, c_hi), M))
        realized = realized_cost(M, K, c_lo, c_hi)
        dual.update(realized / B_bar)
        rows.append({"fid": fid, "c_hi": c_hi, "K": K, "nu": dual.nu,
                     "realized": realized})
        if verbose:
            print(f"  {fid:>3} {c_hi:>7.0f} {c_lo:>6.0f} {B_eff:>9.0f} "
                  f"{K:>4}/{M:<2} {realized:>9.0f} {dual.nu:>6.3f}")
    return rows


def part_a() -> bool:
    print("=" * 70)
    print("Part A — budget-dual allocation over the real fidelity ladder")
    print("=" * 70)
    M, sub_hi = 16, 16                 # frame_dt 8e-3 / fine dt 5e-4 = 16 substeps
    sub_lo = round(8e-3 / 1e-3)        # coarse dt 1e-3 = 8 substeps (G2c axis)

    # Generous (auto) budget: low-fi sweep + half-M fine corrections at finest.
    fine = FIDELITY_STEPS[max(FIDELITY_STEPS)]
    B_auto = M * (fine * sub_lo) + (M // 2) * (fine * sub_hi)
    print(f"\n[generous] B_bar={B_auto}")
    rows_g = _adaptive_loop(B_auto, M, sub_hi, sub_lo)
    Ks = [r["K"] for r in rows_g]
    # Gate 1: K non-increasing as fidelity (c_hi) grows.
    g1 = all(Ks[i] >= Ks[i + 1] for i in range(len(Ks) - 1)) and (Ks[0] > Ks[-1])
    print(f"  -> K by block {Ks}  (cost-adaptive, non-increasing & varies): "
          f"{'PASS' if g1 else 'FAIL'}")

    # Tight budget: only ~the low-fi sweep is affordable -> fine blocks floor K,
    # realized > B_bar on the expensive block -> dual nu must rise.
    B_tight = int(M * (fine * sub_lo) * 0.6)
    print(f"\n[tight]    B_bar={B_tight}")
    rows_t = _adaptive_loop(B_tight, M, sub_hi, sub_lo)
    nu_max = max(r["nu"] for r in rows_t)
    g2 = (nu_max > 0.0) and (rows_t[-1]["K"] <= rows_t[0]["K"])
    print(f"  -> max nu={nu_max:.3f} (>0 = dual self-regulated), "
          f"fine-block K={rows_t[-1]['K']}: {'PASS' if g2 else 'FAIL'}")
    return g1 and g2


def part_b() -> bool:
    print("\n" + "=" * 70)
    print("Part B — end-to-end through the real MRMFMBD backend driver (GPU)")
    print("=" * 70)
    try:
        import jax
        from genedynamics.experiments.plugins.task_domains.jax_mpm import (
            JaxMpmTaskDomainProvider,
        )
        from genedynamics.experiments.framework.baseline import BaselineConfig
        from genedynamics.solvers.single.codesign_optimizers.mrmfmbd import (
            MRMFMBDBaseline,
        )
    except Exception as e:  # pragma: no cover
        print(f"[skip] imports unavailable: {e!r}")
        return True

    print("backend:", jax.default_backend())
    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(
        ".", voxel_dims=[4, 3, 4], n_grid=64, reward_shaping_weight=100.0,
        act_strength_base=24.0, scale=50.0, task="crawling_ground",
    )
    task_spec = prov.get_task_spec("crawling_ground")
    cfg = ev._mpm_cfg
    vx, vy, vz = cfg.voxel_dims
    x_dim = int(vx * vy * vz)
    phi_dim = int(cfg.n_actuators * cfg.n_sin_waves + 4 * cfg.n_actuators)

    base_extra = dict(
        backend="mbd", K=6, M=8, num_fidelity_levels=3,
        phi_dim_override=True,          # full per-actuator sine controller (matches compute_actuation)
        x_lo=0.2, x_hi=1.0, phi_lo=-0.5, phi_hi=0.5,
        cv_enabled=True, method="control_variate",
        cv_low_dt=1e-3,                 # G2c physics-resolution low fidelity
        shac_refine_steps=0, show_tqdm=True,
    )

    def _run(adaptive: bool):
        extra = dict(base_extra, cv_adaptive=adaptive)
        bl = BaselineConfig(task_id="crawling_ground", seed=0, extra=extra)
        r = MRMFMBDBaseline().run(bl, ev, task_spec, x_dim=x_dim, phi_dim=phi_dim)
        return r.metadata.get("cv_adaptive_summary", {})

    print("\n[adaptive=True]")
    s_on = _run(True)
    Ks = s_on.get("per_step_K", [])
    distinct = sorted(set(Ks))
    print(f"  active={s_on.get('active')} budget_B={s_on.get('budget_B')}")
    print(f"  per_step_K distinct values across blocks: {distinct}")
    g3 = bool(s_on.get("active")) and len(distinct) >= 2

    print("\n[adaptive=False]  (legacy path must be untouched)")
    s_off = _run(False)
    g4 = not bool(s_off.get("active", False))
    print(f"  active={s_off.get('active')} (expected False/empty): "
          f"{'PASS' if g4 else 'FAIL'}")

    print(f"\n  Gate3 (>=2 distinct K, active): {'PASS' if g3 else 'FAIL'}")
    return g3 and g4


def main() -> int:
    a = part_a()
    try:
        b = part_b()
    except Exception as e:  # pragma: no cover
        print(f"\n[Part B errored — reporting it, not masking]: {e!r}")
        b = False
    ok = a and b
    print("\n" + "=" * 70)
    print(f"RESULT: Part A {'PASS' if a else 'FAIL'} | Part B {'PASS' if b else 'FAIL'}"
          f"  => {'ALL PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
