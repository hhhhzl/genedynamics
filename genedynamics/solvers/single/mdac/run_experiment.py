"""Run the MDAC arm/humanoid experiment (closed-loop) -> per-method metric tables.

Lightweight driver (experiment_run_plan.md §1): for each (task, surface/level,
method, seed) it builds a fairly-configured solver via ``make_mdac``
(``prior=None`` — no RL prior in this run), runs the receding-horizon closed
loop, extracts the general metrics, and aggregates by method (mean ± std +
CVaR95). Records flush to JSON after every run so a crash/OOM never loses work.

Task matrix (idea.txt):
  * Arm  ``manipulator_surface_scan`` surface families S1-S4:
      S1 = {plane, cylinder}, S2 = convex NURBS, S3 = bumpy NURBS,
      S4 = unseen NURBS + domain randomization. ``surface_seed`` selects the
      random NURBS / DR draw, so each seed is a different surface for S2-S4.
  * Humanoid ``humanoid_box_push`` levels H1 (double-support), H2 (+DR),
      H4 box-unjamming (12D) and H4-B (+v_base, 15D). ``surface_seed`` -> the
      H2 DR draw.

All methods share one CFG so the sample budget (Nsample, Hsample, Ndiffuse) is
identical (``assert_fair``). prior=None => ``mdac_no_rl_prior`` would equal
``mdac``, so it is excluded here. Scale CFG / SEEDS / METHODS at the top.

Run in docker (real brax):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python -m genedynamics.solvers.single.mdac.run_experiment"
"""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import jax

from genedynamics.solvers.single.mdac.experiment import (
    make_mdac, metrics_plugin_for, ARM_TASK, HUMANOID_TASK,
)
from genedynamics.evaluation.aggregate import aggregate_by

# --- run configuration (small pilot; scale up once the pipeline is verified) --
# DIAL-aligned sampling budget (matches the solver's DIAL-inherited defaults:
# Hsample=16, Nsample=2048, Ndiffuse_init=10, temp_sample=0.06) so the comparison
# runs at DIAL's intended planning horizon + sample budget, not a reduced pilot.
CFG = dict(Hsample=16, Hnode=4, Nsample=2048, Ndiffuse_init=10, Ndiffuse=2,
           temp_sample=0.06, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)
N_STEPS = 14
SEEDS = [0, 1]
# prior=None for every method -> no_rl_prior == mdac, so it is excluded.
METHODS = ["mdac", "dial", "mppi",
           "mdac_no_softfeas", "mdac_no_stiffness", "mdac_fixed_stiffness",
           "mdac_euclid_stiffness", "mdac_no_tangent", "mdac_no_retraction",
           "mdac_no_anneal"]

# (task, level, extra make_mdac kwargs, table label). S1 expands to plane+cylinder.
RUNS: List[Tuple[str, str, Dict[str, Any], str]] = (
    [(ARM_TASK, lv, {}, lv) for lv in ("plane", "cylinder", "convex", "bumpy", "unseen")]
    + [(HUMANOID_TASK, "double_support", {}, "double_support"),
       (HUMANOID_TASK, "heavy_dr", {}, "heavy_dr"),
       (HUMANOID_TASK, "unjam", {}, "unjam"),
       (HUMANOID_TASK, "unjam", {"use_base": True}, "unjam_base")]
)
OUT = "docs/mdac/results"


def _run_one(task: str, level: str, kw: Dict[str, Any], seed: int, method: str) -> Dict[str, Any]:
    env, sol = make_mdac(task, method, level=level, surface_seed=seed, prior=None, **kw, **CFG)
    x0 = env.reset(jax.random.PRNGKey(seed))
    t0 = time.time()
    res = sol.run_receding(x0, N_STEPS, jax.random.PRNGKey(1000 + seed))
    dt = time.time() - t0
    rec = metrics_plugin_for(task).compute(res, env, None, None, x0=x0, planning_time=dt)
    return rec


def _table(records: List[Dict[str, Any]], task: str, label: str) -> None:
    rows = [r for r in records if r.get("task") == task and r.get("level") == label]
    if not rows:
        print(f"  (no records for {task}/{label})")
        return
    agg = aggregate_by(rows, "method")
    metrics = [k for k in rows[0] if k not in ("task", "method", "level", "seed")]
    print(f"\n==== {task} / {label}  (n_steps={N_STEPS}, seeds={len(SEEDS)}, "
          f"Nsample={CFG['Nsample']}) ====")
    print("method".ljust(22) + "".join(m[:13].rjust(14) for m in metrics))
    for method in METHODS:
        if method not in agg:
            continue
        cells = "".join(f"{agg[method].get(m, {}).get('mean', float('nan')):14.4g}" for m in metrics)
        print(method.ljust(22) + cells)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, os.environ.get("RESULT_FILE", "pilot_noprior.json"))
    records: List[Dict[str, Any]] = []
    total = len(RUNS) * len(METHODS) * len(SEEDS)
    i = 0
    for task, level, kw, label in RUNS:
        for method in METHODS:
            for seed in SEEDS:
                i += 1
                tag = f"{task.split('_')[0]}/{label}/{method}/seed{seed}"
                try:
                    rec = _run_one(task, level, kw, seed, method)
                    rec.update(task=task, method=method, level=label, seed=seed)
                    records.append(rec)
                    with open(path, "w") as f:
                        json.dump(records, f, indent=2, default=float)
                    jax.clear_caches(); gc.collect()       # release XLA executables (avoid OOM over many runs)
                    key = "violation_rate" if task == HUMANOID_TASK else "surface_tracking_error"
                    print(f"[{i}/{total}] {tag}  {key}~{rec.get(key, float('nan')):.4g}")
                except Exception as e:        # one failure must not kill the matrix
                    print(f"[{i}/{total}] {tag}  FAILED: {type(e).__name__}: {e}")
    for task, _, _, label in RUNS:
        _table(records, task, label)
    print(f"\nsaved {len(records)} records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
