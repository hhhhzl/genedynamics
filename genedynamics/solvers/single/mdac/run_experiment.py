"""Run the MDAC arm/humanoid experiment (closed-loop) -> per-method metric tables.

Config-driven driver: it walks a config tree subdivided BY ENVIRONMENT (default
``configs/arm/impedence/rigid/<environment>/<role>/<method>.yaml``, override the
root with ``MDAC_CONFIG_DIR``). Each method yaml carries a single ``level`` (its
environment) and inherits the shared ``_base.yaml`` (task / seeds / horizon /
sampling budget), so every method in every environment runs at an IDENTICAL budget
(``assert_fair``) — the comparison differs only by the method's component flags.
For each (config, seed) it builds a fairly-configured solver via ``make_mdac``
(``prior=None`` — no RL prior in this run), runs the receding-horizon closed loop,
extracts the general metrics, and writes per-method records to that config's
``output_dir`` after every run (a crash never loses work).

Environments (idea.txt surface families S1-S4): plane / cylinder (S1 analytic),
convex (S2 NURBS), bumpy (S3 NURBS), unseen (S4 NURBS + domain randomization). The
per-seed ``surface_seed`` selects the random NURBS / DR draw, so each seed is a
different surface. (Humanoid ``humanoid_box_push``: add a humanoid config tree.)

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
    make_controller, metrics_plugin_for, ARM_TASK, HUMANOID_TASK,
)
from genedynamics.solvers.single.mdac.config import discover_configs
from genedynamics.solvers.single.mdac.core.method_registry import assert_fair
from genedynamics.evaluation.aggregate import aggregate_by

DEFAULT_CONFIG_DIR = "configs/arm/impedence/rigid"
RESULT_FILE = "metrics.json"             # written under each config's output_dir

# keys in method_params that are make_mdac's own kwargs, not solver **cfg
_MAKE_MDAC_KW = ("aug_lambda", "aug_rho")


def _series(task: str, res: Any, env: Any, x0: Any) -> Dict[str, Any]:
    """Rich per-step time-series for post-hoc plots (force vs time + f_min/f_max bands,
    tracking residuals, contact, stiffness) -- so we keep raw data and decide the best
    view/metric later instead of committing now. Re-uses the metric extractor's signals
    (a cheap env.step roll, NOT the planner)."""
    import numpy as _np
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_signals, humanoid_box_push_signals)
    ex = humanoid_box_push_signals if task == HUMANOID_TASK else arm_surface_scan_signals
    d = ex(res, env, None, None, x0=x0)
    def lst(k):
        v = d.get(k)
        return None if v is None else _np.asarray(v).reshape(len(_np.asarray(v)), -1).squeeze().tolist()
    if task == HUMANOID_TASK:
        return {k: lst(k) for k in ("box_x", "force", "g_bal", "g_fric", "tip_series",
                                    "f_normal", "f_tangential", "slip_speed", "in_contact")}
    sd = {k: lst(k) for k in ("force", "force_cmd", "in_contact", "on_surface")}
    h = _np.asarray(d["h_surf"]); sd["surf_resid"] = _np.linalg.norm(h, axis=1).tolist()
    hn = _np.asarray(d["h_normal"]); sd["normal_resid"] = _np.linalg.norm(hn, axis=1).tolist()
    sd["ee"] = _np.asarray(d["positions"]).tolist()
    sd.update(f_min=d.get("f_min"), f_max=d.get("f_max"),
              f_target=float(getattr(env._config, "f_target", 0.0)))
    return sd


def _solver_cfg(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split method_params into (make_mdac kwargs, solver **cfg sampling budget)."""
    mp = dict(cfg["method_params"])
    mk = {k: mp.pop(k) for k in _MAKE_MDAC_KW if k in mp}
    return mk, mp


def _budget(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """(Nsample, Hsample, Ndiffuse) — the fairness budget that must match across methods."""
    mp = cfg["method_params"]
    return (int(mp["Nsample"]), int(mp["Hsample"]), int(mp["Ndiffuse"]))


def _run_one(cfg: Dict[str, Any], level: str, seed: int) -> Dict[str, Any]:
    task, method = cfg["task"], cfg["method"]
    mk, sampling = _solver_cfg(cfg)
    env, sol = make_controller(task, method, level=level, surface_seed=seed, prior=None,
                               env_overrides=cfg.get("env_params"), **mk, **sampling)
    x0 = env.reset(jax.random.PRNGKey(seed))
    t0 = time.time()
    res = sol.run_receding(x0, int(cfg["n_steps"]), jax.random.PRNGKey(1000 + seed))
    dt = time.time() - t0
    rec = metrics_plugin_for(task).compute(res, env, None, None, x0=x0, planning_time=dt)
    try:
        rec["series"] = _series(task, res, env, x0)     # rich raw data for post-hoc plots
    except Exception as e:
        rec["series"] = {"error": f"{type(e).__name__}: {e}"}
    return rec


def _table(records: List[Dict[str, Any]], methods: List[str], task: str, level: str,
           n_steps: int, nsample: int) -> None:
    rows = [{k: v for k, v in r.items() if k != "series"}        # series = raw arrays, not a metric
            for r in records if r.get("task") == task and r.get("level") == level]
    if not rows:
        return
    agg = aggregate_by(rows, "method")
    metrics = [k for k in rows[0] if k not in ("task", "method", "level", "seed")]
    print(f"\n==== {task} / {level}  (n_steps={n_steps}, Nsample={nsample}) ====")
    print("method".ljust(22) + "".join(m[:13].rjust(14) for m in metrics))
    for method in methods:
        if method not in agg:
            continue
        cells = "".join(f"{agg[method].get(m, {}).get('mean', float('nan')):14.4g}" for m in metrics)
        print(method.ljust(22) + cells)


def main() -> int:
    cfg_dir = os.environ.get("MDAC_CONFIG_DIR", DEFAULT_CONFIG_DIR)
    configs = discover_configs(cfg_dir)
    if not configs:
        print(f"no method configs found under {cfg_dir}/{{baselines,main,ablation}}/*.yaml")
        return 1

    # fairness: every method must share the sampling budget (Nsample, Hsample, Ndiffuse).
    ref = _budget(configs[0])
    for c in configs[1:]:
        assert_fair(ref, _budget(c))

    methods = list(dict.fromkeys(c["method"] for c in configs))   # unique, main->baseline->ablation
    records: List[Dict[str, Any]] = []
    n_runs = sum(len(c["seeds"]) for c in configs)
    i = 0
    for cfg in configs:
        task, method, level = cfg["task"], cfg["method"], cfg["level"]
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, RESULT_FILE)
        method_recs: List[Dict[str, Any]] = []
        for seed in cfg["seeds"]:
            i += 1
            tag = f"{task.split('_')[0]}/{level}/{method}/seed{seed}"
            try:
                rec = _run_one(cfg, level, seed)
                rec.update(task=task, method=method, level=level, seed=seed)
                method_recs.append(rec)
                records.append(rec)
                with open(path, "w") as f:
                    json.dump(method_recs, f, indent=2, default=float)
                jax.clear_caches(); gc.collect()       # release XLA executables (avoid OOM over many runs)
                key = "violation_rate" if task == HUMANOID_TASK else "surface_tracking_error"
                print(f"[{i}/{n_runs}] {tag}  {key}~{rec.get(key, float('nan')):.4g}")
            except Exception as e:        # one failure must not kill the matrix
                print(f"[{i}/{n_runs}] {tag}  FAILED: {type(e).__name__}: {e}")
        print(f"  saved {len(method_recs)} records -> {path}")

    # comparison tables: per (task, environment/level), methods as rows.
    seen: List[Tuple[str, str]] = []
    for c in configs:
        key = (c["task"], c["level"])
        if key not in seen:
            seen.append(key)
    for task, level in seen:
        _table(records, methods, task, level, int(configs[0]["n_steps"]), _budget(configs[0])[0])
    print(f"\nran {len(records)}/{n_runs} records across {len(configs)} methods from {cfg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
