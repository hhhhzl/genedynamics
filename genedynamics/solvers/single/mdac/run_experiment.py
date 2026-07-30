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
import subprocess
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


def _csv_env(name: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def _apply_run_overrides(configs: List[Dict[str, Any]], cfg_dir: str) -> List[Dict[str, Any]]:
    """Optional diagnostic-run overrides without mutating canonical YAML files.

    Supported env vars:
      MDAC_METHODS, MDAC_VARIANTS, MDAC_LEVELS  comma-separated filters
      MDAC_SEEDS                 comma-separated integers
      MDAC_N_STEPS, MDAC_NSAMPLE integer overrides
      MDAC_ENV_OVERRIDES         JSON object merged into env_params
      MDAC_OUTPUT_ROOT           redirects outputs, preserving config-tree layout
    """
    methods = set(_csv_env("MDAC_METHODS"))
    variants = set(_csv_env("MDAC_VARIANTS"))
    levels = set(_csv_env("MDAC_LEVELS"))
    out = [
        dict(c) for c in configs
        if (not methods or c["method"] in methods)
        and (not variants or c.get("variant", c["method"]) in variants)
        and (not levels or c["level"] in levels)
    ]
    seeds = _csv_env("MDAC_SEEDS")
    n_steps = os.environ.get("MDAC_N_STEPS")
    nsample = os.environ.get("MDAC_NSAMPLE")
    env_overrides = os.environ.get("MDAC_ENV_OVERRIDES")
    output_root = os.environ.get("MDAC_OUTPUT_ROOT")
    extra_env = json.loads(env_overrides) if env_overrides else {}
    if not isinstance(extra_env, dict):
        raise ValueError("MDAC_ENV_OVERRIDES must decode to a JSON object")
    for c in out:
        if seeds:
            c["seeds"] = [int(x) for x in seeds]
        if n_steps:
            c["n_steps"] = int(n_steps)
        if nsample:
            c["method_params"] = dict(c["method_params"])
            c["method_params"]["Nsample"] = int(nsample)
        if extra_env:
            c["env_params"] = {**dict(c.get("env_params") or {}), **extra_env}
        if output_root:
            rel = os.path.relpath(c["config_path"], os.path.abspath(cfg_dir))
            c["output_dir"] = os.path.join(
                output_root, os.path.splitext(rel)[0]
            )
    return configs if not any((
        methods, variants, levels, seeds, n_steps, nsample, env_overrides, output_root
    )) else out


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


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
    sd = {k: lst(k) for k in (
        "force", "force_cmd", "in_contact", "on_surface", "on_path_common",
        "deformation", "scan_xi", "normal_offset",
        "gate_scalar", "gate_path", "gate_normal", "gate_stiffness", "gate_force",
        "gate_path_error", "gate_normal_error", "gate_force_error",
        "gate_deformation_risk", "gate_contact_loss", "realization_offset",
    )}
    h = _np.asarray(d["h_surf"]); sd["surf_resid"] = _np.linalg.norm(h, axis=1).tolist()
    ht = _np.asarray(d["h_tangent"])
    sd["tangential_resid"] = _np.linalg.norm(ht, axis=1).tolist()
    hn = _np.asarray(d["h_normal"]); sd["normal_resid"] = _np.linalg.norm(hn, axis=1).tolist()
    sd["ee"] = _np.asarray(d["positions"]).tolist()
    sd.update(f_min=d.get("f_min"), f_max=d.get("f_max"),
              f_target=float(getattr(env._config, "f_target", 0.0)),
              deformation_safe=float(getattr(env._config, "deformation_safe", 0.0)),
              deformation_scale=float(getattr(env._config, "deformation_scale", 1.0)))
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
    # The task metric extractors replay executed actions from x0, so retaining
    # every full MJX State only causes long-run device-memory growth.
    res = sol.run_receding(
        x0, int(cfg["n_steps"]), jax.random.PRNGKey(1000 + seed),
        collect_states=False,
        synchronize_steps=True,
    )
    dt = time.time() - t0
    rec = metrics_plugin_for(task).compute(res, env, None, None, x0=x0, planning_time=dt)
    try:
        rec["series"] = _series(task, res, env, x0)     # rich raw data for post-hoc plots
    except Exception as e:
        rec["series"] = {"error": f"{type(e).__name__}: {e}"}
    return rec


def _table(records: List[Dict[str, Any]], variants: List[str], task: str, level: str,
           n_steps: int, nsample: int) -> None:
    rows = [{k: v for k, v in r.items() if k != "series"}        # series = raw arrays, not a metric
            for r in records if r.get("task") == task and r.get("level") == level]
    if not rows:
        return
    agg = aggregate_by(rows, "variant")
    metrics = [
        k for k in rows[0]
        if k not in ("task", "method", "variant", "level", "seed")
    ]
    print(f"\n==== {task} / {level}  (n_steps={n_steps}, Nsample={nsample}) ====")
    print("method".ljust(22) + "".join(m[:13].rjust(14) for m in metrics))
    for variant in variants:
        if variant not in agg:
            continue
        cells = "".join(f"{agg[variant].get(m, {}).get('mean', float('nan')):14.4g}" for m in metrics)
        print(variant.ljust(22) + cells)


def main() -> int:
    cfg_dir = os.environ.get("MDAC_CONFIG_DIR", DEFAULT_CONFIG_DIR)
    configs = _apply_run_overrides(discover_configs(cfg_dir), cfg_dir)
    if not configs:
        print(f"no selected method configs found under {cfg_dir}")
        return 1

    # fairness: every method must share the sampling budget (Nsample, Hsample, Ndiffuse).
    ref = _budget(configs[0])
    for c in configs[1:]:
        assert_fair(ref, _budget(c))

    variants = list(dict.fromkeys(
        c.get("variant", c["method"]) for c in configs
    ))
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_runs = sum(len(c["seeds"]) for c in configs)
    i = 0
    for cfg in configs:
        task, method, level = cfg["task"], cfg["method"], cfg["level"]
        variant = cfg.get("variant", method)
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, RESULT_FILE)
        manifest = {
            "config": cfg,
            "git_sha": _git_sha(),
            "run_overrides": {
                k: os.environ[k] for k in (
                    "MDAC_METHODS", "MDAC_VARIANTS", "MDAC_LEVELS",
                    "MDAC_SEEDS", "MDAC_N_STEPS",
                    "MDAC_NSAMPLE", "MDAC_ENV_OVERRIDES", "MDAC_OUTPUT_ROOT",
                ) if k in os.environ
            },
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=float)
        method_recs: List[Dict[str, Any]] = []
        for seed in cfg["seeds"]:
            i += 1
            tag = f"{task.split('_')[0]}/{level}/{variant}/seed{seed}"
            try:
                rec = _run_one(cfg, level, seed)
                rec.update(
                    task=task, method=method, variant=variant,
                    level=level, seed=seed,
                )
                method_recs.append(rec)
                records.append(rec)
                with open(path, "w") as f:
                    json.dump(method_recs, f, indent=2, default=float)
                jax.clear_caches(); gc.collect()       # release XLA executables (avoid OOM over many runs)
                key = "violation_rate" if task == HUMANOID_TASK else "surface_tracking_error"
                print(f"[{i}/{n_runs}] {tag}  {key}~{rec.get(key, float('nan')):.4g}")
            except Exception as e:        # one failure must not kill the matrix
                failure = {
                    "task": task, "method": method, "variant": variant,
                    "level": level, "seed": seed,
                    "error_type": type(e).__name__, "error": str(e),
                }
                failures.append(failure)
                with open(os.path.join(out_dir, "failures.json"), "w") as f:
                    json.dump([x for x in failures if x["variant"] == variant and x["level"] == level],
                              f, indent=2)
                print(f"[{i}/{n_runs}] {tag}  FAILED: {type(e).__name__}: {e}")
        print(f"  saved {len(method_recs)} records -> {path}")

    # comparison tables: per (task, environment/level), methods as rows.
    seen: List[Tuple[str, str]] = []
    for c in configs:
        key = (c["task"], c["level"])
        if key not in seen:
            seen.append(key)
    for task, level in seen:
        _table(records, variants, task, level, int(configs[0]["n_steps"]), _budget(configs[0])[0])
    print(f"\nran {len(records)}/{n_runs} records across {len(configs)} methods from {cfg_dir}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
