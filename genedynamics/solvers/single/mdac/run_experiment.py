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

import argparse
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
from genedynamics.solvers.single.mdac.config import (
    deep_merge,
    discover_configs,
    load_experiment_config,
)
from genedynamics.solvers.single.mdac.core.method_registry import assert_fair
from genedynamics.evaluation.aggregate import aggregate_by

DEFAULT_CONFIG_DIR = "configs/arm/impedence/rigid"
RESULT_FILE = "metrics.json"             # written under each config's output_dir

# keys in method_params that belong to the controller factory, not solver **cfg
_MAKE_CONTROLLER_KW = (
    "aug_lambda", "aug_rho", "policy_ckpt", "atacom_policy_ckpt",
    "reliability_ckpt",
)


def _csv_env(name: str) -> List[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


def _apply_run_overrides(configs: List[Dict[str, Any]], cfg_dir: str) -> List[Dict[str, Any]]:
    """Optional diagnostic-run overrides without mutating canonical YAML files.

    Supported env vars:
      MDAC_METHODS, MDAC_VARIANTS, MDAC_LEVELS, MDAC_SUITES filters
      MDAC_SEEDS                 comma-separated integers
      MDAC_N_STEPS, MDAC_NSAMPLE integer overrides
      MDAC_ENV_OVERRIDES         JSON object merged into env_params
      MDAC_METHOD_OVERRIDES      JSON object merged into method_params
      MDAC_OUTPUT_ROOT           redirects outputs, preserving config-tree layout
    """
    methods = set(_csv_env("MDAC_METHODS"))
    variants = set(_csv_env("MDAC_VARIANTS"))
    algorithms = set(_csv_env("MDAC_ALGORITHMS"))
    levels = set(_csv_env("MDAC_LEVELS"))
    suites = set(_csv_env("MDAC_SUITES"))
    out = [
        dict(c) for c in configs
        if (not methods or c["method"] in methods)
        and (not variants or c.get("variant", c["method"]) in variants)
        and (not algorithms or c.get("algorithm", c["method"]) in algorithms)
        and (not levels or c["level"] in levels)
        and (not suites or c.get("suite", c["level"]) in suites)
    ]
    seeds = _csv_env("MDAC_SEEDS")
    n_steps = os.environ.get("MDAC_N_STEPS")
    nsample = os.environ.get("MDAC_NSAMPLE")
    env_overrides = os.environ.get("MDAC_ENV_OVERRIDES")
    method_overrides = os.environ.get("MDAC_METHOD_OVERRIDES")
    output_root = os.environ.get("MDAC_OUTPUT_ROOT")
    extra_env = json.loads(env_overrides) if env_overrides else {}
    extra_method = json.loads(method_overrides) if method_overrides else {}
    if not isinstance(extra_env, dict):
        raise ValueError("MDAC_ENV_OVERRIDES must decode to a JSON object")
    if not isinstance(extra_method, dict):
        raise ValueError("MDAC_METHOD_OVERRIDES must decode to a JSON object")
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
        if extra_method:
            c["method_params"] = {
                **dict(c.get("method_params") or {}), **extra_method,
            }
        if output_root:
            rel = c.get(
                "output_rel",
                os.path.splitext(
                    os.path.relpath(c["config_path"], os.path.abspath(cfg_dir))
                )[0],
            )
            c["output_dir"] = os.path.join(
                output_root, rel
            )
    return configs if not any((
        methods, variants, algorithms, levels, suites, seeds, n_steps, nsample,
        env_overrides, method_overrides, output_root
    )) else out


def _evaluation_configs(
    config_path: str,
    policy_ckpt: str | None,
    policy_seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Expand one algorithm YAML over its declared evaluation environments."""
    base = load_experiment_config(config_path)
    protocol = dict(base.get("evaluation") or {})
    environments = list(protocol.get("environments") or ())
    if not environments:
        raise ValueError(
            "algorithm --config needs evaluation.environments"
        )
    if not base.get("method") or not base.get("name"):
        raise ValueError("algorithm --config needs top-level method and name")
    method_params = dict(base.get("method_params") or {})
    uses_policy = "policy_ckpt" in method_params or policy_ckpt is not None
    training_seed = int(
        protocol.get("policy_seed", base.get("policy_training_seed", 0))
        if policy_seed is None else policy_seed
    )
    if policy_ckpt is not None:
        method_params["policy_ckpt"] = policy_ckpt
    root = str(protocol.get(
        "output_dir",
        base["output_dir"],
    ))
    seeds = [int(x) for x in protocol.get("seeds", (10, 11))]
    n_steps = int(protocol.get("n_steps", base["n_steps"]))
    out: List[Dict[str, Any]] = []
    for environment in environments:
        suite = str(environment["name"])
        level = str(environment["level"])
        env_params = deep_merge(
            dict(base.get("env_params") or {}),
            dict(environment.get("env_params") or {}),
        )
        algorithm = str(base["name"])
        rel = suite
        out.append({
            **base,
            "algorithm": algorithm,
            "suite": suite,
            "level": level,
            "seeds": seeds,
            "n_steps": n_steps,
            "env_params": env_params,
            "method_params": dict(method_params),
            "output_dir": os.path.join(root, rel),
            "output_rel": os.path.join(algorithm, rel),
            "group": str(base.get("group", "baseline")),
            "policy_training_seed": training_seed if uses_policy else None,
            "policy_training_steps": (
                protocol.get("training_steps") if uses_policy else None
            ),
            "policy_training_domains": (
                protocol.get("policy_domain_count") if uses_policy else None
            ),
        })
    return out


def _discover_algorithm_configs(
    config_dir: str,
    policy_ckpt: str | None = None,
    policy_seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Load a flat D3IL-style directory: exactly one YAML per algorithm."""
    out: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(config_dir)):
        if fname.startswith("_") or not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(config_dir, fname)
        out.extend(_evaluation_configs(path, policy_ckpt, policy_seed))
    return out


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
        "force_int", "gravity_normal", "effective_press",
    )}
    h = _np.asarray(d["h_surf"]); sd["surf_resid"] = _np.linalg.norm(h, axis=1).tolist()
    ht = _np.asarray(d["h_tangent"])
    sd["tangential_resid"] = _np.linalg.norm(ht, axis=1).tolist()
    hn = _np.asarray(d["h_normal"]); sd["normal_resid"] = _np.linalg.norm(hn, axis=1).tolist()
    sd["ee"] = _np.asarray(d["positions"]).tolist()
    sd["actions"] = _np.asarray(res.actions, dtype=_np.float32).tolist()
    sd.update(f_min=d.get("f_min"), f_max=d.get("f_max"),
              f_target=float(getattr(env._config, "f_target", 0.0)),
              deformation_safe=float(getattr(env._config, "deformation_safe", 0.0)),
              deformation_scale=float(getattr(env._config, "deformation_scale", 1.0)),
              path_scale=float(getattr(env._config, "geometry_gate_path_scale", 0.02)),
              normal_scale=float(getattr(env._config, "geometry_gate_normal_scale", 0.005)),
              scan_rate=float(getattr(env._config, "scan_rate", 0.0)),
              scan_span=float(getattr(env._config, "scan_span", 1.0)))
    return sd


def _solver_cfg(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split method_params into (make_mdac kwargs, solver **cfg sampling budget)."""
    mp = dict(cfg["method_params"])
    mk = {k: mp.pop(k) for k in _MAKE_CONTROLLER_KW if k in mp}
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
    rec.update(_prior_diagnostics(res))
    try:
        rec["series"] = _series(task, res, env, x0)     # rich raw data for post-hoc plots
    except Exception as e:
        rec["series"] = {"error": f"{type(e).__name__}: {e}"}
    return rec


def _prior_diagnostics(res: Any) -> Dict[str, Any]:
    """Aggregate MDAC-prior and baseline-controller diagnostics."""
    infos = [x for x in getattr(res, "infos", ()) if isinstance(x, dict)]
    accepted = [x["prior_accepted"] for x in infos if "prior_accepted" in x]

    def mean_scalar(key: str) -> float:
        values = [np.asarray(x[key], dtype=float).reshape(-1) for x in infos if key in x]
        return float(np.mean(np.concatenate(values))) if values else float("nan")

    out: Dict[str, Any] = {}
    if accepted:
        out.update({
            "prior_acceptance_rate": mean_scalar("prior_accepted"),
            "prior_predicted_improvement_mean": mean_scalar(
                "prior_predicted_improvement"
            ),
            "prior_risk_ok_rate": mean_scalar("prior_risk_ok"),
            "prior_force_veto_rate": mean_scalar("prior_force_veto"),
        })
        out["prior_fallback_rate"] = 1.0 - out["prior_acceptance_rate"]
        for source in (
            "prior_risk_rl", "prior_risk_incumbent", "prior_risk_refined",
            "prior_risk_atacom",
            "reliability_risk_incumbent", "reliability_risk_refined",
        ):
            vectors = [np.asarray(x[source], dtype=float).reshape(-1) for x in infos if source in x]
            if vectors:
                mean = np.mean(np.stack(vectors), axis=0)
                labels = ("force_violation", "contact_loss", "deformation", "force_mae")
                out.update({
                    f"{source}_{label}": float(value)
                    for label, value in zip(labels, mean)
                })
        out.update({
            key: mean_scalar(key)
            for key in (
                "reliability_support_incumbent", "reliability_support_refined",
                "proposal_stochastic_count", "proposal_atacom_count",
                "proposal_logp_mean",
                "proposal_gaussian_best_reward", "proposal_rl_best_reward",
                "proposal_atacom_best_reward", "proposal_gaussian_weight",
                "proposal_rl_weight", "proposal_atacom_weight",
                "atacom_incumbent_selected", "prior_score_atacom",
            )
            if any(key in x for x in infos)
        })
    if any("failure" in x for x in infos):
        out.update({
            "issa_projection_failure_rate": mean_scalar("failure"),
            "issa_intervention_mean": mean_scalar("intervention"),
            "issa_projected_margin_mean": mean_scalar("margin"),
        })
    if any("atacom_slack_norm" in x for x in infos):
        out.update({
            "atacom_slack_norm_mean": mean_scalar("atacom_slack_norm"),
            "atacom_action_norm_mean": mean_scalar("atacom_action_norm"),
        })
    return out


def _table(records: List[Dict[str, Any]], algorithms: List[str], task: str, suite: str,
           level: str, n_steps: int, nsample: int) -> None:
    rows = [{k: v for k, v in r.items() if k != "series"}        # series = raw arrays, not a metric
            for r in records if r.get("task") == task
            and r.get("suite", r.get("level")) == suite]
    if not rows:
        return
    agg = aggregate_by(rows, "algorithm")
    metrics = [
        k for k in rows[0]
        if k not in (
            "task", "method", "algorithm", "variant", "suite", "level", "seed",
            "policy_training_seed", "policy_training_steps",
        )
    ]
    print(f"\n==== {task} / {suite} [{level}]  (n_steps={n_steps}, Nsample={nsample}) ====")
    print("method".ljust(22) + "".join(m[:13].rjust(14) for m in metrics))
    for algorithm in algorithms:
        if algorithm not in agg:
            continue
        cells = "".join(f"{agg[algorithm].get(m, {}).get('mean', float('nan')):14.4g}" for m in metrics)
        print(algorithm.ljust(22) + cells)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help="one algorithm YAML; expands only its evaluation environments",
    )
    ap.add_argument(
        "--config-dir",
        default=None,
        help="flat D3IL-style directory containing one YAML per algorithm",
    )
    ap.add_argument(
        "--policy-ckpt",
        default=None,
        help="shared standalone-RL/MGA-prior checkpoint override",
    )
    ap.add_argument(
        "--policy-seed",
        type=int,
        default=None,
        help="training-seed metadata for --policy-ckpt (default: config value)",
    )
    args = ap.parse_args(argv)
    if args.config and args.config_dir:
        ap.error("use either --config or --config-dir, not both")
    if args.config:
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        selected = _evaluation_configs(
            args.config, args.policy_ckpt, policy_seed=args.policy_seed
        )
    elif args.config_dir:
        cfg_dir = os.path.abspath(args.config_dir)
        selected = _discover_algorithm_configs(
            cfg_dir, args.policy_ckpt, policy_seed=args.policy_seed
        )
    else:
        cfg_dir = os.environ.get("MDAC_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        selected = discover_configs(cfg_dir)
    configs = _apply_run_overrides(selected, cfg_dir)
    if not configs:
        print(f"no selected method configs found under {cfg_dir}")
        return 1

    # fairness: every method must share the sampling budget (Nsample, Hsample, Ndiffuse).
    ref = _budget(configs[0])
    for c in configs[1:]:
        assert_fair(ref, _budget(c))

    variants = list(dict.fromkeys(
        c.get("algorithm", c.get("variant", c["method"])) for c in configs
    ))
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_runs = sum(len(c["seeds"]) for c in configs)
    i = 0
    for cfg in configs:
        task, method, level = cfg["task"], cfg["method"], cfg["level"]
        variant = cfg.get("algorithm", cfg.get("variant", method))
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, RESULT_FILE)
        manifest = {
            "config": cfg,
            "git_sha": _git_sha(),
            "run_overrides": {
                k: os.environ[k] for k in (
                    "MDAC_METHODS", "MDAC_VARIANTS", "MDAC_ALGORITHMS",
                    "MDAC_LEVELS", "MDAC_SUITES",
                    "MDAC_SEEDS", "MDAC_N_STEPS",
                    "MDAC_NSAMPLE", "MDAC_ENV_OVERRIDES", "MDAC_OUTPUT_ROOT",
                    "MDAC_METHOD_OVERRIDES",
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
                    task=task, method=method, algorithm=variant, variant=variant,
                    suite=cfg.get("suite", level), level=level, seed=seed,
                    policy_training_seed=cfg.get("policy_training_seed"),
                    policy_training_steps=cfg.get("policy_training_steps"),
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
    seen: List[Tuple[str, str, str]] = []
    for c in configs:
        key = (c["task"], c.get("suite", c["level"]), c["level"])
        if key not in seen:
            seen.append(key)
    for task, suite, level in seen:
        _table(
            records, variants, task, suite, level,
            int(configs[0]["n_steps"]), _budget(configs[0])[0],
        )
    print(
        f"\nran {len(records)}/{n_runs} records across "
        f"{len(variants)} algorithms / {len(configs)} evaluations from {cfg_dir}"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
