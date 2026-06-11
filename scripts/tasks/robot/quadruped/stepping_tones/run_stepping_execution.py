#!/usr/bin/env python3
"""
Stepping-stones plan -> governor -> walker: the single entry point.

All execution / evaluation / validation for the stepping reference-governor framework lives
here as subcommands (the previous one-off scripts are folded in):

  exec      Run one deploy config (or seed-dir) through governor + walker; write the full
            ``res`` (qpos/qvel/ctrl/base_rpy + summary + governed step-stats + executed
            footholds + governor ranking) and a replay GIF.
  eval      Planner-SSR vs execution-SSR across methods/seeds (termination reason, goal
            error, touchdown-timeout ratio, executed foothold safety, roll/pitch, progress).
  validate  Staged validation: flat-straight -> straight-stones -> 2GO governed -> baselines,
            each gated on the previous (offline gates always; sim when MuJoCo is available).

Requires native MuJoCo for the sim parts, so run inside the project's arm64 image::

    docker run --rm -i -v "$PWD":/work -w /work -e MUJOCO_GL=osmesa -e PYTHONPATH=/work \
      genedynamics/dev-cpu:torch bash -lc '
        python scripts/tasks/robot/quadruped/stepping_tones/run_stepping_execution.py \
          exec --config configs/quadruped/stepping_stones_2d/deploy/governed_exec.yaml'

    # other modes:
    #   ... run_stepping_execution.py exec --seed-dir <plan_seed_dir> --mode both
    #   ... run_stepping_execution.py eval --methods mbd ebmbd mdoc mdcoas twogo --mode both
    #   ... run_stepping_execution.py validate            (add --no-sim for offline gates only)

YAML schema for `exec` (only env_params.plan_seed_dir is required)::

    output_dir: results/.../deploy/governed/twogo
    exec_mode: both            # governed | raw | both
    env_params: {plan_seed_dir: ..., l_max: 0.35, stance_width: 0.30, start_mid: [...], goal_mid: [...]}
    walker:   {gait: walk, phase_steps: 36, base_kp_xy: 70.0, ...}   # any MinimalFollowerConfig field
    governor: {stride_length: 0.22, swing_step_limit: 0.35, ...}     # any GovernorConfig field
    render:   {enabled: true, width: 480, height: 360, max_frames: 120}
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from genedynamics.deploy.controllers.quadruped_stepping import (
    MinimalFollowerConfig, GovernorConfig, QuadrupedSteppingController, SteppingReferenceGovernor,
    diagnose_plan, ExecGates,
)
from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene, stepping_scene_to_dict
from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np


# ============================================================================ shared helpers
def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _filter_to_dataclass(params: Optional[Dict[str, Any]], dc: type) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only keys that are valid fields of ``dc``; return (kept, ignored_keys)."""
    valid = {f.name for f in dataclasses.fields(dc)}
    params = params or {}
    kept = {k: v for k, v in params.items() if k in valid}
    ignored = [k for k in params if k not in valid]
    if "walk_order" in kept and isinstance(kept["walk_order"], list):
        kept["walk_order"] = tuple(kept["walk_order"])
    kept.pop("rank_gates", None)
    return kept, ignored


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def load_plan(seed_dir: Path) -> Tuple[List[np.ndarray], int, Optional[List[float]]]:
    blob = json.load(open(seed_dir / "trajectory" / "trajectory.json"))
    cand = [np.asarray(c, dtype=np.float32) for c in blob["candidate_states"]]
    return cand, int(blob.get("best_idx", 0)), blob.get("candidate_costs")


def resolve_scene(seed_dir: Path, env: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer the REAL scene the planner used (saved by the experiment) over re-sampling,
    # which can differ (stone count / platform size) from the env's actual geometry.
    ex = seed_dir / "execution_results.json"
    if ex.exists():
        b = json.load(open(ex))
        if isinstance(b.get("stepping_scene"), dict):
            return b["stepping_scene"]
    res = json.load(open(seed_dir / "results.json")) if (seed_dir / "results.json").exists() else {}
    if isinstance(res.get("stepping_scene"), dict):
        return res["stepping_scene"]                      # the actual planner scene
    return stepping_scene_to_dict(sample_stepping_stones_scene(
        level=int(res.get("level", 1)), seed=int(res.get("seed", 0)),
        l_max=float(env.get("l_max", 0.35)), stance_width=float(env.get("stance_width", 0.30)),
        start_mid=tuple(env.get("start_mid", (-1.25, 0.0))),
        goal_mid=tuple(env.get("goal_mid", (1.25, 0.0))),
    ))


def save_res(out_dir: Path, name: str, res: Dict[str, Any], scene: Dict[str, Any]) -> Path:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for k in ("qpos", "qvel", "ctrl", "base_rpy"):
        if k in res and res[k] is not None:
            np.save(d / f"{k}.npy", np.asarray(res[k]))
    with open(d / "res.pkl", "wb") as f:
        pickle.dump(res, f)
    readable = {
        "termination_reason": res.get("termination_reason"),
        "terminated": res.get("terminated"),
        "summary": res.get("summary"),
        "governed": res.get("governed"),
        "governor_ranking": res.get("governor_ranking"),
        "frames": int(np.asarray(res.get("qpos", [])).shape[0]),
    }
    with open(d / "result.json", "w", encoding="utf-8") as f:
        json.dump(readable, f, indent=2, default=_json_default)
    with open(d / "stepping_scene.json", "w", encoding="utf-8") as f:
        json.dump(scene, f)
    return d


def render_gif(d: Path, res: Dict[str, Any], *, sim_dt: float, width: int, height: int, max_frames: int) -> Optional[str]:
    from genedynamics.deploy.observers.mujoco_render import render_episode_to_gif
    qp = np.asarray(res.get("qpos", []))
    qv = np.asarray(res.get("qvel", []))
    if qp.ndim != 2 or qp.shape[0] == 0:
        return None
    n = qp.shape[0]
    step = max(1, int(np.ceil(n / max(1, max_frames))))
    idx = np.arange(0, n, step)
    states = np.concatenate([qp[idx], qv[idx]], axis=1) if qv.shape[0] == n else qp[idx]
    gif = render_episode_to_gif(d, states=states, model="go2", width=width, height=height,
                                fps=1.0 / (sim_dt * step), draw_trajectory=True)
    return str(gif)


def _summ(res: Dict[str, Any]) -> str:
    s = res.get("summary", {}) or {}
    return ("term={} goal_err={:.3f} steps={}/{} timeout_ratio={:.2f} roll={:.3f} pitch={:.3f}".format(
        res.get("termination_reason"), s.get("goal_error_xy", float("nan")),
        s.get("steps_completed", s.get("interval_count", "?")), s.get("steps_planned", "?"),
        s.get("interval_timeout_ratio", 0.0), s.get("roll_abs_max", 0.0), s.get("pitch_abs_max", 0.0)))


# ============================================================================ exec
def cmd_exec(args: argparse.Namespace) -> None:
    if not args.config and not args.seed_dir:
        raise SystemExit("exec: provide --config <yaml> or --seed-dir <plan_seed_dir>")
    cfg_yaml = _load_yaml(args.config) if args.config else {}
    env = dict(cfg_yaml.get("env_params", {}) or {})
    if args.seed_dir:
        env["plan_seed_dir"] = args.seed_dir
    exec_mode = args.mode or str(cfg_yaml.get("exec_mode", "governed"))
    default_out = f"results/quadruped/stepping_stones_2d/deploy/run"
    out_dir = Path(args.output_dir or cfg_yaml.get("output_dir") or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # walker config: `walker:` (or legacy `method_params:`) + CLI overrides
    walker_params = dict(cfg_yaml.get("walker") or cfg_yaml.get("method_params") or {})
    for k in ("gait", "phase_steps", "step_width", "base_kp_xy", "base_kd_xy"):
        v = getattr(args, k, None)
        if v is not None:
            walker_params[k] = v
    wkw, wign = _filter_to_dataclass(walker_params, MinimalFollowerConfig)
    walker_cfg = MinimalFollowerConfig(**wkw)

    gov_params = dict(cfg_yaml.get("governor") or {})
    for k, v in (("gait", walker_cfg.gait), ("step_width", walker_cfg.step_width),
                 ("x_f_nominal", walker_cfg.x_f_nominal), ("x_r_nominal", walker_cfg.x_r_nominal),
                 ("y_L_nominal", walker_cfg.y_L_nominal), ("y_R_nominal", walker_cfg.y_R_nominal),
                 ("centerline_y", walker_cfg.centerline_y), ("min_foot_z", walker_cfg.min_foot_z),
                 ("swing_step_limit", float(env.get("l_max", 0.35)))):
        gov_params.setdefault(k, v)
    gkw, gign = _filter_to_dataclass(gov_params, GovernorConfig)
    governor_cfg = GovernorConfig(**gkw)

    rnd = cfg_yaml.get("render", {}) or {}
    render_enabled = (not args.no_render) and bool(rnd.get("enabled", True))
    rw = int(args.width or rnd.get("width", walker_params.get("render_width", 640)))
    rh = int(args.height or rnd.get("height", walker_params.get("render_height", 480)))
    rmaxf = int(args.max_frames or rnd.get("max_frames", walker_params.get("max_render_frames", 160)))

    seed_dir = Path(env["plan_seed_dir"])
    scene = resolve_scene(seed_dir, env)
    cand, best, costs = load_plan(seed_dir)

    print("plan_seed_dir :", seed_dir, "(", len(cand), "candidates, best_idx", best, ")")
    print("output_dir    :", out_dir, "| exec_mode:", exec_mode)
    if wign:
        print("  [note] walker keys ignored (not MinimalFollowerConfig fields):", wign)
    if gign:
        print("  [note] governor keys ignored:", gign)

    manifest: Dict[str, Any] = {"config": args.config, "seed_dir": str(seed_dir),
                                "exec_mode": exec_mode, "outputs": {}}

    def _do(name: str, res: Dict[str, Any]) -> None:
        print("[{}] {}".format(name.upper(), _summ(res)))
        d = save_res(out_dir, name, res, scene)
        gif = render_gif(d, res, sim_dt=walker_cfg.sim_dt, width=rw, height=rh, max_frames=rmaxf) if render_enabled else None
        if gif:
            print("  GIF  ->", gif)
        print("  data ->", str(d) + "/  (res.pkl, qpos/qvel/ctrl.npy, result.json)")
        if getattr(args, "figures", False) and name == "governed":
            make_tracking_plot(d, sim_dt=walker_cfg.sim_dt)
            make_gait_diagram(d, sim_dt=walker_cfg.sim_dt)
            # Render the strip in a FRESH subprocess: a second in-process MuJoCo GL context
            # (after the GIF renderer) can segfault osmesa. The subprocess has a single context.
            import subprocess, sys
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "viz",
                            "--res-dir", str(d), "--no-tracking", "--no-gait"], check=False)
        manifest["outputs"][name] = {"dir": str(d), "gif": gif,
                                     "termination_reason": res.get("termination_reason"),
                                     "summary": res.get("summary")}

    if exec_mode in ("governed", "both"):
        ctrl = QuadrupedSteppingController(cfg=walker_cfg, stepping_scene=scene)
        res = ctrl.rollout_governed(cand, planner_best_idx=best, candidate_costs=costs, governor_cfg=governor_cfg)
        _do("governed", res)
    if exec_mode in ("raw", "both"):
        ctrl = QuadrupedSteppingController(cfg=walker_cfg, stepping_scene=scene)
        res = ctrl.rollout(cand[best])
        _do("raw", res)

    with open(out_dir / "execution_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    print("Wrote", out_dir / "execution_manifest.json")


# ============================================================================ eval (planner-SSR vs exec-SSR)
def cmd_eval(args: argparse.Namespace) -> None:
    # Use the validated deployed gains so eval reflects the actual tuned controller
    # (bare defaults use base_kp_xy=44, which lags the body ~0.04 m further from goal).
    walker_cfg = MinimalFollowerConfig(gait=args.gait, phase_steps=int(args.phase_steps),
                                       base_kp_xy=float(args.base_kp_xy), base_kd_xy=float(args.base_kd_xy))
    cfg = ExecEvalConfig(
        methods=tuple(args.methods), results_root=args.results_root, level=int(args.level),
        seeds=tuple(int(s) for s in args.seeds), step_width=float(args.step_width),
        mode=args.mode, run_sim=not args.no_sim, out_dir=args.out_dir, walker_cfg=walker_cfg,
        goal_margin=float(args.goal_margin), exec_require_foothold=bool(args.require_foothold),
        exec_foothold_margin=float(args.foothold_margin),
    )
    out = run_eval(cfg)
    print(format_table(out))
    recs = out["records"]
    plan_ssr = sum(1 for r in recs if r.get("planner", {}).get("planner_success")) / max(len(recs), 1)

    def _ssr(key: str) -> Optional[float]:
        blocks = [r[key] for r in recs if isinstance(r.get(key), dict)]
        return (sum(1 for b in blocks if b.get("exec_success")) / len(blocks)) if blocks else None

    def _fmt(x: Optional[float]) -> str:
        return "n/a (sim unavailable)" if x is None else f"{x:.2f}"

    all_blocks = [r[k] for r in recs for k in ("governed_exec", "raw_exec") if isinstance(r.get(k), dict)]
    exec_ssr = (sum(1 for b in all_blocks if b.get("exec_success")) / len(all_blocks)) if all_blocks else None
    print(f"\nplanner-SSR={plan_ssr:.2f}  exec-SSR={_fmt(exec_ssr)}  "
          f"(governed={_fmt(_ssr('governed_exec'))} raw={_fmt(_ssr('raw_exec'))})  n={len(recs)}")
    print("Wrote", Path(cfg.out_dir) / "exec_eval.json")


# ============================================================================ validate (staged)
def straight_plan_16d(n: int = 17, x0: float = -1.25, x1: float = 1.25) -> np.ndarray:
    """Minimal 16D corridor-residual plan: straight body path, zero residual."""
    st = np.zeros((int(n), 16), dtype=np.float32)
    st[:, 0] = np.linspace(float(x0), float(x1), int(n))
    st[:, 15] = np.linspace(0.0, 1.0, int(n))
    return st


def two_row_stone_scene(x0: float = -1.1, x1: float = 1.1, pitch: float = 0.22,
                        y_rows=(0.15, -0.15), radius: float = 0.085) -> Dict[str, Any]:
    xs = np.arange(float(x0), float(x1) + 1e-6, float(pitch))
    centers = [[float(x), float(y)] for y in y_rows for x in xs]
    return {"level": 1, "difficulty": "synthetic", "map_x": [-1.5, 1.5], "map_y": [-0.6, 0.6],
            "river_x": [0.0, 0.0], "has_river": False, "stones_centers": centers,
            "stones_radii": [float(radius)] * len(centers), "support_platforms": []}


def _offline_gate(gait_ref, scene: Dict[str, Any], swing_limit: float, require_on_stone: bool):
    from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np
    centers = np.asarray(scene.get("stones_centers", []), dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(scene.get("stones_radii", []), dtype=np.float32).reshape(-1)
    plat = np.zeros((0, 4), dtype=np.float32)
    fh = 0.0
    if centers.shape[0] > 0:
        for p in gait_ref.phases:
            for lg in ("FL", "FR", "RL", "RR"):
                fh = max(fh, float(foot_stepping_violation_np(
                    np.asarray(p.foot_goal[lg][:2], dtype=np.float32), centers, radii, plat, 0.0)))
    swing_ok = gait_ref.max_swing_distance <= swing_limit + 1e-6
    onstone_ok = (not require_on_stone) or gait_ref.all_on_stone
    goal_ok = gait_ref.diagnostics.get("final_body_goal_error", 0.0) <= 0.05
    info = {"max_swing": gait_ref.max_swing_distance, "swing_ratio": gait_ref.diagnostics.get("swing_ratio", 0.0),
            "all_on_stone": gait_ref.all_on_stone, "planned_foothold_viol_max": fh,
            "final_body_goal_error": gait_ref.diagnostics.get("final_body_goal_error", 0.0),
            "feasible": gait_ref.diagnostics.get("feasible", True), "n_steps": gait_ref.n_steps}
    return bool(swing_ok and onstone_ok and goal_ok), info


def _maybe_sim(scene, walker_cfg, gov_cfg, cands, best, costs, run_sim: bool):
    if not run_sim:
        return None
    try:
        ctrl = QuadrupedSteppingController(cfg=walker_cfg, stepping_scene=scene)
        res = ctrl.rollout_governed(cands, planner_best_idx=best, candidate_costs=costs, governor_cfg=gov_cfg)
    except Exception as e:
        return {"exec_status": "sim_unavailable", "exec_error": str(e)}
    s = res.get("summary", {})
    gv = res.get("governed", {}) or {}
    return {"exec_status": "ok", "termination_reason": res.get("termination_reason"),
            "goal_error_xy": float(s.get("goal_error_xy", float("nan"))),
            "roll_abs_max": float(s.get("roll_abs_max", 0.0)), "pitch_abs_max": float(s.get("pitch_abs_max", 0.0)),
            "steps_completed": int(gv.get("steps_completed", 0)), "steps_planned": int(gv.get("n_steps", 0))}


def _run_stage(name, cands, best, costs, scene, walker_cfg, require_on_stone, run_sim):
    gov_cfg = GovernorConfig(gait=str(walker_cfg.gait), step_width=float(walker_cfg.step_width),
                             x_f_nominal=float(walker_cfg.x_f_nominal), x_r_nominal=float(walker_cfg.x_r_nominal),
                             y_L_nominal=float(walker_cfg.y_L_nominal), y_R_nominal=float(walker_cfg.y_R_nominal),
                             swing_step_limit=0.35)
    gait_ref, _ = SteppingReferenceGovernor(cfg=gov_cfg, stepping_scene=scene).govern(
        cands, planner_best_idx=best, candidate_costs=costs)
    passed, info = _offline_gate(gait_ref, scene, gov_cfg.swing_step_limit, require_on_stone)
    sim = _maybe_sim(scene, walker_cfg, gov_cfg, cands, best, costs, run_sim)
    rec = {"stage": name, "offline_pass": passed, "offline": info, "sim": sim}
    if sim and sim.get("exec_status") == "ok":
        fail = {"roll_pitch_abort", "support_contact_abort", "consecutive_touchdown_timeouts", "touchdown_timeout_abort"}
        rec["sim_pass"] = bool(sim.get("termination_reason") not in fail
                               and sim.get("steps_completed", 0) >= sim.get("steps_planned", 1)
                               and np.isfinite(sim.get("goal_error_xy", np.nan))
                               and sim.get("goal_error_xy", 1e9) <= 0.18
                               and max(sim.get("roll_abs_max", 0.0), sim.get("pitch_abs_max", 0.0)) <= 0.45)
    return rec


def cmd_validate(args: argparse.Namespace) -> None:
    run_sim = not args.no_sim
    wcfg = MinimalFollowerConfig(gait=args.gait, step_width=float(args.step_width), phase_steps=int(args.phase_steps))
    records: List[Dict[str, Any]] = []

    flat = {"level": 1, "difficulty": "flat", "map_x": [-2, 2], "map_y": [-1, 1], "river_x": [0, 0],
            "has_river": False, "stones_centers": [], "stones_radii": [], "support_platforms": []}
    records.append(_run_stage("1_flat_straight", [straight_plan_16d()], 0, None, flat, wcfg, False, run_sim))
    if records[-1]["offline_pass"]:
        records.append(_run_stage("2_straight_stones", [straight_plan_16d()], 0, None,
                                  two_row_stone_scene(), wcfg, True, run_sim))
    if records[-1]["offline_pass"]:
        for tag, method, on_stone in [("3_twogo_governed", "twogo", True)] + \
                [(f"4_baseline_{m}", m, True) for m in args.methods]:
            seed_dir = Path(args.results_root) / method / f"level_{args.level}" / f"seed_{args.seed}"
            scene = stepping_scene_to_dict(sample_stepping_stones_scene(
                level=args.level, seed=args.seed, l_max=0.35, stance_width=float(args.step_width),
                start_mid=(-1.25, 0.0), goal_mid=(1.25, 0.0)))
            try:
                cand, best, costs = load_plan(seed_dir)
                records.append(_run_stage(tag, cand, best, costs, scene, wcfg, on_stone, run_sim))
            except Exception as e:
                records.append({"stage": tag, "offline_pass": False, "error": str(e)})

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "staged_validation.json", "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2, default=_json_default)
    print(f"{'stage':22} {'offline':>8} {'swingR':>7} {'onStone':>7} {'goalErr':>7} {'sim':>16}")
    print("-" * 72)
    for r in records:
        if "error" in r:
            print(f"{r['stage']:22} {'ERROR':>8}  {r['error']}"); continue
        o = r.get("offline", {}); sim = r.get("sim")
        sim_s = ("skipped" if sim is None else (sim.get("exec_status") if sim.get("exec_status") != "ok"
                 else ("PASS" if r.get("sim_pass") else "FAIL") + f"/{sim.get('termination_reason')}"))
        print(f"{r['stage']:22} {('PASS' if r['offline_pass'] else 'FAIL'):>8} "
              f"{o.get('swing_ratio', 0.0):>6.2f}x {str(o.get('all_on_stone', '')):>7} "
              f"{o.get('final_body_goal_error', 0.0):>7.3f} {sim_s:>16}")
    all_ok = all(r.get("offline_pass") for r in records if "error" not in r)
    print(f"\nALL OFFLINE GATES: {'PASS' if all_ok else 'FAIL'}  ({len(records)} stages)  -> {out_dir/'staged_validation.json'}")


# ============================================================================ viz (paper figures)
LEG_ORDER_VIZ = ("FL", "FR", "RL", "RR")


def _movavg(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1 or x.size == 0:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(np.asarray(x, dtype=np.float64), k, mode="same")


def _tracking_signals(res: Dict[str, Any], sim_dt: float):
    """Reconstruct executed vs reference forward position/velocity over the FULL rollout.
    The per-substep base_xy_ref lives in ``debug`` (gait phase only); the goal-hold tail is
    extended at the final reference so the settle is shown."""
    dbg = res.get("debug") or []
    qp = np.asarray(res["qpos"]); qv = np.asarray(res["qvel"])
    T = qp.shape[0]
    n = len(dbg)
    x_ref = np.full(T, np.nan)
    if n:
        x_ref[:n] = np.array([e["base_xy_ref"][0] for e in dbg])
        x_ref[n:] = x_ref[n - 1] if n < T else x_ref[-1]
    t = np.arange(T) * sim_dt
    x_act = qp[:T, 0]
    vx_act = qv[:T, 0]
    n_steps = int((res.get("governed", {}) or {}).get("n_steps", 48) or 48)
    win = max(20, int(4 * n / max(1, n_steps)))        # ~ one gait cycle (4 single-leg steps)
    vx_des = _movavg(np.gradient(np.nan_to_num(x_ref, nan=x_ref[np.isfinite(x_ref)][-1] if np.any(np.isfinite(x_ref)) else 0.0), sim_dt), win)
    vx_act_s = _movavg(vx_act, win)
    return t, x_ref, x_act, vx_act, vx_act_s, vx_des, n, win


def _paper_style() -> None:
    """Match the corridor trajectory_modes.png typography (large labels/ticks, no title)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 26, "axes.titlesize": 24,
        "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22,
        "axes.linewidth": 1.1,
    })


def _add_foot_markers(mujoco, scn, res, scene_dict, *, off_tol: float = 0.05) -> None:
    """Overlay, as translucent 3D geoms: each foot's SWING trajectory (thin green shadow, from
    ``swing_ref``; NaN = stance) and a translucent red blob at any landing that misses every
    support — stones AND the start/goal platforms — matching trajectory_best.png's off-support
    convention (feet on a platform are valid, not flagged)."""
    centers = np.asarray(scene_dict.get("stones_centers", []), dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(scene_dict.get("stones_radii", []), dtype=np.float32).reshape(-1)
    plat = np.asarray(scene_dict.get("support_platforms", []), dtype=np.float32).reshape(-1, 4)
    eye = np.eye(3, dtype=np.float64).flatten()
    GREEN = np.array([0.15, 0.85, 0.30, 0.35], dtype=np.float32)   # translucent
    RED = np.array([0.95, 0.12, 0.12, 0.42], dtype=np.float32)     # translucent

    swing_ref = res.get("swing_ref", {}) or {}
    for leg in LEG_ORDER_VIZ:                            # thin green swing-arc shadows
        arr = np.asarray(swing_ref.get(leg, []), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        valid = np.isfinite(arr).all(axis=1)
        for i in range(0, arr.shape[0] - 2, 2):         # subsample x2 to bound geom count
            if scn.ngeom >= scn.maxgeom - 4:
                break
            j = i + 2
            if not (valid[i] and valid[j]) or np.linalg.norm(arr[j] - arr[i]) > 0.4:
                continue
            try:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.0045, arr[i], arr[j])
                g.rgba[:] = GREEN
                scn.ngeom += 1
            except Exception:
                pass

    for e in (res.get("governed", {}) or {}).get("executed_footholds", []) or []:
        if scn.ngeom >= scn.maxgeom:
            break
        x, y = float(e["landed_xy"][0]), float(e["landed_xy"][1])
        viol = float(foot_stepping_violation_np(np.array([x, y], np.float32), centers, radii, plat, 0.0))
        if viol > off_tol:                               # off every support -> small red dot AT the foot
            zf = float(e.get("landed_z", 0.03)) if isinstance(e, dict) else 0.03
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                np.array([0.032, 0.032, 0.032]),
                                np.array([x, y, float(np.clip(zf, 0.015, 0.05))]), eye, RED)
            scn.ngeom += 1


def make_tracking_plot(res_dir: Path, *, sim_dt: float = 0.01, out: Optional[Path] = None) -> Optional[Path]:
    """Execution-tracks-reference figure.

    Top: forward position s, executed vs the governed reference path (the meaningful tracking
    signal — the base PD tracks POSITION).  Bottom: forward speed — raw actual shows the
    per-step crawl pulses inherent to a one-leg-at-a-time static gait; the cycle-mean (bold)
    tracks the planned average pace (the per-substep reference *velocity* is a stop-go
    smoothstep artifact, so we compare cycle-means, not the raw derivative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_style()
    res = pickle.load(open(res_dir / "res.pkl", "rb"))
    if not (res.get("debug") or []):
        print("  [tracking] no debug stream in res.pkl"); return None
    t, x_ref, x_act, vx_act, vx_act_s, vx_des, n, win = _tracking_signals(res, sim_dt)

    fig, ax = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, constrained_layout=True)
    ax[0].plot(t, x_ref, color="0.35", ls="--", lw=2.2, label="reference")
    ax[0].plot(t, x_act, color="#1f4fd8", lw=2.4, label="executed")
    ax[0].set_ylabel("s (m)"); ax[0].grid(alpha=0.25)
    ax[0].legend(loc="upper left", framealpha=0.9)
    ax[1].plot(t, vx_act, color="#1f4fd8", lw=0.8, alpha=0.30)
    ax[1].plot(t, vx_act_s, color="#1f4fd8", lw=2.4, label="actual")
    ax[1].plot(t, vx_des, color="#e23b3b", ls="--", lw=2.0, label="planned")
    ax[1].axhline(0.0, color="0.8", lw=0.8)
    ax[1].set_ylabel("v (m/s)"); ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper left", framealpha=0.9)
    out = out or (res_dir / "tracking.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("  [tracking] wrote", out)
    return out


def make_gait_diagram(res_dir: Path, *, sim_dt: float = 0.01, out: Optional[Path] = None) -> Optional[Path]:
    """Contact-phase (gait) diagram + forward speed: per-leg stance bars over time with the
    cycle-mean speed on top (the "annotate speed / contact phase" view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_style()
    res = pickle.load(open(res_dir / "res.pkl", "rb"))
    contacts = res.get("contact_legs") or []
    qv = np.asarray(res["qvel"])
    T = min(len(contacts), qv.shape[0])
    if T == 0:
        print("  [gait] no contact stream"); return None
    t = np.arange(T) * sim_dt
    stance = {lg: np.array([lg in (contacts[i] or []) for i in range(T)]) for lg in LEG_ORDER_VIZ}
    n_steps = int((res.get("governed", {}) or {}).get("n_steps", 48) or 48)
    win = max(20, int(4 * len(res.get("debug") or []) / max(1, n_steps)))
    vx_s = _movavg(qv[:T, 0], win)

    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.8), sharex=True, constrained_layout=True,
                           gridspec_kw={"height_ratios": [1.0, 1.4]})
    ax[0].plot(t, vx_s, color="#1f4fd8", lw=2.4)
    ax[0].set_ylabel("v (m/s)"); ax[0].grid(alpha=0.25); ax[0].axhline(0, color="0.85", lw=0.8)
    colors = {"FL": "#d1495b", "FR": "#edae49", "RL": "#00798c", "RR": "#30638e"}
    for row, lg in enumerate(LEG_ORDER_VIZ):
        st = stance[lg]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], st.astype(int), [0]])))
        for a, b in zip(edges[0::2], edges[1::2]):
            ax[1].barh(row, (b - a) * sim_dt, left=a * sim_dt, height=0.62,
                       color=colors[lg], edgecolor="none")
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(LEG_ORDER_VIZ)
    ax[1].set_ylim(-0.6, 3.6); ax[1].invert_yaxis()
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("stance")
    ax[1].grid(alpha=0.2, axis="x")
    out = out or (res_dir / "gait_diagram.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("  [gait] wrote", out)
    return out


def make_motion_strip(res_dir: Path, *, n_poses: int = 6, width: int = 1280, height: int = 360,
                      azimuth: float = 90.0, elevation: float = -18.0, distance: float = 1.45,
                      orthographic: bool = False, overlay_footholds: bool = True,
                      out: Optional[Path] = None) -> Optional[Path]:
    """Ghosted multi-pose strip — the walk shown step-by-step, earlier poses faded -> final
    pose solid, via per-pixel background-subtraction compositing.  Front side-elevation camera
    (azimuth 90), raised + close to just frame the traverse.  ``overlay_footholds`` draws the
    executed footstep trajectory in green and circles off-stone landings in red."""
    import mujoco
    import imageio
    from genedynamics.envs.utils.mujoco_model_generator import create_go2_render_xml_with_trajectory
    from genedynamics.robots.registry import _get_go2_path

    qp = np.load(res_dir / "qpos.npy")
    scene = json.load(open(res_dir / "stepping_scene.json"))
    res = pickle.load(open(res_dir / "res.pkl", "rb")) if (res_dir / "res.pkl").exists() else {}
    do_overlay = bool(overlay_footholds and res)
    model_dir = Path(_get_go2_path()).parent           # tmp must sit beside go2.xml so <include> resolves
    tmp = str(model_dir / "_strip_temp.xml")
    create_go2_render_xml_with_trajectory(tmp, trajectory_positions=[], stepping_scene=scene, swing_trajectories=None)
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
        if orthographic:
            try:
                m.vis.global_.orthographic = 1
            except Exception:
                pass
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x0, x1 = float(qp[:, 0].min()), float(qp[:, 0].max())
        cam.lookat[:] = [0.5 * (x0 + x1), 0.0, 0.12]    # centre the traverse
        cam.azimuth = float(azimuth); cam.elevation = float(elevation); cam.distance = float(distance)
        idx = np.linspace(0, qp.shape[0] - 1, int(n_poses)).astype(int)
        frames = []
        for i in idx:
            d.qpos[:] = qp[i, : m.nq]; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            frames.append(r.render().astype(np.float32))
        frames = np.stack(frames, 0)
        bg = np.median(frames, axis=0)                 # terrain (robot is transient)
        result = bg.copy()
        ramp = np.linspace(0.40, 1.0, int(n_poses))    # faint early ghosts -> solid final pose
        for i in range(int(n_poses)):
            mask = np.abs(frames[i] - bg).max(axis=2) > 16.0
            w = float(ramp[i])
            result[mask] = (1.0 - w) * result[mask] + w * frames[i][mask]

        if do_overlay:                                 # overlay swing arcs + off-support reds
            d.qpos[:] = qp[idx[-1], : m.nq]; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            _add_foot_markers(mujoco, r.scene, res, scene)
            marked = r.render().astype(np.float32)
            mmask = np.abs(marked - frames[-1]).max(axis=2) > 10.0
            result[mmask] = marked[mmask]

        out = out or (res_dir / "motion_strip.png")
        imageio.imwrite(str(out), np.clip(result, 0, 255).astype(np.uint8))
        print("  [strip] wrote", out, f"({n_poses} poses, az={azimuth} el={elevation} dist={distance})")
        return out
    finally:
        Path(tmp).unlink(missing_ok=True)


def make_all_figures(res_dir: Path, *, sim_dt: float = 0.01, poses: int = 6,
                     width: int = 1280, height: int = 360, azimuth: float = 90.0,
                     elevation: float = -18.0, distance: float = 1.45, orthographic: bool = False,
                     tracking: bool = True, strip: bool = True, gait: bool = True) -> None:
    if tracking:
        try:
            make_tracking_plot(res_dir, sim_dt=sim_dt)
        except Exception as e:
            print("  [tracking] failed:", e)
    if gait:
        try:
            make_gait_diagram(res_dir, sim_dt=sim_dt)
        except Exception as e:
            print("  [gait] failed:", e)
    if strip:
        try:
            make_motion_strip(res_dir, n_poses=poses, width=width, height=height,
                              azimuth=azimuth, elevation=elevation, distance=distance, orthographic=orthographic)
        except Exception as e:
            print("  [strip] skipped (needs native MuJoCo):", e)


def cmd_viz(args: argparse.Namespace) -> None:
    res_dir = Path(args.res_dir)
    if not (res_dir / "res.pkl").exists():
        raise SystemExit(f"viz: {res_dir}/res.pkl not found (point --res-dir at an exec governed/ or raw/ dir)")
    make_all_figures(res_dir, sim_dt=float(args.sim_dt), poses=int(args.poses),
                     width=int(args.width), height=int(args.height), azimuth=float(args.azimuth),
                     elevation=float(args.elevation), distance=float(args.distance),
                     orthographic=bool(args.ortho),
                     tracking=not args.no_tracking, strip=not args.no_strip, gait=not args.no_gait)


# ============================================================================ eval engine (Rec. 4)
@dataclass
class ExecEvalConfig:
    methods: Sequence[str] = ("mbd", "ebmbd", "mdoc", "mdcoas", "twogo")
    results_root: str = "results/quadruped/stepping_stones_2d/smoke"
    level: int = 1
    seeds: Sequence[int] = (0,)
    step_width: float = 0.30
    l_max: float = 0.35
    mode: str = "governed"          # governed | raw | both
    run_sim: bool = True
    # exec-SSR success = "walked the course": completed all steps, no fall/abort, reached near
    # goal, stayed upright.  goal_margin reflects the kinematic limit (the last stone row caps
    # how far forward the body can go).  Foothold safety is REPORTED always; it only GATES
    # success when exec_require_foothold is set (e.g. river scenes where a missed foot falls) and
    # then uses CVaR95 (robust to a few off-stone landings) rather than the brittle max.
    goal_margin: float = 0.18
    exec_require_foothold: bool = False
    exec_foothold_margin: float = 0.08    # ~ stone radius; only applied when exec_require_foothold
    roll_pitch_limit: float = 0.45
    out_dir: str = "results/quadruped/stepping_stones_exec_eval"
    walker_cfg: MinimalFollowerConfig = field(default_factory=lambda: MinimalFollowerConfig(gait="walk"))
    governor_cfg: Optional[GovernorConfig] = None
    gates: ExecGates = field(default_factory=ExecGates)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.load(open(path, "r", encoding="utf-8")) or {}
    except Exception:
        return {}


def _eval_seed_dir(cfg: ExecEvalConfig, method: str, seed: int) -> Path:
    return Path(cfg.results_root) / method / f"level_{cfg.level}" / f"seed_{seed}"


def _eval_load_candidates(seed_dir: Path) -> Tuple[List[np.ndarray], int, Optional[List[float]]]:
    blob = _load_json(seed_dir / "trajectory" / "trajectory.json")
    cand = blob.get("candidate_states") or []
    if not cand:
        raise ValueError(f"no candidate_states in {seed_dir}")
    best = max(0, min(int(blob.get("best_idx", 0)), len(cand) - 1))
    return [np.asarray(c, dtype=np.float32) for c in cand], best, blob.get("candidate_costs")


def _eval_resolve_scene(seed_dir: Path, cfg: ExecEvalConfig, seed: int) -> Dict[str, Any]:
    ex = _load_json(seed_dir / "execution_results.json")
    if isinstance(ex.get("stepping_scene"), dict):
        return ex["stepping_scene"]
    res = _load_json(seed_dir / "results.json")
    return stepping_scene_to_dict(sample_stepping_stones_scene(
        level=int(res.get("level", cfg.level)), seed=int(res.get("seed", seed)),
        l_max=float(cfg.l_max), stance_width=float(cfg.step_width),
        start_mid=(-1.25, 0.0), goal_mid=(1.25, 0.0)))


def _planner_metrics(seed_dir: Path) -> Dict[str, Any]:
    res = _load_json(seed_dir / "results.json")
    sm = (res.get("metrics") or {}).get("stepping_metrics") or res.get("stepping_metrics") or {}
    keys = ["success", "final_goal_error", "foothold_violation_mean", "foothold_violation_cvar95",
            "step_violation_cvar95", "action_step_violation_cvar95", "stance_drift_mean",
            "stance_drift_max", "planning_time"]
    out = {k: sm.get(k) for k in keys}
    out["planner_success"] = bool(sm.get("success", False))
    return out


def _executed_foothold_safety(scene: Dict[str, Any], executed: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    centers = np.asarray(scene.get("stones_centers", []), dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(scene.get("stones_radii", []), dtype=np.float32).reshape(-1)
    plat = np.zeros((0, 4), dtype=np.float32)
    viols = [float(foot_stepping_violation_np(np.asarray(e.get("landed_xy", [0.0, 0.0]), dtype=np.float32),
                                              centers, radii, plat, 0.0)) for e in executed]
    if not viols:
        return {"max": 0.0, "mean": 0.0, "cvar95": 0.0}
    arr = np.asarray(viols, dtype=np.float32)
    k = max(1, int(np.ceil(0.05 * arr.size)))
    return {"max": float(np.max(arr)), "mean": float(np.mean(arr)),
            "cvar95": float(np.mean(np.partition(arr, arr.size - k)[arr.size - k:]))}


def _governed_foothold_safety(scene: Dict[str, Any], gait_ref: Any) -> Dict[str, float]:
    centers = np.asarray(scene.get("stones_centers", []), dtype=np.float32).reshape(-1, 2)
    radii = np.asarray(scene.get("stones_radii", []), dtype=np.float32).reshape(-1)
    plat = np.zeros((0, 4), dtype=np.float32)
    viols = [float(foot_stepping_violation_np(np.asarray(p.foot_goal[lg][:2], dtype=np.float32),
                                              centers, radii, plat, 0.0))
             for p in gait_ref.phases for lg in ("FL", "FR", "RL", "RR")]
    arr = np.asarray(viols, dtype=np.float32) if viols else np.zeros((0,), dtype=np.float32)
    return {"max": float(np.max(arr)) if arr.size else 0.0, "mean": float(np.mean(arr)) if arr.size else 0.0}


def _summarize_exec(res: Dict[str, Any], scene: Dict[str, Any], cfg: ExecEvalConfig) -> Dict[str, Any]:
    s = res.get("summary", {}) or {}
    term = res.get("termination_reason")
    governed = res.get("governed", {}) or {}
    executed = governed.get("executed_footholds", []) or res.get("executed_footholds", []) or []
    foothold_checked = len(executed) > 0
    fh = _executed_foothold_safety(scene, executed) if executed else {"max": float("nan"), "mean": float("nan"), "cvar95": float("nan")}
    steps_done = int(governed.get("steps_completed", s.get("interval_count", 0)) or 0)
    steps_plan = int(governed.get("n_steps", steps_done) or steps_done)
    failure_terms = {"roll_pitch_abort", "support_contact_abort",
                     "consecutive_touchdown_timeouts", "touchdown_timeout_abort"}
    goal_err = float(s.get("goal_error_xy", float("nan")))
    roll = float(s.get("roll_abs_max", 0.0)); pitch = float(s.get("pitch_abs_max", 0.0))
    fh_cvar = fh.get("cvar95", float("nan"))
    foothold_ok = (not cfg.exec_require_foothold) or (not np.isfinite(fh_cvar)) or (fh_cvar <= cfg.exec_foothold_margin)
    exec_success = bool(
        (term is None or term not in failure_terms)
        and (steps_plan == 0 or steps_done >= steps_plan)
        and np.isfinite(goal_err) and goal_err <= cfg.goal_margin
        and foothold_ok
        and max(roll, pitch) <= cfg.roll_pitch_limit)
    return {"exec_success": exec_success, "exec_foothold_checked": bool(foothold_checked),
            "termination_reason": term, "goal_error_xy": goal_err,
            "base_progress_xy": float(s.get("base_progress_xy", 0.0)),
            "interval_timeout_ratio": float(s.get("interval_timeout_ratio", 0.0)),
            "touchdown_any_ratio": float(s.get("touchdown_any_ratio", 0.0)),
            "roll_abs_max": roll, "pitch_abs_max": pitch,
            "steps_completed": steps_done, "steps_planned": steps_plan,
            "executed_foothold_viol_max": fh["max"], "executed_foothold_viol_cvar95": fh.get("cvar95", float("nan")),
            "frames": int(np.asarray(res.get("qpos", [])).shape[0])}


def evaluate_one(method: str, seed: int, cfg: ExecEvalConfig) -> Dict[str, Any]:
    seed_dir = _eval_seed_dir(cfg, method, seed)
    rec: Dict[str, Any] = {"method": method, "seed": seed, "level": cfg.level, "seed_dir": str(seed_dir)}
    try:
        cands, best, costs = _eval_load_candidates(seed_dir)
    except Exception as e:
        rec["error"] = f"load_candidates: {e}"
        return rec
    scene = _eval_resolve_scene(seed_dir, cfg, seed)
    rec["planner"] = _planner_metrics(seed_dir)

    gov_cfg = cfg.governor_cfg or GovernorConfig(
        gait=str(cfg.walker_cfg.gait), step_width=float(cfg.step_width),
        x_f_nominal=float(cfg.walker_cfg.x_f_nominal), x_r_nominal=float(cfg.walker_cfg.x_r_nominal),
        y_L_nominal=float(cfg.walker_cfg.y_L_nominal), y_R_nominal=float(cfg.walker_cfg.y_R_nominal),
        swing_step_limit=float(cfg.l_max), rank_gates=cfg.gates)
    raw_diag = diagnose_plan(
        cands[best], gates=cfg.gates, step_width=gov_cfg.step_width, centerline_y=gov_cfg.centerline_y,
        x_f_nominal=gov_cfg.x_f_nominal, x_r_nominal=gov_cfg.x_r_nominal,
        y_L_nominal=gov_cfg.y_L_nominal, y_R_nominal=gov_cfg.y_R_nominal, half_pair_length=gov_cfg.half_pair_length)
    rec["raw_exec_diag"] = {"n_intervals": raw_diag.n_intervals,
                            "mode_interval_swing_max": raw_diag.mode_interval_swing_max,
                            "mode_interval_swing_ratio": raw_diag.mode_interval_swing_ratio,
                            "stance_drift_max": raw_diag.stance_drift_max,
                            "walker_compatible": raw_diag.walker_compatible, "reasons": raw_diag.reasons}

    gov = SteppingReferenceGovernor(cfg=gov_cfg, stepping_scene=scene)
    gait_ref, ranking = gov.govern(cands, planner_best_idx=best, candidate_costs=costs)
    fh = _governed_foothold_safety(scene, gait_ref)
    rec["governed_diag"] = {"source_candidate_idx": gait_ref.source_candidate_idx, "n_steps": gait_ref.n_steps,
                            "max_swing_distance": gait_ref.max_swing_distance,
                            "swing_ratio": gait_ref.diagnostics.get("swing_ratio", 0.0),
                            "all_on_stone": gait_ref.all_on_stone,
                            "feasible": gait_ref.diagnostics.get("feasible", True),
                            "n_stalled_steps": gait_ref.diagnostics.get("n_stalled_steps", 0),
                            "goal_in_support_ratio": gait_ref.diagnostics.get("goal_in_support_ratio", 0.0),
                            "planned_foothold_viol_max": fh["max"],
                            "final_body_goal_error": gait_ref.diagnostics.get("final_body_goal_error", 0.0)}

    if cfg.run_sim:
        try:
            ctrl = QuadrupedSteppingController(cfg=cfg.walker_cfg, stepping_scene=scene)
        except Exception as e:
            rec["exec_status"] = "sim_unavailable"; rec["exec_error"] = str(e)
            return rec
        if cfg.mode in ("governed", "both"):
            try:
                gres = ctrl.rollout_governed(cands, planner_best_idx=best, candidate_costs=costs, governor_cfg=gov_cfg)
                rec["governed_exec"] = _summarize_exec(gres, scene, cfg)
            except Exception as e:
                rec["governed_exec"] = {"exec_status": "error", "exec_error": str(e)}
        if cfg.mode in ("raw", "both"):
            try:
                ctrl.reset(); rres = ctrl.rollout(cands[best])
                rec["raw_exec"] = _summarize_exec(rres, scene, cfg)
            except Exception as e:
                rec["raw_exec"] = {"exec_status": "error", "exec_error": str(e)}
        rec["exec_status"] = "ok"
    else:
        rec["exec_status"] = "skipped"
    return rec


def _eval_config_to_dict(cfg: ExecEvalConfig) -> Dict[str, Any]:
    d = {k: v for k, v in cfg.__dict__.items() if k not in ("walker_cfg", "governor_cfg", "gates")}
    d["walker_cfg"] = asdict(cfg.walker_cfg)
    d["gates"] = asdict(cfg.gates)
    return d


def run_eval(cfg: ExecEvalConfig) -> Dict[str, Any]:
    records = [evaluate_one(m, s, cfg) for m in cfg.methods for s in cfg.seeds]
    out = {"config": _eval_config_to_dict(cfg), "records": records}
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "exec_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    return out


def format_table(out: Dict[str, Any]) -> str:
    rows = out.get("records", [])
    hdr = (f"{'method':8} {'seed':>4} | {'plan':>4} {'pGoal':>6} | "
           f"{'rawSwing':>9} {'rawOK':>5} | {'govSwing':>8} {'onStn':>5} {'goalErr':>7} | "
           f"{'EXEC':>5} {'term':>26} {'timeout':>7} {'rp_max':>6} {'fhViol':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['method']:8} {r['seed']:>4} | ERROR: {r['error']}"); continue
        p = r.get("planner", {}); rd = r.get("raw_exec_diag", {}); gd = r.get("governed_diag", {})
        ge = r.get("governed_exec", {}) or r.get("raw_exec", {}) or {}
        pgoal = p.get("final_goal_error")
        pgoal_s = f"{pgoal:.3f}" if isinstance(pgoal, (int, float)) else " n/a "
        exec_s = ge.get("exec_success")
        exec_str = "PASS" if exec_s is True else ("FAIL" if exec_s is False else (r.get("exec_status", "?")))
        term = str(ge.get("termination_reason")) if ge else r.get("exec_status", "")
        timeout = ge.get("interval_timeout_ratio")
        rp = max(float(ge.get("roll_abs_max", 0.0) or 0.0), float(ge.get("pitch_abs_max", 0.0) or 0.0)) if ge else 0.0
        fh = ge.get("executed_foothold_viol_max")
        lines.append(
            f"{r['method']:8} {r['seed']:>4} | "
            f"{('Y' if p.get('planner_success') else 'N'):>4} {pgoal_s:>6} | "
            f"{rd.get('mode_interval_swing_ratio', 0.0):>7.1f}x {str(rd.get('walker_compatible', '')):>5} | "
            f"{gd.get('swing_ratio', 0.0):>7.2f}x {str(gd.get('all_on_stone', '')):>5} "
            f"{gd.get('final_body_goal_error', 0.0):>7.3f} | "
            f"{exec_str:>5} {term[:26]:>26} "
            f"{(f'{timeout:.2f}' if isinstance(timeout,(int,float)) else ' n/a '):>7} "
            f"{rp:>6.3f} {(f'{fh:.3f}' if isinstance(fh,(int,float)) and np.isfinite(fh) else ' n/a '):>7}")
    return "\n".join(lines)


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Stepping-stones plan -> governor -> walker (exec | eval | validate)")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("exec", help="run one config/seed-dir through governor+walker; full res + GIF")
    pe.add_argument("--config", default=None, help="deploy YAML")
    pe.add_argument("--seed-dir", default=None, help="plan seed dir (instead of/over --config env_params)")
    pe.add_argument("--output-dir", default=None)
    pe.add_argument("--mode", default=None, choices=["governed", "raw", "both"])
    pe.add_argument("--gait", default=None, choices=["walk", "trot"])
    pe.add_argument("--phase-steps", dest="phase_steps", type=int, default=None)
    pe.add_argument("--step-width", dest="step_width", type=float, default=None)
    pe.add_argument("--base-kp-xy", dest="base_kp_xy", type=float, default=None)
    pe.add_argument("--base-kd-xy", dest="base_kd_xy", type=float, default=None)
    pe.add_argument("--width", type=int, default=None)
    pe.add_argument("--height", type=int, default=None)
    pe.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    pe.add_argument("--no-render", action="store_true")
    pe.add_argument("--figures", action="store_true", help="also emit tracking + gait + motion-strip figures for the governed run")
    pe.set_defaults(func=cmd_exec)

    pv = sub.add_parser("eval", help="planner-SSR vs exec-SSR across methods/seeds")
    pv.add_argument("--methods", nargs="+", default=["mbd", "ebmbd", "mdoc", "mdcoas", "twogo"])
    pv.add_argument("--results-root", default="results/quadruped/stepping_stones_2d/smoke")
    pv.add_argument("--level", type=int, default=1)
    pv.add_argument("--seeds", nargs="+", default=["0"])
    pv.add_argument("--mode", default="governed", choices=["governed", "raw", "both"])
    pv.add_argument("--gait", default="walk", choices=["walk", "trot"])
    pv.add_argument("--phase-steps", dest="phase_steps", type=int, default=36)
    pv.add_argument("--step-width", dest="step_width", type=float, default=0.30)
    pv.add_argument("--base-kp-xy", dest="base_kp_xy", type=float, default=70.0)
    pv.add_argument("--base-kd-xy", dest="base_kd_xy", type=float, default=16.0)
    pv.add_argument("--goal-margin", dest="goal_margin", type=float, default=0.18,
                    help="exec-SSR goal tolerance; last stone row at x~1.07 vs goal 1.25 => floor ~0.15-0.18")
    pv.add_argument("--require-foothold", dest="require_foothold", action="store_true",
                    help="also gate exec-SSR on executed-foothold CVaR95 (use for river scenes)")
    pv.add_argument("--foothold-margin", dest="foothold_margin", type=float, default=0.08)
    pv.add_argument("--no-sim", action="store_true")
    pv.add_argument("--out-dir", default="results/quadruped/stepping_stones_exec_eval")
    pv.set_defaults(func=cmd_eval)

    ps = sub.add_parser("validate", help="staged plan->governor->walker validation")
    ps.add_argument("--results-root", default="results/quadruped/stepping_stones_2d/smoke")
    ps.add_argument("--level", type=int, default=1)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--methods", nargs="+", default=["mbd", "ebmbd", "mdoc", "mdcoas", "twogo"])
    ps.add_argument("--gait", default="walk", choices=["walk", "trot"])
    ps.add_argument("--step-width", dest="step_width", type=float, default=0.30)
    ps.add_argument("--phase-steps", dest="phase_steps", type=int, default=36)
    ps.add_argument("--no-sim", action="store_true")
    ps.add_argument("--out-dir", default="results/quadruped/stepping_stones_staged")
    ps.set_defaults(func=cmd_validate)

    pf = sub.add_parser("viz", help="paper figures from an exec result dir: tracking plot + ghost motion strip")
    pf.add_argument("--res-dir", required=True, help="an exec output subdir (…/governed or …/raw) holding res.pkl + qpos.npy")
    pf.add_argument("--poses", type=int, default=6, help="number of ghost poses in the motion strip")
    pf.add_argument("--sim-dt", dest="sim_dt", type=float, default=0.01)
    pf.add_argument("--width", type=int, default=1280)
    pf.add_argument("--height", type=int, default=360)
    pf.add_argument("--azimuth", type=float, default=90.0, help="strip camera azimuth (90 = front side-elevation)")
    pf.add_argument("--elevation", type=float, default=-18.0)
    pf.add_argument("--distance", type=float, default=1.45)
    pf.add_argument("--ortho", action="store_true", help="orthographic strip camera (default: perspective side-elevation)")
    pf.add_argument("--no-tracking", action="store_true")
    pf.add_argument("--no-strip", action="store_true")
    pf.add_argument("--no-gait", action="store_true")
    pf.set_defaults(func=cmd_viz)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
