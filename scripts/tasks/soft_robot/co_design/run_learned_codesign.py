"""Permanent launcher for the learned-controller co-design run (full M3BD stack:
SHAC warmup + learned closed-loop controller + in-loop SHAC + td-lambda critic +
risk-sensitive rho_H + budgeted adaptive multi-fidelity + MF prior + validity).

Unlike the throwaway /tmp launcher, this SAVES the full diagnostics needed to
answer "did it crawl forward?" and "is the fidelity schedule interleaved?":
  - x_latent, c_controller         (the design — re-rollable by the dumper)
  - fidelity_summary               (per_step_level + per_step_nu + nu_final + counts)
  - best_fine_return, shac_summary
  - the trained policy (via policy_warmup_save) for reuse / diagnostics

Run-then-diagnose:
  python scripts/tasks/soft_robot/co_design/run_learned_codesign.py \
    --config configs/soft_robot/co_design/main_v2/crawling_learned.yaml \
    --n-grid 64 --out results/.../crawling_learned/results.json \
    --policy-save data/policies/learned_loco64.npz \
    --override K=40 M=6 num_modes=4 policy_warmup_steps=200
"""
import argparse
import json
import os
import time

import numpy as np
import yaml


def _parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)            # numbers / bools / lists
        except Exception:
            out[k] = v                        # bare string
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--voxel-dims", default="13,8,13")
    ap.add_argument("--mode-friction", default="", help="comma list, e.g. 0.05,0.2,0.4,0.7,1.0,1.3")
    ap.add_argument("--mode-slope-deg", default="", help="comma list of per-mode slope angles (deg)")
    ap.add_argument("--policy-load", default="", help="controller_path: reuse a pretrained policy")
    ap.add_argument("--policy-save", default="", help="policy_warmup_save: persist the warmed policy")
    ap.add_argument("--override", nargs="*", default=[], help="method_param overrides k=v ...")
    args = ap.parse_args()

    from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
    from genedynamics.experiments.framework.baseline import BaselineConfig
    from genedynamics.solvers.single.mrmfmbd.codesign import MRMFMBDBaseline

    cfg = yaml.safe_load(open(args.config))
    mp = dict(cfg["method_params"])
    mp.update(_parse_overrides(args.override))
    if args.policy_load:
        mp["controller_path"] = args.policy_load
    if args.policy_save:
        mp["policy_warmup_save"] = args.policy_save
    mp.setdefault("show_tqdm", True)
    vd = [int(x) for x in args.voxel_dims.split(",")]

    prov = JaxMpmTaskDomainProvider()
    _evkw = {}
    if args.mode_friction:
        _evkw["mode_friction"] = [float(x) for x in args.mode_friction.split(",")]
    if args.mode_slope_deg:
        _evkw["mode_slope_deg"] = [float(x) for x in args.mode_slope_deg.split(",")]
    # Open-loop trajectory controller (MBD-favorable high-dim non-convex landscape).
    _phi_dim = 80
    if str(mp.get("controller_kind", "sinusoid")) == "open_loop":
        _nnodes = int(mp.get("n_control_nodes", 50)); _nact = int(mp.get("n_actuators", 10))
        _evkw["controller_kind"] = "open_loop"; _evkw["n_control_nodes"] = _nnodes
        _evkw["n_actuators"] = _nact; _evkw["env_horizon"] = int(mp.get("env_horizon", 200))
        _phi_dim = _nnodes * _nact
        mp["phi_lo"] = -3.0; mp["phi_hi"] = 3.0; mp["phi_std"] = 1.0
        mp["phi_dim_override"] = True; mp["controller_type"] = "sinusoid"
    ev = prov.create_evaluator(".", voxel_dims=vd, n_grid=args.n_grid,
                               reward_shaping_weight=100.0, act_strength_base=24.0,
                               scale=50.0, task="crawling_ground", **_evkw)
    ts = prov.get_task_spec("crawling_ground")
    print(f"[run] n_grid={args.n_grid} {vd} | K={mp.get('K')} M={mp.get('M')} "
          f"modes={mp.get('num_modes')} warmup={mp.get('policy_warmup_steps')} "
          f"n_particles={ev._scene.n_particles}", flush=True)

    # x_dim: latent w when an A2 decoder is used, else the raw voxel count (prod dims).
    _mld = int(mp.get("morph_latent_dim", 0))
    _xdim = _mld if _mld > 0 else int(np.prod(vd))
    t0 = time.time()
    r = MRMFMBDBaseline().run(BaselineConfig(task_id="crawling_ground", seed=int(mp.get("seed", 0)), extra=mp),
                              ev, ts, x_dim=_xdim, phi_dim=_phi_dim)
    wall = time.time() - t0
    md = r.metadata
    out = {
        "return_rho_H": float(r.return_), "success": bool(r.success), "wall_min": wall / 60.0,
        "config": os.path.basename(args.config), "n_grid": args.n_grid, "voxel_dims": vd,
        "x_latent": np.asarray(r.x).tolist(), "c_controller": np.asarray(r.phi).tolist(),
        "shac_summary": md.get("shac_summary", {}),
        "best_fine_return": md.get("best_fine_return"),
        "fidelity_summary": md.get("fidelity_summary", {}),   # per_step_level + per_step_nu inside
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    fs = out["fidelity_summary"]
    print(f"[run] DONE rho_H={r.return_:.5f} wall={wall/60:.1f}min "
          f"counts={fs.get('level_counts')} nu={fs.get('nu_final')} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
