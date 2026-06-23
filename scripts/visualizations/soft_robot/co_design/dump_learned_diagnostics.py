"""Diagnostics dumper for a learned-controller co-design design.

Given a saved design (results.json with x_latent + c_controller), the trained
policy (controller .npz), and the morphology decoder, this re-rolls the design at
fine fidelity over the friction regimes and produces:

  * diagnostics.json   — per-mode returns, final forward displacement, rho_H
  * com_x_traj.png     — COM forward displacement vs time per mode
                         (rising-monotone = real crawl; spike-then-flat = a lunge)
  * motion.gif         — side-view (x-y) particle animation of the nominal mode
  * fidelity_sched.png — (if --result given) the per-step fidelity level ell* and
                         the dual nu_k trace (shows adaptive interleaving vs blocks)

NOTE: this is the analysis tool — it is NOT auto-run by the campaign. Invoke it
explicitly once a design + policy are saved. Example:

  python scripts/visualizations/soft_robot/co_design/dump_learned_diagnostics.py \
    --design results/.../crawling_learned/results.json \
    --policy data/policies/learned_loco64.npz \
    --decoder data/morph_decoders/loco_big_l32 \
    --result results/.../crawling_learned/results.json \
    --out-dir results/.../crawling_learned/diag --n-grid 128 --steps 200 --modes 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _load_decoder(decoder_dir):
    from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
    from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig
    dc = json.load(open(os.path.join(decoder_dir, "decoder_config.json")))
    mc = MorphDecoderConfig(
        latent_dim=int(dc["latent_dim"]), hidden_dim=int(dc["hidden_dim"]),
        n_voxels=int(dc["n_voxels"]), x_lo=float(dc["x_lo"]), x_hi=float(dc["x_hi"]),
        decode_actuator=bool(dc.get("decode_actuator", False)),
        n_actuators=int(dc.get("n_actuators", 0)))
    return MorphDecoder.load(os.path.join(decoder_dir, "decoder_params.npz"), mc), mc


def _plot_com(com_per_mode, frictions, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for fr, com in zip(frictions, com_per_mode):
        fwd = com[:, 0] - com[0, 0]                      # forward displacement from start
        ax.plot(np.arange(len(fwd)), fwd, label=f"mu={fr:.2f}")
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("env step"); ax.set_ylabel("forward COM displacement")
    ax.set_title("Forward progress per regime (rising = real crawl; spike-flat = lunge)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def _render_gif(x_hist, actuator_id, out_path, stride=4):
    """Side-view (x horizontal, y vertical) 2D scatter animation of the gait."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    x_hist = np.asarray(x_hist)                          # (T, P, 3)
    frames = list(range(0, x_hist.shape[0], stride))
    aid = np.asarray(actuator_id)
    cN = float(max(aid.max(), 1))
    fig, ax = plt.subplots(figsize=(6, 3))
    xmin, xmax = float(x_hist[..., 0].min()), float(x_hist[..., 0].max())
    ymin, ymax = float(x_hist[..., 1].min()), float(x_hist[..., 1].max())
    sc = ax.scatter(x_hist[0][:, 0], x_hist[0][:, 1], c=aid / cN, cmap="viridis", s=6)
    ax.set_xlim(xmin - 0.02, xmax + 0.02); ax.set_ylim(ymin - 0.02, ymax + 0.02)
    ax.set_xlabel("x (forward)"); ax.set_ylabel("y (up)")

    def _update(t):
        sc.set_offsets(x_hist[t][:, :2]); ax.set_title(f"env step {t}")
        return (sc,)

    anim = FuncAnimation(fig, _update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=15)); plt.close(fig)


def _plot_fidelity(result_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(result_path))
    fs = res.get("fidelity_summary") or {}
    levels = fs.get("per_step_level")
    nus = fs.get("per_step_nu")
    if not levels:
        print("[diag] no per_step_level in result; skipping fidelity_sched.png")
        return
    k = np.arange(len(levels))
    fig, ax1 = plt.subplots(figsize=(8, 3.2))
    ax1.step(k, levels, where="mid", color="tab:blue", label="ell* (fidelity level)")
    ax1.set_xlabel("diffusion step k"); ax1.set_ylabel("fidelity level ell*", color="tab:blue")
    ax1.set_yticks(sorted(set(int(x) for x in levels)))
    if nus:
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(nus)), nus, color="tab:red", lw=1.2, label="nu_k (dual)")
        ax2.set_ylabel("nu_k", color="tab:red")
    ax1.set_title("Adaptive fidelity selection (interleaved if ell* jumps up AND down)")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, help="results.json with x_latent + c_controller")
    ap.add_argument("--controller", default="learned", choices=["learned", "sinusoid"],
                    help="learned: c is the latent + needs --policy; sinusoid: c IS the 80-d phi")
    ap.add_argument("--policy", default="", help="trained controller .npz (learned only)")
    ap.add_argument("--decoder", required=True, help="morph decoder dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--result", default="", help="result json to plot per_step_level/nu (optional)")
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--voxel-dims", default="13,8,13")
    ap.add_argument("--mode-friction", default="", help="comma list of friction regimes")
    ap.add_argument("--gif-mode", type=int, default=-1, help="mode index for the GIF (-1 = nominal/last)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import jax.numpy as jnp
    from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
    from genedynamics.envs.external.jax_mpm.scene import (
        rollout_with_positions_closed, rollout_with_positions)
    from genedynamics.solvers.single.mrmfmbd.controller_system import load_controller
    from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
        risk_sensitive_marginalize_np)

    vd = [int(x) for x in args.voxel_dims.split(",")]
    prov = JaxMpmTaskDomainProvider()
    _evkw = {}
    if args.mode_friction:
        _evkw["mode_friction"] = [float(x) for x in args.mode_friction.split(",")]
    ev = prov.create_evaluator(".", voxel_dims=vd, n_grid=args.n_grid, reward_shaping_weight=100.0,
                               act_strength_base=24.0, scale=50.0, task="crawling_ground", **_evkw)
    sc, cfg = ev._scene, ev._mpm_cfg

    design = json.load(open(args.design))
    if "x_latent" in design:                      # launcher layout (ours runs)
        w = np.asarray(design["x_latent"], np.float32)
        c = np.asarray(design.get("c_controller", design.get("phi")), np.float32)
    else:                                          # baseline layout (per_seed list)
        s = design["per_seed"][0]
        w = np.asarray(s["x_latent"], np.float32)
        c = np.asarray(s["phi"], np.float32)
    dec, _mc = _load_decoder(args.decoder)
    occ, act, _ = dec.decode_full_batch(jnp.asarray(w[None]))
    occ = occ[0]; aw = act[0] if act is not None else None
    learned = (args.controller == "learned")
    if learned:
        if not args.policy:
            raise SystemExit("--controller learned requires --policy")
        pp, E, _crit = load_controller(args.policy)

    fr_table = np.asarray(ev._mode_friction, np.float32)[:args.modes]
    rewards, finals, coms = [], [], []
    x_hist_gif = None
    gif_idx = (len(fr_table) - 1) if args.gif_mode < 0 else args.gif_mode
    for i, fr in enumerate(fr_table):
        if learned:
            r, fdisp, com_x, x_hist, _mass = rollout_with_positions_closed(
                occ, jnp.asarray(c), jnp.asarray(float(fr)), sc, cfg, args.steps, pp, E,
                actuator_weight_voxel=aw)
        else:                                            # sinusoid: c IS the phi vector
            r, fdisp, com_x, x_hist, _mass = rollout_with_positions(
                occ, jnp.asarray(c), jnp.asarray(float(fr)), sc, cfg, args.steps,
                actuator_weight_voxel=aw)
        rewards.append(float(r)); finals.append(float(fdisp)); coms.append(np.asarray(com_x))
        if i == gif_idx:
            x_hist_gif = np.asarray(x_hist)
    rho_H = float(risk_sensitive_marginalize_np(np.asarray(rewards), np.zeros(len(rewards)), 1.0))

    json.dump({
        "rho_H": rho_H, "per_mode_returns": rewards, "per_mode_final_disp": finals,
        "frictions": fr_table.tolist(), "worst_mode": float(np.min(rewards)),
        "mean_return": float(np.mean(rewards)), "steps": args.steps, "n_grid": args.n_grid,
    }, open(os.path.join(args.out_dir, "diagnostics.json"), "w"), indent=2)

    _plot_com(coms, fr_table, os.path.join(args.out_dir, "com_x_traj.png"))
    if x_hist_gif is not None:
        _render_gif(x_hist_gif, sc.actuator_id, os.path.join(args.out_dir, "motion.gif"))
    if args.result:
        _plot_fidelity(args.result, os.path.join(args.out_dir, "fidelity_sched.png"))
    print(f"[diag] rho_H={rho_H:.4f} per_mode={[round(r,3) for r in rewards]} "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
