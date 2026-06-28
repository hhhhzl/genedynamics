"""Render a GIF of a DiffuseBot-aligned co-design result (decoder/latent path).

Unlike render_soft_robot_gif.py (which assumes the x-block is voxel occupancy),
this handles the A2 decoder path: the result's x-block is a 32-d shape latent w,
which we decode to per-voxel occupancy (g(w)) + actuator field before rolling out
on the [13,8,13] / n_grid=128 scene.

Usage:
  python render_dbaligned_gif.py <results.json> <out.gif> \
      --decoder data/morph_decoders/loco_big_l32 --voxel-dims 13,8,13 \
      --n-grid 128 --num-env-steps 200 --stride 3 --fps 20
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_with_positions
from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("out_gif")
    ap.add_argument("--decoder", required=True, help="morph decoder dir (decoder_config.json + decoder_params.npz)")
    ap.add_argument("--voxel-dims", default="13,8,13")
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--checkpoint", type=int, default=-1, help="theta_history index (-1=final)")
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--num-env-steps", type=int, default=200)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    vx, vy, vz = (int(v) for v in args.voxel_dims.split(","))

    # --- load decoder ---
    with open(os.path.join(args.decoder, "decoder_config.json")) as f:
        dc = json.load(f)
    mcfg = MorphDecoderConfig(
        latent_dim=int(dc["latent_dim"]), hidden_dim=int(dc["hidden_dim"]),
        n_voxels=int(dc["n_voxels"]), x_lo=float(dc["x_lo"]), x_hi=float(dc["x_hi"]),
        beta_kl=float(dc.get("beta_kl", 1e-3)),
        decode_actuator=bool(dc.get("decode_actuator", False)),
        decode_stiffness=bool(dc.get("decode_stiffness", False)),
        n_actuators=int(dc.get("n_actuators", 0)),
        e_lo=float(dc.get("e_lo", 0.5)), e_hi=float(dc.get("e_hi", 3.0)),
    )
    decoder = MorphDecoder.load(os.path.join(args.decoder, "decoder_params.npz"), mcfg)
    latent_dim = mcfg.latent_dim

    # --- load result theta = [w(latent_dim), phi] ---
    res = json.load(open(args.results_json))
    if isinstance(res, list):
        res = res[0]["result"]
    elif "result" in res:
        res = res["result"]
    th = res.get("theta_history")
    theta = np.asarray(th[args.checkpoint] if th else res["theta"], dtype=np.float32)
    w, phi = theta[:latent_dim], theta[latent_dim:]

    occ = np.asarray(decoder.decode(w), dtype=np.float32)            # (n_voxels,) in [x_lo,x_hi]
    act_field = np.asarray(decoder._decode_act(jnp.asarray(w)))      # (n_voxels, K)
    act_voxel = act_field.argmax(-1).astype(np.int32)               # (n_voxels,)

    # --- scene at aligned resolution ---
    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=[vx, vy, vz], n_grid=args.n_grid,
                               reward_shaping_weight=100.0, act_strength_base=24.0,
                               scale=50.0, task="crawling_ground")
    scene, cfg = ev._scene, ev._mpm_cfg
    K_act = int(cfg.n_actuators)

    # inject the decoded actuator placement (per-voxel argmax → per-particle via voxel_id)
    vid = np.asarray(scene.voxel_id)
    aid_particle = np.where(vid >= 0, act_voxel[np.clip(vid, 0, len(act_voxel) - 1)], -1).astype(np.int32)
    scene = scene._replace(actuator_id=jnp.asarray(aid_particle, dtype=jnp.int32))

    reward, disp, comx, xhist, massf = rollout_with_positions(
        jnp.asarray(occ), jnp.asarray(phi), jnp.asarray(float(args.friction), jnp.float32),
        scene, cfg, args.num_env_steps)
    xhist = np.asarray(xhist); massf = np.asarray(massf)
    occ_n = massf / float(massf.max() + 1e-9)
    keep = occ_n > 0.45
    print(f"reward={float(reward):.2f} Δx={float(disp):.3f} filled={int(keep.sum())}/{keep.size} "
          f"frames={xhist.shape[0]} particles={keep.size}")

    P = xhist[:, keep, :]
    aid = aid_particle[keep]
    base = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:K_act]
    cmap = ListedColormap(np.vstack([[0.82, 0.82, 0.82, 1.0], base]))
    cidx = (aid + 1).clip(0, K_act)
    s = 6.0 + 24.0 * (occ_n[keep] ** 2)

    lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    pad = 0.03
    xlim = (lo[0] - pad, hi[0] + pad); zlim = (lo[2] - pad, hi[2] + pad)
    ylim = (min(lo[1], 0.0), hi[1] + pad)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for t in range(0, P.shape[0], args.stride):
        ax.clear()
        ax.scatter(P[t, :, 0], P[t, :, 2], P[t, :, 1], c=cidx, cmap=cmap, vmin=0, vmax=K_act,
                   s=s, depthshade=False, edgecolors="none", alpha=0.95)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_zlim(*ylim)
        ax.set_box_aspect((xlim[1]-xlim[0], zlim[1]-zlim[0], (ylim[1]-ylim[0]) * 1.6))
        ax.view_init(elev=14, azim=-72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"DiffuseBot-aligned co-design (reward {float(reward):.1f})  t={t}/{P.shape[0]}", fontsize=9)
        ax.set_xlabel("forward →", fontsize=8, labelpad=-10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_gif)), exist_ok=True)
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=0)
    print(f"OK: wrote {args.out_gif} ({len(frames)} frames @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
