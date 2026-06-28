"""High-quality render of A's json-reconstructed rollout (OUR JAX-MPM).

Reconstructs A entirely from results.json (decode latent w -> occupancy +
actuator field, apply the json's phi controller) -> rollout_with_positions (the
SAME path as A_dbaligned_crawl.gif, so it genuinely moves forward) -> then
renders each frame as a SMOOTH marching-cubes surface on a textured ground with
lighting (instead of scatter dots). No SoftZoo: A's controller is valid here.

  python render_A_hq.py <results.json> <out.gif> --decoder <dir> --voxel-dims 13,8,13 --n-grid 128
"""
import argparse, json, os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import imageio.v2 as imageio
from skimage import measure

from genedynamics.experiments.plugins.task_domains.jax_mpm import JaxMpmTaskDomainProvider
from genedynamics.envs.external.jax_mpm.scene import rollout_with_positions
from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("out_gif")
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--voxel-dims", default="13,8,13")
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--num-env-steps", type=int, default=200)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--grid", type=int, default=56, help="marching-cubes voxel resolution")
    args = ap.parse_args()
    vx, vy, vz = (int(v) for v in args.voxel_dims.split(","))

    dc = json.load(open(os.path.join(args.decoder, "decoder_config.json")))
    mcfg = MorphDecoderConfig(latent_dim=int(dc["latent_dim"]), hidden_dim=int(dc["hidden_dim"]),
        n_voxels=int(dc["n_voxels"]), x_lo=float(dc["x_lo"]), x_hi=float(dc["x_hi"]),
        beta_kl=float(dc.get("beta_kl",1e-3)), decode_actuator=bool(dc.get("decode_actuator",False)),
        decode_stiffness=bool(dc.get("decode_stiffness",False)), n_actuators=int(dc.get("n_actuators",0)),
        e_lo=float(dc.get("e_lo",0.5)), e_hi=float(dc.get("e_hi",3.0)))
    decoder = MorphDecoder.load(os.path.join(args.decoder, "decoder_params.npz"), mcfg)
    res = json.load(open(args.results_json))
    res = res[0]["result"] if isinstance(res, list) else res.get("result", res)
    theta = np.asarray(res["theta"], np.float32)
    w, phi = theta[:mcfg.latent_dim], theta[mcfg.latent_dim:]
    occ = np.asarray(decoder.decode(w), np.float32)

    prov = JaxMpmTaskDomainProvider()
    ev = prov.create_evaluator(".", voxel_dims=[vx, vy, vz], n_grid=args.n_grid,
                               reward_shaping_weight=100.0, act_strength_base=24.0, scale=50.0,
                               task="crawling_ground")
    scene, cfg = ev._scene, ev._mpm_cfg
    reward, disp, comx, xhist, massf = rollout_with_positions(
        jnp.asarray(occ), jnp.asarray(phi), jnp.asarray(float(args.friction), jnp.float32),
        scene, cfg, args.num_env_steps)
    xhist = np.asarray(xhist); massf = np.asarray(massf)
    occn = massf / float(massf.max() + 1e-9)
    keep = occn > 0.45
    P = xhist[:, keep, :]                                   # (T, n_keep, 3) — moves forward
    print(f"reward={float(reward):.2f} dx={float(disp):.3f} keep={int(keep.sum())} frames={P.shape[0]}", flush=True)

    # fixed world bounds across trajectory (so forward translation is visible)
    lo = P.reshape(-1, 3).min(0); hi = P.reshape(-1, 3).max(0)
    pad = 0.02
    G = args.grid
    ls = LightSource(azdeg=315, altdeg=45)
    fig = plt.figure(figsize=(7, 4.2)); ax = fig.add_subplot(111, projection="3d")
    frames = []
    for t in range(0, P.shape[0], args.stride):
        pts = P[t]
        # voxelize this frame's particles into a smooth occupancy -> marching cubes
        bmin = pts.min(0) - 1e-3; bmax = pts.max(0) + 1e-3
        ijk = np.clip(((pts - bmin) / (bmax - bmin) * (G - 1)).astype(int), 0, G - 1)
        vol = np.zeros((G, G, G), np.float32)
        vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
        # light smoothing for a solid surface
        from scipy.ndimage import gaussian_filter, binary_dilation
        vol = gaussian_filter(binary_dilation(vol, iterations=1).astype(np.float32), sigma=1.0)
        ax.clear()
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=0.4)
            verts = bmin + verts / (G - 1) * (bmax - bmin)     # back to world coords
            ax.plot_trisurf(verts[:, 0], verts[:, 2], faces, verts[:, 1],
                            color=(0.30, 0.45, 0.80), lw=0.0, antialiased=True, shade=True)
        except Exception:
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=6, c="steelblue")
        # ground plane (soil-colored)
        gx = np.array([[lo[0]-0.1, hi[0]+0.1], [lo[0]-0.1, hi[0]+0.1]])
        gz = np.array([[lo[2]-0.1, lo[2]-0.1], [hi[2]+0.1, hi[2]+0.1]])
        gy = np.full_like(gx, max(lo[1], 0.0))
        ax.plot_surface(gx, gz, gy, color=(0.62, 0.50, 0.38), alpha=0.6, zorder=-1)
        ax.set_xlim(lo[0]-pad, hi[0]+pad); ax.set_ylim(lo[2]-pad, hi[2]+pad); ax.set_zlim(min(lo[1],0.0), hi[1]+pad)
        ax.set_box_aspect((hi[0]-lo[0]+2*pad, hi[2]-lo[2]+2*pad, (hi[1]-lo[1]+pad)*1.5))
        ax.view_init(elev=16, azim=-70)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"A — json-reconstructed crawl (reward {float(reward):.1f})  forward →   t={t}/{P.shape[0]}", fontsize=10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_gif)), exist_ok=True)
    imageio.mimsave(args.out_gif, frames, fps=args.fps, loop=0)
    print(f"OK: wrote {args.out_gif} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
