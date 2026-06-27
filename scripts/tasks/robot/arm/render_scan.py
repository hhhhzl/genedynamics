"""Render an MDAC arm surface-scan rollout to a GIF (run in docker, real brax).

results/arm/impedence/rigid/<level>/arm_scan.gif -- 3D view: the analytic surface mesh, the
Panda arm kinematic chain (body positions; the vendored Panda is mesh-free so a
mujoco render shows nothing), the end-effector and its scan path, and the target.

  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python scripts/tasks/robot/arm/render_scan.py cylinder"
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genedynamics.solvers.single.mdac.experiment import make_mdac, ARM_TASK
import genedynamics.core.coverage.surface_geometry as sg

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "s3"
N_STEPS = 28
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)

env, sol = make_mdac(ARM_TASK, "mdac", level=LEVEL, **CFG)
x0 = env.reset(jax.random.PRNGKey(0))
res = sol.run_receding(x0, N_STEPS, jax.random.PRNGKey(1))

# roll the executed actions, collecting EE, scan target, and the arm chain
s = x0
ee, tgt, arm = [], [], []
for a in res.actions:
    s = env.step(s, jnp.asarray(a))
    ee.append(np.asarray(s.pipeline_state.site_xpos[env._ee_site]))
    xi_t, eta_t = env._target(s.info["step"])
    tgt.append(np.asarray(sg.point(env.surface, xi_t, eta_t)))
    arm.append(np.asarray(s.pipeline_state.x.pos))      # (nbody,3) link positions
ee, tgt, arm = np.array(ee), np.array(tgt), np.array(arm)
base = np.array([0.0, 0.0, 0.33])                       # Panda mount (link0 fused)

# surface mesh over (xi,eta)
g = np.linspace(0.0, 1.0, 24)
P = np.array([[np.asarray(sg.point(env.surface, xi, eta)) for xi in g] for eta in g])

allpts = np.concatenate([P.reshape(-1, 3), ee, arm.reshape(-1, 3), base[None]], axis=0)
lo, hi = allpts.min(0), allpts.max(0)
ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 + 0.05

frames = []
for i in range(len(ee)):
    fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=0.4, color="tan", linewidth=0)
    chain = np.concatenate([base[None], arm[i]], axis=0)        # base -> links
    ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], "-o", color="0.3", lw=2, ms=3, label="Panda arm")
    ax.plot(ee[:i + 1, 0], ee[:i + 1, 1], ee[:i + 1, 2], "b-", lw=2, label="EE path")
    ax.scatter(*ee[i], c="red", s=45, label="EE")
    ax.scatter(*tgt[i], c="green", s=20, label="scan target")
    for d, c in zip(range(3), "xyz"):
        getattr(ax, f"set_{c}lim")(ctr[d] - rad, ctr[d] + rad)
    ax.set_title(f"MDAC surface scan — {LEVEL}  (step {i + 1}/{len(ee)})")
    ax.view_init(elev=20, azim=-60 + i * 1.2)
    if i == 0:
        ax.legend(loc="upper left", fontsize=7)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frames.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
    plt.close(fig)
out_dir = f"results/arm/impedence/rigid/{LEVEL}"
os.makedirs(out_dir, exist_ok=True)
out = f"{out_dir}/arm_scan.gif"
imageio.mimsave(out, frames, fps=6, loop=0)
print("wrote", out, len(frames), "frames")
