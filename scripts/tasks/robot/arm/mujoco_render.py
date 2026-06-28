"""Paper-level mujoco render of an MDAC arm surface-scan (run in docker, real brax).

The vendored Panda used for the mjx primitive is mesh-free, so we render with the
FULL visual Panda (mujoco_menagerie panda_nohand.xml, joints joint1..joint7 in the
same order) driven by the SIMULATED qpos. The TRUE analytic surface p_s(xi,eta) is
triangulated into a mesh (.obj) and added as a mesh geom, so the actual curved /
bumpy surface is rendered (not a placeholder box). The end-effector is a small
probe sphere at its simulated position.

Output: results/arm/impedence/rigid/<level>/arm_render.mp4 (+ .gif).

  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo imageio-ffmpeg >/dev/null 2>&1; \
              python scripts/tasks/robot/arm/mujoco_render.py cylinder"
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import imageio.v2 as imageio
import mujoco

from genedynamics.solvers.single.mdac.experiment import make_mdac, ARM_TASK
import genedynamics.core.coverage.surface_geometry as sg

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "convex"
N_STEPS = 40
GRID = 28                         # surface mesh resolution
W, H, FPS = 960, 720, 8
MENAGERIE = "third_party/mujoco_menagerie/franka_emika_panda"
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)

# --- 1. simulate: collect qpos (7 joints) + EE positions ---
env, sol = make_mdac(ARM_TASK, "mdac", level=LEVEL, **CFG)
x0 = env.reset(jax.random.PRNGKey(0))
res = sol.run_receding(x0, N_STEPS, jax.random.PRNGKey(1))
s = x0
qpos, ee = [], []
for a in res.actions:
    s = env.step(s, jnp.asarray(a))
    qpos.append(np.asarray(s.pipeline_state.qpos))
    ee.append(np.asarray(s.pipeline_state.site_xpos[env._ee_site]))   # ACTUAL end-effector
qpos, ee = np.array(qpos), np.array(ee)
# smooth qpos + EE for a clean render: the sampling-based controller replans every
# step and its joint trajectory is jittery; a small centered moving average
# removes the per-frame shake (cosmetic — the executed motion is unchanged).
def _smooth(a, k=5):
    return np.array([a[max(0, i - k // 2):i + k // 2 + 1].mean(0) for i in range(len(a))])
qpos, ee = _smooth(qpos), _smooth(ee)

# --- 2. triangulate the TRUE surface p_s(xi,eta) into an .obj mesh ---
g = np.linspace(0.0, 1.0, GRID)
V = np.array([[np.asarray(sg.point(env.surface, xi, eta)) for eta in g] for xi in g])  # (G,G,3)
V = V.reshape(-1, 3)
ctr = V.mean(0)
Vloc = V - ctr                                          # mesh-local (geom placed at ctr)
faces = []
for i in range(GRID - 1):
    for j in range(GRID - 1):
        a = i * GRID + j; b = a + 1; c = a + GRID; d = c + 1
        # both windings -> the open surface renders lit from above AND below
        faces += [(a, c, b), (b, c, d), (a, b, c), (b, d, c)]
obj_path = os.path.join(MENAGERIE, "assets", "_mdac_surf.obj")
with open(obj_path, "w") as f:
    for v in Vloc:
        f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
    for t in faces:
        f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")       # .obj is 1-indexed

# EE path trace (static faint spheres along the executed EE trajectory), like the
# matplotlib EE-path line — the moving probe rides this trail on the surface.
trail = "".join(
    f'<geom type="sphere" size="0.007" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
    f'material="trail" contype="0" conaffinity="0"/>\n    ' for p in ee[::2])

# --- 3. render scene around the VISUAL panda ---
scene = f"""<mujoco model="mdac_scan">
  <include file="panda_nohand.xml"/>
  <statistic center="{ctr[0]:.3f} 0 0.5" extent="0.9"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="130" elevation="-22" offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" mark="edge" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.2"/>
    <mesh name="surfmesh" file="_mdac_surf.obj" inertia="shell"/>
    <material name="surf" rgba="0.95 0.62 0.28 0.4" specular="0.3" shininess="0.4"/>
    <material name="probe" rgba="1 0.12 0.08 1" emission="0.45"/>
    <material name="trail" rgba="1 0.2 0.15 0.7"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.3 2.0" dir="-0.2 0.1 -1" directional="true"/>
    <light pos="{ctr[0]:.2f} 0.4 1.4" dir="0 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="grid"/>
    <geom name="surfmesh" type="mesh" mesh="surfmesh" pos="{ctr[0]:.4f} {ctr[1]:.4f} {ctr[2]:.4f}"
          material="surf" contype="0" conaffinity="0"/>
    {trail}
    <body name="probe" mocap="true" pos="{ee[0,0]:.3f} {ee[0,1]:.3f} {ee[0,2]:.3f}">
      <geom type="sphere" size="0.022" material="probe" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
scene_path = os.path.join(MENAGERIE, "_mdac_scan_scene.xml")
with open(scene_path, "w") as f:
    f.write(scene)

try:
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [ctr[0], ctr[1], ctr[2]]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 140, -14   # closer, near-level to show the 3D surface
    nq = min(7, model.nq)
    frames = []
    for i in range(len(qpos)):
        data.qpos[:nq] = qpos[i][:nq]
        data.mocap_pos[0] = ee[i]
        mujoco.mj_forward(model, data)
        cam.azimuth = 140 + i * 0.5
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    out_dir = f"results/arm/impedence/rigid/{LEVEL}"; os.makedirs(out_dir, exist_ok=True)
    out_mp4 = f"{out_dir}/arm_render.mp4"
    out_gif = f"{out_dir}/arm_render.gif"
    try:
        imageio.mimsave(out_mp4, frames, fps=FPS, quality=8); print("wrote", out_mp4, len(frames))
    except Exception as e:
        print("mp4 failed:", e)
    imageio.mimsave(out_gif, frames, fps=FPS, loop=0); print("wrote", out_gif, len(frames))
finally:
    os.remove(scene_path); os.remove(obj_path)
