"""brax-native interactive HTML render of an MDAC arm scan (run in docker, brax).

The env IS brax/mjx, so brax.io.html is the native renderer — but the vendored
Panda used for the mjx primitive is mesh-free (geoms stripped), so rendering the
env System directly shows an invisible arm. We instead load the FULL visual Panda
(menagerie panda_nohand.xml) as a brax System and drive it with the simulated
qpos, then html.render -> an interactive (rotate/zoom) HTML.

Output: results/arm/impedence/rigid/<level>/arm_brax.html
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from brax.io import mjcf, html
from brax.mjx import pipeline

from genedynamics.solvers.single.mdac.experiment import make_mdac, ARM_TASK

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "convex"
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)

env, sol = make_mdac(ARM_TASK, "mdac", level=LEVEL, **CFG)
x0 = env.reset(jax.random.PRNGKey(0))
res = sol.run_receding(x0, 30, jax.random.PRNGKey(1))
s = x0
qpos = []
for a in res.actions:
    s = env.step(s, jnp.asarray(a))
    qpos.append(np.asarray(s.pipeline_state.qpos))
qpos = np.array(qpos)

# full visual Panda as a brax mjx System; FK each qpos into a renderable State.
full = mjcf.load("third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml")
nq = int(full.q_size())
qd0 = jnp.zeros(full.qd_size())
states = [pipeline.init(full, jnp.asarray(q[:nq], jnp.float32), qd0) for q in qpos]

out = html.render(full, states, height=600)
out_dir = f"results/arm/impedence/rigid/{LEVEL}"; os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/arm_brax.html"
with open(path, "w") as f:
    f.write(out)
print("wrote", path, len(states), "states; full Panda nq =", nq)
