"""Geometry seam end-to-end on a REAL brax task (docker only).

Validates the genemetry reuse path (SdfManifold.geometry/project — the metric
tangent projection) running INSIDE the real brax reverse-diffusion kernel
(jit + vmap + lax.scan over brax States), not just a CPU mock.

The env supplies a constraint-geometry fn `geometry_fn(state, Ybar_nodes, t0) ->
a_geom (Hnode+1, nu)` (the per-node constraint proxy vectors, exactly as 2GO's
_constraint_geometry_time_jit). Here the active constraint is "do not move along
control dim 0" (a_geom rows = e_0), so the metric projection must remove the
e_0 component of the diffusion update.

Gates:
  G1) geometry-on plan runs on real brax, all-finite (the SdfManifold/project
      chain compiles + runs inside the jit/vmap/scan kernel over brax States).
  G2) geometry-on != geometry-off (the projection is actually active).
  G3) geometry-on SUPPRESSES the constrained direction: mean|u[:,0]| is lower
      than geometry-off (DIAL) -> the tangent projection is directionally correct.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python scripts/validation/docker/mga_geometry.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mga.mga import MGASolver

ENV_NAME = "quadruped_go2_walk"
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=4, Ndiffuse=2,
           temp_sample=0.06, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _planned_us(solver, x0, rng):
    b = solver._get_backend_impl()
    Y = b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng)
    return np.asarray(b.spline.node2u(Y), np.float32)         # (Hsample+1, nu)


def main():
    backend = get_backend("jax")
    rng = jax.random.PRNGKey(0)
    env = make_env(ENV_NAME)
    x0 = env.reset(jax.random.PRNGKey(7))
    ok = []

    # constraint-geometry fn: active constraint = "no motion along control dim 0".
    # a_geom rows point along e_0 (per node), so SdfManifold.project must remove
    # the e_0 component of the diffusion update at the active nodes.
    def geometry_fn(state, Ybar_nodes, t0):
        a = jnp.zeros_like(Ybar_nodes)
        return a.at[:, 0].set(1.0)                            # (Hnode+1, nu)

    m_off = MGASolver(env, None, backend, method="mga_base", **CFG)         # no geometry_fn => DIAL path
    m_on = MGASolver(env, None, backend, method="mga_base",
                      geometry_fn=geometry_fn, mga_topk_active=4,
                      mga_geom_gain=1.0, **CFG)

    u_off = _planned_us(m_off, x0, rng)
    u_on = _planned_us(m_on, x0, rng)

    okG1 = bool(np.all(np.isfinite(u_on)))
    ok.append(okG1)
    print(f"[G1] geometry-on runs on real brax, finite -> {'PASS' if okG1 else 'FAIL'}")

    dmax = float(np.max(np.abs(u_on - u_off)))
    okG2 = dmax > 1e-4
    ok.append(okG2)
    print(f"[G2] geometry-on != geometry-off: max|Δ|={dmax:.3e} -> {'PASS' if okG2 else 'FAIL'}")

    c_off = float(np.mean(np.abs(u_off[:, 0])))
    c_on = float(np.mean(np.abs(u_on[:, 0])))
    okG3 = c_on < c_off - 1e-6
    ok.append(okG3)
    print(f"[G3] constrained-dir mean|u[:,0]|  off={c_off:.4f}  on={c_on:.4f}  "
          f"(on<off) -> {'PASS' if okG3 else 'FAIL'}")

    print("RESULT:", "ALL PASS" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
