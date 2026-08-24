"""Humanoid box pushing end-to-end on real brax (docker).

The H1 (+ 1-DoF x-slide box) pushes the box toward a goal x with the MDAC
position-stiffness primitive, under the contact manifold (hand–box contact +
balance) via the soft-feasibility (AL) seam.

  H1) env mechanics: action_size == 12, reset/step finite,
      fixed-stance constraint_residual -> h(14), g(3).
  H2) MDAC plans on the humanoid (Nsample fan-out), finite.
  H3) AL active: the formal aug_lambda/aug_rho setting lowers the same full
      equality/inequality residual penalty used by the augmented rollout.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_mdac_humanoid_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mdac.mdac import MDACSolver

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _rollout_residuals(env, x0, us):
    def f(s, u):
        s2 = env.step(s, u)
        h, g = env.constraint_residual(s2, u)
        gp = jax.nn.relu(g)
        residual = jnp.concatenate([jnp.abs(h), gp])
        return s2, jnp.array([
            jnp.sum(jnp.abs(h)),
            jnp.sum(gp),
            jnp.sum(residual * residual),
            jnp.linalg.norm(h[2:5]),
        ])
    _, stats = jax.lax.scan(f, x0, us)
    return np.asarray(jnp.mean(stats, axis=0), np.float64)


def _plan_us(solver, x0, rng):
    b = solver._get_backend_impl()
    Y = b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng)
    return jnp.asarray(b.spline.node2u(Y))


def main():
    backend = get_backend("jax")
    rng = jax.random.PRNGKey(0)
    env = make_env("humanoid_box_push")
    ok = []

    # ---- H1) env mechanics ----
    print("action_size:", env.action_size)
    x0 = env.reset(jax.random.PRNGKey(1))
    s1 = env.step(x0, jnp.zeros(env.action_size))
    h, g = env.constraint_residual(s1, jnp.zeros(env.action_size))
    okH1 = (env.action_size == 12) and bool(jnp.isfinite(s1.reward)) \
        and h.shape == (14,) and g.shape == (3,)
    ok.append(okH1)
    print(f"[H1] env: action_size={env.action_size}, reward finite, h{h.shape} g{g.shape} -> {'PASS' if okH1 else 'FAIL'}")

    # ---- H2) MDAC plans ----
    sol = MDACSolver(env, None, backend, method="mdac", aug_lambda=0.0, aug_rho=0.0, **CFG)
    us = _plan_us(sol, x0, rng)
    okH2 = (us.shape == (CFG["Hsample"] + 1, 12)) and bool(jnp.all(jnp.isfinite(us)))
    ok.append(okH2)
    print(f"[H2] MDAC plan us{tuple(us.shape)} finite -> {'PASS' if okH2 else 'FAIL'}")

    # ---- H3) AL lowers its actual full residual objective ----
    aug_lambda, aug_rho = 2.0, 200.0
    sol_off = MDACSolver(env, None, backend, method="mdac", aug_lambda=0.0, aug_rho=0.0, **CFG)
    sol_on = MDACSolver(env, None, backend, method="mdac",
                        aug_lambda=aug_lambda, aug_rho=aug_rho, **CFG)
    r_off = _rollout_residuals(env, x0, _plan_us(sol_off, x0, rng))
    r_on = _rollout_residuals(env, x0, _plan_us(sol_on, x0, rng))
    v_off = aug_lambda * (r_off[0] + r_off[1]) + 0.5 * aug_rho * r_off[2]
    v_on = aug_lambda * (r_on[0] + r_on[1]) + 0.5 * aug_rho * r_on[2]
    okH3 = v_on < v_off - 1e-6
    ok.append(okH3)
    print(f"[H3] executed AL residual objective off={v_off:.4f} on={v_on:.4f}; "
          f"hand-contact off={r_off[3]:.4f} on={r_on[3]:.4f} "
          f"-> {'PASS' if okH3 else 'FAIL'}")

    print("RESULT:", "ALL PASS" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
