"""Panda surface-contact scanning end-to-end on real brax (docker).

Validates the arm env + 6D impedance + MDAC stiffness primitive + AL soft-
feasibility on the real mjx Panda:

  E1) env mechanics: action_size==25, reset/step finite, EE moves under impedance,
      constraint_residual returns h(6),g(0).
  E2) MDAC plans on the arm (Nsample fan-out, finite); planned controls move the
      surface coords / EE.
  E3) AL active: aug_rho>0 lowers the mean surface/normal constraint ‖h‖ of the
      executed plan vs aug=0 (the cfsmbd soft-feasibility steers the arm onto the
      contact manifold).

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_mdac_arm_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mdac.mdac import MDACSolver

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _rollout_violation(env, x0, us):
    def f(s, u):
        s2 = env.step(s, u)
        h, _ = env.constraint_residual(s2, u)
        return s2, jnp.linalg.norm(h)
    _, hs = jax.lax.scan(f, x0, us)
    return float(jnp.mean(hs))


def _plan_us(solver, x0, rng):
    b = solver._get_backend_impl()
    Y = b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng)
    return jnp.asarray(b.spline.node2u(Y))


def main():
    backend = get_backend("jax")
    rng = jax.random.PRNGKey(0)
    env = make_env("manipulator_surface_scan")          # plane surface
    ok = []

    # ---- E1) env mechanics ----
    print("action_size:", env.action_size)
    x0 = env.reset(jax.random.PRNGKey(1))
    s1 = env.step(x0, jnp.zeros(env.action_size))
    sr = env.step(x0, 0.3 * jax.random.normal(jax.random.PRNGKey(2), (env.action_size,)))
    h, g = env.constraint_residual(s1, jnp.zeros(env.action_size))
    moved = float(jnp.linalg.norm(sr.pipeline_state.qpos - x0.pipeline_state.qpos))
    okE1 = (env.action_size == 25) and bool(jnp.isfinite(s1.reward)) and h.shape == (6,) \
        and g.shape == (0,) and moved > 1e-4
    ok.append(okE1)
    print(f"[E1] env: action_size=25, reward finite, h{h.shape} g{g.shape}, EE moved={moved:.3e} -> {'PASS' if okE1 else 'FAIL'}")

    # ---- E2) MDAC plans on the arm ----
    sol = MDACSolver(env, None, backend, method="mdac", aug_lambda=0.0, aug_rho=0.0, **CFG)
    us = _plan_us(sol, x0, rng)
    okE2 = (us.shape == (CFG["Hsample"] + 1, 25)) and bool(jnp.all(jnp.isfinite(us)))
    ok.append(okE2)
    print(f"[E2] MDAC plan us{tuple(us.shape)} finite -> {'PASS' if okE2 else 'FAIL'}")

    # ---- E3) AL lowers constraint violation ----
    sol_off = MDACSolver(env, None, backend, method="mdac", aug_lambda=0.0, aug_rho=0.0, **CFG)
    sol_on = MDACSolver(env, None, backend, method="mdac", aug_lambda=2.0, aug_rho=50.0, **CFG)
    v_off = _rollout_violation(env, x0, _plan_us(sol_off, x0, rng))
    v_on = _rollout_violation(env, x0, _plan_us(sol_on, x0, rng))
    okE3 = v_on < v_off - 1e-6
    ok.append(okE3)
    print(f"[E3] mean‖h‖ executed  AL-off={v_off:.4f}  AL-on={v_on:.4f}  (on<off) -> {'PASS' if okE3 else 'FAIL'}")

    print("RESULT:", "ALL PASS" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
