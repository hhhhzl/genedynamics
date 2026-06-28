"""MDAC augmented-Lagrangian (cfsmbd/mdcoas pattern) on a REAL brax env.

Runs ONLY in the docker image `genedynamics/dev-cpu:torch` (brax/mjx/mujoco) —
fedguide x86 has no brax. Validates the soft-feasibility seam end-to-end on the
brax substrate:

  A) AL-off (no constraint / empty residual) => MDAC plan() is byte-identical to
     the DIAL backend plan() on the same brax env (the regression gate, on
     real physics, not a mock).
  B) AL-on (synthetic action constraint g=||u||^2 - budget, aug_rho>0) => the
     planned trajectory's mean [g]_+ violation is STRICTLY LOWER than aug=0,
     i.e. the cfsmbd augmented reward actually steers MDAC toward feasibility.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_mdac_al_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.dial.dial import DIALMPCSolver
from genedynamics.solvers.single.mdac.mdac import MDACSolver

ENV_NAME = "quadruped_go2_walk"
CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.06, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)
BUDGET = 0.5            # ||u||^2 budget for the synthetic inequality g = ||u||^2 - budget


def _plan_actions(backend, x0, rng):
    """Open-loop planned controls via the core reverse-diffusion (replan + node2u).
    Avoids plan()'s _rollout_states, which assumes flat (non-brax-State) x0."""
    Y = backend.replan(x0, backend.init_plan_var(),
                       backend.make_schedule(backend.Ndiffuse_init), rng)
    return np.asarray(backend.spline.node2u(Y), np.float32)   # (Hsample+1, nu)


def _violation(actions, budget=BUDGET):
    a = np.asarray(actions, np.float32)            # (Hsample+1, nu)
    g = np.sum(a * a, axis=-1) - budget
    return float(np.mean(np.maximum(g, 0.0)))


def main():
    backend = get_backend("jax")
    rng = jax.random.PRNGKey(0)

    # ---- A) AL-off == DIAL, on real brax ----
    env = make_env(ENV_NAME)
    x0 = env.reset(jax.random.PRNGKey(7))
    dial = DIALMPCSolver(env, None, backend, **CFG)._get_backend_impl()
    # method="dial" => use_soft_feasibility off => plain brax rollout
    mdac_off = MDACSolver(env, None, backend, method="dial", **CFG)._get_backend_impl()
    Yd = _plan_actions(dial, x0, rng)
    Ym = _plan_actions(mdac_off, x0, rng)
    dmax = float(np.max(np.abs(Yd - Ym)))
    okA = dmax < 1e-4
    print(f"[A] AL-off vs DIAL on brax: max|Δactions|={dmax:.2e}  -> {'PASS' if okA else 'FAIL'}")

    # ---- B) AL-on reduces the synthetic constraint violation ----
    # synthetic inequality on the ACTION: g = ||u||^2 - budget  (<=0 feasible)
    def _cres(pipeline_state, action, ctx=None):
        h = jnp.zeros((0,), jnp.float32)
        g = jnp.array([jnp.sum(action * action) - BUDGET], jnp.float32)
        return h, g

    env_c = make_env(ENV_NAME)
    env_c.constraint_residual = _cres            # inject the synthetic constraint
    x0c = env_c.reset(jax.random.PRNGKey(7))

    mdac_aloff = MDACSolver(env_c, None, backend, method="mdac",
                            aug_lambda=0.0, aug_rho=0.0, **CFG)._get_backend_impl()
    mdac_alon = MDACSolver(env_c, None, backend, method="mdac",
                           aug_lambda=5.0, aug_rho=200.0, **CFG)._get_backend_impl()
    v_off = _violation(_plan_actions(mdac_aloff, x0c, rng))
    v_on = _violation(_plan_actions(mdac_alon, x0c, rng))
    okB = v_on < v_off - 1e-6
    print(f"[B] mean[g]_+  AL-off={v_off:.4f}  AL-on={v_on:.4f}  "
          f"(on<off) -> {'PASS' if okB else 'FAIL'}")

    print("RESULT:", "ALL PASS" if (okA and okB) else "FAILED")
    return 0 if (okA and okB) else 1


if __name__ == "__main__":
    raise SystemExit(main())
