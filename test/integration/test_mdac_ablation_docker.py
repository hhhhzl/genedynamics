"""MDAC ablations are ACTIVE + ablatable on arm/humanoid (docker, real brax).

For each task, the full MDAC plan must DIFFER from each single-component ablation
— proving the component is genuinely wired (not a no-op):
  mdac vs mdac_no_softfeas   -> AL soft-feasibility active
  mdac vs mdac_no_stiffness  -> env stiffness primitive active
  mdac vs mdac_no_tangent    -> geometry tangent-projection active
  mdac(+prior) vs mdac_no_rl_prior(+prior) -> RL prior warm-start active
Plus a fairness self-check (same sample budget across methods).

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_mdac_ablation_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.mdac.experiment import make_mdac, ARM_TASK, HUMANOID_TASK
from genedynamics.solvers.single.mdac.core.method_registry import assert_fair

CFG = dict(Hsample=8, Hnode=4, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.1, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


class _ConstPrior:                       # minimal Prior: nonzero constant warm-start
    output_dim = 1

    def __init__(self, hn1, nu):
        self._u = jnp.full((hn1, nu), 0.3, jnp.float32)

    def act(self, obs, *, key=None, deterministic=True):
        return self._u[0]

    def logp_of_sequence(self, o, a):
        return jnp.array(0.0)

    def warm_start(self, state):
        return self._u


def _plan(env, sol, x0, rng):
    b = sol._get_backend_impl()
    return np.asarray(b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng))


def _check_task(task, level=None):
    rng = jax.random.PRNGKey(0)
    prior = _ConstPrior(CFG["Hnode"] + 1, 25)
    env0, full = make_mdac(task, "mdac", level=level, prior=prior, **CFG)
    x0 = env0.reset(jax.random.PRNGKey(1))
    base = _plan(env0, full, x0, rng)

    ok = []
    ablations = {
        "no_softfeas": "mdac_no_softfeas",
        "no_stiffness": "mdac_no_stiffness",
        "no_tangent": "mdac_no_tangent",
        "no_retraction": "mdac_no_retraction",
        "no_anneal": "mdac_no_anneal",
        "no_rl_prior": "mdac_no_rl_prior",
    }
    for tag, method in ablations.items():
        env, sol = make_mdac(task, method, level=level, prior=prior, **CFG)
        x = env.reset(jax.random.PRNGKey(1))
        d = float(np.max(np.abs(_plan(env, sol, x, rng) - base)))
        active = d > 1e-4
        ok.append(active)
        print(f"  [{task}] mdac vs {method}: max|Δplan|={d:.3e} -> {'ACTIVE' if active else 'NO-OP!'}")

    # fairness: same (M,H,K)
    assert_fair((CFG["Nsample"], CFG["Hsample"], CFG["Ndiffuse"]),
                (full.config["Nsample"], full.config["Hsample"], full.config["Ndiffuse"]))
    return all(ok)


def main():
    a = _check_task(ARM_TASK, level="plane")
    h = _check_task(HUMANOID_TASK)
    print("RESULT:", "ALL COMPONENTS ACTIVE" if (a and h) else "SOME NO-OP")
    return 0 if (a and h) else 1


if __name__ == "__main__":
    raise SystemExit(main())
