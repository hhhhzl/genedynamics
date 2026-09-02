"""RL prior (brax integration) end-to-end with MGA, on real brax (docker).

Tiny PPO-trains a policy prior on a brax env (brax.training, the JAX RL backend),
wraps it as an RLPrior, and feeds it to MGA via the prior= seam. Validates:

  P1) the brax RLPrior instantiates from trained params and produces a valid
      warm_start (Hnode+1, action_size) and per-step action (action_size,).
  P2) MGA.replan WITH the prior differs from WITHOUT (the warm-start mix is
      active on real brax) and stays finite; prior=None is unchanged.

Invoke (from repo root):
  docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
    -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
    bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
              python test/integration/test_mga_prior_docker.py"
"""

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core import get_backend
from genedynamics.envs.factories import make_env
from genedynamics.solvers.single.mga.mga import MGASolver
from genedynamics.learning.train_rl_policy import (
    build_policy_prior,
    train_rl_policy,
)

ENV_NAME = "quadruped_go2_walk"
HNODE = 4
CFG = dict(Hsample=16, Hnode=HNODE, Nsample=64, Ndiffuse_init=3, Ndiffuse=2,
           temp_sample=0.06, action_limit=1.0, dt=0.02, ctrl_dt=0.02, seed=0)


def _replan(solver, x0, rng):
    b = solver._get_backend_impl()
    return np.asarray(b.replan(x0, b.init_plan_var(), b.make_schedule(b.Ndiffuse_init), rng))


def main():
    backend = get_backend("jax")
    rng = jax.random.PRNGKey(0)
    env = make_env(ENV_NAME)
    x0 = env.reset(jax.random.PRNGKey(7))
    ok = []

    # tiny PPO train (brax) -> RLPrior
    params, cfg = train_rl_policy(
        env, algo="ppo", num_timesteps=4096, episode_length=50, num_envs=16,
        batch_size=8, num_minibatches=2, unroll_length=10,
        num_updates_per_batch=1, normalize_observations=True, seed=0,
    )
    prior = build_policy_prior(
        params, cfg, env=env, Hsample=CFG["Hsample"], Hnode=HNODE
    )
    A = int(env.action_size)

    ws = np.asarray(prior.warm_start(x0))
    act = np.asarray(prior.act(x0.obs, deterministic=True))
    proposals = prior.sample_horizons(
        x0, key=jax.random.PRNGKey(11), n_samples=3
    )
    okP1 = (ws.shape == (HNODE + 1, A)) and (act.shape == (A,)) \
        and (np.asarray(proposals.trajectories).shape == (3, HNODE + 1, A)) \
        and (np.asarray(proposals.log_prob).shape == (3,)) \
        and bool(np.all(np.isfinite(ws))) and bool(np.all(np.isfinite(act))) \
        and bool(np.all(np.isfinite(np.asarray(proposals.trajectories))))
    ok.append(okP1)
    print(f"[P1] brax RLPrior: warm_start{ws.shape} act{act.shape} finite -> {'PASS' if okP1 else 'FAIL'}")

    y_none = _replan(MGASolver(env, None, backend, method="mga_base", **CFG), x0, rng)
    y_prior = _replan(MGASolver(env, None, backend, method="mga_base",
                                 prior=prior, prior_lambda_shift=0.5, **CFG), x0, rng)
    dmax = float(np.max(np.abs(y_prior - y_none)))
    okP2 = (dmax > 1e-4) and bool(np.all(np.isfinite(y_prior)))
    ok.append(okP2)
    print(f"[P2] MGA with vs without prior: max|Δplan|={dmax:.3e} -> {'PASS' if okP2 else 'FAIL'}")

    structured_solver = MGASolver(
        env, None, backend, method="mga_base", prior=prior,
        prior_lambda_shift=0.5, prior_stochastic_samples=3, **CFG,
    )
    y_structured = _replan(structured_solver, x0, rng)
    okP3 = y_structured.shape == y_prior.shape and bool(
        np.all(np.isfinite(y_structured))
    )
    ok.append(okP3)
    print(f"[P3] stochastic horizon proposals enter fixed-budget MGA -> {'PASS' if okP3 else 'FAIL'}")

    print("RESULT:", "ALL PASS" if all(ok) else "FAILED")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
