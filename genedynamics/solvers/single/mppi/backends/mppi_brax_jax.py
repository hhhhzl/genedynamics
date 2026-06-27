"""JAX (brax-native) backend for the MPPI baseline — path-integral sampling MPC.

Mirrors ``dial/backends/dial_jax.py``: pulls config off the solver, builds the SAME brax
rollout as DIAL/MDAC (``env_rollout.build_brax_rollout`` — real ``env.step`` + reward), and
implements the ``WarmStartPlanner`` capability so it plugs into the shared
``RecedingHorizonController`` natively (warm-started, few iterations per real step).

MPPI here samples over the DENSE control sequence ``U`` (Hsample+1, nu) and refines the mean
by path-integral weighting ``softmax(return/λ)`` — NO manifold tangent projection / retraction
(idea.txt baselines). Distinct from DIAL only in the weighting; same env / rollout / budget.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.common.env_rollout import (
    is_brax_env, build_brax_rollout, build_brax_step,
)


class MPPIBraxBackendJax:
    """WarmStartPlanner + single-shot ``plan()`` for the brax MPPI baseline."""

    def __init__(self, solver: Any = None, *, nu: Optional[int] = None,
                 rollout_fn: Optional[Callable] = None, step_fn: Optional[Callable] = None,
                 Hsample: int = 16, Nsample: int = 2048, noise_sigma: float = 0.3,
                 lambda_: float = 1.0, action_limit: float = 1.0,
                 Ndiffuse: int = 2, Ndiffuse_init: int = 10, seed: int = 0) -> None:
        if solver is not None:
            cfg = solver.config
            nu = nu if nu is not None else int(solver.nu)
            Hsample = int(cfg.get("Hsample", Hsample))
            Nsample = int(cfg.get("Nsample", Nsample))
            noise_sigma = float(cfg.get("noise_sigma", noise_sigma))
            lambda_ = float(cfg.get("lambda_", lambda_))
            action_limit = float(cfg.get("action_limit", action_limit))
            Ndiffuse = int(cfg.get("Ndiffuse", Ndiffuse))
            Ndiffuse_init = int(cfg.get("Ndiffuse_init", Ndiffuse_init))
            seed = int(getattr(solver, "seed", seed))
            rollout_fn = rollout_fn or getattr(solver, "_rollout_fn", None)
            step_fn = step_fn or getattr(solver, "_step_fn", None)
        if nu is None:
            raise ValueError("MPPIBraxBackendJax requires nu (action dim).")

        self.nu, self.Hsample, self.Nsample = int(nu), int(Hsample), int(Nsample)
        self.sigma, self.lam = float(noise_sigma), float(lambda_)
        self.action_limit = float(action_limit)
        self.Ndiffuse, self.Ndiffuse_init, self.seed = int(Ndiffuse), int(Ndiffuse_init), int(seed)

        env = getattr(solver, "dynamics", None)
        self._rollout_fn = rollout_fn or (build_brax_rollout(env) if is_brax_env(env) else None)
        self._step_fn = step_fn or (build_brax_step(env) if is_brax_env(env) else None)
        if self._rollout_fn is None:
            raise ValueError("MPPIBraxBackendJax needs a brax env (or an injected rollout_fn).")
        self._iterate = jax.jit(self._mppi_iter)

    # one path-integral MPPI iteration: sample around mean -> roll -> softmax-weight -> mean
    def _mppi_iter(self, state, mean, rng):
        noise = self.sigma * jax.random.normal(rng, (self.Nsample, self.Hsample + 1, self.nu))
        us = jnp.clip(mean[None] + noise, -self.action_limit, self.action_limit)
        rews = self._rollout_fn(state, us, 0.0)                  # (N, Hsample+1)
        w = jax.nn.softmax(jnp.sum(rews, axis=1) / self.lam)     # (N,)
        return jnp.einsum("n,nhu->hu", w, us)

    # --- WarmStartPlanner protocol (consumed by RecedingHorizonController) ---
    def init_plan_var(self) -> jnp.ndarray:
        return jnp.zeros((self.Hsample + 1, self.nu), jnp.float32)

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        return jnp.arange(int(n_diffuse))                        # MPPI: n iterations (fixed σ)

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        mean = warm_start
        for _ in range(schedule.shape[0]):
            rng, sub = jax.random.split(rng)
            mean = self._iterate(state, mean, sub)
        return jnp.clip(mean, -self.action_limit, self.action_limit)

    def first_action(self, plan_var) -> jnp.ndarray:
        return plan_var[0]

    def shift(self, plan_var) -> jnp.ndarray:
        return jnp.concatenate([plan_var[1:], jnp.zeros((1, self.nu), plan_var.dtype)], axis=0)

    # --- single-shot plan (base Solver path) ---
    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        Y = self.replan(x0, self.init_plan_var(), self.make_schedule(self.Ndiffuse_init), rng_key)
        return {"actions": np.asarray(Y, np.float32)}

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        return [self.plan(x0, rng_key=k) for k in keys]


__all__ = ["MPPIBraxBackendJax"]
