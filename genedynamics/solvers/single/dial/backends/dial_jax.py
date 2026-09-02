"""DIAL-MPC JAX backend (reverse-update kernel + warm-start planner).

Follows the genedynamics multi-backend pattern (cf. mbd/backends/mbd_jax.py):
the SOLVER wrapper (``dial.py``) selects this backend via
``RuntimeBackendManager`` and ``_get_dial_backend``; this class implements both

  * the unified ``plan(x0, rng) -> dict`` interface (single-shot full-horizon,
    for the ``BaseModelBasedDiffusionSolver`` / multirun path), and
  * the ``WarmStartPlanner`` capability (``init_plan_var/make_schedule/replan/
    first_action/shift``) consumed by the backend-agnostic receding-horizon
    bridge (``solvers/common/receding_horizon.py``).

The reverse update is the control-space MBD weighted-mean (proven identical to
mbd's DDPM score-form, see test_mbd_dial_equivalence) expressed with DIAL's
knobs: spline NODES (Hnode<Hsample), a per-node geometric noise schedule, the
incumbent baseline + node-0 pin, and the env's own reward (no MBD guidance/
terminal shaping). So DIAL == this MBD weighted-mean backend + the bridge.

Reward: the candidate rollout uses ``rollout_fn(state, us) -> rews``. When not
injected (CPU tests inject a mock), it is built from the solver's
``DynamicsToEnvAdapter`` transition + ``LegacyEnergyFunctional`` as
``rew = -energy.compute(s, u, ctx={"t": t})`` (DIAL reward = -energy, the
genedynamics-general reward path; mjx-gated).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.dial.spline import NodeSpline

RolloutFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]  # (state, us, t0)->rews


# ---------------------------------------------------------------------------
# Schedule + reverse-update kernel (pure functions)
# ---------------------------------------------------------------------------

def make_sigma_control(Hnode: int, horizon_diffuse_factor: float, sigma_scale: float) -> jnp.ndarray:
    """Per-node base noise scale (dial-mpc ``MBDPI.sigma_control``)."""
    sc = horizon_diffuse_factor ** jnp.arange(Hnode + 1, dtype=jnp.float32)[::-1]
    return sc * float(sigma_scale)


def make_traj_diffuse_factors(sigma_control: jnp.ndarray, traj_diffuse_factor: float, n_diffuse: int) -> jnp.ndarray:
    """Per-step per-node noise schedule ``(n_diffuse, Hnode+1)`` (geometric anneal)."""
    ks = jnp.arange(n_diffuse, dtype=jnp.float32)[:, None]
    return sigma_control[None, :] * (float(traj_diffuse_factor) ** ks)


def reverse_once(
    state: Any,
    rng: jnp.ndarray,
    Ybar_nodes: jnp.ndarray,        # (Hnode+1, nu)
    noise_scale: jnp.ndarray,       # (Hnode+1,)
    *,
    rollout_fn: RolloutFn,
    N2U: jnp.ndarray,               # (Hsample+1, Hnode+1)
    Nsample: int,
    temp_sample: float,
    action_limit: float,
    t0: Any = 0.0,                  # global step offset (gait phase / time-aware reward)
    update_form: str = "weighted_mean",  # "weighted_mean" (DIAL) | "mbd_score" (mbd DDPM form)
) -> tuple:
    """One DIAL reverse-diffusion step in node space (MBD weighted mean)."""
    nu = Ybar_nodes.shape[-1]
    Hn1 = Ybar_nodes.shape[0]

    rng, y_rng = jax.random.split(rng)
    eps = jax.random.normal(y_rng, (Nsample, Hn1, nu))
    Y0s = eps * noise_scale[None, :, None] + Ybar_nodes[None]
    Y0s = Y0s.at[:, 0].set(Ybar_nodes[0])                       # pin node-0 (committed control)
    Y0s = jnp.concatenate([Y0s, Ybar_nodes[None]], axis=0)      # append incumbent (= baseline)
    Y0s = jnp.clip(Y0s, -action_limit, action_limit)

    us = jnp.einsum("hn,bnu->bhu", N2U, Y0s)                    # node -> dense
    rewss = rollout_fn(state, us, t0)                           # (Nsample+1, Hsample+1)
    rew_incumbent = rewss[-1].mean()
    rews = rewss.mean(axis=-1)

    std = rews.std()
    std = jnp.where(std < 1e-6, 1.0, std)
    logp0 = (rews - rew_incumbent) / (std * temp_sample)
    weights = jax.nn.softmax(logp0)

    Ybar_w = jnp.einsum("n,nij->ij", weights, Y0s)             # MBD weighted mean (no abar rescale)
    if update_form == "mbd_score":
        # mbd_jax.py's DDPM score-form (verbatim shape from test_mbd_dial_equivalence;
        # a reference reimpl -- does NOT import or modify mbd_jax.py). With per-call
        # alpha_k = abar_k this reduces algebraically to Ybar_w (the proven,
        # schedule-independent equivalence), exercised end-to-end to confirm
        # "MBD update + bridge == DIAL".
        abar = 0.7
        score = (-Ybar_nodes + jnp.sqrt(abar) * Ybar_w) / (1.0 - abar)
        Ybar_next = (Ybar_nodes + (1.0 - abar) * score) / jnp.sqrt(abar)
    else:
        Ybar_next = Ybar_w
    info = {"rews": rews, "mean_reward": rews.mean(), "weights_max": weights.max()}
    return Ybar_next, info


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class DialBackendJax:
    """JAX backend: WarmStartPlanner + unified plan() for the DIAL solver."""

    def __init__(
        self,
        solver: Any = None,
        *,
        nu: Optional[int] = None,
        rollout_fn: Optional[RolloutFn] = None,
        step_fn: Optional[Callable[[Any, Any], Any]] = None,
        Hnode: int = 4,
        Hsample: int = 16,
        Nsample: int = 2048,
        temp_sample: float = 0.06,
        horizon_diffuse_factor: float = 0.9,
        traj_diffuse_factor: float = 0.5,
        sigma_scale: float = 1.0,
        action_limit: float = 1.0,
        ctrl_dt: float = 0.02,
        Ndiffuse: int = 2,
        Ndiffuse_init: int = 10,
        seed: int = 0,
        update_form: str = "weighted_mean",
    ) -> None:
        # Pull config from the solver when constructed through _get_backend_impl.
        if solver is not None:
            cfg = solver.config
            nu = nu if nu is not None else int(solver.nu)
            Hnode = int(cfg.get("Hnode", Hnode))
            Hsample = int(cfg.get("Hsample", Hsample))
            Nsample = int(cfg.get("Nsample", Nsample))
            temp_sample = float(cfg.get("temp_sample", temp_sample))
            horizon_diffuse_factor = float(cfg.get("horizon_diffuse_factor", horizon_diffuse_factor))
            traj_diffuse_factor = float(cfg.get("traj_diffuse_factor", traj_diffuse_factor))
            sigma_scale = float(cfg.get("sigma_scale", sigma_scale))
            action_limit = float(cfg.get("action_limit", action_limit))
            ctrl_dt = float(cfg.get("ctrl_dt", ctrl_dt))
            Ndiffuse = int(cfg.get("Ndiffuse", Ndiffuse))
            Ndiffuse_init = int(cfg.get("Ndiffuse_init", Ndiffuse_init))
            update_form = str(cfg.get("update_form", update_form))
            seed = int(getattr(solver, "seed", seed))
            rollout_fn = rollout_fn or getattr(solver, "_rollout_fn", None)
            step_fn = step_fn or getattr(solver, "_step_fn", None)

        if nu is None:
            raise ValueError("DialBackendJax requires nu (action dim).")

        self.nu = int(nu)
        self.Hnode, self.Hsample = int(Hnode), int(Hsample)
        self.Nsample = int(Nsample)
        self.temp_sample = float(temp_sample)
        self.traj_diffuse_factor = float(traj_diffuse_factor)
        self.action_limit = float(action_limit)
        self.Ndiffuse = int(Ndiffuse)
        self.Ndiffuse_init = int(Ndiffuse_init)
        self.seed = int(seed)

        self.spline = NodeSpline.build(self.Hnode, self.Hsample, ctrl_dt)
        self.sigma_control = make_sigma_control(self.Hnode, horizon_diffuse_factor, sigma_scale)

        # rollout/step: injected (tests) or built from the solver env (mjx/brax)
        self._rollout_fn = rollout_fn or (self._build_rollout_from_solver(solver) if solver else None)
        self._step_fn = step_fn or self._build_step_from_solver(solver)
        if self._rollout_fn is None:
            raise ValueError("DialBackendJax needs a rollout_fn (inject one or construct via a solver).")

        self.update_form = str(update_form)
        self._reverse = jax.jit(
            functools.partial(
                reverse_once,
                rollout_fn=self._rollout_fn,
                N2U=self.spline.N2U,
                Nsample=self.Nsample,
                temp_sample=self.temp_sample,
                action_limit=self.action_limit,
                update_form=self.update_form,
            )
        )

    def _build_step_from_solver(self, solver: Any):
        """Real one-step dynamics for the receding-horizon bridge. brax env ->
        env.step (brax State); else the flat-state DynamicsToEnvAdapter."""
        if solver is None:
            return None
        from genedynamics.solvers.common.env_rollout import is_brax_env, build_brax_step
        env = getattr(solver, "dynamics", None)
        if is_brax_env(env):
            return build_brax_step(env)
        adapter = getattr(solver, "_env_adapter", None)
        return adapter.jax_transition if adapter is not None else None

    # --- rollout builder (mjx-gated; CPU tests inject a mock) ---
    def _build_rollout_from_solver(self, solver: Any) -> Optional[RolloutFn]:
        """Build ``rollout_fn(state, us, t0) -> rews``.

        Two modes, both time-aware (gait phase / time-dependent rewards use the
        ABSOLUTE step ``t0 + h``, where t0 is the receding-horizon real step):
        - ENV-REWARD: if the env exposes ``reward(state, action, t)`` (e.g. a
          gait reward needing forward kinematics), use it directly -- matching
          dial-mpc's "reward in the env" convention.
        - ENERGY: otherwise reward = -energy.compute(state, action, {"t": t}).
        """
        # BRAX-STATE path (faithful to dial-mpc MBDPI): if the env is a brax
        # PipelineEnv (dial's UnitreeH1/Go2 envs), roll out env.step over the
        # brax State and read state.reward. Shared layer (env_rollout) so MBD /
        # MPPI / MGA can reuse the same env+reward. Time/gait phase lives in
        # state.info (advanced by env.step), so t0 is unused here.
        from genedynamics.solvers.common.env_rollout import is_brax_env, build_brax_rollout
        env = getattr(solver, "dynamics", None)
        if is_brax_env(env):
            return build_brax_rollout(env)

        env_adapter = getattr(solver, "_env_adapter", None)
        if env_adapter is None:
            return None
        trans = env_adapter.jax_transition
        reward_fn = getattr(getattr(solver, "dynamics", None), "reward", None)
        energy = getattr(solver, "_legacy_energy", None)

        if callable(reward_fn):
            def rollout_one(x0, us, t0):  # us:(H+1,nu)
                ts = jnp.asarray(t0, jnp.float32) + jnp.arange(us.shape[0], dtype=jnp.float32)

                def f(s, u_t):
                    u, t = u_t
                    s2 = trans(s, u)
                    return s2, reward_fn(s2, u, t)

                _, rews = jax.lax.scan(f, x0, (us, ts))
                return rews
        elif energy is not None:
            def rollout_one(x0, us, t0):
                ts = jnp.asarray(t0, jnp.float32) + jnp.arange(us.shape[0], dtype=jnp.float32)

                def f(s, u_t):
                    u, t = u_t
                    s2 = trans(s, u)
                    return s2, -energy.compute(s2, u, {"t": t})

                _, rews = jax.lax.scan(f, x0, (us, ts))
                return rews
        else:
            return None

        return jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None)))

    # --- WarmStartPlanner protocol (consumed by the bridge) ---
    def init_plan_var(self) -> jnp.ndarray:
        return jnp.zeros((self.Hnode + 1, self.nu), dtype=jnp.float32)

    def make_schedule(self, n_diffuse: int) -> jnp.ndarray:
        return make_traj_diffuse_factors(self.sigma_control, self.traj_diffuse_factor, int(n_diffuse))

    def replan(self, state, warm_start, schedule, rng, t0=0.0) -> jnp.ndarray:
        # t0 = receding-horizon real-step offset, threaded into the reward so
        # time-dependent terms (gait phase) advance with execution. Same t0 for
        # all diffusion steps of this replan (they plan from the same state).
        t0 = jnp.asarray(t0, jnp.float32)
        Y = warm_start
        for k in range(schedule.shape[0]):
            rng, sub = jax.random.split(rng)
            Y, _ = self._reverse(state, sub, Y, schedule[k], t0=t0)
        return Y

    def first_action(self, plan_var) -> jnp.ndarray:
        return self.spline.node2u(plan_var)[0]

    def shift(self, plan_var) -> jnp.ndarray:
        return self.spline.shift_nodes(plan_var)

    # --- unified backend interface (single-shot, for the base Solver path) ---
    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        """Single-shot full-horizon DIAL plan (Ndiffuse_init steps from cold)."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        Y = self.init_plan_var()
        Y = self.replan(x0, Y, self.make_schedule(self.Ndiffuse_init), rng_key)
        us = self.spline.node2u(Y)                              # (Hsample+1, nu)
        rews = np.asarray(self._rollout_fn(x0, us[None], 0.0)[0], dtype=np.float32)
        actions = np.asarray(us, dtype=np.float32)
        states = self._rollout_states(x0, us)
        return {
            "actions": actions,
            "states": states,
            "rewards": rews,
            "total_reward": float(np.sum(rews)),
            "mean_reward": float(np.mean(rews)) if rews.size else 0.0,
        }

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        return [self.plan(x0, rng_key=k) for k in keys]

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Optional[Any] = None):
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        keys = jax.random.split(rng_key, max(1, int(n_samples)))
        return self.plan_batch(x0, keys)

    def _rollout_states(self, x0: Any, us: jnp.ndarray) -> np.ndarray:
        if self._step_fn is None:
            return np.asarray(x0, dtype=np.float32)[None]
        def f(s, u):
            s2 = self._step_fn(s, u)
            return s2, s2
        _, states = jax.lax.scan(f, x0, us)
        x0a = jnp.asarray(x0)
        return np.asarray(jnp.concatenate([x0a[None], states], axis=0), dtype=np.float32)
