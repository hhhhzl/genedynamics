from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.energy import EnergyFunctional
from enerdynamics.core.metrics import euclidean_metric_inv
from enerdynamics.core.constraints import project_box
from enerdynamics.core.integrators import langevin_step

from enerdynamics.envs.double_integrator_box import DoubleIntegratorBoxEnv
from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv


# =========================================================
# 1. arguments
# =========================================================
@dataclass
class EDOCArgs:
    # exp
    seed: int = 0
    np_random_seed: Optional[int] = None
    # env
    env_name: str = "double_integrator_box"
    horizon: int = 80
    dt: float = 0.1
    # diffusion / ebdc
    noise_std: float = 0.05
    n_particles: int = 1         
    # optimization domain
    action_space: bool = True
    diffusion_mode: str = "reverse"  # "forward" or "reverse"
    action_diffuse_steps: int = 100
    action_beta0: float = 1e-4
    action_betaT: float = 1e-2
    action_temp: float = 0.1
    action_extra_sigma: float = 0.0
    action_stage_ratio: float = 1.0
    action_score_mode: str = "energy"  # ["reward", "energy", "learned"]
    action_nsample: int = 128
    use_antithetic: bool = True
    dyn_loss_coeff: float = 1.0
    dyn_loss_mode: str = "trajectory"  # ["terminal", "trajectory"]
    # state box
    use_state_box: bool = False
    state_low: float = -2.0
    state_high: float = 2.0
    # render / save
    save_path: Optional[str] = None
    verbose: bool = True


# =========================================================
# 2. env / energy factory
# =========================================================
def make_env(name: str):
    if name == "double_integrator_box":
        return DoubleIntegratorBoxEnv()
    if name == "double_integrator_box_2d":
        return DoubleIntegratorBox2DEnv()
    else:
        raise ValueError(f"unknown env {name}")


def make_energy(env_name: str) -> EnergyFunctional:
    from enerdynamics.core.energy import EnergyTerm

    if env_name == "double_integrator_box":
        def task_energy(x, u, ctx):
            # x: (state_dim,)
            pos = x[0]
            vel = x[1]
            return (pos - 0.0) ** 2 + (vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[0]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[1]) - v_max)
            pen = pos_violate ** 2 + vel_violate ** 2
            return pen

        return EnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    if env_name == "double_integrator_box_2d":
        def task_energy(x, u, ctx):
            pos = x[:2]
            vel = x[2:]
            return jnp.sum((pos - jnp.array([0.0, 0.0], dtype=jnp.float32)) ** 2) + jnp.sum(vel ** 2)

        def box_energy(x, u, ctx):
            p_max, v_max = 2.0, 2.0
            pos_violate = jnp.maximum(0.0, jnp.abs(x[:2]) - p_max)
            vel_violate = jnp.maximum(0.0, jnp.abs(x[2:]) - v_max)
            pen = jnp.sum(pos_violate ** 2 + vel_violate ** 2)
            return pen

        return EnergyFunctional({
            "task": EnergyTerm(task_energy, 1.0),
            "box": EnergyTerm(box_energy, 1.0),
        })
    else:
        raise ValueError(f"no energy factory for env {env_name}")


# =========================================================
# 3. Planner
# =========================================================
class EDOCPlanner:

    def __init__(
        self,
        env,
        energy: EnergyFunctional,
        horizon: int,
        dt: float,
        noise_std: float = 0.05,
        n_particles: int = 1,
        state_box: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        action_space: bool = True,
        diffusion_mode: str = "reverse",
        action_diffuse_steps: int = 100,
        action_beta0: float = 1e-4,
        action_betaT: float = 1e-2,
        action_temp: float = 0.1,
        action_extra_sigma: float = 0.0,
        action_stage_ratio: float = 0.5,
        action_score_mode: str = "reward",
        action_nsample: int = 256,
        use_antithetic: bool = False,
        dyn_loss_coeff: float = 1.0,
        dyn_loss_mode: str = "terminal",
        np_random_seed: Optional[int] = None,
    ):
        if action_score_mode not in {"reward", "energy", "learned"}:
            raise ValueError(f"Unknown action_score_mode {action_score_mode}")
        if dyn_loss_mode not in {"terminal", "trajectory"}:
            raise ValueError(f"Unknown dyn_loss_mode {dyn_loss_mode}")

        self.use_antithetic = bool(use_antithetic)
        self.env = env
        self.energy = energy
        self.horizon = horizon
        self.dt = dt
        self.noise_std = noise_std
        self.n_particles = n_particles
        self.state_box = state_box
        self.action_space = action_space
        self.diffusion_mode = diffusion_mode
        self.action_diffuse_steps = action_diffuse_steps
        self.action_beta0 = action_beta0
        self.action_betaT = action_betaT
        self.action_temp = action_temp
        self.action_extra_sigma = action_extra_sigma
        self.action_stage_ratio = np.clip(action_stage_ratio, 0.0, 1.0)
        self.action_score_mode = action_score_mode
        self.action_nsample = max(1, int(action_nsample))
        self.dyn_loss_coeff = float(max(0.0, dyn_loss_coeff))
        self.dyn_loss_mode = dyn_loss_mode
        self._np_rng = np.random.default_rng(np_random_seed)
        self._rollout_states_energy_fn = None
        self._rollout_env_states_fn = None
        self._rollout_energy_fn = None
        self._batch_rollout_energy_fn = None
        self._reward_mean_fn = None
        self._batch_reward_mean_fn = None
        self._reverse_diffuse_jit = None

        # use JAX to automatically
        def energy_x(x, u, ctx):
            return self.energy.compute(x, u, ctx)

        self.grad_energy_x = jax.jit(jax.grad(energy_x, argnums=0))

        if self.action_space:
            if not hasattr(self.env, "jax_transition"):
                raise ValueError("Environment must provide jax_transition when using action_space diffusion.")
            self.jax_transition = jax.jit(self.env.jax_transition)
            if hasattr(self.env, "jax_model_transition"):
                self.jax_model_transition = jax.jit(self.env.jax_model_transition)
            else:
                self.jax_model_transition = self.jax_transition
            self._time_index = jnp.arange(self.horizon, dtype=jnp.int32)

            energy_fn = self.energy
            transition_fn = self.jax_model_transition
            time_index = self._time_index
            dyn_coeff = self.dyn_loss_coeff
            dyn_mode = self.dyn_loss_mode

            if hasattr(self.env, "jax_env_transition"):
                env_transition_fn = jax.jit(self.env.jax_env_transition)
            else:
                env_transition_fn = transition_fn

            def rollout_states_and_energy(state, actions):
                def body(carry, inputs):
                    s = carry
                    t, act = inputs
                    ctx = {"t": t}
                    e_val = energy_fn.compute(s, act, ctx)
                    s_next = transition_fn(s, act)
                    return s_next, (s_next, e_val)

                _, (states_seq, energy_seq) = jax.lax.scan(
                    body, state, (time_index, actions)
                )
                states_full = jnp.concatenate([state[None, :], states_seq], axis=0)
                return states_full, energy_seq

            def rollout_env_states(state, actions):
                def body(carry, inputs):
                    s = carry
                    _, act = inputs
                    s_next = env_transition_fn(s, act)
                    return s_next, s_next

                _, states_seq = jax.lax.scan(
                    body, state, (time_index, actions)
                )
                states_full = jnp.concatenate([state[None, :], states_seq], axis=0)
                return states_full

            def total_energy(state, actions):
                states_full, energy_seq = rollout_states_and_energy(state, actions)
                total = jnp.sum(energy_seq)
                if dyn_coeff > 0.0:
                    env_states_full = rollout_env_states(state, actions)
                    if dyn_mode == "terminal":
                        diff = states_full[-1] - env_states_full[-1]
                        dyn_loss = jnp.sum(diff * diff)
                    else:
                        dyn_loss = jnp.sum((states_full - env_states_full) ** 2)
                    total = total + dyn_coeff * dyn_loss
                return total

            reward_cost_fn = getattr(self.env, "jax_cost", None)

            if reward_cost_fn is not None:

                def mean_reward(state, actions):
                    def body(carry, inputs):
                        s = carry
                        _, act = inputs
                        s_next = env_transition_fn(s, act)
                        reward = -reward_cost_fn(s_next)
                        return s_next, reward

                    _, reward_seq = jax.lax.scan(
                        body, state, (time_index, actions)
                    )
                    return jnp.mean(reward_seq)

                self._reward_mean_fn = jax.jit(mean_reward)
                self._batch_reward_mean_fn = jax.jit(
                    jax.vmap(self._reward_mean_fn, in_axes=(None, 0))
                )

            self._rollout_states_energy_fn = jax.jit(rollout_states_and_energy)
            self._rollout_env_states_fn = jax.jit(rollout_env_states)
            self._rollout_energy_fn = jax.jit(total_energy)
            self._batch_rollout_energy_fn = jax.jit(
                jax.vmap(self._rollout_energy_fn, in_axes=(None, 0))
            )
            self._init_reverse_diffuse_jit()

    def _init_reverse_diffuse_jit(self):
        self._reverse_diffuse_jit = None
        if not self.action_space:
            return
        mode = self.action_score_mode.lower()
        if mode == "learned":
            return
        if self.use_antithetic:
            return
        if mode == "reward" and self._batch_reward_mean_fn is None:
            return
        if mode == "energy" and self._batch_rollout_energy_fn is None:
            return

        horizon = int(self.horizon)
        if horizon <= 0:
            return
        action_dim = int(self.env.act_dim)
        Ndiffuse = int(self.action_diffuse_steps)
        if Ndiffuse <= 1:
            return
        num_particles = int(self.action_nsample)
        if num_particles <= 0:
            return

        temp = float(self.action_temp if self.action_temp > 1e-6 else 1e-6)
        beta0 = float(self.action_beta0)
        betaT = float(self.action_betaT)
        extra_sigma0 = float(self.action_extra_sigma)
        stage_switch = max(1, int(self.action_stage_ratio * (Ndiffuse - 1)))

        control_limit = getattr(self.env, "control_limit", None)
        clip_actions = control_limit is not None
        if clip_actions:
            control_limit = float(control_limit)

        if mode == "reward":
            score_fn = self._batch_reward_mean_fn
            def transform_scores(values):
                return values
        else:
            score_fn = self._batch_rollout_energy_fn
            def transform_scores(values):
                return -values

        betas = jnp.linspace(beta0, betaT, Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        extra_sigmas = jnp.linspace(extra_sigma0, 0.0, Ndiffuse, dtype=jnp.float32)
        diffusion_indices = jnp.arange(Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        stage_switch_val = jnp.int32(stage_switch)
        temp_val = jnp.float32(temp)
        num_particles_f = jnp.array(num_particles, dtype=jnp.float32)
        uniform_logw = jnp.full(
            (num_particles,),
            -jnp.log(num_particles_f),
            dtype=jnp.float32,
        )

        def reverse_diffuse(rng_key, state_init):
            rng_key, init_key = jax.random.split(rng_key)
            Ybar_init = jax.random.normal(
                init_key, (horizon, action_dim), dtype=jnp.float32
            )

            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_next, noise_key, extra_key = jax.random.split(rng_curr, 3)

                sqrt_alpha_bar_i = jnp.sqrt(alphas_bar[idx])
                sigma_i = sigmas[idx]
                Yi = Ybar_curr * sqrt_alpha_bar_i
                eps = jax.random.normal(
                    noise_key, (num_particles, horizon, action_dim), dtype=jnp.float32
                )
                Y0s = eps * sigma_i + Ybar_curr
                if clip_actions:
                    Y0s = jnp.clip(Y0s, -control_limit, control_limit)

                use_uniform = idx >= stage_switch_val

                def score_branch(actions):
                    raw_scores = score_fn(state_init, actions)
                    scores = transform_scores(raw_scores)
                    score_mean = jnp.mean(scores)
                    score_std = jnp.std(scores)
                    score_std = jnp.where(score_std < 1e-4, 1.0, score_std)
                    logw = (scores - score_mean) / (score_std * temp_val)
                    logw = logw - jnp.max(logw)
                    reward_val = jnp.mean(scores)
                    return logw, reward_val

                logw, reward_val = jax.lax.cond(
                    use_uniform,
                    lambda _: (uniform_logw, jnp.nan),
                    lambda acts: score_branch(acts),
                    Y0s,
                )
                weights = jax.nn.softmax(logw)

                Ybar_weighted = jnp.tensordot(weights, Y0s, axes=([0], [0]))

                one_minus_alpha_bar = 1.0 - alphas_bar[idx]
                score_val = (-Yi + sqrt_alpha_bar_i * Ybar_weighted) / one_minus_alpha_bar
                Yim1 = (Yi + one_minus_alpha_bar * score_val) / jnp.sqrt(alphas[idx])
                sqrt_alpha_bar_prev = jnp.sqrt(alphas_bar[idx - 1])
                Ybar_next = Yim1 / sqrt_alpha_bar_prev

                extra_sigma = extra_sigmas[idx]

                def add_noise(y_in):
                    noise = jax.random.normal(
                        extra_key, (horizon, action_dim), dtype=jnp.float32
                    )
                    return y_in + extra_sigma * noise

                Ybar_next = jax.lax.cond(
                    jnp.abs(extra_sigma) > 0.0,
                    add_noise,
                    lambda y_in: y_in,
                    Ybar_next,
                )

                return (rng_next, Ybar_next), (reward_val, Ybar_next, Y0s)

            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_key, Ybar_init), diffusion_indices
            )
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            if clip_actions:
                Ybar_hist = jnp.clip(Ybar_hist, -control_limit, control_limit)
                Ysamples_hist = jnp.clip(Ysamples_hist, -control_limit, control_limit)
            return rng_out, Ybar_final, reward_hist, Ybar_hist, Ysamples_hist

        self._reverse_diffuse_jit = jax.jit(reverse_diffuse)

    # ------------------- main interface -------------------
    def plan(self, rng: jax.Array) -> Dict[str, Any]:
        try:
            x0, info = self.env.reset(rng)
        except TypeError:
            x0, info = self.env.reset()
        if self.action_space:
            if self.diffusion_mode == "reverse":
                return self._run_action_reverse(x0, info, rng)
            return self._run_action_space(x0, info, rng)
        if self.n_particles == 1:
            return self._run_single(x0, info)
        else:
            return self._run_multi(x0, info)

    # ------------------- single particle -------------------
    def _run_single(self, x, info):
        xs = [x]
        Es = []
        term_list = []
        rewards = []

        for t in range(self.horizon):
            u = jnp.zeros((self.env.act_dim,), dtype=jnp.float32)
            ctx = {"t": t, **info}

            E_val = self.energy.compute(x, u, ctx)
            grad_x = self.grad_energy_x(x, u, ctx)

            G_inv = euclidean_metric_inv(x)
            x_new = langevin_step(x, grad_x, G_inv, self.dt, self.noise_std)

            # constraint projection
            if self.state_box is not None:
                low, high = self.state_box
                x_new = project_box(x_new, low, high)

            # really step the environment (to record cost etc.)
            x_env, cost, done, info_n = self.env.step(
                np.array(x_new, dtype=np.float32),
                np.array(u, dtype=np.float32),
                t,
                info,
            )
            reward = -cost

            xs.append(x_env)
            Es.append(E_val)
            term_list.append(self.energy.breakdown(x, u, ctx))
            rewards.append(reward)

            x, info = x_env, info_n
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(Es, axis=0),
            "terms": term_list,
            "rewards": jnp.stack(rewards, axis=0),
            "initial_state": xs[0],
        }

    # ------------------- A-MCSA (multi-particle) -------------------
    def _run_multi(self, x, info):
        xs = [x]
        Es = []
        term_list = []
        rewards = []

        for t in range(self.horizon):
            u = jnp.zeros((self.env.act_dim,), dtype=jnp.float32)
            ctx = {"t": t, **info}

            # extend particles
            x_particles = jnp.repeat(x[None, :], self.n_particles, axis=0)

            # vmap to compute energy and gradient for all particles
            def e_and_g(xp):
                Ev = self.energy.compute(xp, u, ctx)
                Gv = self.grad_energy_x(xp, u, ctx)
                return Ev, Gv

            E_all, G_all = jax.vmap(e_and_g)(x_particles)
            mean_E = jnp.mean(E_all)
            mean_grad = jnp.mean(G_all, axis=0)

            G_inv = euclidean_metric_inv(x)
            x_new = langevin_step(x, mean_grad, G_inv, self.dt, self.noise_std)

            if self.state_box is not None:
                low, high = self.state_box
                x_new = project_box(x_new, low, high)

            x_env, cost, done, info_n = self.env.step(x_new, u, t, info)
            reward = -cost

            xs.append(x_env)
            Es.append(mean_E)
            term_list.append(self.energy.breakdown(x, u, ctx))
            rewards.append(reward)

            x, info = x_env, info_n
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(Es, axis=0),
            "terms": term_list,
            "rewards": jnp.stack(rewards, axis=0),
            "initial_state": xs[0],
        }

    # ------------------- action sequence diffusion -------------------
    def _run_action_space(self, x0, info, rng):
        act_dim = self.env.act_dim
        actions = jnp.zeros((self.horizon, act_dim), dtype=jnp.float32)
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)

        energy_from_actions = lambda seq: self._rollout_energy_fn(x0_jnp, seq)
        energy_and_grad = jax.jit(jax.value_and_grad(energy_from_actions))

        energy_hist = []
        for _ in range(self.horizon):
            E_val, grad_val = energy_and_grad(actions)
            energy_hist.append(float(E_val))

            actions_np = np.asarray(actions, dtype=np.float32)
            grad_np = np.asarray(grad_val, dtype=np.float32)

            flat_actions = actions_np.reshape(-1)
            flat_grad = grad_np.reshape(-1)
            G_inv = np.eye(flat_actions.size, dtype=np.float32)
            updated_flat = langevin_step(flat_actions, flat_grad, G_inv, self.dt, self.noise_std)

            updated = updated_flat.reshape(self.horizon, act_dim)

            if hasattr(self.env, "control_limit") and self.env.control_limit is not None:
                limit = float(self.env.control_limit)
                updated = np.clip(updated, -limit, limit)

            actions = jnp.asarray(updated, dtype=jnp.float32)

        x = np.array(x0, dtype=np.float32)
        xs = [x]
        rewards = []
        terms = []
        energies = []
        info_curr = info

        actions_np_final = np.asarray(actions, dtype=np.float32)

        for t in range(self.horizon):
            u = actions_np_final[t]
            ctx = {"t": t, **info_curr}
            E_val = self.energy.compute(x, u, ctx)
            energies.append(E_val)
            terms.append(self.energy.breakdown(x, u, ctx))

            x_pred = self.env.transition(x, u)
            x_env, cost, done, info_next = self.env.step(
                np.array(x_pred, dtype=np.float32),
                np.array(u, dtype=np.float32),
                t,
                info_curr,
            )

            rewards.append(-cost)
            xs.append(x_env)

            x = x_env
            info_curr = info_next
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(np.array(energies, dtype=np.float32), axis=0),
            "terms": terms,
            "rewards": jnp.stack(np.array(rewards, dtype=np.float32), axis=0),
            "actions": jnp.asarray(actions_np_final, dtype=jnp.float32),
            "initial_state": xs[0],
            "energy_iterations": jnp.asarray(np.array(energy_hist, dtype=np.float32)),
        }

    def _simulate_rewards_numpy(self, state, actions):
        x = np.array(state, dtype=np.float32)
        rewards = []
        for u in np.asarray(actions, dtype=np.float32):
            x = self.env.transition(x, u)
            rewards.append(-self.env.cost(x))
        return np.array(rewards, dtype=np.float32)

    def _simulate_energy_numpy(self, state, actions, info):
        x = np.array(state, dtype=np.float32)
        info_curr = info
        total_energy = 0.0
        for t, u in enumerate(np.asarray(actions, dtype=np.float32)):
            ctx = {"t": t, **info_curr}
            total_energy += float(self.energy.compute(x, u, ctx))
            x = self.env.transition(x, u)
        return total_energy

    def _score_particles(self, x0, info, batch_actions):
        mode = self.action_score_mode.lower()
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        batch_actions_jnp = jnp.asarray(batch_actions, dtype=jnp.float32)

        if mode == "reward":
            if self._batch_reward_mean_fn is not None:
                rewards_mean = self._batch_reward_mean_fn(x0_jnp, batch_actions_jnp)
                return np.asarray(rewards_mean, dtype=np.float32)
            num = batch_actions.shape[0]
            scores = np.zeros((num,), dtype=np.float32)
            for idx in range(num):
                rewards = self._simulate_rewards_numpy(x0, batch_actions[idx])
                scores[idx] = float(np.mean(rewards))
            return scores
        elif mode == "energy":
            if self._batch_rollout_energy_fn is not None:
                energy_vals = self._batch_rollout_energy_fn(x0_jnp, batch_actions_jnp)
                return -np.asarray(energy_vals, dtype=np.float32)
            num = batch_actions.shape[0]
            scores = np.zeros((num,), dtype=np.float32)
            for idx in range(num):
                energy_val = float(
                    self._rollout_energy_fn(
                        x0_jnp, jnp.asarray(batch_actions[idx], dtype=jnp.float32)
                    )
                )
                scores[idx] = -energy_val
            return scores
        elif mode == "learned":
            return np.zeros((batch_actions.shape[0],), dtype=np.float32)
        raise ValueError(f"Unknown action_score_mode {self.action_score_mode}")

    def _add_extra_noise(self, actions_array, sigma):
        if sigma <= 0.0:
            return actions_array
        noise = self._np_rng.standard_normal(size=actions_array.shape).astype(np.float32)
        return actions_array + sigma * noise

    def _run_action_reverse(self, x0, info, rng):
        act_dim = self.env.act_dim
        horizon = self.horizon
        Ndiffuse = self.action_diffuse_steps
        beta0 = self.action_beta0
        betaT = self.action_betaT
        temp = self.action_temp
        control_limit = getattr(self.env, "control_limit", None)
        num_particles = max(1, int(self.action_nsample))
        temp_eps = temp if temp > 1e-6 else 1e-6

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        reward_history_arr = None
        diffusion_actions_traj_arr = None
        diffusion_samples_traj_arr = None

        if self._reverse_diffuse_jit is not None:
            rng, diffuse_key = jax.random.split(rng)
            _, actions_jnp, reward_hist_jnp, traj_jnp, samples_jnp = self._reverse_diffuse_jit(diffuse_key, x0_jnp)
            actions_np_final = np.asarray(actions_jnp, dtype=np.float32)
            reward_history_arr = jnp.asarray(reward_hist_jnp, dtype=jnp.float32)
            diffusion_actions_traj_arr = jnp.asarray(traj_jnp, dtype=jnp.float32)
            diffusion_samples_traj_arr = jnp.asarray(samples_jnp, dtype=jnp.float32)
        else:
            betas = np.linspace(beta0, betaT, Ndiffuse, dtype=np.float32)
            alphas = 1.0 - betas
            alphas_bar = np.cumprod(alphas, axis=0)
            sigmas = np.sqrt(1.0 - alphas_bar)
            extra_sigmas = np.linspace(self.action_extra_sigma, 0.0, Ndiffuse, dtype=np.float32)

            rng, init_key = jax.random.split(rng)
            Ybar = np.array(
                jax.random.normal(init_key, (horizon, act_dim), dtype=jnp.float32),
                dtype=np.float32,
            )

            reward_history = []
            Ybar_history = []
            Ysamples_history = []
            stage_switch = max(1, int(self.action_stage_ratio * (Ndiffuse - 1)))

            for i in range(Ndiffuse - 1, 0, -1):
                Yi = Ybar * np.sqrt(alphas_bar[i])

                rng, sample_key = jax.random.split(rng)
                sigma_i = float(sigmas[i])
                if self.use_antithetic and num_particles > 0:
                    half = num_particles // 2
                    has_extra = num_particles % 2
                    sample_count = half + has_extra
                    if sample_count > 0:
                        eps_core = np.array(
                            jax.random.normal(
                                sample_key, (sample_count, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                    else:
                        eps_core = np.zeros((0, horizon, act_dim), dtype=np.float32)

                    Y_candidates = []
                    if half > 0:
                        eps_half = eps_core[:half]
                        Y_candidates.append(Ybar + sigma_i * eps_half)
                        Y_candidates.append(Ybar - sigma_i * eps_half)
                    if has_extra:
                        eps_extra = eps_core[-1:]
                        Y_candidates.append(Ybar + sigma_i * eps_extra)

                    if not Y_candidates:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()
                    elif len(Y_candidates) == 1:
                        Y0s = Y_candidates[0]
                    else:
                        Y0s = np.concatenate(Y_candidates, axis=0)
                else:
                    if num_particles > 0:
                        eps = np.array(
                            jax.random.normal(
                                sample_key, (num_particles, horizon, act_dim), dtype=jnp.float32
                            ),
                            dtype=np.float32,
                        )
                        Y0s = Ybar + sigma_i * eps
                    else:
                        Y0s = np.broadcast_to(Ybar, (1, horizon, act_dim)).copy()

                if control_limit is not None:
                    limit = float(control_limit)
                    Y0s = np.clip(Y0s, -limit, limit)

                use_uniform = i >= stage_switch and self.action_score_mode != "learned"
                if use_uniform:
                    weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    reward_history.append(np.nan)
                elif self.action_score_mode == "energy":
                    actions_batch = jnp.asarray(Y0s, dtype=jnp.float32)
                    energy_vals = np.asarray(self._batch_rollout_energy_fn(x0_jnp, actions_batch))
                    energy_mean = float(np.mean(energy_vals))
                    energy_std = float(np.std(energy_vals))
                    if energy_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        denom = energy_std * temp_eps
                        logw = -(energy_vals - energy_mean) / denom
                        logw = logw - np.max(logw)
                        weights = np.exp(logw).astype(np.float64)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(-np.mean(energy_vals)))
                else:
                    if self.action_score_mode == "learned":
                        scores = np.zeros((num_particles,), dtype=np.float32)
                    else:
                        scores = self._score_particles(x0, info, Y0s)

                    score_std = float(scores.std())
                    if score_std < 1e-4:
                        weights = np.full((num_particles,), 1.0 / num_particles, dtype=np.float32)
                    else:
                        score_mean = float(scores.mean())
                        denom = score_std * temp_eps
                        logp0 = (scores - score_mean) / denom
                        logp0 = logp0 - np.max(logp0)
                        weights = np.exp(logp0)
                        weights = weights / np.sum(weights)
                    reward_history.append(float(np.mean(scores)))
                
                weights = np.asarray(weights, dtype=np.float32)

                Ybar_weighted = np.tensordot(weights, Y0s, axes=([0], [0]))

                score = (-Yi + np.sqrt(alphas_bar[i]) * Ybar_weighted) / (1.0 - alphas_bar[i])
                Yim1 = (Yi + (1.0 - alphas_bar[i]) * score) / np.sqrt(alphas[i])
                Ybar = Yim1 / np.sqrt(alphas_bar[i - 1])
                Ybar = self._add_extra_noise(Ybar, extra_sigmas[i])

                if control_limit is not None:
                    limit = float(control_limit)
                    Ybar_history.append(np.clip(Ybar, -limit, limit))
                    Ysamples_history.append(np.clip(Y0s, -limit, limit))
                else:
                    Ybar_history.append(np.array(Ybar, copy=True))
                    Ysamples_history.append(np.array(Y0s, copy=True))

            if control_limit is not None:
                limit = float(control_limit)
                actions_final = np.clip(Ybar, -limit, limit)
            else:
                actions_final = Ybar

            actions_np_final = np.asarray(actions_final, dtype=np.float32)
            reward_history_arr = jnp.asarray(np.array(reward_history[::-1], dtype=np.float32))
            diffusion_actions_traj_arr = jnp.asarray(np.array(Ybar_history[::-1], dtype=np.float32))
            diffusion_samples_traj_arr = jnp.asarray(np.array(Ysamples_history[::-1], dtype=np.float32))

        x = np.array(x0, dtype=np.float32)
        xs = [x]
        rewards = []
        energies = []
        terms = []
        info_curr = info

        for t in range(horizon):
            u = actions_np_final[t]
            ctx = {"t": t, **info_curr}
            E_val = self.energy.compute(x, u, ctx)
            energies.append(E_val)
            terms.append(self.energy.breakdown(x, u, ctx))

            x_pred = self.env.transition(x, u)
            x_env, cost, done, info_next = self.env.step(
                np.array(x_pred, dtype=np.float32),
                np.array(u, dtype=np.float32),
                t,
                info_curr,
            )

            rewards.append(-cost)
            xs.append(x_env)
            x = x_env
            info_curr = info_next
            if done:
                break

        return {
            "states": jnp.stack(xs, axis=0),
            "energies": jnp.stack(np.array(energies, dtype=np.float32), axis=0),
            "terms": terms,
            "rewards": jnp.stack(np.array(rewards, dtype=np.float32), axis=0),
            "actions": jnp.asarray(actions_np_final, dtype=jnp.float32),
            "initial_state": xs[0],
            "reward_history": reward_history_arr if reward_history_arr is not None else jnp.asarray([], dtype=jnp.float32),
            "diffusion_actions_traj": diffusion_actions_traj_arr if diffusion_actions_traj_arr is not None else jnp.asarray([], dtype=jnp.float32),
            "diffusion_sampled_actions": diffusion_samples_traj_arr if diffusion_samples_traj_arr is not None else jnp.asarray([], dtype=jnp.float32),
        }


# =========================================================
# 4. main entry
# =========================================================
def run_edoc(args: EDOCArgs):
    rng = jax.random.PRNGKey(args.seed)
    np_seed = args.np_random_seed if args.np_random_seed is not None else args.seed
    if np_seed is not None:
        np.random.seed(np_seed)

    # env & energy
    env = make_env(args.env_name)
    energy = make_energy(args.env_name)

    state_box = None
    if args.use_state_box:
        low = jnp.array([args.state_low, args.state_low], dtype=jnp.float32)
        high = jnp.array([args.state_high, args.state_high], dtype=jnp.float32)
        state_box = (low, high)

    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=args.horizon,
        dt=args.dt,
        noise_std=args.noise_std,
        n_particles=args.n_particles,
        state_box=state_box,
        action_space=args.action_space,
        diffusion_mode=args.diffusion_mode,
        action_diffuse_steps=args.action_diffuse_steps,
        action_beta0=args.action_beta0,
        action_betaT=args.action_betaT,
        action_temp=args.action_temp,
        action_extra_sigma=args.action_extra_sigma,
        action_stage_ratio=args.action_stage_ratio,
        action_score_mode=args.action_score_mode,
        action_nsample=args.action_nsample,
        use_antithetic=args.use_antithetic,
        dyn_loss_coeff=args.dyn_loss_coeff,
        dyn_loss_mode=args.dyn_loss_mode,
        np_random_seed=np_seed,
    )

    out = planner.plan(rng)

    if args.verbose:
        if not out:
            print("Planner returned no output.")
        else:
            energies = out.get("energies")
            if energies is not None:
                print("energies:", energies)
            print("initial state:", out.get("initial_state"))
            states = out.get("states")
            if states is not None:
                print("final state:", states[-1])
            rewards = out.get("rewards")
            if rewards is not None:
                print("rewards:", rewards)
                print("total reward:", float(jnp.sum(rewards)))
            if "actions" in out:
                print("actions:", out["actions"])
            if "energy_iterations" in out:
                print("energy Iter history:", out["energy_iterations"])
            if "reward_history" in out:
                print("reward history:", out["reward_history"])

    return out


# =========================================================
# 5. CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("EBDC Planner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--np_random_seed", type=int, default=None)
    parser.add_argument("--env_name", type=str, default="double_integrator_box")
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--n_particles", type=int, default=32)
    parser.add_argument("--use_state_box", action="store_true", default=True)
    parser.add_argument("--state_low", type=float, default=-2.0)
    parser.add_argument("--state_high", type=float, default=2.0)
    parser.add_argument("--action_space", action="store_true", default=True)
    parser.add_argument("--diffusion_mode", type=str, default="reverse")
    parser.add_argument("--action_diffuse_steps", type=int, default=100)
    parser.add_argument("--action_beta0", type=float, default=1e-4)
    parser.add_argument("--action_betaT", type=float, default=1e-2)
    parser.add_argument("--action_temp", type=float, default=0.1)
    parser.add_argument("--action_extra_sigma", type=float, default=0.01)
    parser.add_argument("--action_stage_ratio", type=float, default=1.0)
    parser.add_argument("--action_score_mode", type=str, default="reward")
    parser.add_argument("--action_nsample", type=int, default=256)
    parser.add_argument("--no_antithetic", action="store_true", default=False, help="Disable antithetic pairing when sampling action trajectories.")
    parser.add_argument("--dyn_loss_coeff", type=float, default=0.0)
    parser.add_argument("--dyn_loss_mode", type=str, default="terminal", choices=["terminal", "trajectory"])
    parser.add_argument("--verbose", action="store_true", default=True)

    cli_args = parser.parse_args()
    args = EDOCArgs(
        seed=cli_args.seed,
        np_random_seed=cli_args.np_random_seed,
        env_name=cli_args.env_name,
        horizon=cli_args.horizon,
        dt=cli_args.dt,
        noise_std=cli_args.noise_std,
        n_particles=cli_args.n_particles,
        use_state_box=cli_args.use_state_box,
        state_low=cli_args.state_low,
        state_high=cli_args.state_high,
        action_space=cli_args.action_space,
        diffusion_mode=cli_args.diffusion_mode,
        action_diffuse_steps=cli_args.action_diffuse_steps,
        action_beta0=cli_args.action_beta0,
        action_betaT=cli_args.action_betaT,
        action_temp=cli_args.action_temp,
        action_extra_sigma=cli_args.action_extra_sigma,
        action_stage_ratio=cli_args.action_stage_ratio,
        action_score_mode=cli_args.action_score_mode,
        action_nsample=cli_args.action_nsample,
        use_antithetic=not cli_args.no_antithetic,
        dyn_loss_coeff=cli_args.dyn_loss_coeff,
        dyn_loss_mode=cli_args.dyn_loss_mode,
        verbose=cli_args.verbose,
    )

    run_edoc(args)
