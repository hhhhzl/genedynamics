"""
NumPy backend implementation for MDOC.

MDOC = MBD-style diffusion + per-step ConstraintFilter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from genedynamics.core.types import Trajectory, State
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class MDOCBackendNumpy:
    """NumPy implementation of MDOC diffusion with action-space filtering."""

    def __init__(
        self,
        *,
        env_adapter,
        legacy_energy,
        horizon: int,
        dt: float,
        Nsample: int,
        Ndiffuse: int,
        temp_sample: float,
        beta0: float,
        betaT: float,
        action_limit: float,
        seed: int = 0,
        scheduler: Any = None,
        show_tqdm: bool = False,
        constraint_filter: Optional[ConstraintFilter] = None,
        obstacles: Any = None,
        **kwargs: Any,
    ):
        self.env = env_adapter
        self.energy = legacy_energy
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.Nsample = int(Nsample)
        self.Ndiffuse = int(Ndiffuse)
        self.temp_sample = float(temp_sample)
        self.beta0 = float(beta0)
        self.betaT = float(betaT)
        self.action_limit = float(action_limit)
        self.seed = int(seed)
        self.act_dim = int(getattr(self.env, "act_dim", 2))
        self.scheduler = scheduler
        self.show_tqdm = bool(show_tqdm)
        self.obstacles = obstacles
        self.constraint_filter = constraint_filter or NoOpConstraintFilter()

        # CBF-related parameters (kept in sync with JAX backend defaults)
        self.cbf_tau = float(kwargs.get("cbf_tau", 0.005))
        self.cbf_eta = float(kwargs.get("cbf_eta", 1.5))
        self.cbf_margin_fixed = float(kwargs.get("cbf_margin", 0.1))
        self.base_beta = float(kwargs.get("base_beta", 0.05))

        # Guidance/terminal costs
        self.terminal_weight = float(kwargs.get("terminal_energy_weight", 100.0))
        self.guide_weight = float(kwargs.get("guide_weight", 20.0))

        self._rng = np.random.default_rng(self.seed)

        # Precompute constraint schedule arrays
        self._margin_arr = None
        self._rho_arr = None
        self._qp_gate_arr = None
        self._qp_prob_arr = None
        self._topK = -1
        try:
            margin_list: List[float] = []
            rho_list: List[float] = []
            qp_gate_list: List[bool] = []
            qp_prob_list: List[float] = []
            topK_val = None
            if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse - 1))
                        d = cs.constraint_params(st) or {}
                        margin_list.append(float(d.get("margin", 0.0)))
                        rho_list.append(float(d.get("rho", 1.0)))
                        qp_gate_list.append(bool(d.get("qp_gate", True)))
                        qp_prob_list.append(float(d.get("qp_prob", 1.0)))
                        topK_val = d.get("topK", topK_val)
            if margin_list:
                self._margin_arr = np.asarray(margin_list, dtype=np.float32)
                self._rho_arr = np.asarray(rho_list, dtype=np.float32)
                self._qp_gate_arr = np.asarray(qp_gate_list, dtype=np.bool_)
                self._qp_prob_arr = np.asarray(qp_prob_list, dtype=np.float32)
            if topK_val is not None:
                self._topK = int(topK_val)
        except Exception:
            self._margin_arr = None
            self._rho_arr = None
            self._qp_gate_arr = None
            self._qp_prob_arr = None

        # Precompute diffusion temperature schedule (T_k) if scheduler provides it.
        self._T_k_arr = None
        try:
            T_k_list: List[float] = []
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
                    for k in range(self.Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, self.Ndiffuse - 1))
                        d = ds.diffusion_params(st) or {}
                        T_k_list.append(float(d.get("T_k", self.temp_sample)))
            if T_k_list:
                self._T_k_arr = np.asarray(T_k_list, dtype=np.float32)
        except Exception:
            self._T_k_arr = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _step_env(self, s: np.ndarray, a: np.ndarray, t: int) -> np.ndarray:
        """
        Best-effort step wrapper handling different adapter signatures.
        Prefers env.step(s, a, t, info) if available, else falls back.
        """

        def _to_state(x):
            if isinstance(x, (tuple, list)) and len(x) > 0:
                return np.asarray(x[0], dtype=np.float32)
            return np.asarray(x, dtype=np.float32)

        if hasattr(self.env, "step"):
            for args in [(s, a, t, {}), (s, a)]:
                try:
                    return _to_state(self.env.step(*args))
                except TypeError:
                    continue
        if hasattr(self.env, "model_transition"):
            try:
                return _to_state(self.env.model_transition(s, a))
            except TypeError:
                pass
        return _to_state(self.env.jax_transition(s, a))

    def _rollout_states(self, state_init: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out states using the environment adapter."""
        states = [np.asarray(state_init, dtype=np.float32)]
        s = np.asarray(state_init, dtype=np.float32)
        for t, a in enumerate(actions):
            s = self._step_env(s, a, t)
            states.append(s)
        return np.stack(states, axis=0)

    def _rollout_rewards(self, state_init: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out rewards with guidance and terminal shaping."""
        rewards = np.zeros((actions.shape[0],), dtype=np.float32)
        s = np.asarray(state_init, dtype=np.float32)

        target_obj = getattr(self.env, "target", None)
        target = np.asarray(target_obj, dtype=np.float32) if target_obj is not None else np.zeros(2, dtype=np.float32)
        d0 = float(np.linalg.norm(s[0:2] - target[0:2]))
        t_star = float(self.horizon - 1)

        for t, a in enumerate(actions):
            s = self._step_env(s, a, t)
            ctx = {"t": int(t)}

            reward = -float(self.energy.compute(s, a, ctx))
            d_hat = d0 * max(0.0, 1.0 - t / (t_star + 1e-6))
            dist_to_goal = float(np.linalg.norm(s[0:2] - target[0:2]))
            guide_reward = -self.guide_weight * np.square(dist_to_goal - d_hat)

            rewards[t] = reward + guide_reward

        terminal_dist = float(np.linalg.norm(s[0:2] - target[0:2]))
        rewards[-1] += -self.terminal_weight * terminal_dist
        return rewards

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        rng = self._rng if rng_key is None else np.random.default_rng(rng_key)
        x0_np = np.asarray(x0, dtype=np.float32)

        betas = np.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(1.0 - alphas_bar)
        diffusion_indices = np.arange(self.Ndiffuse - 1, 0, -1, dtype=np.int32)
        total_steps = int(self.Ndiffuse - 1)

        Ybar = np.zeros((self.horizon, self.act_dim), dtype=np.float32)

        reward_hist: List[float] = []
        actions_traj: List[np.ndarray] = []
        sampled_traj: List[np.ndarray] = []

        diffusion_iter = diffusion_indices
        if HAS_TQDM and self.show_tqdm:
            diffusion_iter = tqdm(
                diffusion_indices,
                desc="MDOC Diffusion (numpy)",
                unit="step",
                total=len(diffusion_indices),
                leave=False,
            )

        for idx in diffusion_iter:
            eps = rng.normal(size=(self.Nsample, self.horizon, self.act_dim)).astype(np.float32)
            Y0s = eps * sigmas[idx] + Ybar
            if self.action_limit is not None:
                Y0s = np.clip(Y0s, -self.action_limit, self.action_limit)

            step_k = int((self.Ndiffuse - 1) - idx)
            if self._margin_arr is not None:
                kk = int(np.clip(step_k, 0, self._margin_arr.shape[0] - 1))
                margin = float(self._margin_arr[kk])
                rho = float(self._rho_arr[kk])
                qp_gate = bool(self._qp_gate_arr[kk])
                qp_prob = float(self._qp_prob_arr[kk])
            else:
                margin = 0.0
                rho = 1.0
                qp_gate = True
                qp_prob = 1.0

            sched_state = {"k": step_k, "K": total_steps}
            sched_params = {
                "margin": margin,
                "rho": rho,
                "qp_gate": qp_gate,
                "qp_prob": qp_prob,
                "topK": int(self._topK),
                "cbf_tau": float(self.cbf_tau),
                "cbf_eta": float(self.cbf_eta),
                "cbf_margin": float(self.cbf_margin_fixed),
                "base_beta": float(self.base_beta),
            }

            Y0s_f = self.constraint_filter.apply_actions_batch(
                x0_np,
                Y0s,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=sched_state,
                schedule_params=sched_params,
            )
            Y0s_f_np = np.asarray(Y0s_f, dtype=np.float32)

            rews_list: List[float] = []
            for n in range(Y0s_f_np.shape[0]):
                rewards_seq = self._rollout_rewards(x0_np, Y0s_f_np[n])
                rews_list.append(float(np.sum(rewards_seq)))
            rews_np = np.asarray(rews_list, dtype=np.float32)

            rew_mean = float(np.mean(rews_np))
            rew_std = float(np.std(rews_np))
            if rew_std < 1e-4:
                rew_std = 1.0

            T_k = self.temp_sample
            if self._T_k_arr is not None:
                kkT = int(np.clip(step_k, 0, self._T_k_arr.shape[0] - 1))
                T_k = float(self._T_k_arr[kkT])
            temp_eps = T_k if T_k > 1e-6 else 1e-6

            logp0 = (rews_np - rew_mean) / (rew_std * temp_eps)
            logp0 = logp0 - np.max(logp0)
            weights = np.exp(logp0)
            weights_sum = float(np.sum(weights)) + 1e-8
            weights = weights / weights_sum

            Ybar_next = np.einsum("n,nij->ij", weights, Y0s_f_np)
            Ybar_next = self.constraint_filter.apply_actions(
                x0_np,
                Ybar_next,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=sched_state,
                schedule_params=sched_params,
            )
            Ybar = np.asarray(Ybar_next, dtype=np.float32)

            reward_hist.append(rew_mean)
            actions_traj.append(Ybar.copy())
            sampled_traj.append(Y0s_f_np.copy())

        if reward_hist:
            reward_hist = reward_hist[::-1]
        if actions_traj:
            actions_traj = actions_traj[::-1]
        if sampled_traj:
            sampled_traj = sampled_traj[::-1]

        final_actions = Ybar
        if self.action_limit is not None:
            final_actions = np.clip(final_actions, -self.action_limit, self.action_limit)

        states = self._rollout_states(x0_np, final_actions)
        rewards = self._rollout_rewards(x0_np, final_actions)

        return {
            "actions": final_actions.astype(np.float32),
            "states": states.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "initial_state": states[0],
            "diffusion_rewards": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(actions_traj, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(sampled_traj, dtype=np.float32),
        }

    def sample_trajectories(
        self, x0: State, n_samples: int, rng_key: Optional[Any] = None
    ) -> List[Trajectory]:
        result = self.plan(x0, rng_key)
        traj = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in result["states"]],
            actions=[np.asarray(a, dtype=np.float32) for a in result["actions"]],
            info=result,
        )
        return [traj for _ in range(max(1, n_samples))]

