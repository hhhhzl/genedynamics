"""
NumPy backend implementation for CFS-MBD.

CFS-MBD = MBD-style diffusion + Augmented Lagrangian objective + CFS-based per-step QP projection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from genedynamics.core.types import Trajectory, State
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.action_filters import ConstraintFilter, NoOpConstraintFilter


class CFSMBDBackendNumpy:
    """NumPy implementation of CFS-MBD diffusion with augmented Lagrangian and CFS-based projection."""

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
        aug_lambda: float = 0.0,
        aug_rho: float = 1.0,
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
        self.aug_lambda = float(aug_lambda)
        self.aug_rho = float(aug_rho)

        self._rng = np.random.default_rng(self.seed)

        # Precompute constraint schedule arrays (same as MDOC)
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
        """Best-effort step wrapper handling different adapter signatures."""
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

    def _rollout_rewards_with_augmented(
        self, state_init: np.ndarray, actions: np.ndarray, step_k: int = 0
    ) -> Tuple[np.ndarray, float]:
        """
        Roll out rewards with augmented Lagrangian penalty.
        
        Returns:
            (rewards_per_step, total_augmented_reward)
        """
        rewards = np.zeros((actions.shape[0],), dtype=np.float32)
        constraint_violations: List[float] = []  # Track [g]_+ for augmented objective
        
        s = np.asarray(state_init, dtype=np.float32)
        states = [s]
        
        # Get clearance/margin for constraint evaluation
        clearance = 0.0
        if self._margin_arr is not None:
            kk = int(np.clip(step_k, 0, self._margin_arr.shape[0] - 1))
            clearance = float(self._margin_arr[kk])
        
        # Rollout loop
        for t, a in enumerate(actions):
            s = self._step_env(s, a, t)
            states.append(s)
            
            # Base reward (negative cost)
            ctx = {"t": int(t)}
            reward = -float(self.energy.compute(s, a, ctx))
            
            # Constraint violation (SDF-based): g_t = clearance - sdf(pos)
            # g_t <= 0 means feasible, so [g_t]_+ = max(0, g_t) is violation
            if self.obstacles is not None:
                pos = s[0:2]  # single_2d: state is position [px, py]
                try:
                    sdf = self.obstacles.sdf(pos)
                    if isinstance(sdf, np.ndarray):
                        sdf = float(sdf[0]) if sdf.size > 0 else float(sdf)
                    else:
                        sdf = float(sdf)
                    g_t = clearance - sdf  # g_t <= 0 means feasible
                    g_plus = max(0.0, g_t)  # [g_t]_+
                except Exception:
                    g_plus = 0.0
            else:
                g_plus = 0.0
            
            constraint_violations.append(g_plus)
            rewards[t] = reward
        
        # Augmented Lagrangian penalty
        # J(τ) + λ^T [g]_+ + (ρ/2) ||[g]_+||^2
        g_plus_array = np.asarray(constraint_violations, dtype=np.float32)
        dual_term = self.aug_lambda * np.sum(g_plus_array)
        penalty_term = (self.aug_rho / 2.0) * np.sum(g_plus_array ** 2)
        augmented_penalty = dual_term + penalty_term
        
        # Total reward = sum of step rewards - augmented penalty
        # (since we maximize reward, penalty is subtracted)
        total_reward = float(np.sum(rewards)) - augmented_penalty
        
        return rewards, total_reward

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
                desc="CFS-MBD Diffusion (numpy)",
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
            }

            # Apply CFS-based filter per diffusion step
            Y0s_f = self.constraint_filter.apply_actions_batch(
                x0_np,
                Y0s,
                env=self.env,
                obstacles=self.obstacles,
                schedule_state=sched_state,
                schedule_params=sched_params,
            )
            Y0s_f_np = np.asarray(Y0s_f, dtype=np.float32)

            # Compute rewards with augmented Lagrangian
            rews_list: List[float] = []
            for n in range(Y0s_f_np.shape[0]):
                _, total_rew = self._rollout_rewards_with_augmented(x0_np, Y0s_f_np[n], step_k=step_k)
                rews_list.append(total_rew)
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
            
            # Filter the mean too
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
        rewards, _ = self._rollout_rewards_with_augmented(x0_np, final_actions, step_k=0)

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
