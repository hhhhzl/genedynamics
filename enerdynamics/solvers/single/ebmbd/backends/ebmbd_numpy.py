"""
NumPy backend for EB-MBD (Emerging-Barrier Model-Based Diffusion).

This backend is intended primarily for debugging/validation against EDOC NumPy.
It mirrors the EB-MBD reverse diffusion loop but uses:
  - env.transition (NumPy)
  - energy.compute (Python / NumPy)
  - obstacles.sdf (exact geometric SDF) when available

It is not optimized for paper-level performance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np

try:
    from enerdynamics.core.registry.edoc_backends import register_edoc_backend
except ImportError:  # pragma: no cover
    def register_edoc_backend(name):
        def deco(cls):
            return cls
        return deco


@register_edoc_backend("ebmbd_numpy")
class EBMBDBackendNumpy:
    """EB-MBD NumPy backend."""

    def __init__(self, solver: Any):
        self.env = solver._env_adapter
        self.energy = solver._legacy_energy
        self.horizon = int(solver.horizon)
        self.Nsample = int(solver.Nsample)
        self.Ndiffuse = int(solver.Ndiffuse)
        self.temp = float(solver.temp_sample)
        self.beta0 = float(solver.beta0)
        self.betaT = float(solver.betaT)
        self.action_limit = float(solver.action_limit)
        self.action_extra_sigma = float(getattr(solver, "action_extra_sigma", 0.0))
        self.mu = float(solver.mu)
        self.alpha = float(solver.alpha)
        self.bound = float(solver.bound)
        self.use_min_over_time = bool(solver.use_min_over_time)
        self.terminal_energy_weight = float(getattr(solver, "terminal_energy_weight", 0.0))

        # Obstacles (exact SDF) and config
        self.obstacles = getattr(solver, "_obstacles", None)
        self.obstacle_config = getattr(solver, "_obstacle_config", {}) or {}
        self.robot_radius = float(self.obstacle_config.get("robot_radius", 0.05))

        # RNG (deterministic given solver.seed)
        self._np_rng = np.random.default_rng(int(getattr(solver, "seed", 0)))
        self.scheduler = getattr(solver, "scheduler", None)
        self.show_tqdm = bool(getattr(solver, "show_tqdm", False))

        # Allow constraint scheduler override (emerging_barrier)
        if self.scheduler is not None and hasattr(self.scheduler, "constraint_schedulers"):
            try:
                cs_list = getattr(self.scheduler, "constraint_schedulers", [])
                if cs_list:
                    cs = cs_list[0]
                    try:
                        from enerdynamics.core.constraints.core.types import ScheduleState
                        params_cs = cs.constraint_params(ScheduleState(k=0, K=max(self.Ndiffuse - 1, 1))) or {}
                    except Exception:
                        params_cs = cs.constraint_params(None) if hasattr(cs, "constraint_params") else {}
                    self.mu = float(params_cs.get("mu", self.mu))
                    self.alpha = float(params_cs.get("alpha", self.alpha))
                    self.bound = float(params_cs.get("bound", self.bound))
                    self.use_min_over_time = bool(params_cs.get("use_min_over_time", self.use_min_over_time))
                    self.terminal_energy_weight = float(
                        params_cs.get("terminal_energy_weight", self.terminal_energy_weight)
                    )
            except Exception:
                pass

        # Validate env adapter
        if not hasattr(self.env, "transition"):
            raise ValueError("EB-MBD NumPy backend requires env.transition (NumPy transition).")

    # --------------------------------------------------------------------- SDF
    def _box_sdf(self, pos: np.ndarray) -> np.ndarray:
        """
        Box boundary SDF used by box environments (positive inside).
        pos: (..., 2)
        """
        # Try to infer p_max from wrapped env/dynamics
        p_max = None
        dyn = getattr(self.env, "dynamics", None)
        if dyn is not None and hasattr(dyn, "p_max"):
            p_max = float(dyn.p_max)
        elif hasattr(self.env, "p_max"):
            p_max = float(self.env.p_max)
        else:
            p_max = 2.0
        pos = np.asarray(pos, dtype=np.float32)
        dist = p_max - np.abs(pos)
        return np.min(dist, axis=-1)

    def _obstacles_sdf(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Exact obstacle SDF using ObstacleManager.sdf.
        Returns (...,) with sdf > 0 outside, < 0 inside.
        """
        if self.obstacles is None or not hasattr(self.obstacles, "sdf"):
            return None
        pos2 = np.asarray(pos, dtype=np.float32)
        shp = pos2.shape[:-1]
        flat = pos2.reshape(-1, 2)
        sdf = self.obstacles.sdf(flat)
        sdf = np.asarray(sdf, dtype=np.float32).reshape(shp)
        # Account for robot radius (clearance): sdf - robot_radius
        return sdf - self.robot_radius

    def _compute_min_sdf_one(self, states: np.ndarray) -> float:
        """
        states: (H+1, state_dim)
        """
        pos = np.asarray(states[..., :2], dtype=np.float32)  # (H+1,2)
        sdf_terms: List[np.ndarray] = []
        sdf_terms.append(self._box_sdf(pos))
        obs_sdf = self._obstacles_sdf(pos)
        if obs_sdf is not None:
            sdf_terms.append(obs_sdf)
        sdf = sdf_terms[0] if len(sdf_terms) == 1 else np.minimum.reduce(sdf_terms)
        if self.use_min_over_time:
            return float(np.min(sdf))
        return float(sdf[-1])

    # ---------------------------------------------------------------- rollout
    def _rollout_states(self, state_init: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.asarray(state_init, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        states = np.zeros((actions.shape[0] + 1, x.shape[0]), dtype=np.float32)
        states[0] = x
        for t in range(actions.shape[0]):
            x = self.env.transition(x, actions[t])
            states[t + 1] = x
        return states

    def _rollout_rewards(self, state_init: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.asarray(state_init, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.zeros((actions.shape[0],), dtype=np.float32)
        for t in range(actions.shape[0]):
            x = self.env.transition(x, actions[t])
            ctx = {"t": int(t)}
            c = float(self.energy.compute(x, actions[t], ctx))
            rewards[t] = -c
        return rewards

    def _rollout_total_cost(self, state_init: np.ndarray, actions: np.ndarray) -> Tuple[float, float]:
        """
        Paper-aligned total cost:
          - Stage costs for actions[:-1]
          - Last action only advances to terminal state
          - Terminal cost weighted by terminal_energy_weight
        Returns: (total_cost, rews_mean_stage)
        """
        x = np.asarray(state_init, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        H = actions.shape[0]
        total_stage = 0.0

        if H > 1:
            for t in range(H - 1):
                x = self.env.transition(x, actions[t])
                total_stage += float(self.energy.compute(x, actions[t], {"t": t}))
            # advance with last action for terminal state
            x = self.env.transition(x, actions[-1])
            rews_mean_stage = -total_stage / float(H - 1)
        else:
            # Only one action: no stage cost, only terminal
            x = self.env.transition(x, actions[0])
            rews_mean_stage = 0.0

        total = total_stage
        if self.terminal_energy_weight > 0.0:
            u0 = np.zeros((self.env.act_dim,), dtype=np.float32)
            total += float(self.terminal_energy_weight) * float(self.energy.compute(x, u0, {"t": H}))
        return float(total), float(rews_mean_stage)

    # ---------------------------------------------------------------- diffusion
    def reverse_diffuse(self, rng_key: Any, state_init: np.ndarray) -> Dict[str, Any]:
        # rng_key is ignored for NumPy backend; we keep deterministic np_rng
        horizon = self.horizon
        act_dim = int(self.env.act_dim)
        Ndiffuse = self.Ndiffuse

        # Allow diffusion scheduler override M_k / T_k / betas / Ndiffuse if present
        local_Nsample = self.Nsample
        local_temp = self.temp
        try:
            from enerdynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule

            beta0 = self.beta0
            betaT = self.betaT
            Ndiffuse_override = Ndiffuse
            ds = None
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
            if ds is not None:
                try:
                    params = ds.diffusion_params(None) or {}
                    if "M_k" in params:
                        local_Nsample = int(params["M_k"])
                    if "T_k" in params:
                        local_temp = float(params["T_k"])
                    beta0 = float(params.get("beta0", beta0))
                    betaT = float(params.get("betaT", betaT))
                    Ndiffuse_override = int(params.get("Ndiffuse", Ndiffuse_override))
                except Exception:
                    pass

            betas = DiffusionNoiseSchedule.from_betas(
                np.linspace(beta0, betaT, Ndiffuse_override, dtype=np.float32)
            ).betas.astype(np.float32)
            Ndiffuse = int(len(betas))
        except Exception:
            betas = np.linspace(self.beta0, self.betaT, Ndiffuse, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(1.0 - alphas_bar)

        # idx semantics: idx=N-1 is initial (most noisy), idx=0 is final (least noisy)
        denom = max(float(Ndiffuse - 1), 1.0)
        progress_inc_by_idx = 1.0 - (np.arange(Ndiffuse, dtype=np.float32) / denom)  # idx high->0, idx low->1
        offset_by_idx = self.bound * (1.0 - (progress_inc_by_idx ** self.alpha))     # idx high->bound, idx low->0
        extra_sigmas_by_idx = self.action_extra_sigma * (1.0 - progress_inc_by_idx)  # idx high->extra, idx low->0

        Ybar = np.zeros((horizon, act_dim), dtype=np.float32)
        temp_eps = max(local_temp, 1e-6)

        reward_hist = []
        Ybar_hist = []
        Ysamples_hist = []

        # Reverse diffusion loop: N-1 -> 1
        for idx in range(Ndiffuse - 1, 0, -1):
            sqrt_alpha_bar = float(np.sqrt(alphas_bar[idx]))
            Yi = Ybar * sqrt_alpha_bar
            sigma_i = float(sigmas[idx])
            eps = self._np_rng.standard_normal(size=(local_Nsample, horizon, act_dim)).astype(np.float32)
            Y0s = Ybar[None, :, :] + sigma_i * eps
            Y0s = np.clip(Y0s, -self.action_limit, self.action_limit)

            # Score samples
            total_costs = np.zeros((local_Nsample,), dtype=np.float32)
            rews_mean = np.zeros((local_Nsample,), dtype=np.float32)

            for m in range(local_Nsample):
                acts = Y0s[m]
                # rollout once (states for SDF; cost for task objective)
                states = self._rollout_states(state_init, acts)
                min_sdf = self._compute_min_sdf_one(states)

                # barrier (non-negative; only active near boundary; infeasible -> inf)
                z_raw = float(min_sdf + float(offset_by_idx[idx]))
                if z_raw <= 0.0:
                    barrier = float("inf")
                else:
                    z = max(z_raw, 1e-6)
                    z = min(z, 1.0)
                    barrier = -self.mu * float(np.log(z))

                task_cost, rew_mean_stage = self._rollout_total_cost(state_init, acts)
                total = task_cost + barrier
                total_costs[m] = float(total)

                # for logging only (stage-only mean reward)
                rews_mean[m] = float(rew_mean_stage)

            scores = -total_costs
            scores = np.where(np.isfinite(scores), scores, -1e9).astype(np.float32)

            # Direct softmax on negative cost (energy mode): logw = scores / temp
            logw = scores / temp_eps
            logw = logw - float(np.max(logw))
            w = np.exp(logw).astype(np.float32)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 1e-12:
                w = np.full((local_Nsample,), 1.0 / float(local_Nsample), dtype=np.float32)
            else:
                w = w / w_sum

            Ybar_weighted = np.tensordot(w, Y0s, axes=([0], [0])).astype(np.float32)  # (H,act_dim)

            one_minus_alpha_bar = float(1.0 - alphas_bar[idx])
            score_val = (-Yi + sqrt_alpha_bar * Ybar_weighted) / max(one_minus_alpha_bar, 1e-12)
            Yim1 = (Yi + one_minus_alpha_bar * score_val) / float(np.sqrt(alphas[idx]))
            Ybar_next = Yim1 / float(np.sqrt(alphas_bar[idx - 1]))

            # extra noise (early large -> late 0)
            extra_sigma = float(extra_sigmas_by_idx[idx])
            if abs(extra_sigma) > 0.0:
                noise = self._np_rng.standard_normal(size=(horizon, act_dim)).astype(np.float32)
                Ybar_next = Ybar_next + extra_sigma * noise

            Ybar_next = np.clip(Ybar_next, -self.action_limit, self.action_limit)

            reward_hist.append(float(np.mean(rews_mean)))
            Ybar_hist.append(Ybar_next.copy())
            Ysamples_hist.append(Y0s.copy())
            Ybar = Ybar_next

        # Match visualization convention: index 0 = final, last = initial
        reward_hist = reward_hist[::-1]
        Ybar_hist = Ybar_hist[::-1]
        Ysamples_hist = Ysamples_hist[::-1]

        final_states = self._rollout_states(state_init, Ybar)
        rewards_final = self._rollout_rewards(state_init, Ybar)
        energies_final = -rewards_final

        return {
            "states": np.asarray(final_states, dtype=np.float32),
            "actions": np.asarray(Ybar, dtype=np.float32),
            "rewards": np.asarray(rewards_final, dtype=np.float32),
            "energies": np.asarray(energies_final, dtype=np.float32),
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(Ybar_hist, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(Ysamples_hist, dtype=np.float32),
            "scheduler_params_history": [],
            "rng": rng_key,
        }


