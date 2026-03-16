"""
NumPy backend implementation for MBD.
"""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from genedynamics.core.types import Trajectory, State
from genedynamics.core.task_spec import legacy_extract_position


class MBDBackendNumpy:
    """NumPy implementation of multi-scale barrier diffusion."""

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
        terminal_energy_weight: float = 100.0,
        position_extractor=None,
        position_dim: int = 2,
    ):
        self.env = env_adapter
        self.position_extractor = position_extractor or legacy_extract_position
        self.position_dim = int(position_dim)
        self.energy = legacy_energy
        self.terminal_energy_weight = float(terminal_energy_weight)
        self.horizon = horizon
        self.dt = dt
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = beta0
        self.betaT = betaT
        self.action_limit = action_limit
        self.seed = seed
        self.act_dim = self.env.act_dim

        self._rng = np.random.default_rng(self.seed)
        self.show_tqdm = bool(show_tqdm)
        self.scheduler = scheduler

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _step_env(self, s: np.ndarray, a: np.ndarray, t: int):
        """
        Best-effort step wrapper handling different adapter signatures.
        Prefers env.step(s, a, t, info) if available, else falls back.
        """
        def _to_state(x):
            # If step returns (state, info) or list/tuple, take first element
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
        # Last resort: jax_transition
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
        """Roll out rewards (negative cost): stage cost + terminal cost."""
        rewards = np.zeros((actions.shape[0],), dtype=np.float32)
        s = np.asarray(state_init, dtype=np.float32)
        for t, a in enumerate(actions):
            s = self._step_env(s, a, t)
            ctx = {"t": int(t)}
            rewards[t] = -float(self.energy.compute(s, a, ctx))
        # Terminal cost: -terminal_weight * dist(final_state, target)
        t = getattr(self.env, "target", None)
        if t is None:
            target_pos = np.zeros(self.position_dim, dtype=np.float32)
        else:
            target_pos = np.asarray(self.position_extractor(t), dtype=np.float32).reshape(-1)[: self.position_dim]
        final_pos = np.asarray(self.position_extractor(s), dtype=np.float32).reshape(-1)[: self.position_dim]
        terminal_dist = float(np.linalg.norm(final_pos - target_pos))
        terminal_reward = -self.terminal_energy_weight * terminal_dist
        rewards[-1] += terminal_reward
        return rewards

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self, x0: State, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        rng = self._rng if rng_key is None else np.random.default_rng(rng_key)

        try:
            from genedynamics.core.constraints.schedulers.utils import DiffusionNoiseSchedule

            # Allow scheduler to override beta0/betaT/Ndiffuse if provided
            beta0 = self.beta0
            betaT = self.betaT
            Ndiffuse = self.Ndiffuse

            # If diffusion scheduler is available, read overrides
            diffusion_scheduler = None
            if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
                try:
                    ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                    if ds_list:
                        diffusion_scheduler = ds_list[0]
                        ds_params = diffusion_scheduler.diffusion_params(None) or {}
                        beta0 = float(ds_params.get("beta0", beta0))
                        betaT = float(ds_params.get("betaT", betaT))
                        Ndiffuse = int(ds_params.get("Ndiffuse", Ndiffuse))
                except Exception:
                    diffusion_scheduler = None

            betas_np = np.linspace(beta0, betaT, Ndiffuse, dtype=np.float32)
            betas = DiffusionNoiseSchedule.from_betas(betas_np).betas.astype(np.float32)
            # Update local Ndiffuse if overridden
            self.Ndiffuse = int(len(betas))
        except Exception:
            betas = np.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(1.0 - alphas_bar)
        diffusion_indices = np.arange(self.Ndiffuse - 1, 0, -1, dtype=np.int32)
        total_steps = self.Ndiffuse - 1

        Ybar = np.zeros((self.horizon, self.act_dim), dtype=np.float32)

        reward_hist = []
        actions_traj = []
        sampled_traj = []

        # Diffusion scheduler (e.g., fixed_diffusion numpy backend)
        diffusion_scheduler = None
        if self.scheduler is not None and hasattr(self.scheduler, "diffusion_schedulers"):
            try:
                ds_list = getattr(self.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    diffusion_scheduler = ds_list[0]
            except Exception:
                diffusion_scheduler = None

        def _get_diffusion_params(step_k: int) -> Dict[str, Any]:
            if diffusion_scheduler is None:
                return {"M_k": self.Nsample, "T_k": self.temp_sample}
            try:
                from genedynamics.core.constraints.core.types import ScheduleState

                state = ScheduleState(k=step_k, K=total_steps)
                params = diffusion_scheduler.diffusion_params(state)
                return params or {}
            except Exception:
                return {"M_k": self.Nsample, "T_k": self.temp_sample}

        diffusion_iter = diffusion_indices
        if HAS_TQDM and getattr(self, "show_tqdm", False):
            diffusion_iter = tqdm(diffusion_indices, desc="MBD Diffusion (numpy)", unit="step",
                                  total=len(diffusion_indices), leave=False)

        for idx in diffusion_iter:
            Yi = Ybar * np.sqrt(alphas_bar[idx])
            eps = rng.normal(size=(self.Nsample, self.horizon, self.act_dim)).astype(np.float32)
            # Per-step diffusion params
            step_k = int((self.Ndiffuse - 1) - idx)
            diff_params = _get_diffusion_params(step_k)
            M_k = int(diff_params.get("M_k", self.Nsample))
            T_k = float(diff_params.get("T_k", self.temp_sample))
            num_particles_current = max(1, M_k)
            temp_eps = T_k if T_k > 1e-6 else 1e-6

            if num_particles_current != self.Nsample:
                eps = rng.normal(size=(num_particles_current, self.horizon, self.act_dim)).astype(np.float32)

            Y0s = eps * sigmas[idx] + Ybar
            if self.action_limit is not None:
                Y0s = np.clip(Y0s, -self.action_limit, self.action_limit)

            # Score particles
            rews_mean = []
            for n in range(num_particles_current):
                rewards = self._rollout_rewards(x0, Y0s[n])
                rews_mean.append(np.mean(rewards))
            rews_mean = np.asarray(rews_mean, dtype=np.float32)

            rew_mean = float(np.mean(rews_mean))
            rew_std = float(np.std(rews_mean))
            if rew_std < 1e-4:
                rew_std = 1.0

            logp0 = (rews_mean - rew_mean) / (rew_std * temp_eps)
            logp0 = logp0 - np.max(logp0)
            weights = np.exp(logp0)
            weights_sum = np.sum(weights) + 1e-8
            weights = weights / weights_sum

            Ybar_weighted = np.einsum("n,nij->ij", weights, Y0s)

            score = (-Yi + np.sqrt(alphas_bar[idx]) * Ybar_weighted) / (1.0 - alphas_bar[idx])
            Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / np.sqrt(alphas[idx])
            Ybar = Yim1 / np.sqrt(alphas_bar[idx - 1])

            reward_hist.append(np.mean(rews_mean))
            actions_traj.append(Ybar.copy())
            sampled_traj.append(Y0s.copy())

        # Reverse histories so index 0 = final (least noisy), matching visualization convention
        if len(reward_hist) > 0:
            reward_hist = reward_hist[::-1]
        if len(actions_traj) > 0:
            actions_traj = actions_traj[::-1]
        if len(sampled_traj) > 0:
            sampled_traj = sampled_traj[::-1]

        final_actions = Ybar
        if self.action_limit is not None:
            final_actions = np.clip(final_actions, -self.action_limit, self.action_limit)

        states = self._rollout_states(x0, final_actions)
        rewards = self._rollout_rewards(x0, final_actions)

        energy_vals = []
        for t in range(final_actions.shape[0]):
            ctx = {"t": int(t)}
            energy_vals.append(float(self.energy.compute(states[t], final_actions[t], ctx)))
        energies_np = np.asarray(energy_vals, dtype=np.float32)

        return {
            "actions": final_actions.astype(np.float32),
            "states": states.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
            "initial_state": states[0],
            "energies": energies_np,
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(actions_traj, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(sampled_traj, dtype=np.float32),
        }

    def sample_trajectories(
        self,
        x0: State,
        n_samples: int,
        rng_key: Optional[Any] = None,
    ) -> List[Trajectory]:
        result = self.plan(x0, rng_key)
        traj = Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in result["states"]],
            actions=[np.asarray(a, dtype=np.float32) for a in result["actions"]],
            info=result,
        )
        return [traj for _ in range(max(1, n_samples))]

