"""
NumPy backend for CEM.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from enerdynamics.solvers.single.cem.backend_impl import CEMBackendBase
from enerdynamics.core.types import Trajectory
from enerdynamics.core.constraints.core.registry import register


@register("cem_backend", "numpy", "numpy")
class CEMBackendNumpy(CEMBackendBase):
    """Pure NumPy implementation of CEM."""

    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed if rng_key is None else int(np.asarray(rng_key).sum()))
        x0_np = np.asarray(x0, dtype=np.float32)
        act_dim = self.env.act_dim

        elite_count = max(1, int(self.num_samples * self.elite_frac))
        mean = np.zeros((self.horizon, act_dim), dtype=np.float32)
        std = np.full((self.horizon, act_dim), float(self.init_std), dtype=np.float32)
        min_std_val = float(self.min_std)
        best_actions = mean.copy()
        best_return = -np.inf

        for _ in range(self.num_iterations):
            samples = rng.normal(size=(self.num_samples, self.horizon, act_dim)).astype(np.float32) * std[None, :, :] + mean[None, :, :]
            if self.action_limit is not None:
                samples = np.clip(samples, -self.action_limit, self.action_limit)

            total_returns = []
            rewards_seq = []
            for s in samples:
                st = x0_np
                rewards = []
                for a in s:
                    st = np.asarray(self.env.transition(st, a), dtype=np.float32)
                    ctx = {"t": 0}
                    rewards.append(-float(self.energy.compute(st, a, ctx)))
                rewards_seq.append(rewards)
                total_returns.append(np.sum(rewards))
            total_returns = np.asarray(total_returns, dtype=np.float32)

            best_idx = int(np.argmax(total_returns))
            if total_returns[best_idx] > best_return:
                best_return = float(total_returns[best_idx])
                best_actions = samples[best_idx]

            elite_idx = np.argsort(total_returns)[-elite_count:]
            elites = samples[elite_idx]
            mean = np.mean(elites, axis=0).astype(np.float32)
            std = np.clip(np.std(elites, axis=0), min_std_val, None).astype(np.float32)

        if self.action_limit is not None:
            best_actions = np.clip(best_actions, -self.action_limit, self.action_limit)

        # Rollout final trajectory
        st = x0_np
        states = [st]
        rewards = []
        for a in best_actions:
            st = np.asarray(self.env.transition(st, a), dtype=np.float32)
            ctx = {"t": 0}
            rewards.append(-float(self.energy.compute(st, a, ctx)))
            states.append(st)

        states = np.asarray(states, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        energies = -rewards  # per-step stage energy = cost

        return {
            "actions": np.asarray(best_actions, dtype=np.float32),
            "states": states,
            "rewards": rewards,
            "energies": energies,
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        }

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Optional[Any] = None
    ) -> List[Trajectory]:
        res = self.plan(x0, rng_key)
        states_list = [np.asarray(s, dtype=np.float32) for s in res["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in res["actions"]]
        traj = Trajectory(states=states_list, actions=actions_list, info=res)
        return [traj for _ in range(max(1, n_samples))]

