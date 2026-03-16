"""
NumPy backend for MPPI.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from genedynamics.solvers.single.mppi.backend_impl import MPPIBackendBase
from genedynamics.core.types import Trajectory
from genedynamics.core.constraints.core.registry import register


@register("mppi_backend", "numpy", "numpy")
class MPPIBackendNumpy(MPPIBackendBase):
    """Pure NumPy implementation of MPPI."""

    def plan(self, x0: Any, rng_key: Optional[Any] = None) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed if rng_key is None else int(np.asarray(rng_key).sum()))
        x0_np = np.asarray(x0, dtype=np.float32)
        act_dim = self.env.act_dim

        mean_actions = np.zeros((self.horizon, act_dim), dtype=np.float32)
        best_actions = mean_actions.copy()
        best_return = -np.inf

        for _ in range(self.num_iterations):
            noise = rng.normal(size=(self.num_samples, self.horizon, act_dim)).astype(np.float32) * self.noise_sigma
            candidates = mean_actions[None, :, :] + noise
            if self.action_limit is not None:
                candidates = np.clip(candidates, -self.action_limit, self.action_limit)

            # rollout rewards
            rewards_seq = []
            for c in candidates:
                s = x0_np
                rewards = []
                for a in c:
                    s = np.asarray(self.env.transition(s, a), dtype=np.float32)
                    ctx = {"t": 0}
                    rewards.append(-float(self.energy.compute(s, a, ctx)))
                rewards_seq.append(rewards)
            rewards_seq = np.asarray(rewards_seq, dtype=np.float32)
            total_returns = np.sum(rewards_seq, axis=1)

            best_idx = int(np.argmax(total_returns))
            if total_returns[best_idx] > best_return:
                best_return = float(total_returns[best_idx])
                best_actions = candidates[best_idx]

            costs = -total_returns
            beta = np.min(costs)
            weights = np.exp(-(costs - beta) / max(self.lambda_, 1e-6))
            weights_sum = np.sum(weights) + 1e-8
            mean_actions = np.einsum("i,ijk->jk", weights, candidates) / weights_sum

        if self.action_limit is not None:
            best_actions = np.clip(best_actions, -self.action_limit, self.action_limit)

        # rollout final trajectory
        s = x0_np
        states = [s]
        for a in best_actions:
            s = np.asarray(self.env.transition(s, a), dtype=np.float32)
            states.append(s)
        states = np.asarray(states, dtype=np.float32)

        return {
            "actions": np.asarray(best_actions, dtype=np.float32),
            "states": states,
            "rewards": np.asarray(rewards_seq[best_idx], dtype=np.float32) if len(rewards_seq) > 0 else np.array([], dtype=np.float32),
            "total_reward": float(best_return),
        }

    def sample_trajectories(
        self, x0: Any, n_samples: int, rng_key: Optional[Any] = None
    ) -> List[Trajectory]:
        traj = self.plan(x0, rng_key)
        states_list = [np.asarray(s, dtype=np.float32) for s in traj["states"]]
        actions_list = [np.asarray(a, dtype=np.float32) for a in traj["actions"]]
        t = Trajectory(states=states_list, actions=actions_list, info=traj)
        return [t for _ in range(max(1, n_samples))]

