"""
Environment-backed fidelity simulator for MRMFMBD.

Wraps env with jax_transition; supports single-level (all fidelities same)
or multi-level when env provides fidelity-aware transition.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from ..types import FidelityLevel
from ..protocols import BaseFidelitySimulator


class EnvFidelitySimulator(BaseFidelitySimulator):
    """
    Fidelity simulator wrapping an environment.

    If env has jax_transition_fidelity(state, action, level), uses it.
    Otherwise all fidelity levels use the same jax_transition (single-level).
    """

    def __init__(
        self,
        env: Any,
        energy: Any,
        position_extractor: Any,
        position_dim: int = 2,
        terminal_weight: float = 100.0,
        num_levels: int = 3,
    ):
        self.env = env
        self.energy = energy
        self.position_extractor = position_extractor
        self.position_dim = position_dim
        self.terminal_weight = terminal_weight
        self.num_levels = num_levels
        self._act_dim = getattr(env, "act_dim", getattr(env, "action_size", 1))
        self._state_dim = getattr(env, "state_dim", 128)
        self._supports_fidelity = hasattr(env, "jax_transition_fidelity")

        self._levels = [
            FidelityLevel(level=i, state_dim=self._state_dim, act_dim=self._act_dim)
            for i in range(num_levels)
        ]

    def _transition(self, state: Any, action: Any, level: int) -> Any:
        if self._supports_fidelity:
            return self.env.jax_transition_fidelity(state, action, level)
        return self.env.jax_transition(state, action)

    def rollout_rewards(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        use_jax = JAX_AVAILABLE
        if use_jax:
            x0 = jnp.asarray(x0, dtype=jnp.float32)
            actions = jnp.asarray(actions, dtype=jnp.float32)
        else:
            x0 = np.asarray(x0, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)

        target = getattr(self.env, "target", None)
        if target is None:
            goal = jnp.zeros(self.position_dim, dtype=jnp.float32) if use_jax else np.zeros(self.position_dim, dtype=np.float32)
        else:
            pos = np.asarray(self.position_extractor(target), dtype=np.float32).reshape(-1)[: self.position_dim]
            goal = jnp.asarray(pos, dtype=jnp.float32) if use_jax else np.asarray(pos, dtype=np.float32)

        def step_fn(carry, action):
            next_state = self._transition(carry, action, fidelity_level)
            ctx = {"t": 0, "target_xy": goal}
            reward = -self.energy.compute(next_state, action, ctx)
            return next_state, reward

        if use_jax:
            final_state, rewards = jax.lax.scan(step_fn, x0, actions)
            terminal_dist = jnp.linalg.norm(final_state[: self.position_dim] - goal[: self.position_dim])
            terminal_reward = -jnp.asarray(self.terminal_weight, dtype=jnp.float32) * terminal_dist
            rewards = rewards.at[-1].add(terminal_reward)
            return rewards
        else:
            state = x0
            rewards_list = []
            for t in range(actions.shape[0]):
                action = actions[t]
                next_state = self._transition(state, action, fidelity_level)
                ctx = {"t": t, "target_xy": np.asarray(goal)}
                reward = -float(self.energy.compute(next_state, action, ctx))
                rewards_list.append(reward)
                state = next_state
            terminal_dist = float(np.linalg.norm(np.asarray(state[: self.position_dim]) - np.asarray(goal[: self.position_dim])))
            rewards_list[-1] = rewards_list[-1] - self.terminal_weight * terminal_dist
            return np.asarray(rewards_list, dtype=np.float32)

    def rollout_states(
        self,
        x0: Any,
        actions: Any,
        fidelity_level: int,
        **kwargs: Any,
    ) -> Any:
        use_jax = JAX_AVAILABLE
        if use_jax:
            x0 = jnp.asarray(x0, dtype=jnp.float32)
            actions = jnp.asarray(actions, dtype=jnp.float32)
        else:
            x0 = np.asarray(x0, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.float32)

        def step_fn(carry, action):
            next_state = self._transition(carry, action, fidelity_level)
            return next_state, next_state

        if use_jax:
            final_state, states = jax.lax.scan(step_fn, x0, actions)
            return jnp.concatenate([x0[None, :], states], axis=0)
        else:
            state = x0
            states_list = [np.asarray(state)]
            for t in range(actions.shape[0]):
                state = self._transition(state, actions[t], fidelity_level)
                states_list.append(np.asarray(state))
            return np.stack(states_list, axis=0)

    def get_fidelity_levels(self) -> List[FidelityLevel]:
        return self._levels

    def state_dim(self, fidelity_level: int) -> int:
        return self._state_dim

    def act_dim(self) -> int:
        return self._act_dim
