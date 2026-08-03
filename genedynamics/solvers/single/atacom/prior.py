"""ATACOM tangent-policy adapter for horizon proposal consumers.

The network remains a 7-D tangent-space policy.  This adapter rolls it through
the same stateful ATACOM transform and task dynamics used by the standalone
baseline, then exposes the resulting 10-D node trajectories through the shared
``StructuredPrior`` contract.  It owns no MDAC logic.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from genedynamics.learning.priors.base import ProposalBatch
from genedynamics.solvers.single.atacom.backends.atacom_jax import (
    init_slack,
    make_atacom_transform,
)
from genedynamics.solvers.single.dial.spline import NodeSpline


class AtacomHorizonPrior:
    """Roll a stochastic tangent policy into full task-control horizons."""

    def __init__(
        self,
        env: Any,
        tangent_prior: Any,
        *,
        Hsample: int = 16,
        Hnode: int = 4,
        ctrl_dt: float = 0.02,
        Kc: float = 1.0,
        action_limit: float = 1.0,
    ) -> None:
        self.env = env
        self.tangent_prior = tangent_prior
        self.output_dim = int(env.action_size)
        self._dense_horizon = int(Hsample) + 1
        self._alpha_max = float(action_limit)
        self._spline = NodeSpline.build(
            int(Hnode), int(Hsample), float(ctrl_dt)
        )
        dt = float(getattr(getattr(env, "_config", None), "dt", ctrl_dt))
        self._transform = make_atacom_transform(
            env, Kc=float(Kc), time_step=dt,
            action_limit=float(action_limit),
        )
        self._policy_batch = jax.jit(jax.vmap(
            lambda obs, key, deterministic: tangent_prior.act(
                obs, key=key, deterministic=deterministic
            ),
            in_axes=(0, 0, None),
        ), static_argnums=(2,))
        self._transform_batch = jax.jit(jax.vmap(self._transform))
        self._step_batch = jax.jit(jax.vmap(env.step))

    @staticmethod
    def _obs_of(state):
        obs = getattr(state, "obs", state)
        if isinstance(obs, dict):
            obs = obs.get("state", next(iter(obs.values())))
        return jnp.asarray(obs)

    def act(self, obs, *, key=None, deterministic=True):
        """Expose the tangent action for protocol compatibility."""
        return self.tangent_prior.act(
            obs, key=key, deterministic=deterministic
        )

    def logp_of_sequence(self, obs_seq, act_seq):
        """Density is defined in ATACOM's tangent coordinates."""
        return self.tangent_prior.logp_of_sequence(obs_seq, act_seq)

    def _rollout(self, state, *, key, n_samples: int, deterministic: bool):
        n_samples = int(n_samples)
        batch_state = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x), (n_samples,) + jnp.asarray(x).shape
            ),
            state,
        )
        slack0 = init_slack(self.env, state)
        slack = jnp.broadcast_to(
            slack0, (n_samples,) + slack0.shape
        )
        keys = jax.random.split(key, self._dense_horizon * n_samples)
        keys = keys.reshape((self._dense_horizon, n_samples, -1))
        observations = []
        alphas = []
        controls = []
        current = batch_state
        for h in range(self._dense_horizon):
            obs = self._obs_of(current)
            alpha = self._policy_batch(obs, keys[h], deterministic)
            control, slack = self._transform_batch(
                current, alpha * self._alpha_max, slack
            )
            observations.append(obs)
            alphas.append(alpha)
            controls.append(control)
            current = self._step_batch(current, control)

        obs_seq = jnp.stack(observations, axis=1)
        alpha_seq = jnp.stack(alphas, axis=1)
        dense_controls = jnp.stack(controls, axis=1)
        nodes = jnp.einsum(
            "nh,bhu->bnu", self._spline.U2N, dense_controls
        )
        log_prob = jax.vmap(
            self.tangent_prior.logp_of_sequence
        )(obs_seq, alpha_seq)
        return nodes, log_prob

    def warm_start(self, state):
        nodes, _ = self._rollout(
            state, key=jax.random.PRNGKey(0), n_samples=1,
            deterministic=True,
        )
        return nodes[0]

    def sample_horizons(self, state, *, key, n_samples: int):
        if int(n_samples) <= 0:
            raise ValueError("n_samples must be positive")
        nodes, log_prob = self._rollout(
            state, key=key, n_samples=int(n_samples), deterministic=False,
        )
        return ProposalBatch(
            trajectories=nodes,
            log_prob=log_prob,
            expert_id=jnp.full((int(n_samples),), 2, jnp.int32),
        )


__all__ = ["AtacomHorizonPrior"]
