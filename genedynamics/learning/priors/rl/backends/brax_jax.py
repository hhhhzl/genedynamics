"""JAX RL-prior backend — brax.training integration.

Wraps a brax PPO/SAC policy network (the JAX backend uses brax's training
integration, per the structure decision). Holds the brax `PPONetworks` +
trained `params` (normalizer + policy) and exposes the `Prior` surface:

  act(obs)            -> brax `make_inference_fn` policy (mode if deterministic)
  logp_of_sequence    -> sum_t dist.log_prob(logits_t, inverse_postprocess(a_t))
                         (jit-safe: depends only on obs/action, not state.info)
  warm_start(state)   -> closed-loop policy horizon mapped to solver nodes when a
                         rollout step is supplied; otherwise the legacy tiled
                         action used by lightweight integration tests

brax is imported lazily (only when this backend is INSTANTIATED) so the priors
package imports fine on fedguide (jax, no brax); training/inference run in the
docker brax image.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp


class BraxRLPrior:
    """brax PPO-network policy prior (JAX)."""

    def __init__(
        self,
        *,
        observation_size: int,
        action_size: int,
        params: Any,                              # (normalizer_params, policy_params[, value_params])
        n_warm_nodes: int = 5,
        normalize_observations: bool = True,
        policy_hidden_layer_sizes: Sequence[int] = (32, 32, 32, 32),
        distribution_type: str = "tanh_normal",
        deterministic: bool = True,
        obs_key: str = "state",
        rollout_step: Any = None,
        Hsample: int = 16,
        Hnode: int = 4,
        ctrl_dt: float = 0.02,
        action_bias: Any = None,
        action_scale: Any = None,
    ) -> None:
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.acme import running_statistics

        preprocess = (
            running_statistics.normalize
            if normalize_observations else (lambda x, y: x)
        )
        self._nets = ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess,
            policy_hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
            distribution_type=distribution_type,
        )
        self._dist = self._nets.parametric_action_distribution
        self._policy_apply = self._nets.policy_network.apply
        self._make_policy = ppo_networks.make_inference_fn(self._nets)
        self.params = params
        self.output_dim = int(action_size)
        self.n_warm_nodes = int(n_warm_nodes)
        self.deterministic = bool(deterministic)
        self._obs_key = obs_key
        self._rollout_step = rollout_step
        self._action_bias = (
            None if action_bias is None
            else jnp.asarray(action_bias, dtype=jnp.float32)
        )
        if self._action_bias is not None and self._action_bias.shape != (self.output_dim,):
            raise ValueError(
                "action_bias shape mismatch: "
                f"{self._action_bias.shape} != {(self.output_dim,)}"
            )
        self._action_scale = (
            None if action_scale is None
            else jnp.asarray(action_scale, dtype=jnp.float32)
        )
        if self._action_scale is not None:
            if self._action_scale.shape != (self.output_dim,):
                raise ValueError(
                    "action_scale shape mismatch: "
                    f"{self._action_scale.shape} != {(self.output_dim,)}"
                )
            if bool(jnp.any(self._action_scale <= 0.0)):
                raise ValueError("action_scale entries must be positive")
        self._dense_horizon = int(Hsample) + 1
        self._spline = None
        self._cpu_stepwise_rollout = False
        self._policy_action_jit = None
        self._rollout_step_jit = None
        if rollout_step is not None:
            from genedynamics.solvers.single.dial.spline import NodeSpline

            self._spline = NodeSpline.build(
                int(Hnode), int(Hsample), float(ctrl_dt)
            )
            if self.n_warm_nodes != int(Hnode) + 1:
                raise ValueError(
                    "n_warm_nodes must equal Hnode + 1 for horizon proposals"
                )
            # On CPU, fusing the policy and the full MJX horizon into one XLA
            # scan makes that executable coexist with MGA's batched rollout
            # executable and can exceed the host memory limit.  Keep the exact
            # same closed-loop recurrence, but compile policy inference and one
            # dynamics step independently and reuse them from the Python loop.
            # GPU keeps the fused scan, where launch overhead matters and the
            # compiler/device memory budget is normally larger.
            self._cpu_stepwise_rollout = jax.default_backend() == "cpu"
            if self._cpu_stepwise_rollout:
                self._policy_action_jit = jax.jit(
                    lambda obs: self.act(obs, deterministic=True)
                )
                self._rollout_step_jit = jax.jit(self._rollout_step)
                self._policy_action_batch_jit = jax.jit(jax.vmap(
                    lambda obs, key: self.act(
                        obs, key=key, deterministic=False
                    )
                ))
                self._rollout_step_batch_jit = jax.jit(jax.vmap(
                    self._rollout_step
                ))

    # --- helpers ---
    def _norm_pol(self):
        return (self.params[0], self.params[1])

    def _logits(self, obs):
        n, p = self._norm_pol()
        return self._policy_apply(n, p, obs)

    @staticmethod
    def _obs_of(state, key):
        obs = getattr(state, "obs", state)
        if isinstance(obs, dict):
            obs = obs.get(key, next(iter(obs.values())))
        return jnp.asarray(obs)

    # --- Prior surface ---
    def act(self, obs, *, key: Optional[Any] = None, deterministic: Optional[bool] = None) -> Any:
        det = self.deterministic if deterministic is None else deterministic
        policy = self._make_policy(self.params, deterministic=det)
        if key is None:
            key = jax.random.PRNGKey(0)
        action, _ = policy(jnp.asarray(obs), key)
        if self._action_scale is not None:
            action = action * self._action_scale
        if self._action_bias is not None:
            action = jnp.clip(action + self._action_bias, -1.0, 1.0)
        return action

    def logp_of_sequence(self, obs_seq, act_seq) -> Any:
        """sum_t log pi(a_t | o_t). `obs_seq`:(H,obs), `act_seq`:(H,A) postprocessed."""
        def step_logp(o, a):
            logits = self._logits(o)
            if self._action_bias is not None:
                a = a - self._action_bias
            if self._action_scale is not None:
                a = a / self._action_scale
            a = jnp.clip(a, -0.999999, 0.999999)
            raw = self._dist.inverse_postprocess(a)
            return self._dist.log_prob(logits, raw)
        return jnp.sum(jax.vmap(step_logp)(jnp.asarray(obs_seq), jnp.asarray(act_seq)))

    def warm_start(self, state) -> Any:
        """Return the policy proposal in the solver's node parameterization."""
        if self._rollout_step is not None:
            if self._cpu_stepwise_rollout:
                current = state
                dense_actions = []
                for _ in range(self._dense_horizon):
                    obs = self._obs_of(current, self._obs_key)
                    action = self._policy_action_jit(obs)
                    dense_actions.append(action)
                    current = self._rollout_step_jit(current, action)
                return self._spline.u2node(jnp.stack(dense_actions))

            def body(s, _):
                obs = self._obs_of(s, self._obs_key)
                action = self.act(obs, deterministic=True)
                return self._rollout_step(s, action), action

            _, dense_actions = jax.lax.scan(
                body, state, None, length=self._dense_horizon
            )
            return self._spline.u2node(dense_actions)

        # Compatibility path for policy-only tests and callers that do not own
        # dynamics. MGA always supplies rollout_step.
        obs = self._obs_of(state, self._obs_key)
        a = self.act(obs, deterministic=True)
        return jnp.tile(a[None, :], (self.n_warm_nodes, 1))

    def sample_horizons(self, state, *, key, n_samples: int):
        """Sample stochastic closed-loop policy horizons in solver node space."""
        from genedynamics.learning.priors.base import ProposalBatch

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self._rollout_step is None or self._spline is None:
            raise ValueError(
                "structured horizon sampling requires rollout_step"
            )

        # CPU deliberately reuses the same small batched policy/one-step MJX
        # executables at every horizon position.  This avoids fusing an extra
        # policy x horizon x environment executable beside MGA's rollout.
        batch_state = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(
                jnp.asarray(x), (n_samples,) + jnp.asarray(x).shape
            ),
            state,
        )
        keys = jax.random.split(key, self._dense_horizon * n_samples)
        keys = keys.reshape((self._dense_horizon, n_samples, -1))
        observations = []
        actions = []

        if self._cpu_stepwise_rollout:
            current = batch_state
            for h in range(self._dense_horizon):
                obs = self._obs_of(current, self._obs_key)
                action = self._policy_action_batch_jit(obs, keys[h])
                observations.append(obs)
                actions.append(action)
                current = self._rollout_step_batch_jit(current, action)
            obs_seq = jnp.stack(observations, axis=1)
            dense_actions = jnp.stack(actions, axis=1)
        else:
            def body(current, key_h):
                obs = self._obs_of(current, self._obs_key)
                action = jax.vmap(
                    lambda o, k: self.act(o, key=k, deterministic=False)
                )(obs, key_h)
                next_state = jax.vmap(self._rollout_step)(current, action)
                return next_state, (obs, action)

            _, (obs_h, action_h) = jax.lax.scan(body, batch_state, keys)
            obs_seq = jnp.swapaxes(obs_h, 0, 1)
            dense_actions = jnp.swapaxes(action_h, 0, 1)

        nodes = jnp.einsum(
            "nh,bhu->bnu", self._spline.U2N, dense_actions
        )
        log_prob = jax.vmap(self.logp_of_sequence)(obs_seq, dense_actions)
        return ProposalBatch(
            trajectories=nodes,
            log_prob=log_prob,
            expert_id=jnp.ones((n_samples,), dtype=jnp.int32),
        )


__all__ = ["BraxRLPrior"]
