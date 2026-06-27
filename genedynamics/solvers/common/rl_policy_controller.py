"""Standalone closed-loop RL controller (shared base for the RL baselines).

Wraps a trained policy's ``act(obs, key) -> action`` as a ``run_receding(x0, n_steps, rng)``
controller (the same interface as ``MDACSolver`` / the sampling baselines), so an RL policy
plugs into the MDAC harness comparison like any other method. There is NO online planning —
the policy is queried each real step. An optional ``action_projection(state, action) ->
action`` hook is applied before stepping, which is exactly where the RL BASELINES add their
contribution:

  * ATACOM — project the action onto the tangent space of the env constraint manifold {C=0}
             (``env.manifold_geometry`` / ``env.manifold_residual``);
  * ISSA   — project the action onto the safe set via AdamBA on the env inequality ``g``
             (``env.constraint_residual``).

Returns a ``RecedingHorizonResult`` so the metrics pipeline is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class RLPolicyController:
    """Closed-loop RL controller: ``act`` each step (+ optional projection), on the real env."""

    def __init__(self, env: Any, act_fn: Callable[[Any, Any], Any], *,
                 action_projection: Optional[Callable[[Any, Any], Any]] = None,
                 seed: int = 0) -> None:
        self.env = env
        self.act_fn = act_fn
        self.action_projection = action_projection
        self.seed = int(seed)

    def run_receding(self, x0: Any, n_steps: int, rng: Any):
        import jax
        from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult

        state = x0
        states, actions = [state], []
        key = rng
        for _ in range(int(n_steps)):
            key, k = jax.random.split(key)
            a = self.act_fn(state.obs, k)                       # policy action
            if self.action_projection is not None:
                a = self.action_projection(state, a)            # ATACOM tangent / ISSA safe-set
            state = self.env.step(state, a)
            actions.append(a)
            states.append(state)
        return RecedingHorizonResult(states=states, actions=actions)


__all__ = ["RLPolicyController"]
