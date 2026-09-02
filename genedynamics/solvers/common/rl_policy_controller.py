"""Standalone closed-loop RL controller (shared base for the RL baselines).

Wraps a trained policy's ``act(obs, key) -> action`` as a ``run_receding(x0, n_steps, rng)``
controller (the same interface as ``MGASolver`` / the sampling baselines), so an RL policy
plugs into the MGA harness comparison like any other method. There is NO online planning —
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

    def run_receding(
        self,
        x0: Any,
        n_steps: int,
        rng: Any,
        *,
        collect_states: bool = True,
        synchronize_steps: bool = False,
    ):
        import jax
        from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult

        project_jax = getattr(self.action_projection, "project_jax", None)
        if not collect_states and (
            self.action_projection is None or callable(project_jax)
        ):
            # JAX-native evaluation: compile policy, optional pure-JAX safety
            # projection, and the true env step into one closed-loop scan.  The
            # Python loop remains for non-JAX projections and callers that
            # explicitly request every intermediate State.
            def rollout(state, key):
                def body(carry, _):
                    current, current_key = carry
                    current_key, action_key = jax.random.split(current_key)
                    action = self.act_fn(current.obs, action_key)

                    if callable(project_jax):
                        action, projection_info = project_jax(current, action)
                    else:
                        # A scalar placeholder keeps the scan output signature
                        # uniform without requiring optional State.info on raw
                        # RL environments.  It is discarded below.
                        projection_info = jax.numpy.asarray(0, dtype=jax.numpy.int32)
                    return (
                        self.env.step(current, action), current_key
                    ), (action, projection_info)

                (final_state, _), (actions, projection_info) = jax.lax.scan(
                    body, (state, key), None, length=int(n_steps)
                )
                return final_state, actions, projection_info

            state, actions, projection_info = jax.jit(rollout)(x0, rng)
            if synchronize_steps:
                jax.block_until_ready((state, actions, projection_info))
            return RecedingHorizonResult(
                states=[x0, state], actions=actions,
                infos=[projection_info] if callable(project_jax) else [],
            )

        state = x0
        states, actions, infos = [state], [], []
        key = rng
        closed_loop_step = None
        if self.action_projection is None:
            # Contact tasks retain every executed State for force/torque
            # metrics.  Compile one policy+dynamics transition and invoke it
            # from the host loop; compiling a full 50-step scan exceeds the
            # memory budget of a 16-GB CPU Mac, while leaving the two calls
            # separate accumulates many small MJX executables.
            def _closed_loop_step(current, action_key):
                action = self.act_fn(current.obs, action_key)
                return self.env.step(current, action), action

            closed_loop_step = jax.jit(_closed_loop_step)
        for _ in range(int(n_steps)):
            key, k = jax.random.split(key)
            if closed_loop_step is not None:
                state, a = closed_loop_step(state, k)
            else:
                a = self.act_fn(state.obs, k)                   # policy action
            if self.action_projection is not None:
                project_with_info = getattr(
                    self.action_projection, "project_with_info", None
                )
                if callable(project_with_info):
                    a, projection_info = project_with_info(state, a)
                    infos.append(projection_info)
                else:
                    a = self.action_projection(state, a)        # ATACOM tangent / ISSA safe-set
                state = self.env.step(state, a)
            if synchronize_steps:
                jax.block_until_ready(state)
            actions.append(a)
            if collect_states:
                states.append(state)
        if not collect_states:
            states.append(state)
        return RecedingHorizonResult(
            states=states, actions=actions, infos=infos
        )


__all__ = ["RLPolicyController"]
