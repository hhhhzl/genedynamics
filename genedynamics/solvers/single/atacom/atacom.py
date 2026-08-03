"""ATACOM RL baseline solver — Acting on the Tangent space of the Constraint Manifold (CoRL'21).

Faithful: the policy is trained ON the manifold via ``AtacomEnvWrapper`` (its action space is
the tangent dimension ``null = nu − n_f``; the jax backend ``make_atacom_transform`` maps the
tangent action to a full env control inside ``env.step`` — `backends/atacom_jax.py`). At deploy
this solver runs the policy through the wrapper and records the executed FULL-dim controls (and
the inner env states) so the comparison metrics are computed on the original env / control space.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:                                   # pragma: no cover
    register_solver = None


class AtacomSolver:
    """Tangent-space (manifold-resident) policy, deployed closed-loop on the inner env."""

    def __init__(self, env: Any, act_fn: Callable[[Any, Any], Any], *, Kc: float = 1.0,
                 action_limit: float = 1.0, seed: int = 0) -> None:
        from genedynamics.solvers.single.atacom.wrapper import AtacomEnvWrapper
        self.env = env                                   # inner env (for metrics / control space)
        self.wrapper = AtacomEnvWrapper(env, Kc=Kc, action_limit=action_limit)
        self.act_fn = act_fn                             # obs -> tangent action α (null dim)
        self.seed = int(seed)

        def policy_step(state, key):
            alpha = self.act_fn(state.obs, key)
            next_state = self.wrapper.step(state, alpha)
            return next_state, next_state.info["atacom_u"]

        # Deployment has no PPO outer scan to provide compilation. Cache the
        # complete policy -> ATACOM -> MJX step as one executable.
        self._policy_step = jax.jit(policy_step)

    def run_receding(self, x0: Any, n_steps: int, rng: Any, *,
                     collect_states: bool = True,
                     synchronize_steps: bool = False):
        from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult
        state = self.wrapper._augment(x0)                # seed the slack on the (inner) reset state
        states, actions, infos = [state], [], []
        key = rng
        for _ in range(int(n_steps)):
            key, k = jax.random.split(key)
            state, executed_u = self._policy_step(state, k)
            if synchronize_steps:
                state = jax.block_until_ready(state)
            actions.append(executed_u)                    # executed full-dim control u
            infos.append({
                "atacom_slack_norm": jax.numpy.linalg.norm(
                    state.info["atacom_s"]
                ),
                "atacom_action_norm": jax.numpy.linalg.norm(
                    state.info["atacom_u"]
                ),
            })
            if collect_states:
                states.append(state)
        if not collect_states:
            states.append(state)
        return RecedingHorizonResult(
            states=states, actions=actions, infos=infos
        )


if register_solver is not None:
    try:
        register_solver("atacom", AtacomSolver)
    except Exception:                                # pragma: no cover
        pass


__all__ = ["AtacomSolver"]
