"""ATACOM env wrapper — the RL policy lives ON the constraint manifold.

A brax-compatible wrapper (`baselines/rl_on_manifold/atacom/atacom.py::AtacomEnvWrapper`):
the policy's action space is the TANGENT dimension ``null = nu − n_f`` (`atacom.py:51`); each
``step`` runs the ATACOM transform (``backends/atacom_jax``) to map the tangent action ``α``
to a full env control ``u`` that keeps the system on ``{C=0}`` (+ a slack update), then steps
the inner env. Because the transform is INSIDE ``step``, the policy is TRAINED on the manifold
(via ``learning/train_rl_policy`` on this wrapped env) — not post-hoc projected.

Slack ``s`` and the realized control ``u`` ride in ``state.info`` (``atacom_s`` / ``atacom_u``)
so the receding-horizon driver can recover the executed full-dim controls for the metrics.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from genedynamics.solvers.single.atacom.backends.atacom_jax import (
    make_atacom_transform, atacom_null_dim, init_slack,
)


class AtacomEnvWrapper:
    """Wrap a brax arm env so the policy acts in the constraint-manifold tangent space."""

    def __init__(self, env: Any, *, Kc: float = 1.0, action_limit: float = 1.0) -> None:
        self.env = env
        self._nu = int(env.action_size)
        self._null = atacom_null_dim(env)
        self.alpha_max = float(action_limit)
        dt = float(getattr(getattr(env, "_config", None), "dt", 0.02))
        self._transform = make_atacom_transform(env, Kc=Kc, time_step=dt, action_limit=action_limit)

    # brax-env duck-typed surface (reset/step/observation_size/action_size)
    @property
    def observation_size(self) -> int:
        return int(self.env.observation_size)

    @property
    def action_size(self) -> int:
        return self._null                                    # the TANGENT (policy) dimension

    def _augment(self, state):
        s = init_slack(self.env, state)
        return state.replace(info={**state.info, "atacom_s": s,
                                   "atacom_u": jnp.zeros((self._nu,), jnp.float32)})

    def reset(self, rng):
        return self._augment(self.env.reset(rng))

    def step(self, state, alpha):
        s = state.info["atacom_s"]
        u, s_new = self._transform(state, jnp.asarray(alpha) * self.alpha_max, s)
        new = self.env.step(state, u)
        return new.replace(info={**new.info, "atacom_s": s_new, "atacom_u": u})


__all__ = ["AtacomEnvWrapper"]
