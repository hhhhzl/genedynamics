"""Shared env rollout builders for receding-horizon diffusion solvers.

The brax-State path rolls ``env.step`` over a brax ``State`` and reads
``state.reward`` (dial-mpc ``MBDPI`` style; time / gait phase carried inside
``state.info`` by the env). Factoring it out of the DIAL backend lets MBD / MPPI
/ CEM / MDAC run the *same* environment and reward and differ only in their
reverse-diffusion update -- the fair-baseline setup for the MDAC experiments.

Flat-state envs keep using ``DynamicsToEnvAdapter.jax_transition`` + a separate
energy; this module only adds the brax-State option.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def is_brax_env(env: Any) -> bool:
    """True for a brax ``PipelineEnv`` (env.step -> State with .reward)."""
    return env is not None and hasattr(env, "step") and hasattr(env, "pipeline_step")


def build_brax_rollout(env: Any) -> Callable[[Any, Any, Any], Any]:
    """Build ``rollout_fn(state, us, t0) -> rews``.

    ``state`` is a brax ``State`` (current), ``us`` is a batch of dense control
    sequences ``(Nsample, Hsample+1, nu)``; returns per-candidate per-step
    rewards ``(Nsample, Hsample+1)`` read from ``state.reward``. ``t0`` is unused
    (the brax env advances ``state.info["step"]`` itself).
    """
    step = env.step

    def rollout_one(state, us, t0):  # us: (Hsample+1, nu)
        def f(s, u):
            s2 = step(s, u)
            return s2, s2.reward
        _, rews = jax.lax.scan(f, state, us)
        return rews

    return jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None)))


def build_brax_step(env: Any) -> Callable[[Any, Any], Any]:
    """Real one-step dynamics for the receding-horizon bridge (brax env.step)."""
    return env.step
