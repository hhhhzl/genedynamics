"""Shared env rollout builders for receding-horizon diffusion solvers.

The brax-State path rolls ``env.step`` over a brax ``State`` and reads
``state.reward`` (dial-mpc ``MBDPI`` style; time / gait phase carried inside
``state.info`` by the env). Factoring it out of the DIAL backend lets MBD / MPPI
/ CEM / MGA run the *same* environment and reward and differ only in their
reverse-diffusion update -- the fair-baseline setup for the MGA experiments.

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


def build_brax_rollout_augmented(env: Any) -> Callable[..., Any]:
    """Augmented-Lagrangian brax rollout (the cfsmbd AL pattern, on a brax env).

    Same as :func:`build_brax_rollout` but the per-step reward is augmented with
    the soft-feasibility penalty (cfsmbd ``rollout_augmented_reward_and_v``):

        r_h        = [ |h_h| ; [g_h]_+ ]        from env.constraint_residual(state, u)
        penalty_h  = aug_lambda * sum(r_h) + (aug_rho/2) * sum(r_h^2)
        reward_h  <- state.reward - penalty_h

    ``aug_lambda`` / ``aug_rho`` are passed at CALL time so they can vary across
    reverse-diffusion steps (the coupled-annealing schedule, eq:coupled_schedule).
    Returns ``rollout_fn(state, us, t0, aug_lambda, aug_rho) -> rews``. With the
    default no-op ``constraint_residual`` (empty h, g) the penalty is exactly 0,
    so this is byte-identical to :func:`build_brax_rollout` on unconstrained tasks.
    """
    step = env.step
    # A task may expose a narrower residual for soft feasibility than for
    # constraint-aware controllers such as ATACOM.  This matters when an
    # equality describes execution tracking rather than safety: penalising that
    # lag in the AL makes forward progress artificially infeasible.  Existing
    # environments keep their byte-identical behaviour through the fallback.
    cres = getattr(env, "soft_feasibility_residual", env.constraint_residual)

    def rollout_one(state, us, t0, aug_lambda, aug_rho):  # us: (Hsample+1, nu)
        def f(s, u):
            s2 = step(s, u)
            h, g = cres(s2, u)                 # full brax State (has .pipeline_state AND .info)
            r = jnp.concatenate([jnp.abs(h), jax.nn.relu(g)], axis=-1)
            penalty = aug_lambda * jnp.sum(r) + 0.5 * aug_rho * jnp.sum(r * r)
            return s2, s2.reward - penalty
        _, rews = jax.lax.scan(f, state, us)
        return rews

    return jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None, None, None)))


def build_brax_step(env: Any) -> Callable[[Any, Any], Any]:
    """JIT-cached real one-step dynamics for the receding-horizon bridge.

    Returning the raw bound method makes MJX dispatch compile the locally
    defined substep scan repeatedly during a long Python receding-horizon loop.
    A single jitted callable preserves identical dynamics while compiling once
    per environment/state structure.
    """
    return jax.jit(env.step)


__all__ = [
    "is_brax_env",
    "build_brax_rollout",
    "build_brax_rollout_augmented",
    "build_brax_step",
]
