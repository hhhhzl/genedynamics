"""JAX ISSA/AdamBA projection using true one-step realized safety.

The vendored ISSA implementation classifies a candidate by simulating it and
checking the change of a state safety index.  This task-adapted port preserves
that contract: the Panda environment owns ``safety_index(state)`` and every ray
candidate passes through ``env.step``.  Command bounds alone are not a safety
index because the impedance servo clips them before execution.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


class IssaProjection:
    """Derivative-free AdamBA projection with a safety-first fallback."""

    def __init__(self, env: Any, *, n_dirs: int = 20, n_iters: int = 50,
                 bound: float = 1e-4, threshold: float = 0.0,
                 enforce_absolute: bool = True, action_limit: float = 1.0,
                 seed: int = 0) -> None:
        if not hasattr(env, "safety_index"):
            raise ValueError("ISSA requires env.safety_index(state)")
        self.env = env
        self.n_dirs = int(n_dirs)
        self.n_iters = int(n_iters)
        self.bound = float(bound)
        self.threshold = float(threshold)
        self.enforce_absolute = bool(enforce_absolute)
        self.action_limit = float(action_limit)
        self.seed = int(seed)
        cfg = getattr(env, "_config", None)
        self.dt = float(getattr(cfg, "dt", 0.02))
        self._key0 = jax.random.PRNGKey(self.seed)
        self.last_info: Dict[str, Any] = {}
        self._margin_jit = jax.jit(self._transition_margin)
        self._adamba_jit = jax.jit(self._adamba)

    def _transition_margin(self, state: Any, action: jax.Array) -> jax.Array:
        """Positive means the candidate violates ISSA's one-step condition."""
        phi_now = self.env.safety_index(state)
        next_state = self.env.step(state, action)
        phi_next = self.env.safety_index(next_state)
        change = phi_next - phi_now - self.threshold * self.dt
        if self.enforce_absolute:
            # Do not leave phi<=0.  If already unsafe, retain the ISSA recovery
            # rule (non-increasing phi) instead of demanding a one-step miracle.
            crossing = jnp.where(phi_now <= 0.0, phi_next, -jnp.inf)
            return jnp.maximum(change, crossing)
        return change

    def _adamba(self, state: Any, nominal: jax.Array,
                key: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        nu = nominal.shape[0]
        dirs = jax.random.normal(key, (self.n_dirs, nu))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
        # Each margin evaluates a complete realized Brax/MJX transition.  A
        # vmap materializes all ray transitions in parallel and exceeds the
        # memory budget of the H1 environment even for the canonical eight
        # directions.  lax.map preserves the exact ray set and AdamBA update
        # while evaluating one transition at a time, bounding peak memory by
        # one environment step rather than ``n_dirs`` steps.
        margins = lambda actions: jax.lax.map(
            lambda u: self._transition_margin(state, u), actions
        )

        def body(carry):
            iteration, current, eta, refining, valid, done = carry
            margin = margins(current)
            out_of_bounds = jnp.any(
                jnp.abs(current) > self.action_limit, axis=1
            )
            valid = valid & (~out_of_bounds)
            safe = margin <= 0.0
            done = done | (safe & (eta <= self.bound))
            active = (~done) & valid
            expand = active & (~safe) & (~refining)
            start_refine = active & safe & (~refining)
            refine_unsafe = active & (~safe) & refining
            refine_safe = active & safe & refining
            signed_step = (
                jnp.where(expand | refine_unsafe, eta, 0.0)
                - jnp.where(refine_safe, eta, 0.0)
            )
            current = current + signed_step[:, None] * dirs
            eta = eta * jnp.where(
                expand, 2.0, jnp.where(start_refine, 0.25, 0.5)
            )
            refining = refining | start_refine
            return (
                iteration + 1, current, eta, refining, valid, done
            )

        def keep_searching(carry):
            iteration, _current, _eta, _refining, valid, done = carry
            unresolved = valid & (~done)
            return (iteration < self.n_iters) & jnp.any(unresolved)

        initial = (
            jnp.asarray(0, jnp.int32),
            jnp.broadcast_to(nominal, (self.n_dirs, nu)),
            jnp.full((self.n_dirs,), self.bound),
            jnp.zeros((self.n_dirs,), dtype=bool),
            jnp.ones((self.n_dirs,), dtype=bool),
            jnp.zeros((self.n_dirs,), dtype=bool),
        )
        (_iterations, ray_actions, eta, _refining, valid, done) = jax.lax.while_loop(
            keep_searching, body, initial
        )
        final_margin = margins(ray_actions)
        converged = done | ((final_margin <= 0.0) & (eta <= self.bound))

        # A zero-action hold is an explicit recovery candidate.  The nominal is
        # included so the fallback is never worse merely because all rays exit
        # the action box.
        candidates = jnp.concatenate([
            jnp.clip(ray_actions, -self.action_limit, self.action_limit),
            nominal[None],
            jnp.zeros((1, nu), nominal.dtype),
        ], axis=0)
        candidate_valid = jnp.concatenate([
            valid & converged,
            jnp.ones((2,), dtype=bool),
        ])
        candidate_margin = margins(candidates)
        candidate_safe = candidate_valid & (candidate_margin <= 0.0)
        distance = jnp.sum((candidates - nominal[None]) ** 2, axis=1)
        closest_safe = jnp.argmin(jnp.where(candidate_safe, distance, jnp.inf))
        safest = jnp.argmin(jnp.where(candidate_valid, candidate_margin, jnp.inf))
        found_safe = jnp.any(candidate_safe)
        index = jnp.where(found_safe, closest_safe, safest)
        projected = candidates[index]
        return projected, {
            "found_safe": found_safe,
            "failure": ~found_safe,
            "margin": candidate_margin[index],
            "intervention": jnp.linalg.norm(projected - nominal),
        }

    def project_with_info(self, state: Any, action: Any):
        nominal = jnp.clip(
            jnp.asarray(action), -self.action_limit, self.action_limit
        )
        nominal_margin = self._margin_jit(state, nominal)
        if float(nominal_margin) <= 0.0:
            info = {
                "found_safe": jnp.asarray(True),
                "failure": jnp.asarray(False),
                "margin": nominal_margin,
                "intervention": jnp.asarray(0.0, nominal.dtype),
            }
            self.last_info = info
            return nominal, info
        step = state.info.get("step", jnp.asarray(0, jnp.int32))
        key = jax.random.fold_in(self._key0, step.astype(jnp.uint32))
        projected, info = self._adamba_jit(state, nominal, key)
        self.last_info = info
        return projected, info

    def project_jax(self, state: Any, action: Any):
        """Pure-JAX projection for a compiled closed-loop rollout.

        ``project_with_info`` deliberately offers an eager Python interface for
        debugging and unit tests.  Production evaluation instead calls this
        method from ``lax.scan`` so policy, AdamBA and the realized MJX step are
        compiled as one control loop rather than dispatched separately at every
        step.  Both paths use the exact same transition margin and AdamBA body.
        """
        nominal = jnp.clip(
            jnp.asarray(action), -self.action_limit, self.action_limit
        )
        nominal_margin = self._transition_margin(state, nominal)
        step = state.info.get("step", jnp.asarray(0, jnp.int32))
        key = jax.random.fold_in(self._key0, step.astype(jnp.uint32))

        def accept(_):
            return nominal, {
                "found_safe": jnp.asarray(True),
                "failure": jnp.asarray(False),
                "margin": nominal_margin,
                "intervention": jnp.asarray(0.0, nominal.dtype),
            }

        def project(_):
            return self._adamba(state, nominal, key)

        return jax.lax.cond(nominal_margin <= 0.0, accept, project, operand=None)

    def __call__(self, state: Any, action: Any):
        return self.project_with_info(state, action)[0]


def make_issa_projection(env: Any, *, n_dirs: int = 20, n_iters: int = 50,
                         bound: float = 1e-4, threshold: float = 0.0,
                         enforce_absolute: bool = True,
                         action_limit: float = 1.0, seed: int = 0) -> IssaProjection:
    return IssaProjection(
        env, n_dirs=n_dirs, n_iters=n_iters, bound=bound,
        threshold=threshold, enforce_absolute=enforce_absolute,
        action_limit=action_limit, seed=seed,
    )


__all__ = ["IssaProjection", "make_issa_projection"]
