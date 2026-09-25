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
import numpy as np


class IssaProjection:
    """Derivative-free AdamBA projection with a safety-first fallback."""

    def __init__(self, env: Any, *, n_dirs: int = 20, n_iters: int = 50,
                 bound: float = 1e-4, threshold: float = 0.0,
                 enforce_absolute: bool = True, action_limit: float = 1.0,
                 seed: int = 0, step_env: Any = None) -> None:
        if not hasattr(env, "safety_index"):
            raise ValueError("ISSA requires env.safety_index(state)")
        self.env = env
        self.step_env = step_env or env
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
        self._transition_jit = jax.jit(self._transition)
        self._execution_step_jit = jax.jit(self.step_env.step)

    def _transition(self, state: Any, action: jax.Array):
        """Return the realized next state and its ISSA transition margin."""
        phi_now = self.env.safety_index(state)
        next_state = self.env.step(state, action)
        phi_next = self.env.safety_index(next_state)
        change = phi_next - phi_now - self.threshold * self.dt
        if self.enforce_absolute:
            # Do not leave phi<=0.  If already unsafe, retain the ISSA recovery
            # rule (non-increasing phi) instead of demanding a one-step miracle.
            crossing = jnp.where(phi_now <= 0.0, phi_next, -jnp.inf)
            change = jnp.maximum(change, crossing)
        return next_state, change

    def _transition_margin(self, state: Any, action: jax.Array) -> jax.Array:
        """Positive means the candidate violates ISSA's one-step condition."""
        return self._transition(state, action)[1]

    def _candidate_margins(
        self, state: Any, actions: jax.Array
    ) -> jax.Array:
        """Evaluate realized margins sequentially to cap H1 peak memory."""
        return jax.lax.map(
            lambda u: self._transition_margin(state, u), actions
        )

    def _search_rays(self, state: Any, nominal: jax.Array,
                     key: jax.Array):
        """Run AdamBA ray search without compiling final checks into it."""
        nu = nominal.shape[0]
        dirs = jax.random.normal(key, (self.n_dirs, nu))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)

        def body(carry):
            iteration, current, eta, refining, valid, done = carry
            margin = self._candidate_margins(state, current)
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
        return ray_actions, eta, valid, done

    def _candidate_actions(
        self, ray_actions: jax.Array, nominal: jax.Array
    ) -> jax.Array:
        """Append nominal and zero-action recovery candidates."""
        nu = nominal.shape[0]
        return jnp.concatenate([
            jnp.clip(ray_actions, -self.action_limit, self.action_limit),
            nominal[None],
            jnp.zeros((1, nu), nominal.dtype),
        ], axis=0)

    def _select_candidate(
        self,
        nominal: jax.Array,
        candidates: jax.Array,
        eta: jax.Array,
        valid: jax.Array,
        done: jax.Array,
        candidate_margin: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        final_margin = candidate_margin[:self.n_dirs]
        converged = done | ((final_margin <= 0.0) & (eta <= self.bound))
        candidate_valid = jnp.concatenate([
            valid & converged,
            jnp.ones((2,), dtype=bool),
        ])
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

    def _adamba(self, state: Any, nominal: jax.Array,
                key: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Pure-JAX AdamBA path used by compiled callers."""
        ray_actions, eta, valid, done = self._search_rays(state, nominal, key)
        candidates = self._candidate_actions(ray_actions, nominal)
        candidate_margin = self._candidate_margins(state, candidates)
        return self._select_candidate(
            nominal, candidates, eta, valid, done, candidate_margin
        )

    def _host_margins(self, state: Any, actions: np.ndarray) -> np.ndarray:
        """Evaluate one cached transition executable at a time.

        H1's realized dynamics are large enough that retaining separate search,
        margin and execution kernels exceeds a 16-GB CPU runtime.  Explicit
        host synchronization reuses only ``_transition_jit`` and releases each
        candidate next-state buffer before the next direction.
        """
        values = []
        for action in actions:
            next_state, margin = self._transition_jit(
                state, jnp.asarray(action)
            )
            jax.block_until_ready((next_state, margin))
            values.append(np.asarray(jax.device_get(margin)))
            del next_state, margin
        return np.asarray(values, dtype=np.float32)

    def _adamba_host(
        self, state: Any, nominal: jax.Array, key: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Memory-bounded AdamBA with the same rays and update rules."""
        nominal_np = np.asarray(jax.device_get(nominal))
        dtype = nominal_np.dtype
        nu = nominal_np.shape[0]
        dirs = jax.random.normal(key, (self.n_dirs, nu), dtype=nominal.dtype)
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
        dirs_np = np.asarray(jax.device_get(dirs), dtype=dtype)

        current = np.broadcast_to(nominal_np, (self.n_dirs, nu)).copy()
        eta = np.full((self.n_dirs,), self.bound, dtype=dtype)
        refining = np.zeros((self.n_dirs,), dtype=bool)
        valid = np.ones((self.n_dirs,), dtype=bool)
        done = np.zeros((self.n_dirs,), dtype=bool)

        for _ in range(self.n_iters):
            if not np.any(valid & ~done):
                break
            margin = self._host_margins(state, current)
            out_of_bounds = np.any(
                np.abs(current) > self.action_limit, axis=1
            )
            valid &= ~out_of_bounds
            safe = margin <= 0.0
            done |= safe & (eta <= self.bound)
            active = (~done) & valid
            expand = active & (~safe) & (~refining)
            start_refine = active & safe & (~refining)
            refine_unsafe = active & (~safe) & refining
            refine_safe = active & safe & refining
            signed_step = (
                np.where(expand | refine_unsafe, eta, 0.0)
                - np.where(refine_safe, eta, 0.0)
            ).astype(dtype, copy=False)
            current += signed_step[:, None] * dirs_np
            eta *= np.where(
                expand, np.asarray(2.0, dtype=dtype),
                np.where(
                    start_refine,
                    np.asarray(0.25, dtype=dtype),
                    np.asarray(0.5, dtype=dtype),
                ),
            )
            refining |= start_refine

        candidates = np.concatenate([
            np.clip(current, -self.action_limit, self.action_limit),
            nominal_np[None],
            np.zeros((1, nu), dtype=dtype),
        ], axis=0)
        candidate_margin = self._host_margins(state, candidates)
        converged = done | (
            (candidate_margin[:self.n_dirs] <= 0.0) & (eta <= self.bound)
        )
        candidate_valid = np.concatenate([
            valid & converged,
            np.ones((2,), dtype=bool),
        ])
        candidate_safe = candidate_valid & (candidate_margin <= 0.0)
        distance = np.sum((candidates - nominal_np[None]) ** 2, axis=1)
        if np.any(candidate_safe):
            index = int(np.argmin(np.where(candidate_safe, distance, np.inf)))
            found_safe = True
        else:
            index = int(np.argmin(np.where(
                candidate_valid, candidate_margin, np.inf
            )))
            found_safe = False
        projected = jnp.asarray(candidates[index], dtype=nominal.dtype)
        info = {
            "found_safe": jnp.asarray(found_safe),
            "failure": jnp.asarray(not found_safe),
            "margin": jnp.asarray(candidate_margin[index], dtype=nominal.dtype),
            "intervention": jnp.linalg.norm(projected - nominal),
        }
        return projected, info

    def project_and_step_with_info(self, state: Any, action: Any):
        """Project an action and return the already-realized next state.

        Candidate certification stays on the nominal model.  The selected
        action advances ``step_env`` so hidden OOD dynamics remain execution
        effects instead of becoming the ISSA projection model.
        """
        nominal = jnp.clip(
            jnp.asarray(action), -self.action_limit, self.action_limit
        )
        model_next_state, nominal_margin = self._transition_jit(state, nominal)
        jax.block_until_ready((model_next_state, nominal_margin))
        if float(nominal_margin) <= 0.0:
            info = {
                "found_safe": jnp.asarray(True),
                "failure": jnp.asarray(False),
                "margin": nominal_margin,
                "intervention": jnp.asarray(0.0, nominal.dtype),
            }
            next_state = (
                model_next_state
                if self.step_env is self.env
                else self._execution_step_jit(state, nominal)
            )
            jax.block_until_ready(next_state)
            self.last_info = info
            return nominal, next_state, info
        del model_next_state
        step = state.info.get("step", jnp.asarray(0, jnp.int32))
        key = jax.random.fold_in(self._key0, step.astype(jnp.uint32))
        projected, info = self._adamba_host(state, nominal, key)
        next_state = self._execution_step_jit(state, projected)
        jax.block_until_ready(next_state)
        self.last_info = info
        return projected, next_state, info

    def project_with_info(self, state: Any, action: Any):
        projected, _next_state, info = self.project_and_step_with_info(
            state, action
        )
        return projected, info

    def project_jax(self, state: Any, action: Any):
        """Pure-JAX projection for a compiled closed-loop rollout.

        ``project_with_info`` deliberately offers a memory-bounded Python
        interface for state-collecting evaluation.  Callers that do not retain
        intermediate states may compile this pure-JAX path.  Both paths use the
        same transition margin, rays and AdamBA update rules.
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
                         action_limit: float = 1.0, seed: int = 0,
                         step_env: Any = None) -> IssaProjection:
    return IssaProjection(
        env, n_dirs=n_dirs, n_iters=n_iters, bound=bound,
        threshold=threshold, enforce_absolute=enforce_absolute,
        action_limit=action_limit, seed=seed, step_env=step_env,
    )


__all__ = ["IssaProjection", "make_issa_projection"]
