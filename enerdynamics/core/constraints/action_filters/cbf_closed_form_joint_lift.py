"""
Closed-form CBF projection filter for joint-space (e.g. 7D qdot) with 2D task-space SDF.

Lifts 2D CBF to action_dim via A = J_xy(q)^T grad_xy; condition A^T qdot >= b.
SDF/grad evaluated only at pos = state[:2]. Batch-optimal: one scan over time
with batched state/action when actions.ndim == 3.
"""

from __future__ import annotations

from typing import Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from enerdynamics.core.constraints.action_filters.base import ConstraintFilter

# Reuse same helpers as QP joint-lift for J_xy
from enerdynamics.core.constraints.action_filters.cbf_qp_joint_lift import (
    _get_jacobian_xy_batch,
    _get_jacobian_xy_single,
)


class ClosedFormCBFFilterJointLift(ConstraintFilter):
    """
    Closed-form CBF filter lifted to joint space: A = J_xy(q)^T grad_xy, A^T qdot >= b.

    Same tau gating and two-pass projection as 2D; SDF/grad at pos = state[:2] only.
    Batch-optimal: one scan over time with batched (B, state_dim) / (B, action_dim).
    """

    def apply_actions(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(actions, np.ndarray):
            return self._apply_actions_numpy(
                x0, actions, env=env, obstacles=obstacles,
                schedule_state=schedule_state, schedule_params=schedule_params,
            )

        params = schedule_params or {}
        tau = params.get("cbf_tau", 0.005)
        eta = params.get("cbf_eta", 1.5)
        margin = params.get("cbf_margin", 0.1)
        base_beta = params.get("base_beta", 0.05)
        robot_radius = getattr(env, "robot_radius", 0.05)
        dt = getattr(env, "dt", 0.05)
        control_limit = float(getattr(env, "control_limit", 1.0))

        if actions.ndim == 3:
            return self._apply_jax_batch(
                x0, actions, obstacles, env,
                tau, eta, margin, base_beta, robot_radius, dt, control_limit,
            )
        return self._apply_jax_single(
            x0, actions, obstacles, env,
            tau, eta, margin, base_beta, robot_radius, dt, control_limit,
        )

    def _apply_jax_single(
        self,
        x0: Any,
        u_seq: Any,
        obstacles: Any,
        env: Any,
        tau: float,
        eta: float,
        margin: float,
        base_beta: float,
        robot_radius: float,
        dt: float,
        control_limit: float,
    ) -> Any:
        act_dim = u_seq.shape[-1]
        out_shape = jax.ShapeDtypeStruct((2, 7), jnp.float32)

        def body_fn(carry, u):
            x = carry
            pos = x[:2]
            sdf, grad_xy = obstacles.sample_sdf_and_grad_2d(pos, backend="jax")
            h = sdf - (robot_radius + margin)
            grad_xy = jnp.reshape(grad_xy, (2,))
            J_xy = jax.pure_callback(
                lambda s: _get_jacobian_xy_single(env, np.asarray(s)),
                out_shape,
                x,
                vmap_method="sequential",
            )
            A = jnp.dot(J_xy.T, grad_xy)
            A = jnp.reshape(A, (act_dim,)) if A.size == act_dim else jnp.pad(A, (0, max(0, act_dim - A.size)))
            b = -(eta / dt) * h + base_beta

            lhs = jnp.dot(A, u)
            den = jnp.dot(A, A) + 1e-9
            need = (h < tau) & (jnp.linalg.norm(A) > 1e-6)
            alpha = jnp.maximum(0.0, (b - lhs) / den) * need
            u_safe = u + alpha * A
            lhs2 = jnp.dot(A, u_safe)
            alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
            u_safe = u_safe + alpha2 * A
            u_safe = jnp.clip(u_safe, -control_limit, control_limit)

            x_next = env.jax_transition(x, u_safe)
            return x_next, u_safe

        _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
        return u_seq_safe

    def _apply_jax_batch(
        self,
        x0: Any,
        actions: Any,
        obstacles: Any,
        env: Any,
        tau: float,
        eta: float,
        margin: float,
        base_beta: float,
        robot_radius: float,
        dt: float,
        control_limit: float,
    ) -> Any:
        B, H, act_dim = actions.shape
        state_dim = x0.shape[-1]
        out_shape_single = jax.ShapeDtypeStruct((2, 7), jnp.float32)

        def body_fn(carry, u):
            x_batch = carry
            pos_batch = x_batch[:, :2]
            sdf, grad_xy = obstacles.sample_sdf_and_grad_2d(pos_batch, backend="jax")
            h = sdf - (robot_radius + margin)
            if grad_xy.ndim == 1:
                grad_xy = jnp.broadcast_to(grad_xy, (B, 2))
            # vmap_method='sequential' so callback works when this scan is under vmap (e.g. MDOC plan_batch)
            J_xy_batch = jax.pure_callback(
                lambda s: _get_jacobian_xy_single(env, np.asarray(s)),
                out_shape_single,
                x_batch,
                vmap_method="sequential",
            )
            # Support both (B, 2, 7) and (outer, B, 2, 7) when scan is under vmap
            A_batch = jnp.einsum("...ij,...i->...j", J_xy_batch, grad_xy)
            if A_batch.shape[-1] < act_dim:
                pad_width = [(0, 0)] * (A_batch.ndim - 1) + [(0, act_dim - A_batch.shape[-1])]
                A_batch = jnp.pad(A_batch, pad_width)
            A_batch = A_batch[..., :act_dim]

            b = -(eta / dt) * h + base_beta
            lhs = jnp.sum(A_batch * u, axis=-1)
            den = jnp.sum(A_batch * A_batch, axis=-1) + 1e-9
            need = (h < tau) & (jnp.linalg.norm(A_batch, axis=-1) > 1e-6)
            alpha = jnp.maximum(0.0, (b - lhs) / den) * need
            u_safe = u + alpha[..., None] * A_batch
            lhs2 = jnp.sum(A_batch * u_safe, axis=-1)
            alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
            u_safe = u_safe + alpha2[..., None] * A_batch
            u_safe = jnp.clip(u_safe, -control_limit, control_limit)

            batch_shape = x_batch.shape[:-1]
            x_flat = x_batch.reshape(-1, state_dim)
            u_flat = u_safe.reshape(-1, act_dim)
            x_next_flat = jax.vmap(env.jax_transition, (0, 0))(x_flat, u_flat)
            x_next = x_next_flat.reshape(batch_shape + (state_dim,))
            return x_next, u_safe

        x0_flat = jnp.reshape(x0, (-1, state_dim))
        carry0 = jnp.broadcast_to(x0_flat[0:1], (B, state_dim)) if x0_flat.shape[0] == 1 else x0_flat
        # Scan over time (H): pass (H, B, act_dim) so body receives u of shape (B, act_dim)
        actions_t = jnp.swapaxes(actions, 0, 1)
        _, u_seq_safe = jax.lax.scan(body_fn, carry0, actions_t)
        return jnp.swapaxes(u_seq_safe, 0, 1)

    def _apply_actions_numpy(
        self,
        x0: Any,
        actions: np.ndarray,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
    ) -> np.ndarray:
        params = schedule_params or {}
        tau = float(params.get("cbf_tau", 0.005))
        eta = float(params.get("cbf_eta", 1.5))
        margin = float(params.get("cbf_margin", 0.1))
        base_beta = float(params.get("base_beta", 0.05))
        robot_radius = float(getattr(env, "robot_radius", 0.05))
        dt = float(getattr(env, "dt", 0.05))
        control_limit = float(getattr(env, "control_limit", 1.0))

        def _step_np(s: np.ndarray, a: np.ndarray) -> np.ndarray:
            if isinstance(s, (tuple, list)) and len(s) > 0:
                s = np.asarray(s[0], dtype=np.float32)
            s = np.asarray(s, dtype=np.float32)
            if hasattr(env, "step"):
                for args in [(s, a, 0, {}), (s, a)]:
                    try:
                        out = env.step(*args)
                        st = out[0] if isinstance(out, tuple) else out
                        return np.asarray(st, dtype=np.float32)
                    except TypeError:
                        continue
            if hasattr(env, "model_transition"):
                return np.asarray(env.model_transition(s, a), dtype=np.float32)
            return np.asarray(env.jax_transition(s, a), dtype=np.float32)

        if actions.ndim == 2:
            x = np.asarray(x0, dtype=np.float32).reshape(-1)
            safe_list = []
            for u in actions:
                pos = x[:2]
                sdf, grad = obstacles.sample_sdf_and_grad_2d(pos, backend="numpy")
                sdf = np.asarray(sdf, dtype=np.float32)
                grad = np.asarray(grad, dtype=np.float32).flatten()[:2]
                J_xy = _get_jacobian_xy_single(env, x)
                A = np.dot(J_xy.T, grad)
                act_dim = u.size
                if A.size < act_dim:
                    A = np.pad(A.astype(np.float32), (0, act_dim - A.size))
                A = np.asarray(A, dtype=np.float32)[:act_dim]

                h = float(sdf) - (robot_radius + margin)
                b = -(eta / dt) * h + base_beta
                lhs = float(np.dot(A, u))
                den = float(np.dot(A, A) + 1e-9)
                need = (np.linalg.norm(A) > 1e-6) and (h < tau)
                alpha = max(0.0, (b - lhs) / den) * (1.0 if need else 0.0)
                u_safe = u + alpha * A
                lhs2 = float(np.dot(A, u_safe))
                alpha2 = max(0.0, (b - lhs2) / den) * (1.0 if (need and lhs2 < b) else 0.0)
                u_safe = u_safe + alpha2 * A
                u_safe = np.clip(u_safe, -control_limit, control_limit).astype(np.float32)
                safe_list.append(u_safe)
                x = _step_np(x, u_safe)
            return np.stack(safe_list, axis=0)

        B, H, act_dim = actions.shape
        x_batch = np.tile(np.asarray(x0, dtype=np.float32).reshape(1, -1), (B, 1))
        safe = np.zeros_like(actions, dtype=np.float32)
        for t in range(H):
            pos_batch = x_batch[:, :2]
            sdf, grad_xy = obstacles.sample_sdf_and_grad_2d(pos_batch, backend="numpy")
            sdf = np.asarray(sdf, dtype=np.float32)
            grad_xy = np.asarray(grad_xy, dtype=np.float32)
            if grad_xy.ndim == 1:
                grad_xy = np.broadcast_to(grad_xy, (B, 2))
            J_xy_batch = _get_jacobian_xy_batch(env, x_batch)
            A_batch = np.einsum("bij,bi->bj", J_xy_batch, grad_xy)
            if A_batch.shape[-1] < act_dim:
                A_batch = np.pad(A_batch, ((0, 0), (0, act_dim - A_batch.shape[-1])))
            A_batch = A_batch.astype(np.float32)

            h = sdf - (robot_radius + margin)
            b = -(eta / dt) * h + base_beta
            u_t = actions[:, t, :]
            lhs = np.sum(A_batch * u_t, axis=-1)
            den = np.sum(A_batch * A_batch, axis=-1) + 1e-9
            need = (np.linalg.norm(A_batch, axis=-1) > 1e-6) & (h < tau)
            alpha = np.maximum(0.0, (b - lhs) / den) * np.where(need, 1.0, 0.0)
            u_safe = u_t + alpha[:, None] * A_batch
            lhs2 = np.sum(A_batch * u_safe, axis=-1)
            alpha2 = np.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
            u_safe = u_safe + alpha2[:, None] * A_batch
            u_safe = np.clip(u_safe, -control_limit, control_limit).astype(np.float32)
            safe[:, t, :] = u_safe
            for b in range(B):
                x_batch[b] = _step_np(x_batch[b], u_safe[b])
        return safe
