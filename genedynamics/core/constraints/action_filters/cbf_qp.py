"""
QP-based CBF projection filter for single-integrator dynamics.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import jax
import jax.numpy as jnp
import numpy as np
from genedynamics.core.constraints.action_filters.base import ConstraintFilter

class QPBasedCBFFilter(ConstraintFilter):
    """
    QP-based projection filter based on Control Barrier Functions.
    
    Aligned with Torch mdoc.py _rollout_single_batch CBF behavior:
    - Tau gating: only apply when h < cbf_tau (near obstacle), including Pass 2
    - Two-pass hard projection (Pass 1 + Pass 2 re-check)
    - b = -(eta/dt)*h + base_beta
    - Action clipping to control_limit after projection
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
        # NumPy fallback to avoid JAX tracers when backend="numpy"
        if isinstance(actions, np.ndarray):
            return self._apply_actions_numpy(
                x0,
                actions,
                env=env,
                obstacles=obstacles,
                schedule_state=schedule_state,
                schedule_params=schedule_params,
            )

        params = schedule_params or {}
        tau = params.get("cbf_tau", 0.005)
        eta = params.get("cbf_eta", 1.5)
        margin = params.get("cbf_margin", 0.1)
        base_beta = params.get("base_beta", 0.05)

        robot_radius = getattr(env, "robot_radius", 0.05)
        dt = getattr(env, "dt", 0.05)
        control_limit = float(getattr(env, "control_limit", 1.0))

        def filter_single(u_seq):
            def body_fn(carry, u):
                x = carry
                sdf, grad = obstacles.sample_sdf_and_grad_2d(x, backend="jax")

                h = sdf - (robot_radius + margin)
                A = grad
                b = -(eta / dt) * h + base_beta

                lhs = jnp.dot(A, u)
                den = jnp.dot(A, A) + 1e-9
                goodA = jnp.linalg.norm(A) > 1e-6
                # Tau gating: only apply when h < tau (near obstacle boundary)
                need = goodA & (h < tau)

                # Pass 1 (hard projection, no slack)
                alpha = jnp.maximum(0.0, (b - lhs) / den) * need
                u_safe = u + alpha * A

                # Pass 2 (re-check with updated u; also gated by h < tau)
                lhs2 = jnp.dot(A, u_safe)
                bad = need & (lhs2 < b)
                alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * bad
                u_safe = u_safe + alpha2 * A

                # Clip to control limit (filter output must be valid)
                u_safe = jnp.clip(u_safe, -control_limit, control_limit)

                x_next = env.jax_transition(x, u_safe)
                return x_next, u_safe

            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe

        if actions.ndim == 3:
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)

    # ---------------- NumPy fallback (for numpy backend) ---------------- #
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

        def _step_env_np(s: np.ndarray, a: np.ndarray) -> np.ndarray:
            def _to_state(x):
                if isinstance(x, (tuple, list)) and len(x) > 0:
                    return np.asarray(x[0], dtype=np.float32)
                return np.asarray(x, dtype=np.float32)

            if hasattr(env, "step"):
                for args in [(s, a, int(0), {}), (s, a)]:
                    try:
                        return _to_state(env.step(*args))
                    except TypeError:
                        continue
            if hasattr(env, "model_transition"):
                try:
                    return _to_state(env.model_transition(s, a))
                except TypeError:
                    pass
            return _to_state(env.jax_transition(s, a))

        def filter_single_np(u_seq: np.ndarray) -> np.ndarray:
            x = np.asarray(x0, dtype=np.float32)
            safe_actions = []
            for u in u_seq:
                sdf, grad = obstacles.sample_sdf_and_grad_2d(x, backend="numpy")
                sdf = np.asarray(sdf, dtype=np.float32)
                grad = np.asarray(grad, dtype=np.float32)

                h = float(sdf) - (robot_radius + margin)
                A = np.asarray(grad, dtype=np.float32).flatten()
                b = -(eta / dt) * h + base_beta

                lhs = float(np.dot(A, u))
                den = float(np.dot(A, A) + 1e-9)
                goodA = float(np.linalg.norm(A)) > 1e-6
                need = goodA and (h < tau)

                # Pass 1 (hard projection)
                alpha = max(0.0, (b - lhs) / den) * (1.0 if need else 0.0)
                u_safe = u + alpha * A

                # Pass 2 (re-check; also gated by h < tau)
                lhs2 = float(np.dot(A, u_safe))
                bad = need and (lhs2 < b)
                alpha2 = max(0.0, (b - lhs2) / den) * (1.0 if bad else 0.0)
                u_safe = u_safe + alpha2 * A

                u_safe = np.clip(u_safe, -control_limit, control_limit)

                safe_actions.append(u_safe.astype(np.float32))
                x = _step_env_np(x, u_safe)
            return np.stack(safe_actions, axis=0)

        if actions.ndim == 3:
            return np.stack([filter_single_np(a) for a in actions], axis=0)
        return filter_single_np(actions)
