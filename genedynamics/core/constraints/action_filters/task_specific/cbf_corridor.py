"""
Per-body-part CBF projection filter for humanoid corridor obstacle avoidance.

For each collision body (torso, arm_L, arm_R), computes a barrier
    h_k = sdf(body_k, obstacle) - margin
and projects the action to satisfy  dh_k + alpha * h_k >= 0.

The filter:
- Extracts 3 body positions from the 14D state via env.jax_safety_points()
- Computes per-body SDF to all obstacles and corridor walls
- Lifts the 2D spatial gradient to the 9D action space via chain rule
- Applies two-pass closed-form CBF projection with tau gating
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from genedynamics.core.constraints.action_filters.base import ConstraintFilter


class CorridorCBFFilter(ConstraintFilter):
    """Per-body closed-form CBF projection for corridor obstacle avoidance."""

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
        params = schedule_params or {}
        tau = jnp.asarray(params.get("cbf_tau", 0.02), dtype=jnp.float32)
        eta = jnp.asarray(params.get("cbf_eta", 1.2), dtype=jnp.float32)
        margin = jnp.asarray(params.get("cbf_margin", 0.03), dtype=jnp.float32)
        base_beta = jnp.asarray(params.get("base_beta", 0.01), dtype=jnp.float32)
        dt = jnp.asarray(getattr(env, "dt", 0.2), dtype=jnp.float32)

        u_lo = jnp.asarray(env._u_lo, dtype=jnp.float32)
        u_hi = jnp.asarray(env._u_hi, dtype=jnp.float32)
        act_dim = int(env.act_dim)

        scene = getattr(env, "scene", None)
        if scene is None:
            return actions

        n_obs = int(env._n_obs)
        obs_xmin = jnp.asarray(env._obs_xmin, dtype=jnp.float32)
        obs_xmax = jnp.asarray(env._obs_xmax, dtype=jnp.float32)
        obs_ymin = jnp.asarray(env._obs_ymin, dtype=jnp.float32)
        obs_ymax = jnp.asarray(env._obs_ymax, dtype=jnp.float32)
        obs_zmin = jnp.asarray(env._obs_zmin, dtype=jnp.float32)
        obs_zmax = jnp.asarray(env._obs_zmax, dtype=jnp.float32)
        w_lo = jnp.asarray(scene.wall_y_min, dtype=jnp.float32)
        w_hi = jnp.asarray(scene.wall_y_max, dtype=jnp.float32)

        TORSO_A = 0.20
        TORSO_B = 0.11
        TORSO_CROUCH_EXTRA = 0.03
        ARM_RADIUS = 0.05
        ARM_REACH_OPEN = 0.35
        ARM_REACH_TUCKED = 0.10
        BODY_HALF_H = 0.25
        H_NOMINAL = 0.75
        _S_X, _S_Y, _S_PSI, _S_H, _S_PSI_T = 0, 1, 2, 3, 4
        _S_AL, _S_AR = 5, 6
        STATE_DIM = 14

        def _box_sdf(px, py, xmin, xmax, ymin, ymax):
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            hx = 0.5 * (xmax - xmin)
            hy = 0.5 * (ymax - ymin)
            dx = jnp.abs(px - cx) - hx
            dy = jnp.abs(py - cy) - hy
            outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2 + 1e-12)
            inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
            return outside + inside

        def _z_overlap(h, z_lo, z_hi):
            body_lo = h - BODY_HALF_H
            body_hi = h + BODY_HALF_H
            ov = jnp.minimum(body_hi, z_hi) - jnp.maximum(body_lo, z_lo)
            return jnp.clip(ov / (2.0 * BODY_HALF_H), 0.0, 1.0)

        def _body_min_sdf(x):
            """Differentiable min SDF across all bodies and obstacles."""
            px, py = x[_S_X], x[_S_Y]
            h = x[_S_H]
            heading = x[_S_PSI] + x[_S_PSI_T]

            # Torso effective radius (use max for conservative bound).
            a_eff = TORSO_A + TORSO_CROUCH_EXTRA * jnp.maximum(0.0, H_NOMINAL - h)

            # Wall SDF
            d_min = jnp.minimum(py - w_lo - a_eff, w_hi - py - a_eff)

            # Arm positions
            def _arm_pos(a_tuck, sign):
                reach = ARM_REACH_OPEN + (ARM_REACH_TUCKED - ARM_REACH_OPEN) * jnp.clip(a_tuck, 0.0, 1.0)
                c = jnp.cos(heading)
                s = jnp.sin(heading)
                return jnp.stack([px - s * sign * reach, py + c * sign * reach])

            arm_L = _arm_pos(x[_S_AL], 1.0)
            arm_R = _arm_pos(x[_S_AR], -1.0)

            # Arm wall SDF
            for ap in [arm_L, arm_R]:
                d_min = jnp.minimum(d_min, ap[1] - w_lo - ARM_RADIUS)
                d_min = jnp.minimum(d_min, w_hi - ap[1] - ARM_RADIUS)

            if n_obs > 0:
                z_ov = _z_overlap(h, obs_zmin, obs_zmax)
                z_act = z_ov > 1e-4
                z_s = jnp.maximum(z_ov, 1e-6)

                obs_cx = 0.5 * (obs_xmin + obs_xmax)
                obs_cy = 0.5 * (obs_ymin + obs_ymax)
                obs_hx = 0.5 * (obs_xmax - obs_xmin)
                obs_hy = 0.5 * (obs_ymax - obs_ymin)

                # Torso to each obstacle
                dx_t = jnp.abs(px - obs_cx) - obs_hx
                dy_t = jnp.abs(py - obs_cy) - obs_hy
                out_t = jnp.sqrt(jnp.maximum(dx_t, 0.0) ** 2 + jnp.maximum(dy_t, 0.0) ** 2 + 1e-12)
                in_t = jnp.minimum(jnp.maximum(dx_t, dy_t), 0.0)
                box_t = out_t + in_t
                dp = jnp.arctan2(obs_cy - py, obs_cx - px) - heading
                r_e = 1.0 / jnp.sqrt((jnp.cos(dp) / a_eff) ** 2 + (jnp.sin(dp) / TORSO_B) ** 2 + 1e-12)
                tsdf = jnp.where(z_act, (box_t - r_e) / z_s, 1e6)
                d_min = jnp.minimum(d_min, jnp.min(tsdf))

                # Arms to each obstacle
                for ap in [arm_L, arm_R]:
                    dx_a = jnp.abs(ap[0] - obs_cx) - obs_hx
                    dy_a = jnp.abs(ap[1] - obs_cy) - obs_hy
                    out_a = jnp.sqrt(jnp.maximum(dx_a, 0.0) ** 2 + jnp.maximum(dy_a, 0.0) ** 2 + 1e-12)
                    in_a = jnp.minimum(jnp.maximum(dx_a, dy_a), 0.0)
                    box_a = out_a + in_a
                    asdf = jnp.where(z_act, (box_a - ARM_RADIUS) / z_s, 1e6)
                    d_min = jnp.minimum(d_min, jnp.min(asdf))

            return d_min

        # Gradient of min SDF w.r.t. state.
        _sdf_grad_raw = jax.grad(_body_min_sdf)

        def _sdf_grad(x):
            g = _sdf_grad_raw(x)
            return jnp.where(jnp.isnan(g), 0.0, g)

        def filter_single(u_seq):
            def body_fn(carry, u):
                x = carry

                sdf_val = _body_min_sdf(x)
                h_barrier = sdf_val - margin

                # Gradient of SDF w.r.t. state.
                g_state = _sdf_grad(x)  # (STATE_DIM,)

                # Lift to action space via chain rule: d(sdf)/d(u) = d(sdf)/d(x_next) * d(x_next)/d(u).
                # For velocity-rate dynamics, d(x_next)/d(u) is sparse and analytic.
                # We use autodiff for correctness.
                def sdf_of_action(u_):
                    x_next = env.jax_model_transition(x, u_)
                    return _body_min_sdf(x_next)

                A = jax.grad(sdf_of_action)(u)  # (act_dim,)
                A = jnp.where(jnp.isnan(A), 0.0, A)

                # CBF condition: A · u >= b  where b = -(eta/dt) * h * 1 + base_beta
                b = -(eta / dt) * h_barrier + base_beta

                lhs = jnp.dot(A, u)
                den = jnp.dot(A, A) + 1e-9
                need = (h_barrier < tau) & (jnp.linalg.norm(A) > 1e-6)

                # Two-pass projection.
                alpha1 = jnp.maximum(0.0, (b - lhs) / den) * need
                u_new = u + alpha1 * A
                lhs2 = jnp.dot(A, u_new)
                alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
                u_safe = u_new + alpha2 * A

                u_safe = jnp.clip(u_safe, u_lo, u_hi)
                x_next = env.jax_model_transition(x, u_safe)
                return x_next, u_safe

            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe

        if actions.ndim == 3:
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)
