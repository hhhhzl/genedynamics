"""
Per-foot CBF projection filter for stepping-stones tasks.

For each foot, computes a barrier h_i = sdf_stone(foot_i) - margin,
and projects the action to satisfy ḣ_i + α·h_i ≥ 0 per step.

Unlike the generic CBF (which uses body-position SDF), this filter:
- Decodes 4 foot positions from the 16D state
- Computes per-foot SDF to nearest stone
- Projects via per-foot barrier gradients lifted to the 12D action space
- Respects phase-dependent weights (support vs swing)
"""

from __future__ import annotations

from typing import Any, Optional
import jax
import jax.numpy as jnp
from genedynamics.core.constraints.action_filters.base import ConstraintFilter


class SteppingCBFFilter(ConstraintFilter):
    """Per-foot closed-form CBF projection for stepping stones."""

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
        tau = jnp.asarray(params.get("cbf_tau", 0.01), dtype=jnp.float32)
        eta = jnp.asarray(params.get("cbf_eta", 1.0), dtype=jnp.float32)
        margin = jnp.asarray(params.get("cbf_margin", 0.01), dtype=jnp.float32)
        base_beta = jnp.asarray(params.get("base_beta", 0.02), dtype=jnp.float32)
        dt = jnp.asarray(getattr(env, "dt", 1.0), dtype=jnp.float32)
        control_limit = float(getattr(env, "control_limit", 0.45))

        # Pre-fetch stone geometry from env scene.
        scene = getattr(env, "scene", None)
        if scene is None:
            # No stepping scene → noop.
            return actions
        centers = jnp.asarray(scene.stones_centers, dtype=jnp.float32)
        radii = jnp.asarray(scene.stones_radii, dtype=jnp.float32)
        stone_margin = jnp.asarray(
            float(getattr(env, "stone_margin", 0.03)), dtype=jnp.float32
        )

        # Check for support platforms (static Python check, not traced).
        platforms = getattr(scene, "support_platforms", None)
        has_platforms = platforms is not None and len(platforms) > 0
        if has_platforms:
            plats = jnp.asarray(platforms, dtype=jnp.float32)
        else:
            plats = jnp.zeros((1, 4), dtype=jnp.float32)  # dummy

        has_stones = int(centers.shape[0]) > 0

        def _foot_sdf(p):
            """Signed distance of foot position p to nearest stone/platform."""
            # Distance to circular stones (eps for grad stability at center).
            d_stones = jnp.sqrt(
                jnp.sum((p[None, :] - centers) ** 2, axis=-1) + 1e-10
            ) - (radii - stone_margin)
            min_stone = jnp.min(d_stones) if has_stones else jnp.asarray(1e9, dtype=jnp.float32)

            # Distance to rectangular platforms.
            def _plat_dist(bounds):
                xm, xM, ym, yM = bounds[0], bounds[1], bounds[2], bounds[3]
                xl, xR = xm + stone_margin, xM - stone_margin
                yb, yT = ym + stone_margin, yM - stone_margin
                dx = jnp.maximum(0.0, jnp.maximum(xl - p[0], p[0] - xR))
                dy = jnp.maximum(0.0, jnp.maximum(yb - p[1], p[1] - yT))
                return jnp.sqrt(dx * dx + dy * dy + 1e-10)

            if has_platforms:
                plat_dists = jax.vmap(_plat_dist)(plats)
                min_plat = jnp.min(plat_dists)
            else:
                min_plat = jnp.asarray(1e9, dtype=jnp.float32)

            return jnp.minimum(min_stone, min_plat)

        # Safe gradient: replace NaN with zero.
        _foot_sdf_grad_raw = jax.grad(_foot_sdf)

        def _foot_sdf_grad(p):
            g = _foot_sdf_grad_raw(p)
            return jnp.where(jnp.isnan(g), 0.0, g)

        def filter_single(u_seq):
            def body_fn(carry, u):
                x = carry

                # Decode feet from state.
                feet4 = env.jax_safety_points(x)        # (4, 2)
                w = env.jax_cfs_safety_point_weights(x)  # (4,)

                u_safe = u

                # Per-foot CBF projection (sequential over 4 feet).
                # Action layout: [body_xy(2), yaw(1), res_FL(2), res_FR(2),
                #                  res_RL(2), res_RR(2), dtau(1)] = 12D
                # Foot i's residual lives at dims 3+2*i : 3+2*i+2.
                foot_res_offsets = jnp.array([3, 5, 7, 9], dtype=jnp.int32)

                def _project_foot(carry_u, foot_info):
                    u_cur = carry_u
                    foot_pos, weight, foot_idx = foot_info

                    sdf = _foot_sdf(foot_pos)
                    grad_p = _foot_sdf_grad(foot_pos)    # (2,)
                    h = sdf - margin

                    # Lift barrier gradient to 12D action space.
                    # d(foot_i)/d(u) has contributions from:
                    #   body rate (dims 0:2) — shared, moves all feet
                    #   residual rate (dims 3+2i : 3+2i+2) — per-foot
                    # Project onto RESIDUAL dims only to avoid
                    # cross-foot interference through body rate.
                    res_off = foot_res_offsets[foot_idx]
                    A = jnp.zeros(u_cur.shape[-1], dtype=jnp.float32)
                    A = A.at[res_off].set(dt * grad_p[0])
                    A = A.at[res_off + 1].set(dt * grad_p[1])

                    b = -(eta / dt) * h * weight + base_beta

                    lhs = jnp.dot(A, u_cur)
                    den = jnp.dot(A, A) + 1e-9
                    need = (h < tau) & (jnp.linalg.norm(grad_p) > 1e-6) & (weight > 0.1)

                    # Two-pass projection.
                    alpha = jnp.maximum(0.0, (b - lhs) / den) * need
                    u_new = u_cur + alpha * A
                    lhs2 = jnp.dot(A, u_new)
                    alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
                    u_new = u_new + alpha2 * A
                    return u_new, None

                foot_indices = jnp.arange(4, dtype=jnp.int32)
                u_safe, _ = jax.lax.scan(
                    _project_foot,
                    u_safe,
                    (feet4, w, foot_indices),
                )
                u_safe = jnp.clip(u_safe, -control_limit, control_limit)
                x_next = env.jax_transition(x, u_safe)
                return x_next, u_safe

            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe

        if actions.ndim == 3:
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)
