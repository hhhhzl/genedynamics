"""
JAX implementation of stepping-stone retraction.

Extracted from ``twogo_jax.py._build_twogo_scan_kernels``.
Projects each foot onto the nearest stone and clips step length
using ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from genedynamics.genemetry.base import RetractionOperator
from genedynamics.genemetry.types import RetractionResult
from genedynamics.genemetry.registry import register_genemetry


@register_genemetry("retraction", "stepping", "jax")
class SteppingRetractionJax(RetractionOperator):
    """Stepping-stone retraction (JAX).

    For bipedal stepping-stone locomotion.  Each action encodes
    left-foot (dims 0:2) and right-foot (dims 2:4) displacements.

    The retraction:

    1. Proposes the raw displacement for each foot.
    2. Projects the resulting position onto the nearest stone.
    3. Clips the step length to ``l_max``.
    4. Clips the final actions to ``[-action_limit, action_limit]``.

    Parameters
    ----------
    stone_centers : (N, 2)
        Stone center positions (JAX array).
    stone_radii : (N,)
        Stone radii (JAX array).
    l_max : scalar
        Maximum per-step displacement (JAX scalar).
    action_limit : float
        Symmetric action clipping bound.
    """

    def __init__(
        self,
        *,
        stone_centers: Any,
        stone_radii: Any,
        l_max: Any,
        action_limit: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._stone_centers = stone_centers
        self._stone_radii = stone_radii
        self._l_max = l_max
        self._action_limit = float(action_limit)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_to_nearest_stone(
        self, point_xy: jnp.ndarray
    ) -> jnp.ndarray:
        """Project a 2-D point onto the nearest stone surface."""
        d = jnp.linalg.norm(
            point_xy[None, :] - self._stone_centers, axis=-1
        )
        i = jnp.argmin(d)
        c = self._stone_centers[i]
        r = self._stone_radii[i]
        v = point_xy - c
        n = jnp.linalg.norm(v)
        return jnp.where(
            n <= r,
            point_xy,
            c + (r / jnp.maximum(n, 1e-6)) * v,
        )

    @staticmethod
    def _clip_step(
        delta_xy: jnp.ndarray, lmax_xy: jnp.ndarray
    ) -> jnp.ndarray:
        """Clip a 2-D displacement vector to maximum length."""
        n = jnp.linalg.norm(delta_xy)
        scale = jnp.where(
            n > lmax_xy,
            lmax_xy / jnp.maximum(n, 1e-6),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        return delta_xy * scale

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retract(
        self,
        state: Any,
        trajectory: Any,
        params: Optional[Any] = None,
    ) -> RetractionResult:
        x0f = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
        a_in = jnp.asarray(trajectory, dtype=jnp.float32)
        H_local = a_in.shape[0]

        p0_l = x0f[:2]
        p0_r = x0f[2:4]
        step_lmax = self._l_max
        project = self._project_to_nearest_stone
        clip = self._clip_step

        def _step(carry, t):
            p_l, p_r, a_prev = carry
            u_t = a_in[t]

            # Left foot
            p_l_prop = p_l + u_t[:2]
            p_l_proj = project(p_l_prop)
            d_l = clip(p_l_proj - p_l, step_lmax)
            p_l_n = p_l + d_l

            # Right foot
            p_r_prop = p_r + u_t[2:4]
            p_r_proj = project(p_r_prop)
            d_r = clip(p_r_proj - p_r, step_lmax)
            p_r_n = p_r + d_r

            u_new = u_t.at[:2].set(d_l)
            u_new = u_new.at[2:4].set(d_r)
            a_next = a_prev.at[t].set(u_new)
            return (p_l_n, p_r_n, a_next), None

        (_, _, a_out), _ = jax.lax.scan(
            _step,
            (p0_l, p0_r, a_in),
            jnp.arange(H_local, dtype=jnp.int32),
        )
        clipped = jnp.clip(a_out, -self._action_limit, self._action_limit)
        return RetractionResult(trajectory=clipped)
