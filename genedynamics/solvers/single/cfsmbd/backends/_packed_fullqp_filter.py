"""
Packed-array obstacle path for CFSQPFullFilter (JAX).

Goal (Phase 5):
- Remove Python obstacle branch trees (`lax.switch` over Python objects) from the hot path.
- Represent obstacles as pure arrays (circle/box primitives) and compute SDF+grad in one vectorized pass.

Goal (Phase 6 support):
- Keep logging host-side and optional; return histories via the existing planner results.

Implementation strategy:
- Provide a function that can monkey-patch a `CFSQPFullFilter` instance's `_apply_actions_jax`.
- The patched implementation only activates when obstacles are ALL (SphereObstacle, BoxObstacle) and
  these are axis-aligned primitives with `center`, `radius` / `half_extents`.
- Otherwise, it falls back to the original `_apply_actions_jax` to preserve behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PackedObstacles2D:
    circle_centers: jnp.ndarray  # (Nc,2)
    circle_radii: jnp.ndarray  # (Nc,)
    box_centers: jnp.ndarray  # (Nb,2)
    box_half: jnp.ndarray  # (Nb,2)


def _pack_obstacles_2d(filter_obj: Any, obstacles: Any) -> Optional[PackedObstacles2D]:
    """Return packed arrays if all obstacles are Sphere/Box; else None."""
    obstacles_list = filter_obj._get_obstacles_list(obstacles)
    if not obstacles_list:
        return PackedObstacles2D(
            circle_centers=jnp.zeros((0, 2), dtype=jnp.float32),
            circle_radii=jnp.zeros((0,), dtype=jnp.float32),
            box_centers=jnp.zeros((0, 2), dtype=jnp.float32),
            box_half=jnp.zeros((0, 2), dtype=jnp.float32),
        )
    try:
        from genedynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
    except Exception:
        return None

    circles = [o for o in obstacles_list if isinstance(o, SphereObstacle)]
    boxes = [o for o in obstacles_list if isinstance(o, BoxObstacle)]
    others = [o for o in obstacles_list if not isinstance(o, (SphereObstacle, BoxObstacle))]
    if len(others) != 0:
        return None

    import numpy as np

    cc = (
        np.asarray([np.asarray(o.center, dtype=np.float32)[:2] for o in circles], dtype=np.float32)
        if circles
        else np.zeros((0, 2), dtype=np.float32)
    )
    cr = np.asarray([float(o.radius) for o in circles], dtype=np.float32) if circles else np.zeros((0,), dtype=np.float32)
    bc = (
        np.asarray([np.asarray(o.center, dtype=np.float32)[:2] for o in boxes], dtype=np.float32)
        if boxes
        else np.zeros((0, 2), dtype=np.float32)
    )
    bh = (
        np.asarray([np.asarray(o.half_extents, dtype=np.float32)[:2] for o in boxes], dtype=np.float32)
        if boxes
        else np.zeros((0, 2), dtype=np.float32)
    )

    return PackedObstacles2D(
        circle_centers=jnp.asarray(cc, dtype=jnp.float32),
        circle_radii=jnp.asarray(cr, dtype=jnp.float32),
        box_centers=jnp.asarray(bc, dtype=jnp.float32),
        box_half=jnp.asarray(bh, dtype=jnp.float32),
    )


def _circle_sdf_grad(p: jnp.ndarray, centers: jnp.ndarray, radii: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # p: (H,2), centers: (Nc,2)
    eps = jnp.asarray(1e-6, dtype=jnp.float32)
    if centers.shape[0] == 0:
        sdf = jnp.full((0, p.shape[0]), jnp.inf, dtype=jnp.float32)
        grad = jnp.zeros((0, p.shape[0], 2), dtype=jnp.float32)
        return sdf, grad
    d = p[None, :, :] - centers[:, None, :]  # (Nc,H,2)
    dist = jnp.linalg.norm(d, axis=-1)  # (Nc,H)
    sdf = dist - radii[:, None]
    grad = d / (dist[:, :, None] + eps)
    return sdf, grad


def _box_sdf_grad(p: jnp.ndarray, centers: jnp.ndarray, half: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Axis-aligned box SDF + subgradient; matches the logic in CFSQPFullFilter fast path.
    eps = jnp.asarray(1e-6, dtype=jnp.float32)
    if centers.shape[0] == 0:
        sdf = jnp.full((0, p.shape[0]), jnp.inf, dtype=jnp.float32)
        grad = jnp.zeros((0, p.shape[0], 2), dtype=jnp.float32)
        return sdf, grad
    rel = p[None, :, :] - centers[:, None, :]  # (Nb,H,2)
    q = jnp.abs(rel) - half[:, None, :]  # (Nb,H,2)
    outside = jnp.maximum(q, 0.0)
    outside_dist = jnp.linalg.norm(outside, axis=-1)  # (Nb,H)
    inside_dist = jnp.max(q, axis=-1)  # <=0 inside
    outside_mask = outside_dist > eps
    sdf = jnp.where(outside_mask, outside_dist, inside_dist)

    # Outside gradient via closest point
    closest = centers[:, None, :] + jnp.clip(rel, -half[:, None, :], half[:, None, :])
    d = p[None, :, :] - closest
    grad_out = d / (outside_dist[:, :, None] + eps)

    # Inside gradient: axis of max(q), direction = sign(rel) but avoid 0 -> 0
    axis = jnp.argmax(q, axis=-1)  # (Nb,H)
    onehot = jax.nn.one_hot(axis, 2, dtype=jnp.float32)  # (Nb,H,2)
    rel_sign = jnp.where(rel >= 0.0, 1.0, -1.0)
    grad_in = onehot * rel_sign
    grad = jnp.where(outside_mask[:, :, None], grad_out, grad_in)
    return sdf, grad


def patch_cfsqp_full_filter_to_packed_obstacles(filter_obj: Any) -> None:
    """
    Monkey-patch a `CFSQPFullFilter` instance to use packed-array obstacles when possible.
    """
    orig_apply = getattr(filter_obj, "_apply_actions_jax")

    def _apply_actions_jax_packed(
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ):
        # If no obstacles, match original behavior (no-op).
        if obstacles is None:
            return actions

        packed = _pack_obstacles_2d(filter_obj, obstacles)
        if packed is None:
            return orig_apply(x0, actions, env=env, obstacles=obstacles, schedule_state=schedule_state, schedule_params=schedule_params, **kwargs)

        params_dict = schedule_params or {}
        margin_raw = params_dict.get("margin", 0.0)
        rho_raw = params_dict.get("rho", 10.0)
        I_QP_raw = params_dict.get("I_QP", params_dict.get("cfs_outer_iters", 1))
        qp_gate = params_dict.get("qp_gate", True)
        qp_prob = params_dict.get("qp_prob", 1.0)
        rng_key = params_dict.get("rng_key", kwargs.get("rng_key", None))
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        margin = margin_raw if isinstance(margin_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(float(margin_raw), dtype=jnp.float32)
        rho = rho_raw if isinstance(rho_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(float(rho_raw), dtype=jnp.float32)
        I_QP = jnp.asarray(I_QP_raw, dtype=jnp.int32)
        I_QP = jnp.clip(I_QP, 1, 64)

        # JAX-friendly gating
        qp_gate_pred = jnp.asarray(qp_gate, dtype=jnp.bool_)
        qp_prob_jax = jnp.asarray(qp_prob, dtype=jnp.float32)
        do_qp = jnp.logical_and(qp_gate_pred, jax.random.uniform(rng_key) < qp_prob_jax)

        x0 = jnp.asarray(x0, dtype=jnp.float32)
        dt = jnp.asarray(float(getattr(env, "dt", 0.05)), dtype=jnp.float32)
        robot_radius = jnp.asarray(float(getattr(env, "robot_radius", 0.05)), dtype=jnp.float32)
        constraint_margin = jnp.asarray(float(getattr(filter_obj, "constraint_margin", 0.25)), dtype=jnp.float32)
        control_limit = float(getattr(env, "control_limit", 1.0))

        def filter_single(u_seq: jnp.ndarray, margin_s: jnp.ndarray, rho_s: jnp.ndarray, I_s: jnp.ndarray) -> jnp.ndarray:
            margin_s = jnp.asarray(margin_s, dtype=jnp.float32)
            rho_s = jnp.asarray(rho_s, dtype=jnp.float32)
            I_s = jnp.asarray(I_s, dtype=jnp.int32)
            clearance = margin_s + robot_radius
            threshold = clearance + constraint_margin

            # Outer CFS iterations: fixed upper bound with masking (batch-friendly)
            MAX_OUTER = 64
            I_s = jnp.clip(I_s, 1, MAX_OUTER)

            time_idx = jnp.arange(u_seq.shape[0], dtype=jnp.int32)

            def one_outer(i, u_curr):
                # Rollout states from u_curr
                def step_fn(carry, a):
                    s_next = env.jax_transition(carry, a)
                    return s_next, s_next
                _, states = jax.lax.scan(step_fn, x0, u_curr)
                states = jnp.concatenate([x0[None, :], states], axis=0)  # (H+1,sdim)

                # Constrain p_{t+1}
                pos = states[1:, 0:2]  # (H,2)
                p0 = x0[0:2]

                sdf_c, grad_c = _circle_sdf_grad(pos, packed.circle_centers, packed.circle_radii)  # (Nc,H), (Nc,H,2)
                sdf_b, grad_b = _box_sdf_grad(pos, packed.box_centers, packed.box_half)  # (Nb,H), (Nb,H,2)
                sdf_all = jnp.concatenate([sdf_c, sdf_b], axis=0)  # (Nobs,H)
                grad_all = jnp.concatenate([grad_c, grad_b], axis=0)  # (Nobs,H,2)

                Nobs = sdf_all.shape[0]
                Ksel = jnp.asarray(int(getattr(filter_obj, "max_constraints_per_point", 8)), dtype=jnp.int32)
                Ksel = jnp.minimum(Ksel, jnp.asarray(max(1, int(Nobs)), dtype=jnp.int32))
                Ksel = jnp.maximum(Ksel, 1)

                # Top-K closest per timestep
                sdf_t = sdf_all.T  # (H,Nobs)
                grad_t = jnp.transpose(grad_all, (1, 0, 2))  # (H,Nobs,2)
                cand_mask = sdf_t < threshold
                sdf_for_sort = jnp.where(cand_mask, sdf_t, jnp.inf)
                neg = -sdf_for_sort

                def topk_idx(v):
                    return jax.lax.top_k(v, int(getattr(filter_obj, "max_constraints_per_point", 8)))[1]
                idx_sel = jax.vmap(topk_idx)(neg)  # (H,Ksel) (Ksel is Python int here)
                sdf_sel = jnp.take_along_axis(sdf_t, idx_sel, axis=1)  # (H,Ksel)
                grad_sel = jnp.take_along_axis(grad_t, idx_sel[:, :, None], axis=1)  # (H,Ksel,2)
                valid_sel = jnp.isfinite(sdf_sel) & (sdf_sel < threshold)
                grad_sel = jnp.where(valid_sel[:, :, None], grad_sel, 0.0)

                rhs = clearance - sdf_sel + jnp.einsum("hkd,hd->hk", grad_sel, pos)
                b = rhs - jnp.einsum("hkd,d->hk", grad_sel, p0)
                b = jnp.where(valid_sel, b, -jnp.inf)

                # Solve structured slack-QP in prefix-sum form
                from genedynamics.core.constraints.solvers.jaxopt_osqp_solver import solve_slack_qp_prefixsum_jax
                rho_eff = jax.lax.cond(
                    jnp.asarray(bool(getattr(filter_obj, "use_slack", True)), dtype=jnp.bool_) & (rho_s > 0),
                    lambda _: rho_s,
                    lambda _: jnp.asarray(1e9, dtype=jnp.float32),
                    operand=None,
                )
                solver_iters = jnp.maximum(jnp.asarray(10, dtype=jnp.int32), I_s * 20)
                u_next, _v = solve_slack_qp_prefixsum_jax(
                    u_curr,
                    (dt * grad_sel).astype(jnp.float32),
                    b.astype(jnp.float32),
                    rho_eff,
                    control_limit=control_limit,
                    tol=1e-7,
                    maxiter=solver_iters,
                )
                # Mask outer loop
                return jax.lax.cond(i < I_s, lambda _: u_next, lambda _: u_curr, operand=None)

            u_out = jax.lax.fori_loop(0, 64, one_outer, jnp.asarray(u_seq, dtype=jnp.float32))
            return u_out

        def do_filter(_):
            # Batch support (C,H,D) with per-sample params
            if actions.ndim == 3:
                if getattr(margin, "ndim", 0) > 0:
                    return jax.vmap(filter_single, in_axes=(0, 0, 0, 0))(actions, margin, rho, I_QP)
                return jax.vmap(lambda u: filter_single(u, margin, rho, I_QP))(actions)
            return filter_single(actions, margin, rho, I_QP)

        return jax.lax.cond(do_qp, do_filter, lambda _: actions, operand=None)

    # Attach patched method
    import types
    filter_obj._apply_actions_jax = types.MethodType(_apply_actions_jax_packed, filter_obj)

