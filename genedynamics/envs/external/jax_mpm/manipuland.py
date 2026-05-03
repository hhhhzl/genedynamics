"""Kinematic AABB manipuland — minimal soft-rigid coupling for the push task.

The block is treated as a moving boundary on the MPM grid: cells inside the
AABB get their velocity overridden to the box velocity (kinematic constraint).
The momentum the soft body would have transferred into those cells is summed
into an impulse on the box (Newton's 3rd law), and the box state is integrated
forward by simple Euler. Translational DOFs only (writeup §11 Task 2).

This is NOT a full rigid-body simulator — there's no rotation, no contact
manifold, no mu friction model on the box surface. It's the smallest
self-consistent thing that lets the push reward respond to the soft robot's
contact pattern. Carry / drop tasks (writeup Task 3) need a richer model and
land in Phase 5.

Conventions
-----------
- Position is the AABB center in world coords.
- half_extent is the box half-size in (x, y, z).
- The box always rests on the local terrain height (y_box bottom = terrain).
- Impulse is in mass·velocity units (consistent with grid_m * grid_v).
- The integration step is the MPM substep (cfg.dt), called once per substep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp


DTYPE = jnp.float32


@dataclass(frozen=True)
class ManipulandConfig:
    """Static description of the push object.

    init_pos : (cx, cy, cz)
        World-frame center at t=0. cy is auto-snapped to terrain at init.
    half_extent : (hx, hy, hz)
        AABB half-sizes. Total volume is 8 * hx * hy * hz.
    mass : float
        Used for v_box += impulse / mass; ignored for vertical motion (clamped
        to terrain).
    object_friction : float
        Coefficient applied to horizontal box velocity each substep when the
        box is resting on terrain. 0.0 = frictionless slide, 1.0 = static.
    horizontal_only : bool
        When True, vy_box is clamped to 0 (writeup Task 2 push regime). Phase 5
        carry/lift will flip this off.
    """

    init_pos: Tuple[float, float, float] = (0.55, 0.10, 0.50)
    half_extent: Tuple[float, float, float] = (0.04, 0.04, 0.04)
    mass: float = 1.0
    object_friction: float = 0.5
    horizontal_only: bool = True


class ManipulandState(NamedTuple):
    """Per-substep box state. Both fields are (3,) jnp arrays."""
    pos: jnp.ndarray
    vel: jnp.ndarray


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def init_manipuland_state(
    manip_cfg: ManipulandConfig,
    terrain_height: jnp.ndarray = None,
    n_grid: int = 64,
) -> ManipulandState:
    """Snap initial center so the box's bottom face rests on local terrain."""
    cx, cy_init, cz = manip_cfg.init_pos
    hy = manip_cfg.half_extent[1]
    if terrain_height is None:
        floor_y = DTYPE(0.0)
    else:
        # Sample terrain at (cx, cz) — nearest cell is good enough for init.
        ix = jnp.clip(jnp.int32(jnp.round(cx * n_grid - 0.5)), 0, n_grid - 1)
        iz = jnp.clip(jnp.int32(jnp.round(cz * n_grid - 0.5)), 0, n_grid - 1)
        floor_y = terrain_height[ix, iz]
    cy = jnp.maximum(jnp.asarray(cy_init, dtype=DTYPE), floor_y + DTYPE(hy))
    return ManipulandState(
        pos=jnp.stack([jnp.asarray(cx, DTYPE), cy, jnp.asarray(cz, DTYPE)]),
        vel=jnp.zeros(3, dtype=DTYPE),
    )


# ---------------------------------------------------------------------------
# Per-substep coupling: grid ↔ box
# ---------------------------------------------------------------------------


def _box_mask(
    pos: jnp.ndarray,
    half_extent: Tuple[float, float, float],
    n_grid: int,
    dx: float,
) -> jnp.ndarray:
    """Boolean (n, n, n) mask: cells whose centers lie inside the AABB."""
    Gi = jnp.arange(n_grid, dtype=DTYPE)[:, None, None]
    Gj = jnp.arange(n_grid, dtype=DTYPE)[None, :, None]
    Gk = jnp.arange(n_grid, dtype=DTYPE)[None, None, :]
    cx, cy, cz = pos[0], pos[1], pos[2]
    hx, hy, hz = half_extent
    in_x = jnp.abs((Gi + DTYPE(0.5)) * DTYPE(dx) - cx) < DTYPE(hx)
    in_y = jnp.abs((Gj + DTYPE(0.5)) * DTYPE(dx) - cy) < DTYPE(hy)
    in_z = jnp.abs((Gk + DTYPE(0.5)) * DTYPE(dx) - cz) < DTYPE(hz)
    return in_x & in_y & in_z


def apply_box_to_grid(
    grid_v: jnp.ndarray,           # (n, n, n, 3)
    grid_m: jnp.ndarray,           # (n, n, n)
    manip_state: ManipulandState,
    manip_cfg: ManipulandConfig,
    n_grid: int,
    dx: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Override box-cell velocities to box velocity; return (v_new, impulse).

    Impulse on the box = sum over inside cells of m_cell * (v_cell - v_box).
    By Newton's 3rd law the box receives this impulse; the soft body sees
    its in-cell velocity replaced by v_box (a kinematic boundary condition).
    """
    in_box = _box_mask(manip_state.pos, manip_cfg.half_extent, n_grid, dx)  # (n, n, n)
    v_box = manip_state.vel                                                  # (3,)
    delta_v = grid_v - v_box[None, None, None, :]                            # (n, n, n, 3)
    impulse_field = (in_box[..., None] * grid_m[..., None]) * delta_v        # (n, n, n, 3)
    impulse = jnp.sum(impulse_field, axis=(0, 1, 2))                         # (3,)
    v_box_broadcast = jnp.broadcast_to(v_box[None, None, None, :], grid_v.shape)
    grid_v_new = jnp.where(in_box[..., None], v_box_broadcast, grid_v)
    return grid_v_new, impulse


def step_manipuland(
    manip_state: ManipulandState,
    impulse: jnp.ndarray,
    dt: float,
    gravity: float,
    manip_cfg: ManipulandConfig,
    terrain_height: jnp.ndarray = None,
    n_grid: int = 64,
) -> ManipulandState:
    """Integrate box state forward by one substep.

    1. v += impulse / mass  (Newton 2nd from soft-body contact)
    2. v.y += -g * dt       (gravity)
    3. v.x, v.z *= damping  (object_friction proxy when on ground)
    4. p += v * dt
    5. Clamp p.y to terrain top (and zero vy if hit floor)
    6. Optionally zero vy when horizontal_only.
    """
    mass = jnp.asarray(manip_cfg.mass, dtype=DTYPE)
    v = manip_state.vel + impulse / jnp.maximum(mass, DTYPE(1e-6))
    v = v.at[1].add(-DTYPE(gravity) * DTYPE(dt))
    # Horizontal friction proxy. Only applied when on terrain (cheap heuristic:
    # always apply small damping; a full Coulomb model is Phase 5 work).
    damp = jnp.exp(-DTYPE(manip_cfg.object_friction) * DTYPE(dt) * DTYPE(20.0))
    v = v.at[0].multiply(damp)
    v = v.at[2].multiply(damp)

    p = manip_state.pos + v * DTYPE(dt)

    # Floor clamp: keep box bottom on terrain.
    hy = DTYPE(manip_cfg.half_extent[1])
    if terrain_height is None:
        floor_y = DTYPE(0.0)
    else:
        ix = jnp.clip(jnp.int32(jnp.round(p[0] * n_grid - 0.5)), 0, n_grid - 1)
        iz = jnp.clip(jnp.int32(jnp.round(p[2] * n_grid - 0.5)), 0, n_grid - 1)
        floor_y = terrain_height[ix, iz]
    on_ground = p[1] - hy <= floor_y
    p = p.at[1].set(jnp.where(on_ground, floor_y + hy, p[1]))
    v = v.at[1].set(jnp.where(on_ground, jnp.maximum(v[1], DTYPE(0.0)), v[1]))

    if manip_cfg.horizontal_only:
        # Push regime: lock vertical motion so the optimizer isn't rewarded for
        # tossing the object into the air.
        v = v.at[1].set(DTYPE(0.0))
        p = p.at[1].set(floor_y + hy)

    return ManipulandState(pos=p, vel=v)
