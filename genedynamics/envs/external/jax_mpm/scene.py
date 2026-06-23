"""JAX 3D MPM for soft-robot co-design — crawling_ground task.

Adapted from the hw4 JAX MPM reference; kept single-file so the whole
pipeline (p2g, grid, g2p, rollout, reward) is easy to audit / JIT.

Pipeline per env step:
    p2g_3d  : particles → grid momentum/mass
    grid_op : momentum/mass → velocity, add gravity, walls, floor-friction
    g2p_3d  : grid velocity → new particle (x, v, C)

theta = (x_morph, phi_ctrl) convention:
    x_morph = voxel occupancy (bounds (0.2, 1.0))
    phi_ctrl = [W.flat, b, g, a, c]  where W ∈ R^(n_act×K), b,g,a,c ∈ R^(n_act)
               g = velocity feedback gain, a/c = time envelope params
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp


DTYPE = jnp.float32
I3 = jnp.eye(3, dtype=DTYPE)

# Per-actuator "muscle direction" d_i in the X-Y plane: front half rotates
# from +X toward +Y (lift-forward), back half rotates from +X toward -Y
# (press-forward). When an actuator fires, stress eigen-vector = d_i ⊗ d_i,
# so each group pushes its local particles along its own direction. This
# breaks the whole-body symmetry that kept the robot pancaking in place
# and enables a real crawling gait under sinusoidal activation.
def _muscle_directions(n_actuators: int) -> jnp.ndarray:
    """Lift-push muscle pattern: back half (+X, +Y), front half (+X, -Y),
    unit-normalized. Breaks the pure-axial peristaltic ceiling (handcrafted
    traveling-wave goes from disp=0.00078 to 0.00428 at act=12 — ~5.5x).

    Actuator layout (see build_scene): bin 0 is at x_min (back/trailing edge),
    bin n-1 is at x_max (front/leading edge). Back half pushes forward-and-up
    (lift trailing edge off the ground between strides); front half pushes
    forward-and-down (press leading edge into ground as an anchor).
    """
    half = n_actuators // 2
    dx = jnp.ones((n_actuators,), dtype=DTYPE)
    idx = jnp.arange(n_actuators)
    dy = jnp.where(idx < half, DTYPE(1.0), DTYPE(-1.0))
    dz = jnp.zeros((n_actuators,), dtype=DTYPE)
    raw = jnp.stack([dx, dy, dz], axis=-1)
    return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)


# Unit-direction fallback (used when actuator_id == -1 / passive).
_ZERO_DIR = jnp.zeros(3, dtype=DTYPE)

# 3×3×3 = 27 neighbour offsets used by the quadratic MPM kernel.
OFFSETS_I_3 = jnp.array(
    [(i, j, k) for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.int32,
)
OFFSETS_F_3 = OFFSETS_I_3.astype(DTYPE)


@dataclass(frozen=True)
class MPMConfig:
    """Simulation hyperparameters.

    Defaults are tuned for `crawling_ground` at grid resolution 64 so the
    whole rollout fits comfortably on one 3090 even when vmap'd over 32
    proposals.
    """

    n_grid: int = 64
    # CFL: dt <= dx / sqrt(E*scale). With E=1, scale=100, mu~100, c~10, dx=1/64
    # -> dt <= ~1.5e-3. Use 5e-4 for safety.
    dt: float = 5e-4
    gravity: float = 3.8
    # Stress coefficient dt * V * 4 * inv_dx^3 controls how hard actuators
    # push. p_vol=5e-4 → coef ~ 0.26 (10x previous), peak actuator tau ~ 5 →
    # peak grid accel ~ 1.3 / substep.
    # scale=50 keeps the body stiff enough that aggressive actuators deform
    # it into a gait cycle rather than splatting it flat on the ground
    # (observed at scale=10 + high act_mul: body pancaked to 170% width).
    p_vol: float = 5e-4
    scale: float = 50.0
    j_min: float = 1e-6
    bound: int = 3
    friction_coeff: float = 0.5

    # Robot box.
    # box_origin.y=0.05 places the body just above the floor (bound*dx=0.047),
    # so the replay starts immediately in contact — avoids the first ~25 env
    # steps of free-fall that previously showed up as a big Y drop in gifs.
    box_size: Tuple[float, float, float] = (0.10, 0.06, 0.10)
    box_origin: Tuple[float, float, float] = (0.30, 0.05, 0.40)
    particle_spacing: float = 1.0 / 128.0  # sampling density (independent of n_grid)

    # Actuator setup
    n_actuators: int = 10
    act_strength_base: float = 4.0  # multiplied by act_mul (θ[2])

    # Controller: [tanh(W·sin + b + g·v_com_x_scaled)] * envelope(a, c, t).
    # phi layout: W(n_act*K) | b(n_act) | g(n_act) | a(n_act) | c(n_act)
    #           → phi_dim = n_act*K + 4*n_act
    n_sin_waves: int = 4
    # Controller family: "sinusoid" (smooth 80-d, default) | "open_loop" (a full per-node
    # actuation TRAJECTORY -> high-dim, non-convex contact-timing landscape, the MBD-paper
    # regime where model-based diffusion beats CEM/CMA-ES). open_loop phi = (n_control_nodes,
    # n_actuators) flat; act[t] = tanh(traj[node(t)]) via zero-order hold over the horizon.
    controller_kind: str = "sinusoid"
    n_control_nodes: int = 0           # 0 -> env_horizon (one node per env step)
    feedback_v_scale: float = 1000.0   # scale v_com_x so feedback is O(0.1), not O(1e-4)
    env_horizon: int = 200             # normalizes t_frac = env_t / env_horizon for envelope
    actuation_omega: float = 20.0
    actuation_strength_scale: float = 1.0   # bumped from 0.3; combined with
                                             # act_strength_base gives peak act≈4,
                                             # enough to drive visible stride.
    frame_dt: float = 8e-3                  # one env step (reported) in seconds

    # Voxel morphology grid: (vx,vy,vz) cells over the box. Each cell carries
    # an occupancy ∈ [0,1]; per-particle mass = occupancy[voxel_of_particle].
    # 3×3×3 = 27 morphology dims keeps total θ at 77 (controller 50 + voxel 27),
    # so M=128 gives ~1.66 sample/dim — comfortable for MBD weighted-mean denoise.
    voxel_dims: Tuple[int, int, int] = (3, 3, 3)

    # Reward shaping
    shaping_weight: float = 100.0
    # Penalty on backward COM velocity (sum of max(0, -v_x[t]) over rollout).
    # 0 = main behaviour; smoothness variants set >0 so the optimizer is
    # pressured to avoid "lunge-and-recoil" patterns.
    backward_penalty_weight: float = 0.0

    @property
    def dx(self) -> float:
        return 1.0 / self.n_grid

    @property
    def inv_dx(self) -> float:
        return float(self.n_grid)

    @property
    def substeps_per_env_step(self) -> int:
        return max(1, int(self.frame_dt / self.dt))


class SceneData(NamedTuple):
    x0: jnp.ndarray          # (N, 3)
    actuator_id: jnp.ndarray  # (N,) int32 in [-1, n_actuators-1]; -1 = passive
    voxel_id: jnp.ndarray     # (N,) int32 in [0, prod(voxel_dims)) — which voxel each particle sits in
    n_particles: int
    n_actuators: int
    n_voxels: int
    # Robotization output: (n_actuators, 3) muscle direction per actuator group.
    # Defaults to None for backward-compat with old call sites; rollout falls
    # back to lift-push pattern when missing.
    fiber_dirs: jnp.ndarray = None
    # Optional per-particle Young's modulus override (None → uniform E0).
    E_per_particle: jnp.ndarray = None
    # Optional continuous per-particle actuation weights (N, n_actuators).
    # None → eigen-stress uses the hard one-hot from actuator_id (legacy).
    actuator_weight: jnp.ndarray = None


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_scene_from_spec(spec, cfg: MPMConfig) -> SceneData:
    """Convert a SoftBodySpec (numpy) into the JAX SceneData consumed by rollouts.

    This is the canonical scene constructor — `build_scene(cfg)` below is a
    thin wrapper that calls the default robotization first.
    """
    return SceneData(
        x0=jnp.asarray(spec.particles_x0, dtype=DTYPE),
        actuator_id=jnp.asarray(spec.actuator_id, dtype=jnp.int32),
        voxel_id=jnp.asarray(spec.voxel_id, dtype=jnp.int32),
        n_particles=int(spec.n_particles),
        n_actuators=int(spec.n_actuators),
        n_voxels=int(spec.n_voxels),
        fiber_dirs=jnp.asarray(spec.fiber_dirs, dtype=DTYPE),
        E_per_particle=(
            jnp.asarray(spec.E_per_particle, dtype=DTYPE)
            if spec.E_per_particle is not None
            else None
        ),
        actuator_weight=(
            jnp.asarray(spec.actuator_weight, dtype=DTYPE)
            if getattr(spec, "actuator_weight", None) is not None
            else None
        ),
    )


def build_scene(cfg: MPMConfig) -> SceneData:
    """Build the default crawling_ground soft body (X-bin actuators, lift-push fibers).

    This is the historical entry point; new code should prefer building a
    `SoftBodySpec` (via genedynamics.morphology.default_robotize or a future
    mesh_robotize) and calling build_scene_from_spec(). Kept as a thin shim
    so existing call sites (jax_mpm_evaluator, adapters) work unchanged.

    Actuator 0 is back (smallest X), actuator n-1 is front. The lift-push
    fiber pattern (back lifts, front presses) breaks pure-axial peristaltic
    symmetry and produces a locomotion gait along +X.
    """
    from genedynamics.morphology import default_robotize
    spec = default_robotize(cfg)
    return build_scene_from_spec(spec, cfg)


# ---------------------------------------------------------------------------
# MPM kernels (3D)
# ---------------------------------------------------------------------------


def det3(F: jnp.ndarray) -> jnp.ndarray:
    return (
        F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
        - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
        + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
    )


def kernel_weights_3d(fx: jnp.ndarray) -> jnp.ndarray:
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    wx = jnp.stack([w0[0], w1[0], w2[0]])
    wy = jnp.stack([w0[1], w1[1], w2[1]])
    wz = jnp.stack([w0[2], w1[2], w2[2]])
    return (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).reshape(27)


def _stress_and_J(
    new_F: jnp.ndarray,
    E: jnp.ndarray,
    actuator_id: jnp.ndarray,
    act_t: jnp.ndarray,
    muscle_dirs: jnp.ndarray,   # (n_actuators, 3)
    cfg: MPMConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Neo-Hookean + per-actuator-direction eigen-stress, 3D.

    Eigen-stress A = act * (d ⊗ d) where d is the actuator's muscle direction,
    so each actuator group pushes particles in its own body-local direction.
    """
    J_raw = det3(new_F)
    J = jnp.maximum(J_raw, DTYPE(cfg.j_min))
    safe_aid = jnp.maximum(actuator_id, 0)
    act = jnp.where(actuator_id == -1, 0.0, act_t[safe_aid])
    d = jnp.where(actuator_id == -1, _ZERO_DIR, muscle_dirs[safe_aid])  # (3,)
    A = act * jnp.outer(d, d)
    mu = E * DTYPE(cfg.scale)
    la = E * DTYPE(cfg.scale)
    B = new_F @ new_F.T
    tau = mu * (B - I3) + la * jnp.log(J) * I3
    tau = tau + new_F @ A @ new_F.T
    return tau, J_raw


def _stress_and_J_weighted(
    new_F: jnp.ndarray,
    E: jnp.ndarray,
    actuator_weight: jnp.ndarray,  # (n_actuators,) continuous weights for THIS particle
    act_t: jnp.ndarray,            # (n_actuators,) activation signal at this env step
    muscle_dirs: jnp.ndarray,      # (n_actuators, 3)
    cfg: MPMConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-actuator variant of `_stress_and_J`, 3D.

    Eigen-stress is the weighted sum over actuator groups:
        A = Σ_i w_i · act_t[i] · (d_i ⊗ d_i)
    With a one-hot ``actuator_weight`` (1 at the particle's group, 0 elsewhere;
    all-zero for a passive particle), this is numerically identical to the
    hard-index `_stress_and_J`. With a continuous weight it lets actuator
    placement be a co-designed field (A2 / DiffuseBot-style baseline).
    """
    J_raw = det3(new_F)
    J = jnp.maximum(J_raw, DTYPE(cfg.j_min))
    acts = actuator_weight * act_t                    # (n_actuators,)
    # A = Σ_i acts_i d_i d_i^T = D^T diag(acts) D, with D = muscle_dirs (K,3).
    A = muscle_dirs.T @ (acts[:, None] * muscle_dirs)  # (3, 3)
    mu = E * DTYPE(cfg.scale)
    la = E * DTYPE(cfg.scale)
    B = new_F @ new_F.T
    tau = mu * (B - I3) + la * jnp.log(J) * I3
    tau = tau + new_F @ A @ new_F.T
    return tau, J_raw


def _p2g_single(
    x: jnp.ndarray,
    v: jnp.ndarray,
    C: jnp.ndarray,
    F: jnp.ndarray,
    E: jnp.ndarray,
    actuator_id: jnp.ndarray,
    act_t: jnp.ndarray,
    muscle_dirs: jnp.ndarray,
    mass: jnp.ndarray,
    cfg: MPMConfig,
    actuator_weight: jnp.ndarray = None,   # (n_actuators,) for THIS particle; None → hard one-hot path
):
    base = jnp.floor(x * cfg.inv_dx - 0.5).astype(jnp.int32)
    fx = x * cfg.inv_dx - base.astype(DTYPE)
    weights = kernel_weights_3d(fx)

    new_F = (I3 + cfg.dt * C) @ F
    if actuator_weight is None:
        tau, J_raw = _stress_and_J(new_F, E, actuator_id, act_t, muscle_dirs, cfg)
    else:
        tau, J_raw = _stress_and_J_weighted(new_F, E, actuator_weight, act_t, muscle_dirs, cfg)

    inv_cube = cfg.inv_dx ** 3
    # Stress and momentum scale linearly with per-particle mass so ghost particles
    # (mass≈0 from voxel occupancy=0) contribute essentially nothing to the grid.
    stress = -(cfg.dt * cfg.p_vol * 4.0 * inv_cube) * tau * mass
    affine = stress + mass * C

    idx = base[None, :] + OFFSETS_I_3                          # (27, 3)
    dpos = (OFFSETS_F_3 - fx[None, :]) * cfg.dx                # (27, 3)
    affine_dpos = jnp.einsum("ab,kb->ka", affine, dpos)
    momentum = weights[:, None] * (mass * v[None, :] + affine_dpos)
    mass_contrib = weights * mass
    return idx, momentum, mass_contrib, new_F, J_raw


def p2g_3d(
    x: jnp.ndarray,
    v: jnp.ndarray,
    C: jnp.ndarray,
    F: jnp.ndarray,
    E_field: jnp.ndarray,
    actuator_id: jnp.ndarray,
    act_t: jnp.ndarray,
    muscle_dirs: jnp.ndarray,
    mass_field: jnp.ndarray,
    cfg: MPMConfig,
    actuator_weight: jnp.ndarray = None,   # (N, n_actuators) continuous; None → hard one-hot path
):
    if actuator_weight is None:
        vmap_fn = jax.vmap(
            lambda xp, vp, Cp, Fp, Ep, aid, mp: _p2g_single(
                xp, vp, Cp, Fp, Ep, aid, act_t, muscle_dirs, mp, cfg
            )
        )
        idx, momentum, mass_contrib, new_F, J_raw = vmap_fn(x, v, C, F, E_field, actuator_id, mass_field)
    else:
        vmap_fn = jax.vmap(
            lambda xp, vp, Cp, Fp, Ep, aid, wp, mp: _p2g_single(
                xp, vp, Cp, Fp, Ep, aid, act_t, muscle_dirs, mp, cfg, actuator_weight=wp
            )
        )
        idx, momentum, mass_contrib, new_F, J_raw = vmap_fn(
            x, v, C, F, E_field, actuator_id, actuator_weight, mass_field
        )
    n = cfg.n_grid
    ii = idx[:, :, 0].reshape(-1)
    jj = idx[:, :, 1].reshape(-1)
    kk = idx[:, :, 2].reshape(-1)
    grid_v = jnp.zeros((n, n, n, 3), dtype=DTYPE)
    grid_m = jnp.zeros((n, n, n), dtype=DTYPE)
    grid_v = grid_v.at[ii, jj, kk].add(
        momentum.reshape(-1, 3), mode="drop")
    grid_m = grid_m.at[ii, jj, kk].add(
        mass_contrib.reshape(-1), mode="drop")
    return grid_v, grid_m, new_F, J_raw


def _boundary_masks_3d(cfg: MPMConfig):
    n = cfg.n_grid
    bnd = cfg.bound
    Gi = jnp.arange(n, dtype=jnp.int32)[:, None, None]
    Gj = jnp.arange(n, dtype=jnp.int32)[None, :, None]
    Gk = jnp.arange(n, dtype=jnp.int32)[None, None, :]
    left = Gi < bnd
    right = Gi > n - bnd
    bottom = Gj < bnd
    top = Gj > n - bnd
    front = Gk < bnd
    back = Gk > n - bnd
    return left, right, bottom, top, front, back


def grid_op_3d(
    grid_v_in: jnp.ndarray,
    grid_m_in: jnp.ndarray,
    friction: jnp.ndarray,
    cfg: MPMConfig,
    terrain_height: jnp.ndarray = None,
    gravity_vec: jnp.ndarray = None,
) -> jnp.ndarray:
    """Grid update: scatter momentum → velocity, gravity, walls, floor friction.

    Phase 2.1: optional ``terrain_height`` of shape (n_grid, n_grid) gives
    per-(i_x, k_z) world-y floor height. When None, falls back to the legacy
    flat floor mask (Gj < bound). Friction is applied to any cell whose
    world-y is below the local floor and whose vertical velocity is downward.

    Per-mode regime: optional ``gravity_vec`` (3,) world-frame acceleration lets a
    mode encode SLOPE as a gravity tilt (e.g. uphill +theta -> [-g*sin, -g*cos, 0]).
    None -> the legacy y-only gravity (byte-identical trace).
    """
    inv_m = 1.0 / (grid_m_in + 1e-10)
    v_out = grid_v_in * inv_m[..., None]
    if gravity_vec is None:
        v_out = v_out.at[..., 1].add(-cfg.dt * cfg.gravity)
    else:
        v_out = v_out + cfg.dt * gravity_vec[None, None, None, :]

    left, right, bottom, top, front, back = _boundary_masks_3d(cfg)
    vx = v_out[..., 0]
    vy = v_out[..., 1]
    vz = v_out[..., 2]
    walls = (
        (left & (vx < 0.0))
        | (right & (vx > 0.0))
        | (front & (vz < 0.0))
        | (back & (vz > 0.0))
        | (top & (vy > 0.0))
    )
    v_out = jnp.where(walls[..., None], 0.0, v_out)

    # Floor mask: legacy flat floor OR per-column terrain height.
    if terrain_height is None:
        floor_mask = bottom
    else:
        n = cfg.n_grid
        # World-y of each cell: Gj * dx. terrain_height is (n, n) on (i_x, k_z).
        Gj = jnp.arange(n, dtype=DTYPE)[None, :, None]
        y_world = Gj * DTYPE(cfg.dx)
        floor_y = terrain_height[:, None, :]   # broadcast to (n, n, n)
        below_terrain = y_world < floor_y
        # Always preserve the absolute lower world-boundary as a hard floor.
        floor_mask = bottom | below_terrain

    # Coulomb friction wherever the cell is at/below the local floor and moving down.
    bottom_hit = floor_mask & (v_out[..., 1] < 0.0)
    vn = v_out[..., 1]
    vtx = v_out[..., 0]
    vtz = v_out[..., 2]
    # speed_t with sqrt-of-sum-plus-eps so the gradient at speed_t=0 is a
    # well-defined finite (zero) instead of NaN. The forward value at v=0
    # is sqrt(eps) ≈ 3e-7, which is below any physical threshold.
    eps_speed = DTYPE(1e-12)
    speed_t = jnp.sqrt(vtx * vtx + vtz * vtz + eps_speed)
    friction_limit = friction * (-vn)
    # Safe division: when the cell is moving, scale by max(0, speed - μ|vn|)/speed.
    # When speed is ~0, force scale=0 (no friction force when no tangential
    # motion — physically correct). Using jnp.where + a "safe denominator"
    # is the JAX-friendly idiom for keeping reverse-mode gradients finite at
    # the singular point (Phase 4 SHAC backprop hits this on every fresh
    # init carry where v_grid = 0).
    nonzero_speed = speed_t > DTYPE(1e-6)
    safe_speed = jnp.where(nonzero_speed, speed_t, DTYPE(1.0))
    raw_scale = jnp.maximum(speed_t - friction_limit, 0.0) / safe_speed
    scale = jnp.where(nonzero_speed, raw_scale, DTYPE(0.0))
    floor_v = jnp.stack([vtx * scale, jnp.zeros_like(vn), vtz * scale], axis=-1)
    v_out = jnp.where(bottom_hit[..., None], floor_v, v_out)
    return v_out


def _g2p_single(x: jnp.ndarray, grid_v_out: jnp.ndarray, cfg: MPMConfig):
    base = jnp.floor(x * cfg.inv_dx - 0.5).astype(jnp.int32)
    fx = x * cfg.inv_dx - base.astype(DTYPE)
    weights = kernel_weights_3d(fx)
    idx = base[None, :] + OFFSETS_I_3
    samples = grid_v_out.at[idx[:, 0], idx[:, 1], idx[:, 2]].get(
        mode="fill", fill_value=0.0)
    new_v = jnp.sum(weights[:, None] * samples, axis=0)
    dpos = OFFSETS_F_3 - fx[None, :]
    new_C = 4.0 * cfg.inv_dx * jnp.einsum("k,ka,kb->ab", weights, samples, dpos)
    x_next = x + cfg.dt * new_v
    return x_next, new_v, new_C


def g2p_3d(x: jnp.ndarray, grid_v_out: jnp.ndarray, cfg: MPMConfig):
    return jax.vmap(lambda xp: _g2p_single(xp, grid_v_out, cfg))(x)


# ---------------------------------------------------------------------------
# Controller: deterministic sin-wave from phi
# ---------------------------------------------------------------------------


def compute_actuation(
    phi: jnp.ndarray,
    env_t: jnp.ndarray,
    cfg: MPMConfig,
    v_com_x: jnp.ndarray = DTYPE(0.0),
) -> jnp.ndarray:
    """Return actuation signal (n_actuators,) at env-time env_t.

    Controller:
        raw[a] = tanh( W[a,:] · sin(ω·t + phases) + b[a] + g[a] · v_com_x_scaled )
        act[a] = raw[a] * sigmoid( a[a] + c[a] · t_frac )

    phi layout (n_act=10, K=4 → phi_dim=80):
        phi[:40]  = W.flat    (sin-wave weights)
        phi[40:50] = b        (bias)
        phi[50:60] = g        (velocity feedback gain)
        phi[60:70] = a        (envelope offset)
        phi[70:80] = c        (envelope slope)
    """
    n_act = cfg.n_actuators
    # Open-loop trajectory controller (static branch on cfg.controller_kind): phi is a flat
    # (n_nodes, n_act) actuation trajectory; the actuation at env-step t is a zero-order hold
    # tanh(traj[node(t)]). High-dim + non-convex (contact timing) = the MBD-favorable regime.
    if cfg.controller_kind == "open_loop":
        n_nodes = cfg.n_control_nodes if cfg.n_control_nodes > 0 else cfg.env_horizon
        traj = phi[: n_nodes * n_act].reshape(n_nodes, n_act)
        node = jnp.clip((env_t.astype(jnp.int32) * n_nodes) // max(int(cfg.env_horizon), 1),
                        0, n_nodes - 1)
        return jnp.tanh(traj[node])                       # (n_act,)
    K = cfg.n_sin_waves
    W = phi[: n_act * K].reshape(n_act, K)
    b = phi[n_act * K : n_act * K + n_act]
    g = phi[n_act * K + n_act : n_act * K + 2 * n_act]
    a = phi[n_act * K + 2 * n_act : n_act * K + 3 * n_act]
    c = phi[n_act * K + 3 * n_act : n_act * K + 4 * n_act]

    t_seconds = env_t.astype(DTYPE) * DTYPE(cfg.frame_dt)
    phases = 2.0 * math.pi / DTYPE(K) * jnp.arange(K, dtype=DTYPE)
    basis = jnp.sin(DTYPE(cfg.actuation_omega) * t_seconds + phases)  # (K,)

    # Velocity feedback scaled so the term is O(0.1) instead of O(1e-4).
    v_scaled = v_com_x * DTYPE(cfg.feedback_v_scale)
    raw = jnp.tanh(W @ basis + b + g * v_scaled)  # (n_act,)

    # Time envelope: per-actuator sigmoid modulation over the rollout.
    t_frac = env_t.astype(DTYPE) / DTYPE(cfg.env_horizon)
    envelope = jax.nn.sigmoid(a + c * t_frac)  # (n_act,)
    return raw * envelope  # (n_act,)


# ---------------------------------------------------------------------------
# Rollout and reward
# ---------------------------------------------------------------------------


def _init_carry(scene: SceneData):
    F0 = jnp.broadcast_to(I3, (scene.n_particles, 3, 3))
    v0 = jnp.zeros((scene.n_particles, 3), dtype=DTYPE)
    C0 = jnp.zeros((scene.n_particles, 3, 3), dtype=DTYPE)
    return scene.x0, v0, C0, F0


def _env_step(
    carry,
    env_t: jnp.ndarray,
    phi: jnp.ndarray,
    E_field: jnp.ndarray,
    actuator_id: jnp.ndarray,
    friction: jnp.ndarray,
    muscle_dirs: jnp.ndarray,
    mass_field: jnp.ndarray,
    cfg: MPMConfig,
    terrain_height: jnp.ndarray = None,
    actuator_weight: jnp.ndarray = None,
    gravity_vec: jnp.ndarray = None,
):
    """One env step = `substeps_per_env_step` MPM substeps + reward probe."""
    n_sub = cfg.substeps_per_env_step
    # COM velocity feedback: compute v_com_x from current state for the controller.
    x, v, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    v_com_x = jnp.sum(v[:, 0] * w, axis=0)
    act_env = compute_actuation(phi, env_t, cfg, v_com_x=v_com_x)
    act_t = act_env * DTYPE(cfg.act_strength_base) * DTYPE(cfg.actuation_strength_scale)

    def substep(c, _):
        x, v, C, F = c
        grid_v, grid_m, F_next, _ = p2g_3d(
            x, v, C, F, E_field, actuator_id, act_t, muscle_dirs, mass_field, cfg,
            actuator_weight=actuator_weight,
        )
        grid_v_out = grid_op_3d(grid_v, grid_m, friction, cfg, terrain_height=terrain_height,
                                gravity_vec=gravity_vec)
        x_next, v_next, C_next = g2p_3d(x, grid_v_out, cfg)
        return (x_next, v_next, C_next, F_next), None

    carry, _ = jax.lax.scan(substep, carry, jnp.arange(n_sub, dtype=jnp.int32))
    # Probe: mass-weighted COM velocity / position so ghost particles don't bias it.
    x, v, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    com_v = jnp.sum(v * w[:, None], axis=0)
    com_x = jnp.sum(x * w[:, None], axis=0)
    return carry, (com_v, com_x)


def _voxel_mass_field(voxel_occ: jnp.ndarray, scene: SceneData) -> jnp.ndarray:
    """Map θ-voxel-occupancy (n_voxels,) → per-particle mass (N,) ∈ [0, 1]."""
    occ = jnp.clip(voxel_occ, DTYPE(0.0), DTYPE(1.0))
    return occ[scene.voxel_id]


def _fiber_dirs_or_default(scene: SceneData, cfg: MPMConfig) -> jnp.ndarray:
    """Use scene-supplied fibers if present (new API), else fall back to lift-push."""
    if scene.fiber_dirs is None:
        return _muscle_directions(cfg.n_actuators)
    return scene.fiber_dirs


def _E_field_or_default(scene: SceneData, cfg: MPMConfig, E0: float) -> jnp.ndarray:
    if scene.E_per_particle is None:
        return jnp.full((scene.n_particles,), DTYPE(E0), dtype=DTYPE)
    return scene.E_per_particle


def _actuator_weight_or_none(scene: SceneData) -> jnp.ndarray:
    """Continuous per-particle actuator weights (N, n_actuators), or None for
    the legacy hard one-hot eigen-stress path. Kept as a helper so all rollout
    call sites read uniformly and a future derive-from-actuator_id default has
    one home."""
    return scene.actuator_weight


def rollout_return(
    x_morph: jnp.ndarray,     # (n_voxels,) ∈ [0,1] occupancy per voxel
    phi: jnp.ndarray,          # (phi_dim,) controller params
    friction: jnp.ndarray,     # scalar
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    actuator_weight_voxel: jnp.ndarray = None,   # (n_voxels, n_actuators) co-designed actuator field
    E_voxel: jnp.ndarray = None,                 # (n_voxels,) co-designed stiffness field
    objective: str = "crawling",                 # DiffuseBot task: crawling|balancing|landing|hurdling
    gravity_vec: jnp.ndarray = None,             # (3,) per-mode gravity tilt = SLOPE; None -> y-only
    mass_scale: jnp.ndarray = None,              # per-mode body-mass multiplier; None -> 1.0
    init_vel: jnp.ndarray = None,                # (3,) additive initial COM velocity; None -> 0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one rollout. Returns (blended_reward, final_forward_disp, com_x_traj).

    Phase 2.1: optional ``terrain_height`` (n_grid, n_grid) makes the floor
    follow a height field. None → flat floor (backward-compat).

    Task-2: optional per-voxel ``actuator_weight_voxel`` / ``E_voxel`` are the
    co-designed actuator-placement and stiffness fields decoded from the shape
    latent w. They are gathered to per-particle via ``scene.voxel_id`` and
    OVERRIDE the scene's fixed fields. None → use the scene's (legacy path).
    """
    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    if E_voxel is not None:
        E_field = E_voxel[scene.voxel_id]
    if actuator_weight_voxel is not None:
        actuator_weight = actuator_weight_voxel[scene.voxel_id]
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)
    if mass_scale is not None:
        mass_field = mass_field * jnp.asarray(mass_scale, DTYPE)   # per-mode payload/mass regime

    carry = _init_carry(scene)
    # DiffuseBot Passive-Dynamics initial conditions:
    #   landing  → spawn elevated so the body DROPS and must settle softly.
    #   balancing→ apply a horizontal velocity perturbation the body must resist.
    if objective == "landing":
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0 + jnp.asarray([0.0, 0.08, 0.0], DTYPE), _v0, _C0, _F0)
    elif objective == "balancing":
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0, _v0.at[:, 0].add(DTYPE(0.5)), _C0, _F0)
    if init_vel is not None:                                       # per-mode initial-condition regime
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0, _v0 + jnp.asarray(init_vel, DTYPE)[None, :], _C0, _F0)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        return _env_step(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg,
            terrain_height=terrain_height, actuator_weight=actuator_weight,
            gravity_vec=gravity_vec,
        )

    carry, (com_v_hist, com_x_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32)
    )
    # Integrated forward displacement: rewards spending TIME ahead of start,
    # not just ending ahead. Fixes the "lunge then stall/reverse" strategy the
    # per_step_v_sum reward was indifferent to (Σv telescopes to final_disp/dt,
    # so it is redundant with shaping_weight * final_disp).
    per_step_forward_disp = jnp.sum(com_x_hist[:, 0] - init_com_x[0])
    final_disp = com_x_hist[-1, 0] - init_com_x[0]
    # Penalize empty/near-empty robots (no mass = trivially "fast" zeros).
    total_mass = jnp.sum(mass_field)
    mass_penalty = DTYPE(0.5) * jnp.maximum(DTYPE(0.1) * scene.n_particles - total_mass, 0.0)
    # Backward-motion penalty: Σ_t max(0, -v_x[t]). Active only when
    # cfg.backward_penalty_weight > 0 (smoothness variant).
    backward_sum = jnp.sum(jnp.maximum(-com_v_hist[:, 0], 0.0))
    smooth_pen = DTYPE(cfg.backward_penalty_weight) * backward_sum

    if objective == "crawling":
        # forward locomotion (DiffuseBot Crawling) — time-ahead + final displacement.
        reward = (per_step_forward_disp + DTYPE(cfg.shaping_weight) * final_disp
                  - mass_penalty - smooth_pen)
    elif objective == "balancing":
        # DiffuseBot Passive-Dynamics "balance": stay put — penalize horizontal
        # COM drift (x,z) and loss of height (y) over the whole episode.
        drift = jnp.sum(jnp.abs(com_x_hist[:, 0] - init_com_x[0])
                        + jnp.abs(com_x_hist[:, 2] - init_com_x[2]))
        height_drop = jnp.maximum(init_com_x[1] - com_x_hist[-1, 1], 0.0)
        reward = -(drift + DTYPE(cfg.shaping_weight) * height_drop) - mass_penalty
    elif objective == "landing":
        # DiffuseBot Passive-Dynamics "landing": settle softly — minimize total
        # COM speed over the episode and the final velocity (no bouncing away).
        speed = jnp.sum(jnp.linalg.norm(com_v_hist, axis=1))
        final_speed = jnp.linalg.norm(com_v_hist[-1])
        reward = -(speed + DTYPE(cfg.shaping_weight) * final_speed) - mass_penalty
    elif objective == "hurdling":
        # DiffuseBot Locomotion "hurdle": move forward AND clear height — reward
        # forward displacement + peak COM height (jump over the hurdle).
        peak_h = jnp.max(com_x_hist[:, 1] - init_com_x[1])
        reward = (final_disp + DTYPE(cfg.shaping_weight) * peak_h
                  - mass_penalty - smooth_pen)
    else:
        reward = (per_step_forward_disp + DTYPE(cfg.shaping_weight) * final_disp
                  - mass_penalty - smooth_pen)
    return reward, final_disp, com_x_hist


def rollout_with_positions(
    x_morph: jnp.ndarray,
    phi: jnp.ndarray,
    friction: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    actuator_weight_voxel: jnp.ndarray = None,
    E_voxel: jnp.ndarray = None,
):
    """Like `rollout_return` but also returns particle positions per env step (for GIF)."""
    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    if E_voxel is not None:
        E_field = E_voxel[scene.voxel_id]
    if actuator_weight_voxel is not None:
        actuator_weight = actuator_weight_voxel[scene.voxel_id]
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)
    carry = _init_carry(scene)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        new_carry, (com_v, com_x) = _env_step(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg,
            terrain_height=terrain_height, actuator_weight=actuator_weight,
        )
        return new_carry, (com_v, com_x, new_carry[0])

    carry, (com_v_hist, com_x_hist, x_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32)
    )
    per_step_forward_disp = jnp.sum(com_x_hist[:, 0] - init_com_x[0])
    final_disp = com_x_hist[-1, 0] - init_com_x[0]
    backward_sum = jnp.sum(jnp.maximum(-com_v_hist[:, 0], 0.0))
    smooth_pen = DTYPE(cfg.backward_penalty_weight) * backward_sum
    reward = (per_step_forward_disp
              + DTYPE(cfg.shaping_weight) * final_disp
              - smooth_pen)
    return reward, final_disp, com_x_hist, x_hist, mass_field


def _env_step_with_manip(
    carry,
    env_t: jnp.ndarray,
    phi: jnp.ndarray,
    E_field: jnp.ndarray,
    actuator_id: jnp.ndarray,
    friction: jnp.ndarray,
    muscle_dirs: jnp.ndarray,
    mass_field: jnp.ndarray,
    cfg: MPMConfig,
    terrain_height: jnp.ndarray = None,
    actuator_weight: jnp.ndarray = None,
    manip_cfg=None,
):
    """Variant of `_env_step` that maintains a kinematic AABB box in the carry.

    Carry layout:
        (x, v, C, F, manip_state)   where manip_state is ManipulandState.

    Per substep: p2g → grid_op (floor + walls + terrain) → box override
    (apply_box_to_grid) → g2p; manip_state updated by step_manipuland from the
    grid impulse. Per env step we report (com_v, com_x, manip_pos).
    """
    from genedynamics.envs.external.jax_mpm.manipuland import (
        apply_box_to_grid,
        step_manipuland,
    )

    n_sub = cfg.substeps_per_env_step
    x, v, _, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    v_com_x = jnp.sum(v[:, 0] * w, axis=0)
    act_env = compute_actuation(phi, env_t, cfg, v_com_x=v_com_x)
    act_t = act_env * DTYPE(cfg.act_strength_base) * DTYPE(cfg.actuation_strength_scale)

    def substep(c, _):
        x, v, C, F, manip_state = c
        grid_v, grid_m, F_next, _ = p2g_3d(
            x, v, C, F, E_field, actuator_id, act_t, muscle_dirs, mass_field, cfg,
            actuator_weight=actuator_weight,
        )
        grid_v_floor = grid_op_3d(grid_v, grid_m, friction, cfg, terrain_height=terrain_height)
        grid_v_box, impulse = apply_box_to_grid(
            grid_v_floor, grid_m, manip_state, manip_cfg, cfg.n_grid, cfg.dx
        )
        manip_state_next = step_manipuland(
            manip_state, impulse, cfg.dt, cfg.gravity, manip_cfg,
            terrain_height=terrain_height, n_grid=cfg.n_grid,
        )
        x_next, v_next, C_next = g2p_3d(x, grid_v_box, cfg)
        return (x_next, v_next, C_next, F_next, manip_state_next), None

    carry, _ = jax.lax.scan(substep, carry, jnp.arange(n_sub, dtype=jnp.int32))
    x, v, _, _, manip_state = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    com_v = jnp.sum(v * w[:, None], axis=0)
    com_x = jnp.sum(x * w[:, None], axis=0)
    return carry, (com_v, com_x, manip_state.pos)


def rollout_return_push(
    x_morph: jnp.ndarray,
    phi: jnp.ndarray,
    friction: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    manip_cfg,
    goal_x: float,
    *,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    weights: Tuple[float, float, float, float] = (1.0, 5.0, 0.01, 50.0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Push-task rollout. Returns (reward, dist_to_goal_T, com_x_traj, manip_x_traj).

    Reward (writeup §11 Task 2, light-weight version):
        R = w0 * Δx_obj                   # forward object displacement
          + w1 * (d_0 - d_T)              # closing distance to goal
          - w2 * Σ‖u_t‖²                  # control penalty (proxy via φ norm)
          - w3 * 1[Δx_obj < ε]            # sticky penalty when object never moved
    """
    from genedynamics.envs.external.jax_mpm.manipuland import init_manipuland_state

    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)

    manip_state0 = init_manipuland_state(manip_cfg, terrain_height=terrain_height, n_grid=cfg.n_grid)
    x0, v0, C0, F0 = _init_carry(scene)
    carry = (x0, v0, C0, F0, manip_state0)

    def body(carry, env_t):
        return _env_step_with_manip(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg,
            terrain_height=terrain_height, actuator_weight=actuator_weight, manip_cfg=manip_cfg,
        )

    carry, (com_v_hist, com_x_hist, manip_x_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32),
    )

    init_obj_x = manip_state0.pos[0]
    final_obj_x = manip_x_hist[-1, 0]
    delta_obj_x = final_obj_x - init_obj_x

    goal_x_d = DTYPE(goal_x)
    d0 = jnp.abs(init_obj_x - goal_x_d)
    dT = jnp.abs(final_obj_x - goal_x_d)
    closing = d0 - dT

    w0_, w1_, w2_, w3_ = weights
    eps_x = DTYPE(1e-3)
    sticky = jnp.where(jnp.abs(delta_obj_x) < eps_x, DTYPE(1.0), DTYPE(0.0))
    ctrl_pen = jnp.sum(phi * phi)

    reward = (
        DTYPE(w0_) * delta_obj_x
        + DTYPE(w1_) * closing
        - DTYPE(w2_) * ctrl_pen
        - DTYPE(w3_) * sticky
    )
    return reward, dT, com_x_hist, manip_x_hist


# ---------------------------------------------------------------------------
# Phase 4 (SHAC) — differentiable h-step rollout from arbitrary carry state.
#
# SHAC needs a function that:
#   - starts from a SAVED simulator state (not always _init_carry)
#   - runs h env steps with the current controller params phi
#   - returns the rewards-per-step, observations-per-step, and final carry
#
# Critically: differentiable in `phi` so jax.grad(loss, argnums=...) works.
# The non-smooth contact friction in grid_op_3d gives a sub-gradient at
# threshold points, which is what SHAC's stochastic policy averages over.
# ---------------------------------------------------------------------------


_OBS_DIM = 9   # COM (3) + COM vel (3) + mean speed (1) + sin/cos time (2)


def observation_from_carry(
    carry,
    env_t: jnp.ndarray,
    mass_field: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    """Compact obs vector for the SHAC critic. Mass-weighted so ghost
    particles (occupancy ≈ 0) don't bias the moments.

    Returns (_OBS_DIM,) float32. This is the input dim for critic.MLP.
    """
    x, v, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    com_x = jnp.sum(x * w[:, None], axis=0)         # (3,)
    com_v = jnp.sum(v * w[:, None], axis=0)         # (3,)
    speed = jnp.sqrt(jnp.sum(v * v, axis=-1) + DTYPE(1e-12))
    mean_speed = jnp.sum(speed * w)
    t_frac = jnp.asarray(env_t, dtype=DTYPE) / DTYPE(max(int(horizon), 1))
    sin_t = jnp.sin(DTYPE(2.0 * math.pi) * t_frac)
    cos_t = jnp.cos(DTYPE(2.0 * math.pi) * t_frac)
    return jnp.stack([
        com_x[0], com_x[1], com_x[2],
        com_v[0], com_v[1], com_v[2],
        mean_speed, sin_t, cos_t,
    ]).astype(DTYPE)


def rollout_h_from_state(
    initial_carry,                         # (x, v, C, F) jnp tuple
    initial_t: jnp.ndarray,                # scalar int env-step at start
    phi: jnp.ndarray,                      # (phi_dim,) controller params
    h: int,                                # static Python int — short horizon
    friction: jnp.ndarray,                 # scalar
    scene: SceneData,
    cfg: MPMConfig,
    *,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    x_morph: jnp.ndarray = None,           # (n_voxels,) occupancy; None → all 1
    horizon_for_time: int = None,
):
    """Run ``h`` env steps from ``initial_carry``; differentiable in ``phi``.

    Returns
    -------
    final_carry : (x, v, C, F) tuple (same shape as initial_carry)
    rewards : (h,) float32 — per-step shaped reward (forward-disp delta - …)
    observations : (h+1, _OBS_DIM) float32 — obs at the start of each step + terminal
    """
    if x_morph is None:
        x_morph = jnp.ones((scene.n_voxels,), dtype=DTYPE)
    horizon_for_time = int(horizon_for_time or cfg.env_horizon)

    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)

    # Initial COM (for shaped reward delta).
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(initial_carry[0] * w0[:, None], axis=0)[0]

    def body(carry_state, dt_step):
        carry, prev_com_x = carry_state
        env_t = initial_t + dt_step
        new_carry, (com_v, com_x) = _env_step(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs,
            mass_field, cfg, terrain_height=terrain_height, actuator_weight=actuator_weight,
        )
        # Shaped per-step reward: positive forward COM displacement minus a
        # tiny control penalty (matches `rollout_return`'s integrated reward
        # pattern but split per step so SHAC's discounted sum is meaningful).
        delta_x = com_x[0] - prev_com_x
        ctrl_pen = DTYPE(1e-4) * jnp.sum(phi * phi)
        reward = delta_x - ctrl_pen
        obs = observation_from_carry(new_carry, env_t, mass_field, horizon_for_time)
        return (new_carry, com_x[0]), (reward, obs)

    init_obs = observation_from_carry(initial_carry, initial_t, mass_field, horizon_for_time)
    (final_state, _final_com_x), (rewards, obs_seq) = jax.lax.scan(
        body, (initial_carry, init_com_x), jnp.arange(h, dtype=jnp.int32),
    )
    # Stack initial obs + per-step obs so the critic sees s_0..s_h (h+1 entries).
    observations = jnp.concatenate([init_obs[None, :], obs_seq], axis=0)
    return final_state, rewards, observations


def rollout_return_batch(
    x_morph_batch: jnp.ndarray,    # (B, n_voxels)
    phi_batch: jnp.ndarray,         # (B, phi_dim)
    friction_batch: jnp.ndarray,    # (B,)
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    actuator_weight_voxel_batch: jnp.ndarray = None,   # (B, n_voxels, n_actuators) or None
    E_voxel_batch: jnp.ndarray = None,                 # (B, n_voxels) or None
    objective: str = "crawling",                       # DiffuseBot task objective
    terrain_height_batch: jnp.ndarray = None,          # (B, n, n) per-mode terrain | None
    gravity_vec_batch: jnp.ndarray = None,             # (B, 3) per-mode gravity tilt (slope) | None
    mass_scale_batch: jnp.ndarray = None,              # (B,) per-mode mass multiplier | None
    init_vel_batch: jnp.ndarray = None,                # (B, 3) per-mode initial COM vel | None
):
    """Vmap rollout_return over a batch of (morphology, controller, friction) tuples.

    Per-mode regime dispatch: ``*_batch`` args (terrain/gravity/mass/init) vary PER
    ROLLOUT in the vmap (in_axes 0 when provided, None otherwise) — so the risk
    marginalizer can roll one candidate over C regimes that differ in slope (gravity
    tilt), terrain, payload mass, or initial condition, not just friction. All None
    -> identical to the legacy friction-only path. ``terrain_height`` (scalar) stays
    the shared/legacy arg, used only when ``terrain_height_batch`` is None.

    Task-2: ``actuator_weight_voxel_batch`` / ``E_voxel_batch`` carry the per-candidate
    co-designed actuator / stiffness fields (vmapped alongside the morphology).
    """
    aw_ax = 0 if actuator_weight_voxel_batch is not None else None
    ev_ax = 0 if E_voxel_batch is not None else None
    th_ax = 0 if terrain_height_batch is not None else None
    gv_ax = 0 if gravity_vec_batch is not None else None
    ms_ax = 0 if mass_scale_batch is not None else None
    iv_ax = 0 if init_vel_batch is not None else None
    _th_batched = terrain_height_batch is not None

    def _one(xm, ph, fr, aw, ev, th, gv, ms, iv):
        r, disp, _ = rollout_return(
            xm, ph, fr, scene, cfg, num_env_steps, E0,
            terrain_height=(th if _th_batched else terrain_height),
            actuator_weight_voxel=aw, E_voxel=ev, objective=objective,
            gravity_vec=gv, mass_scale=ms, init_vel=iv,
        )
        return r, disp

    rs, disps = jax.vmap(_one, in_axes=(0, 0, 0, aw_ax, ev_ax, th_ax, gv_ax, ms_ax, iv_ax))(
        x_morph_batch, phi_batch, friction_batch,
        actuator_weight_voxel_batch, E_voxel_batch,
        terrain_height_batch, gravity_vec_batch, mass_scale_batch, init_vel_batch,
    )
    return rs, disps


def compute_actuation_learned(c, e_x, env_t, cfg, policy_params, v_com_x=DTYPE(0.0)):
    """Learned closed-loop controller (Stage 5; new_version.txt §IV-A):
        a = pi_beta([proprio, psi(t), e_x, c])  in [-1,1]^n_act.
    beta = policy_params is fixed; the controller LATENT c (the z^MB diffusion
    block) and morphology embedding e_x = E_chi(x) condition it by input
    concatenation. Self-contained (no solver import) so the env stays dep-free."""
    t = env_t.astype(DTYPE) / DTYPE(cfg.env_horizon)
    ang = DTYPE(2.0 * math.pi) * t
    psi = jnp.stack([jnp.sin(ang), jnp.cos(ang)])                 # (2,)
    v_scaled = (v_com_x * DTYPE(cfg.feedback_v_scale)).reshape(1)  # (1,)
    obs = jnp.concatenate([v_scaled, psi, e_x, c])               # (d_obs,)
    h = jnp.tanh(obs @ policy_params["w1"] + policy_params["b1"])
    return jnp.tanh(h @ policy_params["w2"] + policy_params["b2"])  # (n_act,)


def _env_step_closed(carry, env_t, c, e_x, policy_params, E_field, actuator_id,
                     friction, muscle_dirs, mass_field, cfg, terrain_height=None,
                     actuator_weight=None, gravity_vec=None):
    """One env step with the LEARNED closed-loop controller (mirror of _env_step)."""
    n_sub = cfg.substeps_per_env_step
    x, v, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    v_com_x = jnp.sum(v[:, 0] * w, axis=0)
    act_env = compute_actuation_learned(c, e_x, env_t, cfg, policy_params, v_com_x=v_com_x)
    act_t = act_env * DTYPE(cfg.act_strength_base) * DTYPE(cfg.actuation_strength_scale)

    def substep(cc, _):
        x, v, C, F = cc
        grid_v, grid_m, F_next, _ = p2g_3d(
            x, v, C, F, E_field, actuator_id, act_t, muscle_dirs, mass_field, cfg,
            actuator_weight=actuator_weight)
        grid_v_out = grid_op_3d(grid_v, grid_m, friction, cfg, terrain_height=terrain_height,
                                gravity_vec=gravity_vec)
        x_next, v_next, C_next = g2p_3d(x, grid_v_out, cfg)
        return (x_next, v_next, C_next, F_next), None

    carry, _ = jax.lax.scan(substep, carry, jnp.arange(n_sub, dtype=jnp.int32))
    x, v, _, _ = carry
    w = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    com_v = jnp.sum(v * w[:, None], axis=0)
    com_x = jnp.sum(x * w[:, None], axis=0)
    return carry, (com_v, com_x)


def _setup_fields_closed(x_morph, scene, cfg, E0, E_voxel, actuator_weight_voxel):
    """Shared field setup for the closed-loop rollouts (mirror of rollout_return head)."""
    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    if E_voxel is not None:
        E_field = E_voxel[scene.voxel_id]
    if actuator_weight_voxel is not None:
        actuator_weight = actuator_weight_voxel[scene.voxel_id]
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)
    return E_field, actuator_id, actuator_weight, muscle_dirs, mass_field


def rollout_return_closed(x_morph, c, friction, scene, cfg, num_env_steps,
                          policy_params, E_proj, E0=1.0, terrain_height=None,
                          actuator_weight_voxel=None, E_voxel=None, objective="crawling",
                          gravity_vec=None, mass_scale=None, init_vel=None):
    """Closed-loop analogue of rollout_return: controller = the learned policy
    conditioned on the latent c and morphology embedding e_x = tanh(occ @ E_chi).
    Per-mode regime: gravity_vec (slope), mass_scale, init_vel — all None -> legacy."""
    E_field, actuator_id, actuator_weight, muscle_dirs, mass_field = _setup_fields_closed(
        x_morph, scene, cfg, E0, E_voxel, actuator_weight_voxel)
    if mass_scale is not None:
        mass_field = mass_field * jnp.asarray(mass_scale, DTYPE)
    e_x = jnp.tanh(x_morph @ E_proj)                             # (d_e,)
    carry = _init_carry(scene)
    if objective == "landing":
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0 + jnp.asarray([0.0, 0.08, 0.0], DTYPE), _v0, _C0, _F0)
    elif objective == "balancing":
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0, _v0.at[:, 0].add(DTYPE(0.5)), _C0, _F0)
    if init_vel is not None:
        _x0, _v0, _C0, _F0 = carry
        carry = (_x0, _v0 + jnp.asarray(init_vel, DTYPE)[None, :], _C0, _F0)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        return _env_step_closed(
            carry, env_t, c, e_x, policy_params, E_field, actuator_id, friction,
            muscle_dirs, mass_field, cfg, terrain_height=terrain_height,
            actuator_weight=actuator_weight, gravity_vec=gravity_vec)

    carry, (com_v_hist, com_x_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32))
    per_step_forward_disp = jnp.sum(com_x_hist[:, 0] - init_com_x[0])
    final_disp = com_x_hist[-1, 0] - init_com_x[0]
    total_mass = jnp.sum(mass_field)
    mass_penalty = DTYPE(0.5) * jnp.maximum(DTYPE(0.1) * scene.n_particles - total_mass, 0.0)
    backward_sum = jnp.sum(jnp.maximum(-com_v_hist[:, 0], 0.0))
    smooth_pen = DTYPE(cfg.backward_penalty_weight) * backward_sum
    if objective == "balancing":
        drift = jnp.sum(jnp.abs(com_x_hist[:, 0] - init_com_x[0])
                        + jnp.abs(com_x_hist[:, 2] - init_com_x[2]))
        height_drop = jnp.maximum(init_com_x[1] - com_x_hist[-1, 1], 0.0)
        reward = -(drift + DTYPE(cfg.shaping_weight) * height_drop) - mass_penalty
    elif objective == "landing":
        speed = jnp.sum(jnp.linalg.norm(com_v_hist, axis=1))
        final_speed = jnp.linalg.norm(com_v_hist[-1])
        reward = -(speed + DTYPE(cfg.shaping_weight) * final_speed) - mass_penalty
    elif objective == "hurdling":
        peak_h = jnp.max(com_x_hist[:, 1] - init_com_x[1])
        reward = (final_disp + DTYPE(cfg.shaping_weight) * peak_h - mass_penalty - smooth_pen)
    else:  # crawling / default
        reward = (per_step_forward_disp + DTYPE(cfg.shaping_weight) * final_disp
                  - mass_penalty - smooth_pen)
    return reward, final_disp, com_x_hist


def rollout_return_closed_batch(x_morph_batch, c_batch, friction_batch, scene, cfg,
                                num_env_steps, policy_params, E_proj, E0=1.0,
                                terrain_height=None, actuator_weight_voxel_batch=None,
                                E_voxel_batch=None, objective="crawling",
                                terrain_height_batch=None, gravity_vec_batch=None,
                                mass_scale_batch=None, init_vel_batch=None):
    """Vmap rollout_return_closed over (morphology, controller-latent, friction).
    policy_params + E_proj shared. Per-mode regime: terrain/gravity(slope)/mass/init
    vary per rollout (in_axes 0 when provided); all None -> legacy friction-only."""
    aw_ax = 0 if actuator_weight_voxel_batch is not None else None
    ev_ax = 0 if E_voxel_batch is not None else None
    th_ax = 0 if terrain_height_batch is not None else None
    gv_ax = 0 if gravity_vec_batch is not None else None
    ms_ax = 0 if mass_scale_batch is not None else None
    iv_ax = 0 if init_vel_batch is not None else None
    _th_batched = terrain_height_batch is not None

    def _one(xm, c, fr, aw, ev, th, gv, ms, iv):
        r, disp, _ = rollout_return_closed(
            xm, c, fr, scene, cfg, num_env_steps, policy_params, E_proj, E0,
            terrain_height=(th if _th_batched else terrain_height),
            actuator_weight_voxel=aw, E_voxel=ev, objective=objective,
            gravity_vec=gv, mass_scale=ms, init_vel=iv)
        return r, disp

    rs, disps = jax.vmap(_one, in_axes=(0, 0, 0, aw_ax, ev_ax, th_ax, gv_ax, ms_ax, iv_ax))(
        x_morph_batch, c_batch, friction_batch,
        actuator_weight_voxel_batch, E_voxel_batch,
        terrain_height_batch, gravity_vec_batch, mass_scale_batch, init_vel_batch)
    return rs, disps


def rollout_with_positions_closed(x_morph, c, friction, scene, cfg, num_env_steps,
                                  policy_params, E_proj, E0=1.0, terrain_height=None,
                                  actuator_weight_voxel=None, E_voxel=None):
    """Closed-loop analogue of rollout_with_positions (crawling): returns the reward,
    final forward displacement, the COM-x trajectory, the particle positions per env
    step (for GIF), and the mass field — for diagnostics of the LEARNED controller."""
    E_field, actuator_id, actuator_weight, muscle_dirs, mass_field = _setup_fields_closed(
        x_morph, scene, cfg, E0, E_voxel, actuator_weight_voxel)
    e_x = jnp.tanh(x_morph @ E_proj)
    carry = _init_carry(scene)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        new_carry, (com_v, com_x) = _env_step_closed(
            carry, env_t, c, e_x, policy_params, E_field, actuator_id, friction,
            muscle_dirs, mass_field, cfg, terrain_height=terrain_height,
            actuator_weight=actuator_weight)
        return new_carry, (com_v, com_x, new_carry[0])      # also emit particle positions x

    carry, (com_v_hist, com_x_hist, x_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32))
    per_step_forward_disp = jnp.sum(com_x_hist[:, 0] - init_com_x[0])
    final_disp = com_x_hist[-1, 0] - init_com_x[0]
    total_mass = jnp.sum(mass_field)
    mass_penalty = DTYPE(0.5) * jnp.maximum(DTYPE(0.1) * scene.n_particles - total_mass, 0.0)
    backward_sum = jnp.sum(jnp.maximum(-com_v_hist[:, 0], 0.0))
    smooth_pen = DTYPE(cfg.backward_penalty_weight) * backward_sum
    reward = (per_step_forward_disp + DTYPE(cfg.shaping_weight) * final_disp
              - mass_penalty - smooth_pen)
    return reward, final_disp, com_x_hist, x_hist, mass_field


def rollout_h_closed_from_state(carry, start_t, c, h, friction, scene, cfg, x_morph,
                                policy_params, E_proj, E0=1.0, terrain_height=None,
                                actuator_weight_voxel=None, E_voxel=None):
    """h-step closed-loop rollout from a given state (for Stage-5b in-loop SHAC).
    Returns (carry_h, rewards_h (h,), com_x_h (h,3)); rewards_h[t] = forward COM
    velocity (a dense per-step signal the short-horizon critic objective sums)."""
    E_field, actuator_id, actuator_weight, muscle_dirs, mass_field = _setup_fields_closed(
        x_morph, scene, cfg, E0, E_voxel, actuator_weight_voxel)
    e_x = jnp.tanh(x_morph @ E_proj)

    def body(carry, env_t):
        carry, (com_v, com_x) = _env_step_closed(
            carry, env_t, c, e_x, policy_params, E_field, actuator_id, friction,
            muscle_dirs, mass_field, cfg, terrain_height=terrain_height,
            actuator_weight=actuator_weight)
        return carry, (com_v, com_x)

    env_ts = start_t + jnp.arange(h, dtype=jnp.int32)
    # jax.checkpoint the env-step body: under the SHAC backward pass the scan stores
    # only the h env-step boundary carries and REMATERIALIZES each step's MPM
    # substep grids, so peak memory is one env-step's tape (not h*n_sub * n_grid^3).
    # Essential to avoid OOM for the in-loop SHAC at production resolution.
    carry, (com_v_h, com_x_h) = jax.lax.scan(jax.checkpoint(body), carry, env_ts)
    return carry, com_v_h, com_x_h            # (h,3),(h,3); reward = com_v_h[:,0]


def rollout_return_push_batch(
    x_morph_batch: jnp.ndarray,    # (B, n_voxels)
    phi_batch: jnp.ndarray,         # (B, phi_dim)
    friction_batch: jnp.ndarray,    # (B,)
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    manip_cfg,
    goal_x: float,
    *,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    weights=(1.0, 5.0, 0.01, 50.0),
):
    """Vmap rollout_return_push over a (morphology, controller, friction) batch.

    Manipuland config, terrain, and goal are shared across the batch (single
    regime). Returns (rewards, dist_to_goal_T) — both (B,).
    """
    def _one(xm, ph, fr):
        r, dT, _, _ = rollout_return_push(
            xm, ph, fr, scene, cfg, num_env_steps,
            manip_cfg=manip_cfg, goal_x=goal_x,
            E0=E0, terrain_height=terrain_height, weights=weights,
        )
        return r, dT

    rs, dTs = jax.vmap(_one)(x_morph_batch, phi_batch, friction_batch)
    return rs, dTs


# ---------------------------------------------------------------------------
# Carry / transport task (writeup §11 Task 3). Like push, but the manipuland is
# gravity-loaded (manip_cfg.horizontal_only = False) so the BODY must support &
# transport it. Reward adds a contact-stability term G_T = fraction of the
# rollout the object stays carried (above carry_y_min), with a drop penalty.
# ---------------------------------------------------------------------------


def carry_reward(
    manip_xyz: jnp.ndarray,        # (T, 3) object trajectory
    init_pos: jnp.ndarray,         # (3,) object start
    *,
    goal_x: float,
    weights: Tuple[float, float, float, float] = (1.0, 3.0, 5.0, 50.0),
    carry_y_min: float = 0.10,
    floor_y: float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Carry reward = forward transport + closing-to-goal + contact stability G_T
    − drop penalty. Returns (reward, dist_to_goal_T, carried_frac).

        R = w0·Δx_obj + w1·(d0 − dT) + w2·G_T − w3·1[object on floor at T]
        G_T = mean_t 1[obj_y[t] ≥ carry_y_min]      (fraction of time carried)
    """
    obj_x = manip_xyz[:, 0]
    obj_y = manip_xyz[:, 1]
    final_x = obj_x[-1]
    delta_x = final_x - init_pos[0]
    d0 = jnp.abs(init_pos[0] - DTYPE(goal_x))
    dT = jnp.abs(final_x - DTYPE(goal_x))
    closing = d0 - dT
    carried_frac = jnp.mean((obj_y >= DTYPE(carry_y_min)).astype(obj_y.dtype))
    dropped = (obj_y[-1] <= DTYPE(floor_y) + DTYPE(1e-3)).astype(obj_y.dtype)
    w0, w1, w2, w3 = weights
    reward = (DTYPE(w0) * delta_x + DTYPE(w1) * closing
              + DTYPE(w2) * carried_frac - DTYPE(w3) * dropped)
    return reward, dT, carried_frac


def grip_reward(
    manip_xyz: jnp.ndarray,        # (T, 3) object trajectory
    init_pos: jnp.ndarray,         # (3,) object start
    *,
    lift_min: float = 0.12,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """DiffuseBot Gripping reward (objective="grasp", reward_mode="last_step_height"):
    grasp + LIFT the object — reward its final height gain plus the fraction of the
    rollout it stays lifted (grasp stability). Returns (reward, neg_final_height, lifted_frac)
    to match the (reward, dist, aux) signature of carry_reward.
    """
    obj_y = manip_xyz[:, 1]
    final_gain = obj_y[-1] - init_pos[1]
    peak_gain = jnp.max(obj_y) - init_pos[1]
    lifted_frac = jnp.mean((obj_y >= DTYPE(lift_min)).astype(obj_y.dtype))
    reward = DTYPE(50.0) * final_gain + DTYPE(10.0) * peak_gain + DTYPE(5.0) * lifted_frac
    return reward, -obj_y[-1], lifted_frac


def rollout_return_carry(
    x_morph: jnp.ndarray,
    phi: jnp.ndarray,
    friction: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    manip_cfg,
    goal_x: float,
    *,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    weights: Tuple[float, float, float, float] = (1.0, 3.0, 5.0, 50.0),
    carry_y_min: float = 0.10,
    floor_y: float = 0.05,
    objective: str = "carry",   # "carry"/"carry_terrain" → transport; "gripping" → grasp-lift
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Carry rollout. ``manip_cfg`` should have ``horizontal_only=False`` so the
    object is gravity-loaded and the body must carry it. Returns
    (reward, dist_to_goal_T, com_x_traj, manip_xyz_traj)."""
    from genedynamics.envs.external.jax_mpm.manipuland import init_manipuland_state

    E_field = _E_field_or_default(scene, cfg, E0)
    actuator_id = scene.actuator_id
    actuator_weight = _actuator_weight_or_none(scene)
    muscle_dirs = _fiber_dirs_or_default(scene, cfg)
    mass_field = _voxel_mass_field(x_morph, scene)

    manip_state0 = init_manipuland_state(manip_cfg, terrain_height=terrain_height, n_grid=cfg.n_grid)
    init_pos = manip_state0.pos
    x0, v0, C0, F0 = _init_carry(scene)
    carry = (x0, v0, C0, F0, manip_state0)

    def body(carry, env_t):
        return _env_step_with_manip(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg,
            terrain_height=terrain_height, actuator_weight=actuator_weight, manip_cfg=manip_cfg,
        )

    carry, (com_v_hist, com_x_hist, manip_xyz_hist) = jax.lax.scan(
        body, carry, jnp.arange(num_env_steps, dtype=jnp.int32),
    )
    if objective == "gripping":
        reward, dT, _aux = grip_reward(manip_xyz_hist, init_pos)
    else:
        reward, dT, _carried = carry_reward(
            manip_xyz_hist, init_pos, goal_x=goal_x, weights=weights,
            carry_y_min=carry_y_min, floor_y=floor_y,
        )
    return reward, dT, com_x_hist, manip_xyz_hist


def rollout_return_carry_batch(
    x_morph_batch: jnp.ndarray,
    phi_batch: jnp.ndarray,
    friction_batch: jnp.ndarray,
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    manip_cfg,
    goal_x: float,
    *,
    E0: float = 1.0,
    terrain_height: jnp.ndarray = None,
    weights: Tuple[float, float, float, float] = (1.0, 3.0, 5.0, 50.0),
    carry_y_min: float = 0.10,
    floor_y: float = 0.05,
    objective: str = "carry",
):
    """Vmap rollout_return_carry over a (morphology, controller, friction) batch.
    Returns (rewards, dist_to_goal_T), both (B,)."""
    def _one(xm, ph, fr):
        r, dT, _, _ = rollout_return_carry(
            xm, ph, fr, scene, cfg, num_env_steps,
            manip_cfg=manip_cfg, goal_x=goal_x,
            E0=E0, terrain_height=terrain_height, weights=weights,
            carry_y_min=carry_y_min, floor_y=floor_y, objective=objective,
        )
        return r, dT

    return jax.vmap(_one)(x_morph_batch, phi_batch, friction_batch)
