"""JAX 3D MPM for soft-robot co-design — crawling_ground task.

Adapted from the hw4 JAX MPM reference; kept single-file so the whole
pipeline (p2g, grid, g2p, rollout, reward) is easy to audit / JIT.

Pipeline per env step:
    p2g_3d  : particles → grid momentum/mass
    grid_op : momentum/mass → velocity, add gravity, walls, floor-friction
    g2p_3d  : grid velocity → new particle (x, v, C)

theta = (x_morph, phi_ctrl) convention, same as the softzoo backend:
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

    Defaults are tuned to approximate softzoo's `crawling_ground` at reduced
    grid resolution (64 instead of 128) so the whole rollout fits comfortably
    on one 3090 even when vmap'd over 32 proposals.
    """

    n_grid: int = 64
    # CFL: dt <= dx / sqrt(E*scale). With E=1, scale=100, mu~100, c~10, dx=1/64
    # -> dt <= ~1.5e-3. Match softzoo's 5e-4 for safety.
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

    # Robot box (matches softzoo's ground.yaml Primitive.Box).
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


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_scene(cfg: MPMConfig) -> SceneData:
    """Build a soft box with actuators assigned by X-bin.

    Actuator 0 is front (largest X), actuator n-1 is back. A traveling-wave
    controller (phase along actuator index) therefore produces a locomotion
    gait along +X.
    """
    sx, sy, sz = cfg.box_size
    ox, oy, oz = cfg.box_origin
    s = cfg.particle_spacing
    nx = max(1, int(sx / s))
    ny = max(1, int(sy / s))
    nz = max(1, int(sz / s))
    rx = sx / nx
    ry = sy / ny
    rz = sz / nz

    # Generate particles on a regular grid centred inside each sub-cell.
    i_idx = np.arange(nx, dtype=np.float32)
    j_idx = np.arange(ny, dtype=np.float32)
    k_idx = np.arange(nz, dtype=np.float32)
    ii, jj, kk = np.meshgrid(i_idx, j_idx, k_idx, indexing="ij")
    xs = ox + (ii + 0.5) * rx
    ys = oy + (jj + 0.5) * ry
    zs = oz + (kk + 0.5) * rz
    pts = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=-1)

    # Actuator group by X-bin (front → back). Keep a thin passive dorsal strip
    # on top so there's a neutral core (like softzoo's table-top).
    xs_flat = pts[:, 0]
    x_min, x_max = xs_flat.min(), xs_flat.max()
    bin_idx = np.floor(
        (xs_flat - x_min) / max(x_max - x_min, 1e-6) * cfg.n_actuators
    ).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, cfg.n_actuators - 1)

    # Top 20% of Y becomes passive (−1) so the robot has an inert spine.
    ys_flat = pts[:, 1]
    y_top_thresh = np.quantile(ys_flat, 0.80)
    act_id = np.where(ys_flat >= y_top_thresh, -1, bin_idx).astype(np.int32)

    # Per-particle voxel index (for occupancy-based morphology). The voxel grid
    # tiles the bounding box uniformly with cfg.voxel_dims cells.
    vx, vy, vz = cfg.voxel_dims
    n_voxels = vx * vy * vz
    zs_flat = pts[:, 2]
    y_min, y_max = ys_flat.min(), ys_flat.max()
    z_min, z_max = zs_flat.min(), zs_flat.max()
    vi = np.clip(np.floor((xs_flat - x_min) / max(x_max - x_min, 1e-6) * vx).astype(np.int32), 0, vx - 1)
    vj = np.clip(np.floor((ys_flat - y_min) / max(y_max - y_min, 1e-6) * vy).astype(np.int32), 0, vy - 1)
    vk = np.clip(np.floor((zs_flat - z_min) / max(z_max - z_min, 1e-6) * vz).astype(np.int32), 0, vz - 1)
    voxel_id = (vi * vy * vz + vj * vz + vk).astype(np.int32)

    return SceneData(
        x0=jnp.asarray(pts, dtype=DTYPE),
        actuator_id=jnp.asarray(act_id, dtype=jnp.int32),
        voxel_id=jnp.asarray(voxel_id, dtype=jnp.int32),
        n_particles=int(pts.shape[0]),
        n_actuators=int(cfg.n_actuators),
        n_voxels=int(n_voxels),
    )


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
):
    base = jnp.floor(x * cfg.inv_dx - 0.5).astype(jnp.int32)
    fx = x * cfg.inv_dx - base.astype(DTYPE)
    weights = kernel_weights_3d(fx)

    new_F = (I3 + cfg.dt * C) @ F
    tau, J_raw = _stress_and_J(new_F, E, actuator_id, act_t, muscle_dirs, cfg)

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
):
    vmap_fn = jax.vmap(
        lambda xp, vp, Cp, Fp, Ep, aid, mp: _p2g_single(
            xp, vp, Cp, Fp, Ep, aid, act_t, muscle_dirs, mp, cfg
        )
    )
    idx, momentum, mass_contrib, new_F, J_raw = vmap_fn(x, v, C, F, E_field, actuator_id, mass_field)
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
) -> jnp.ndarray:
    inv_m = 1.0 / (grid_m_in + 1e-10)
    v_out = grid_v_in * inv_m[..., None]
    v_out = v_out.at[..., 1].add(-cfg.dt * cfg.gravity)

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

    # Coulomb friction on the floor.
    bottom_hit = bottom & (v_out[..., 1] < 0.0)
    vn = v_out[..., 1]
    vtx = v_out[..., 0]
    vtz = v_out[..., 2]
    speed_t = jnp.sqrt(vtx * vtx + vtz * vtz)
    friction_limit = friction * (-vn)
    scale = jnp.maximum(speed_t - friction_limit, 0.0) / (speed_t + DTYPE(1e-8))
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
            x, v, C, F, E_field, actuator_id, act_t, muscle_dirs, mass_field, cfg
        )
        grid_v_out = grid_op_3d(grid_v, grid_m, friction, cfg)
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


def rollout_return(
    x_morph: jnp.ndarray,     # (n_voxels,) ∈ [0,1] occupancy per voxel
    phi: jnp.ndarray,          # (phi_dim,) controller params
    friction: jnp.ndarray,     # scalar
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one rollout. Returns (blended_reward, final_forward_disp, com_x_traj)."""
    E_field = jnp.full((scene.n_particles,), DTYPE(E0), dtype=DTYPE)
    actuator_id = scene.actuator_id
    muscle_dirs = _muscle_directions(cfg.n_actuators)
    mass_field = _voxel_mass_field(x_morph, scene)

    carry = _init_carry(scene)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        return _env_step(carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg)

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
    reward = (per_step_forward_disp
              + DTYPE(cfg.shaping_weight) * final_disp
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
):
    """Like `rollout_return` but also returns particle positions per env step (for GIF)."""
    E_field = jnp.full((scene.n_particles,), DTYPE(E0), dtype=DTYPE)
    actuator_id = scene.actuator_id
    muscle_dirs = _muscle_directions(cfg.n_actuators)
    mass_field = _voxel_mass_field(x_morph, scene)
    carry = _init_carry(scene)
    w0 = mass_field / (jnp.sum(mass_field) + DTYPE(1e-8))
    init_com_x = jnp.sum(scene.x0 * w0[:, None], axis=0)

    def body(carry, env_t):
        new_carry, (com_v, com_x) = _env_step(
            carry, env_t, phi, E_field, actuator_id, friction, muscle_dirs, mass_field, cfg
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


def rollout_return_batch(
    x_morph_batch: jnp.ndarray,    # (B, n_voxels)
    phi_batch: jnp.ndarray,         # (B, phi_dim)
    friction_batch: jnp.ndarray,    # (B,)
    scene: SceneData,
    cfg: MPMConfig,
    num_env_steps: int,
    E0: float = 1.0,
):
    def _one(xm, ph, fr):
        r, disp, _ = rollout_return(xm, ph, fr, scene, cfg, num_env_steps, E0)
        return r, disp

    rs, disps = jax.vmap(_one)(x_morph_batch, phi_batch, friction_batch)
    return rs, disps
