"""Procedural terrain height-fields for the soft-robot MPM scene.

Each generator returns a (n_grid, n_grid) float32 array giving the floor
height (in world-y) at MPM grid column (i_x, k_z). Heights live in domain
units (the grid spans [0, 1]^3 by convention); a "small" bump amplitude is
~0.02 (≈1/50 of the body box).

The MPM grid_op consumes this directly; flat terrain is the all-zero
field — equivalent to "no terrain" — so configs can opt in by passing a
TerrainSpec while the legacy code path (terrain_height=None) is unchanged.

Writeup §10.3 mandates the regime bank:
    train: {flat, slope-5deg, small bumps, soft patch}
    test : {slope-10deg, large bumps, ridge, gap, mixed}
The named generators below match that taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import math
import numpy as np


# All terrain heights are normalized so the maximum bump amplitude in the
# domain is ~0.02. Bigger amplitudes risk MPM CFL violations at dt=5e-4.
DEFAULT_BASE_FLOOR: float = 0.05    # matches scene.bound * dx ≈ 3/64 = 0.047
SAFE_AMP: float = 0.02


@dataclass(frozen=True)
class TerrainSpec:
    """Terrain regime descriptor.

    Fields
    ------
    name : str
        Identifier ("flat", "slope_5deg", ...). Used by the regime bank for
        filtering / serialization.
    height : (n, n) float32
        Per-(i_x, k_z) world-y floor height. n must match the MPM grid resolution
        when fed to grid_op_3d. Use `to_grid(spec, n_grid)` to resample to a
        different resolution.
    base_floor : float
        Constant offset applied beneath the variation; default matches the
        existing flat scene's bound*dx.
    """
    name: str
    height: np.ndarray
    base_floor: float = DEFAULT_BASE_FLOOR


# ---------------------------------------------------------------------------
# Coordinate helper: meshgrid in [0, 1]^2 over the (i_x, k_z) plane.
# ---------------------------------------------------------------------------


def _xz_meshgrid(n: int) -> Tuple[np.ndarray, np.ndarray]:
    g = (np.arange(n, dtype=np.float32) + 0.5) / float(n)
    X, Z = np.meshgrid(g, g, indexing="ij")
    return X, Z  # both (n, n) in [0, 1]


# ---------------------------------------------------------------------------
# Named terrain generators — all return TerrainSpec
# ---------------------------------------------------------------------------


def flat_terrain(n: int = 64) -> TerrainSpec:
    """Constant floor at base_floor. Equivalent to no-terrain code path."""
    h = np.zeros((n, n), dtype=np.float32)
    return TerrainSpec(name="flat", height=h)


def slope_terrain(n: int = 64, angle_deg: float = 5.0,
                  axis: str = "x") -> TerrainSpec:
    """Linear slope along +x (or +z). At angle 5°, height rises ~0.087 across
    the unit domain — capped to SAFE_AMP to stay numerically friendly.
    """
    X, Z = _xz_meshgrid(n)
    g = X if axis == "x" else Z
    rise = math.tan(math.radians(angle_deg)) * 1.0  # rise across the unit domain
    rise = float(np.clip(rise, -SAFE_AMP * 4, SAFE_AMP * 4))
    h = (g - 0.5) * rise
    return TerrainSpec(name=f"slope_{int(angle_deg)}deg_{axis}", height=h.astype(np.float32))


def gauss_bumps_terrain(
    n: int = 64,
    n_bumps: int = 6,
    amplitude: float = SAFE_AMP,
    sigma: float = 0.06,
    seed: int = 0,
) -> TerrainSpec:
    """Random Gaussian bumps. Centers drawn uniformly; heights summed."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(n_bumps, 2)).astype(np.float32)
    X, Z = _xz_meshgrid(n)
    h = np.zeros_like(X)
    for cx, cz in centers:
        d2 = (X - cx) ** 2 + (Z - cz) ** 2
        h = h + amplitude * np.exp(-d2 / (2.0 * sigma * sigma))
    return TerrainSpec(name=f"gauss_bumps_{n_bumps}", height=h.astype(np.float32))


def sin_bumps_terrain(
    n: int = 64,
    omega_x: float = 6.0,
    omega_z: float = 6.0,
    amplitude: float = SAFE_AMP * 0.5,
    phase_x: float = 0.0,
    phase_z: float = 0.0,
) -> TerrainSpec:
    """Periodic sinusoidal bump field — predictable, useful for ablations."""
    X, Z = _xz_meshgrid(n)
    h = amplitude * np.sin(omega_x * X + phase_x) * np.sin(omega_z * Z + phase_z)
    return TerrainSpec(name="sin_bumps", height=h.astype(np.float32))


def ridge_terrain(
    n: int = 64,
    center: float = 0.5,
    half_width: float = 0.05,
    height: float = SAFE_AMP,
) -> TerrainSpec:
    """Single transverse ridge (a wall along z) crossing x = center."""
    X, _ = _xz_meshgrid(n)
    band = np.abs(X - center) < half_width
    # Smooth top: triangle profile so MPM doesn't hit a discontinuity.
    h = np.where(
        band,
        height * (1.0 - np.abs(X - center) / max(half_width, 1e-6)),
        0.0,
    )
    return TerrainSpec(name="ridge", height=h.astype(np.float32))


def hurdle_terrain(
    n: int = 64,
    center: float = 0.55,
    half_width: float = 0.035,
    height: float = 2.0 * SAFE_AMP,
) -> TerrainSpec:
    """DiffuseBot Hurdling obstacle: a tall, narrow transverse wall ahead of the
    robot start that it must clear to advance. Like ridge_terrain but taller and
    placed forward of the spawn (center≈0.55) so it is a genuine hurdle."""
    X, _ = _xz_meshgrid(n)
    band = np.abs(X - center) < half_width
    h = np.where(band, height * (1.0 - np.abs(X - center) / max(half_width, 1e-6)), 0.0)
    return TerrainSpec(name="hurdle", height=h.astype(np.float32))


def gap_terrain(
    n: int = 64,
    center: float = 0.5,
    half_width: float = 0.06,
    depth: float = SAFE_AMP,
) -> TerrainSpec:
    """Single transverse gap (a slot) — local depression below base floor.

    Note: depth is bounded by base_floor so the local floor never goes below 0
    (otherwise particles would fall out of the world). The grid clamps to 0
    automatically.
    """
    X, _ = _xz_meshgrid(n)
    band = np.abs(X - center) < half_width
    h = np.where(
        band,
        -depth * (1.0 - np.abs(X - center) / max(half_width, 1e-6)),
        0.0,
    )
    return TerrainSpec(name="gap", height=h.astype(np.float32))


def soft_patch_terrain(
    n: int = 64,
    *,
    amplitude: float = SAFE_AMP * 0.25,
    seed: int = 0,
) -> TerrainSpec:
    """Low-frequency noise field — locally compliant, no large-scale features."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(-1.0, 1.0, size=(8, 8)).astype(np.float32)
    # Bilinear upsample to (n, n).
    cs = np.linspace(0, 1, 8, dtype=np.float32)
    fs = np.linspace(0, 1, n, dtype=np.float32)
    # Manual 2D linear interp via numpy.
    from numpy import interp
    rows = np.stack([interp(fs, cs, coarse[i]) for i in range(8)], axis=0)
    h = np.stack([interp(fs, cs, rows[:, j]) for j in range(n)], axis=1)
    return TerrainSpec(name="soft_patch", height=(amplitude * h).astype(np.float32))


def mixed_terrain(n: int = 64, seed: int = 0) -> TerrainSpec:
    """Sum of slope + bumps + ridge — combined regime for the test bank."""
    s = slope_terrain(n, angle_deg=5.0).height
    b = gauss_bumps_terrain(n, n_bumps=4, amplitude=SAFE_AMP * 0.5, seed=seed).height
    r = ridge_terrain(n, center=0.7, half_width=0.04, height=SAFE_AMP * 0.5).height
    return TerrainSpec(name="mixed", height=(s + b + r).astype(np.float32))


# ---------------------------------------------------------------------------
# Regime bank: writeup §10.3 train / test split, named for serialization.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainRegime:
    """Index entry for a single (terrain, friction) regime."""
    terrain: TerrainSpec
    friction: float
    mass_o: float = 1.0
    object_friction: float = 0.5

    @property
    def name(self) -> str:
        return f"{self.terrain.name}__mu{self.friction:.2f}__M{self.mass_o:.2f}"


def default_train_regimes(n_grid: int = 64) -> List[TerrainRegime]:
    """Writeup §10.3 training regime bank (4 terrains × 3 frictions = 12)."""
    terrains = [
        flat_terrain(n_grid),
        slope_terrain(n_grid, angle_deg=5.0),
        gauss_bumps_terrain(n_grid, n_bumps=4, amplitude=SAFE_AMP * 0.5, seed=1),
        soft_patch_terrain(n_grid, seed=2),
    ]
    bank: List[TerrainRegime] = []
    for t in terrains:
        for mu in (0.4, 0.6, 0.8):
            bank.append(TerrainRegime(terrain=t, friction=mu, mass_o=1.0))
    return bank


def default_test_regimes(n_grid: int = 64) -> List[TerrainRegime]:
    """Writeup §10.3 held-out regime bank (5 terrains × 4 frictions = 20)."""
    terrains = [
        slope_terrain(n_grid, angle_deg=10.0),
        gauss_bumps_terrain(n_grid, n_bumps=8, amplitude=SAFE_AMP, seed=11),
        ridge_terrain(n_grid),
        gap_terrain(n_grid),
        mixed_terrain(n_grid, seed=12),
    ]
    bank: List[TerrainRegime] = []
    for t in terrains:
        for mu in (0.3, 0.5, 0.7, 0.9):
            bank.append(TerrainRegime(terrain=t, friction=mu, mass_o=1.0))
    return bank


def to_grid(spec: TerrainSpec, n_grid: int) -> np.ndarray:
    """Resample a terrain height field to (n_grid, n_grid).

    Uses bilinear interpolation. Returns absolute world-y (with base_floor
    folded in) so the MPM grid kernel can use it directly.
    """
    h_src = spec.height
    n_src = h_src.shape[0]
    if n_src == n_grid:
        return (h_src + spec.base_floor).astype(np.float32)
    cs = np.linspace(0, 1, n_src, dtype=np.float32)
    fs = np.linspace(0, 1, n_grid, dtype=np.float32)
    rows = np.stack([np.interp(fs, cs, h_src[i]) for i in range(n_src)], axis=0)
    out = np.stack([np.interp(fs, cs, rows[:, j]) for j in range(n_grid)], axis=1)
    return (out + spec.base_floor).astype(np.float32)
