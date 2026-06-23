"""Position-stiffness control primitive (log-Euclidean SPD parametrisation).

Upstream, solver-agnostic control-action primitive: a lower-control primitive
`u_h = (r_h, K_h, nu_h)` — a position/reference command `r_h`, a task-space
stiffness `K_h`, and a feedforward `nu_h` — with the stiffness in the
log-Euclidean chart `K_h = exp(S_h)`, `S_h` symmetric, so that

  * sampling / optimization happen over the UNCONSTRAINED symmetric coordinate
    `S_h` (flattened to its `svec`), and
  * the EXECUTED `K_h` is SPD by construction (`exp(lambda_i) > 0`), with
    `jax.grad` finite (matrix-exp of a symmetric matrix).

This lives in `genedynamics/core/control/` (not inside any solver) because the
primitive is reusable by ANY env (an impedance `act2impedance` law) and ANY
solver (MDAC samples it; others may too). The MDAC solver imports it; it is not
MDAC-private.

`PrimitiveSpec.total_width` is the SINGLE coupling point that widens a sampler's
`action_size`: an env that opts into the impedance primitive reports
`total_width`; torque/position envs keep their native width (no stiffness block).

The `svec` packing is upper-triangular row-major (plain, round-trip-exact:
`svec2sym(sym2svec(M)) == M` for symmetric `M`); no Mandel scaling, since any
metric weight over these coordinates is supplied separately.

Pure JAX, no mjx/brax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Symmetric (svec) vectorization — linear, round-trip exact
# ---------------------------------------------------------------------------

def svec_len(d: int) -> int:
    """Number of free entries of a d x d symmetric matrix = d(d+1)/2."""
    return d * (d + 1) // 2


def _triu(d: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.triu_indices(d)  # includes the diagonal, row-major


def sym2svec(M: jnp.ndarray) -> jnp.ndarray:
    """(..., d, d) symmetric -> (..., L) upper-triangular (incl. diag) vector."""
    d = M.shape[-1]
    r, c = _triu(d)
    return M[..., r, c]


def svec2sym(v: jnp.ndarray, d: int) -> jnp.ndarray:
    """(..., L) -> (..., d, d) symmetric. Inverse of :func:`sym2svec`."""
    r, c = _triu(d)
    out = jnp.zeros(v.shape[:-1] + (d, d), dtype=v.dtype)
    out = out.at[..., r, c].set(v)            # fill upper triangle (incl diag)
    diag = jnp.arange(d)
    upper_diag = out[..., diag, diag]         # the diagonal we just placed
    out = out + jnp.swapaxes(out, -1, -2)     # mirror -> off-diag ok, diag doubled
    out = out.at[..., diag, diag].add(-upper_diag)  # undo diagonal double-count
    return out


# ---------------------------------------------------------------------------
# Log-Euclidean SPD stiffness  K = exp(S)
# ---------------------------------------------------------------------------

def stiffness_log_to_pd(s_vec: jnp.ndarray, d: int) -> jnp.ndarray:
    """`s_vec` (svec of symmetric S) -> SPD `K = exp(S)` via symmetric eigh.

    `S = Q diag(lambda) Q^T` (eigh), `K = Q diag(exp(lambda)) Q^T`. SPD by
    construction; differentiable (distinct eigenvalues a.s.).
    """
    S = svec2sym(s_vec, d)
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2))   # numerical symmetrization
    lam, Q = jnp.linalg.eigh(S)
    return (Q * jnp.exp(lam)[..., None, :]) @ jnp.swapaxes(Q, -1, -2)


def stiffness_pd_to_log(K: jnp.ndarray) -> jnp.ndarray:
    """Inverse chart `S = log(K)` (svec) for an SPD `K` (round-trip with
    :func:`stiffness_log_to_pd`). Used to seed `S` from a target stiffness."""
    lam, Q = jnp.linalg.eigh(0.5 * (K + jnp.swapaxes(K, -1, -2)))
    S = (Q * jnp.log(jnp.clip(lam, 1e-12, None))[..., None, :]) @ jnp.swapaxes(Q, -1, -2)
    return sym2svec(S)


# ---------------------------------------------------------------------------
# Primitive layout: u_h = [ r (pos) | s_vec (stiffness chart) | nu (feedforward) ]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveSpec:
    """Layout of one lower-control primitive `u_h`.

    Concatenation order (per node/step): `[ r_h ; svec(S_h) ; nu_h ]`.
    `stiff_dim == 0` disables the stiffness block (degenerates to a flat
    position+feedforward control of width `pos_dim + feed_dim`).
    """

    pos_dim: int = 0        # reference / position command width  (r)
    stiff_dim: int = 0      # stiffness matrix dimension d (svec width = d(d+1)/2); 0 = none
    feed_dim: int = 0       # feedforward / residual torque width (nu)

    @property
    def stiff_width(self) -> int:
        return svec_len(self.stiff_dim) if self.stiff_dim > 0 else 0

    @property
    def total_width(self) -> int:
        """The widened `action_size` an impedance env reports."""
        return self.pos_dim + self.stiff_width + self.feed_dim

    @property
    def r_slice(self) -> slice:
        return slice(0, self.pos_dim)

    @property
    def s_slice(self) -> slice:
        return slice(self.pos_dim, self.pos_dim + self.stiff_width)

    @property
    def nu_slice(self) -> slice:
        return slice(self.pos_dim + self.stiff_width, self.total_width)


def unpack_primitive(u: jnp.ndarray, spec: PrimitiveSpec):
    """Split a primitive vector `u` (..., total_width) into (r, s_vec, nu).

    `s_vec` is the raw stiffness chart coordinate (call
    :func:`stiffness_log_to_pd` to get the executed SPD `K`).
    """
    return u[..., spec.r_slice], u[..., spec.s_slice], u[..., spec.nu_slice]


def metric_GK_diag(spec: PrimitiveSpec, w_S: float) -> jnp.ndarray:
    """Per-primitive diagonal of a stiffness-coordinate metric `G_K`.

    Acts ONLY on the stiffness `svec` coordinates: `diag = w_S` on `s_slice`,
    `0` elsewhere. Length `total_width`; tile across nodes/steps for the full `U`.
    """
    diag = jnp.zeros((spec.total_width,), dtype=jnp.float32)
    if spec.stiff_width > 0 and w_S != 0.0:
        diag = diag.at[spec.s_slice].set(float(w_S))
    return diag


def metric_GK_diag_full(spec: PrimitiveSpec, w_S: float, n_steps: int) -> jnp.ndarray:
    """`G_K` diagonal over the flattened `U` of all steps ((n_steps*total_width,))."""
    return jnp.tile(metric_GK_diag(spec, w_S), int(n_steps))


__all__ = [
    "svec_len",
    "sym2svec",
    "svec2sym",
    "stiffness_log_to_pd",
    "stiffness_pd_to_log",
    "PrimitiveSpec",
    "unpack_primitive",
    "metric_GK_diag",
    "metric_GK_diag_full",
]
