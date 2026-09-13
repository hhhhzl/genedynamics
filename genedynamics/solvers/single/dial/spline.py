"""Node <-> dense control spline for the DIAL-MPC solver.

DIAL-MPC parametrises the control sequence by ``Hnode+1`` spline *nodes* and
interpolates them to ``Hsample+1`` dense controls with a quadratic
(``k=2``) :class:`jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline`
(see dial-mpc ``core/dial_core.py`` ``MBDPI.node2u`` / ``u2node``). This
node-spline parametrisation is the ONE solver primitive genedynamics did not
already have (0 hits across the repo); every other DIAL behaviour reuses
existing MBD machinery.

Because the ``k=2`` spline is *linear in the y-values* for fixed knots, both
``node2u`` and ``u2node`` are constant linear maps. We materialise them ONCE as
matrices

    us    = N2U @ nodes        N2U : (Hsample+1, Hnode+1)
    nodes = U2N @ us           U2N : (Hnode+1, Hsample+1)

built by applying the reference jax_cosmo spline to each basis vector. The
cached matrices make the per-node constraint Jacobian chaining (needed later by
MGA) trivial and avoid re-fitting a spline inside the diffusion scan. We keep
jax_cosmo only as the build-time/parity oracle, exactly as planned (the hot path
is a single matmul).

All functions are pure JAX, run on CPU (no mjx), and are unit-tested for
byte-parity (rtol 1e-5) against jax_cosmo in ``test/unit/test_dial_spline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp


def _spline_1d(x_in: jnp.ndarray, y_in: jnp.ndarray, x_out: jnp.ndarray) -> jnp.ndarray:
    """Reference k=2 spline eval, matching dial-mpc exactly (single dim).

    jax_cosmo is a build-time oracle (used only to construct the cached spline
    matrices in NodeSpline.build), imported lazily so that importing the DIAL
    solver / the solvers package does NOT hard-require jax_cosmo. Only DIAL
    solver *construction* needs it.
    """
    from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
    return InterpolatedUnivariateSpline(x_in, y_in, k=2)(x_out)


def build_interp_matrix(x_in: jnp.ndarray, x_out: jnp.ndarray) -> jnp.ndarray:
    """Constant matrix ``M`` s.t. ``M @ y_in == spline(x_in, y_in)(x_out)``.

    Built by applying the k=2 spline to each basis vector ``e_i`` of the input
    grid; valid because the spline is linear in ``y_in`` for fixed knots.

    Returns: ``M`` of shape ``(len(x_out), len(x_in))``.
    """
    n_in = x_in.shape[0]
    basis = jnp.eye(n_in, dtype=jnp.float64)
    # cols[i] = spline(x_in, e_i)(x_out)  ->  (len(x_out),)
    cols = jax.vmap(lambda e: _spline_1d(x_in, e, x_out), in_axes=0, out_axes=1)(basis)
    return cols  # (len(x_out), len(x_in))


_CACHE_PATH = __import__("pathlib").Path(__file__).resolve().parent / "spline_cache.npz"
_CACHE = None


def _load_cached_matrices(Hnode: int, Hsample: int):
    """Return (N2U, U2N) jnp arrays from the vendored cache, or None if absent.

    The cache (generated once via the jax_cosmo oracle) removes the jax_cosmo
    runtime dependency for the standard DIAL grids, so the solver imports and
    runs in environments without jax_cosmo (e.g. the CPU docker image).
    """
    global _CACHE
    try:
        if _CACHE is None:
            import numpy as _np
            _CACHE = dict(_np.load(_CACHE_PATH))
        kn, ku = f"n2u_{Hnode}_{Hsample}", f"u2n_{Hnode}_{Hsample}"
        if kn in _CACHE and ku in _CACHE:
            return jnp.asarray(_CACHE[kn]), jnp.asarray(_CACHE[ku])
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class NodeSpline:
    """Cached node<->dense quadratic spline maps for one (Hnode, Hsample) grid.

    Mirrors dial-mpc's grids: ``ctrl_dt`` and total span cancel in the linear
    map, but we keep them so the knot layout is faithful to upstream.
    """

    Hnode: int
    Hsample: int
    ctrl_dt: float = 0.02

    N2U: jnp.ndarray = None  # (Hsample+1, Hnode+1)  nodes -> dense us
    U2N: jnp.ndarray = None  # (Hnode+1, Hsample+1)  us -> nodes

    @staticmethod
    def build(Hnode: int, Hsample: int, ctrl_dt: float = 0.02) -> "NodeSpline":
        # Fast path: load the vendored cached matrices (no jax_cosmo at runtime).
        # The k=2 uniform-knot interpolation matrix is scale-invariant in ctrl_dt
        # (verified), so the cache is keyed by (Hnode, Hsample) only.
        cached = _load_cached_matrices(Hnode, Hsample)
        if cached is not None:
            n2u, u2n = cached
        else:
            # Uncached grid: build via the jax_cosmo oracle (lazy import inside).
            span = ctrl_dt * Hsample
            step_us = jnp.linspace(0.0, span, Hsample + 1, dtype=jnp.float64)
            step_nodes = jnp.linspace(0.0, span, Hnode + 1, dtype=jnp.float64)
            n2u = build_interp_matrix(step_nodes, step_us).astype(jnp.float32)
            u2n = build_interp_matrix(step_us, step_nodes).astype(jnp.float32)
        return NodeSpline(Hnode=Hnode, Hsample=Hsample, ctrl_dt=ctrl_dt, N2U=n2u, U2N=u2n)

    # --- hot-path maps (single matmul, jit-friendly) ---

    def node2u(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """(Hnode+1, nu) -> (Hsample+1, nu). Matches dial-mpc node2u_vmap."""
        return self.N2U @ nodes

    def u2node(self, us: jnp.ndarray) -> jnp.ndarray:
        """(Hsample+1, nu) -> (Hnode+1, nu). Matches dial-mpc u2node_vmap."""
        return self.U2N @ us

    def node2u_batch(self, nodes_b: jnp.ndarray) -> jnp.ndarray:
        """(B, Hnode+1, nu) -> (B, Hsample+1, nu)."""
        return jnp.einsum("hn,bnu->bhu", self.N2U, nodes_b)

    def shift_nodes(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """Receding-horizon shift in DIAL's convention: expand to dense, roll
        one control step, zero the tail, refit to nodes
        (dial-mpc ``MBDPI.shift``)."""
        us = self.node2u(nodes)
        us = jnp.roll(us, -1, axis=0)
        us = us.at[-1].set(jnp.zeros_like(us[-1]))
        return self.u2node(us)

    def shift_nodes_terminal_hold(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """Shift while holding the last previously planned control at the tail.

        A guarded receding incumbent can legitimately survive several rejected
        replans.  Zero-padding would then erase it one control at a time and
        confound rejection with an artificial stop command.  Terminal hold is
        the state-free backup extension for that guarded path; canonical DIAL
        continues to use :meth:`shift_nodes` and its exact zero-tail convention.
        """
        us = self.node2u(nodes)
        terminal = us[-1]
        us = jnp.roll(us, -1, axis=0)
        us = us.at[-1].set(terminal)
        return self.u2node(us)

    def shift_nodes_certified_terminal_hold(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """Shift with an exact first action and a terminal-hold extension.

        Refitting a one-step-shifted dense sequence into fewer spline nodes is
        only a least-squares approximation.  For a receding safety certificate,
        that approximation must not rewrite the immediate backup action that
        was certified in the preceding cycle.  The spline interpolates its
        first endpoint exactly, so pinning node zero to the old dense action at
        index one preserves that deployed successor while retaining the usual
        terminal-hold fit for the remaining horizon.
        """
        us = self.node2u(nodes)
        first_backup = us[1]
        terminal = us[-1]
        shifted_us = jnp.roll(us, -1, axis=0).at[-1].set(terminal)
        shifted_nodes = self.u2node(shifted_us)
        return shifted_nodes.at[0].set(first_backup)
