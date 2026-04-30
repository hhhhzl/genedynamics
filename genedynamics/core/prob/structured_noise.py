"""Geometry-aware structured noise sampler.

Decomposes joint-space exploration into three orthogonal channels using a
state-dependent Jacobian and (optional) surface normal:

.. math::
    u = J^\\dagger (P_T \\eta_T + n \\eta_n) + N(q) \\eta_N

where

  * :math:`\\eta_T` — tangential noise (along the surface, ``σ_tangent``);
  * :math:`\\eta_n` — normal noise (perpendicular to the surface, ``σ_normal``);
  * :math:`\\eta_N` — null-space noise (joint-space redundancy, ``σ_nullspace``).

Callbacks supply ``J(q)`` and (optionally) ``n(q)``; the sampler is
agnostic of how they are computed. Hand it ``JAXKinematics.jacobian`` and
``SDFGrid3D.sdf_and_grad``-derived normals to get the canonical use case;
hand it any other callbacks for visual servoing, dual-arm coordination, etc.

When ``normal_fn=None`` the sampler degenerates to "task-isotropic +
null-isotropic" (still strictly more structured than raw control-space
Gaussian).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from genedynamics.core.prob.noise_sampler import Backend, NoiseSampler


JacobianFn = Callable[[Any], Any]      # q -> (m, n)
NormalFn = Callable[[Any], Any]        # q -> (m,) unit, or None


def _xp_for(backend: Backend):
    if backend == "jax":
        if jnp is None:
            raise RuntimeError("backend='jax' but jax is not installed.")
        return jnp
    if backend == "torch":
        if torch is None:
            raise RuntimeError("backend='torch' but torch is not installed.")
        return torch
    return np


class StructuredNoise(NoiseSampler):
    """Structured noise via tangent / normal / null-space decomposition.

    Args:
        jacobian_fn: callable ``q -> J(q)``; ``J`` shape ``(m, n)``.
        normal_fn: callable ``q -> n(q)``; ``n`` shape ``(m,)``, unit norm.
            Pass ``None`` to skip the tangent/normal split (sampler then
            falls back to task-isotropic + null-isotropic).
        sigma_tangent: stddev of tangential noise ``η_T``.
        sigma_normal: stddev of normal noise ``η_n``.
        sigma_nullspace: stddev of null-space noise ``η_N``.
        damping: Tikhonov damping for ``J^†`` (stabilise near singularities).
        backend: ``"numpy"`` / ``"jax"`` / ``"torch"``. Must be consistent
            with what ``jacobian_fn`` / ``normal_fn`` return.

    Notes:
        ``state`` passed to :meth:`sample` is interpreted as the current
        joint configuration ``q``. Callers in MPPI/MBD pass the latest
        nominal ``q`` per planning step; under jit/vmap the state can be
        a leading-batched array of shape ``(B, n_dof)``.
    """

    backend: Backend

    def __init__(
        self,
        *,
        jacobian_fn: JacobianFn,
        normal_fn: Optional[NormalFn],
        sigma_tangent: float,
        sigma_normal: float = 0.0,
        sigma_nullspace: float = 0.0,
        damping: float = 1e-3,
        backend: Backend = "numpy",
    ) -> None:
        self.jacobian_fn = jacobian_fn
        self.normal_fn = normal_fn
        self.sigma_tangent = float(sigma_tangent)
        self.sigma_normal = float(sigma_normal)
        self.sigma_nullspace = float(sigma_nullspace)
        self.damping = float(damping)
        self.backend = backend
        self._xp = _xp_for(backend)

    # ----- helpers ----------------------------------------------------------

    def _jacobian_pinv(self, J: Any) -> Any:
        xp = self._xp
        m, n = J.shape[-2], J.shape[-1]
        lam2 = xp.asarray(self.damping ** 2, dtype=J.dtype)
        if n > m:                               # wide / redundant
            JJt = J @ self._T(J)
            reg = JJt + lam2 * xp.eye(m, dtype=J.dtype)
            return self._T(J) @ self._inv(reg)
        else:                                   # tall / square
            JtJ = self._T(J) @ J
            reg = JtJ + lam2 * xp.eye(n, dtype=J.dtype)
            return self._inv(reg) @ self._T(J)

    @staticmethod
    def _T(M: Any) -> Any:
        return M.T if M.ndim == 2 else M.swapaxes(-1, -2)

    def _inv(self, M: Any) -> Any:
        # Solve via identity RHS, robust across backends.
        n = M.shape[-1]
        if self.backend == "jax":
            return jnp.linalg.solve(M, jnp.eye(n, dtype=M.dtype))
        if self.backend == "torch":
            return torch.linalg.solve(M, torch.eye(n, dtype=M.dtype, device=M.device))
        return np.linalg.solve(M, np.eye(n, dtype=M.dtype))

    def _randn(self, shape: tuple, key: Any, dtype: Any) -> Any:
        if self.backend == "jax":
            if key is None:
                raise ValueError("jax backend requires a PRNGKey via key=")
            return jax.random.normal(key, shape).astype(dtype)
        if self.backend == "torch":
            return torch.randn(*shape, generator=key).to(dtype=dtype)
        rng = key if isinstance(key, np.random.Generator) else np.random.default_rng(key)
        return rng.standard_normal(shape).astype(dtype)

    # ----- public API -------------------------------------------------------

    def sample(
        self,
        shape: tuple,
        *,
        key: Any = None,
        state: Any = None,
    ) -> Any:
        """Draw structured noise.

        Args:
            shape: trailing axis must equal ``n_dof`` (control dimension).
                Leading axes are treated as a sample batch.
            key: backend-specific RNG handle (``PRNGKey`` for jax,
                ``np.random.Generator`` or seed-int for numpy,
                ``torch.Generator`` for torch).
            state: current joint configuration ``q`` (shape ``(n_dof,)``).
                Required — the sampler is state-dependent.

        Returns:
            Noise array of shape ``shape``.
        """
        if state is None:
            raise ValueError("StructuredNoise requires `state` (current q).")
        xp = self._xp
        q = xp.asarray(state)
        J = self.jacobian_fn(q)                              # (m, n)
        m, n = J.shape[-2], J.shape[-1]
        Jp = self._jacobian_pinv(J)                          # (n, m)
        # Null-space projector N = I - J† J  ∈ R^{n×n}.
        N = xp.eye(n, dtype=J.dtype) - Jp @ J

        if shape[-1] != n:
            raise ValueError(
                f"trailing axis of shape {shape} does not match n_dof={n}"
            )
        batch_shape = shape[:-1]

        # Split RNG for the three independent noise channels (jax-style for
        # jax, separate generators for numpy/torch).
        keys = self._split_keys(key, 3)
        # η_τ ∈ R^{m}: full m-dim task-space noise; gets projected by either
        # P_T (tangent) or onto n (normal). When normal_fn is None we skip
        # the split entirely and treat the whole m-dim as task-isotropic.
        eta_task = self._randn(batch_shape + (m,), keys[0], J.dtype)
        eta_null = self._randn(batch_shape + (n,), keys[2], J.dtype)

        if self.normal_fn is None:
            # Task-isotropic: scale all m components by sigma_tangent.
            task = self.sigma_tangent * eta_task
        else:
            n_vec = self.normal_fn(q)                         # (m,)
            n_vec = n_vec / xp.maximum(xp.linalg.norm(n_vec), xp.asarray(1e-8, dtype=J.dtype))
            # Tangential part: P_T η = η - (n^T η) n.
            inner = (eta_task * n_vec).sum(axis=-1, keepdims=True)
            tan = eta_task - inner * n_vec
            # Normal part: a fresh scalar noise scaled along n.
            eta_n_scalar = self._randn(batch_shape + (1,), keys[1], J.dtype)
            normal = eta_n_scalar * n_vec
            task = self.sigma_tangent * tan + self.sigma_normal * normal

        # Project task noise into joint space and add null-space noise.
        u = (Jp @ task[..., :, None]).squeeze(-1) + self.sigma_nullspace * (N @ eta_null[..., :, None]).squeeze(-1)
        return u

    # ----- key management ---------------------------------------------------

    def _split_keys(self, key: Any, n: int):
        if self.backend == "jax":
            if key is None:
                raise ValueError("jax backend requires a PRNGKey via key=")
            return jax.random.split(key, n)
        # numpy / torch don't need real splitting; reuse the generator.
        return [key] * n


__all__ = ["StructuredNoise", "JacobianFn", "NormalFn"]
