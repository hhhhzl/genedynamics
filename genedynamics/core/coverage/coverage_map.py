"""Stateful coverage-map primitive for path-coverage tasks.

A :class:`CoverageMap` discretises a target region into ``N`` sample points
and maintains a per-rollout ``visit_strength`` array tracking how thoroughly
each sample has been visited along an evolving path. The update rule is

.. math::
    s_i \\leftarrow \\max\\bigl(s_i, \\exp(-\\|p_t - c_i\\|^2 / 2\\sigma^2)\\bigr).

Generic primitive: applies to any "cover-the-region" task — cleaning,
painting, search-and-rescue, mowing, inspection, ultrasound scanning. The
configuration (samples / sigma / threshold) is stored once on the
:class:`CoverageMap` instance, but the **per-rollout** ``visit_strength``
state is held by the caller and updated functionally — so MBD/MPPI scan
bodies can ``vmap`` independent rollouts each carrying their own state.

Backend-agnostic via the same ``Backend = Literal["numpy", "jax", "torch"]``
pattern used by ``envs.obstacles.SDFGrid3D`` and ``core.contact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


Backend = Literal["numpy", "jax", "torch"]
ArrayLike = Any


_BACKEND_MOD = {"numpy": np}
if jnp is not None:
    _BACKEND_MOD["jax"] = jnp
if torch is not None:
    _BACKEND_MOD["torch"] = torch


def _xp(backend: Backend):
    mod = _BACKEND_MOD.get(backend)
    if mod is None:
        raise RuntimeError(
            f"backend {backend!r} not available; install jax or torch to enable it."
        )
    return mod


@dataclass(frozen=True)
class CoverageMap:
    """Coverage-map config + functional update operators.

    Args:
        samples: ``(N, D)`` sample points (``D`` is task-specific — typically
            3 for surface coverage, 2 for floor cleaning).
        sigma: Gaussian kernel width controlling how "wide" each visit
            counts. Smaller σ → tighter spatial resolution.
        threshold: cell-covered cutoff for :meth:`coverage_fraction`.
        backend: ``"numpy"`` / ``"jax"`` / ``"torch"``; ``samples`` should
            already live in the chosen backend.

    The ``visit_strength`` state is held by the caller (see :meth:`init_state`).
    All operators are pure / functional so they jit and vmap.
    """

    samples: ArrayLike
    sigma: float = 0.01
    threshold: float = 0.5
    backend: Backend = "numpy"

    def init_state(self) -> ArrayLike:
        """Return a fresh zeroed ``visit_strength`` array, shape ``(N,)``."""
        xp = _xp(self.backend)
        return xp.zeros(self.samples.shape[0], dtype=self.samples.dtype)

    def update(self, visit_strength: ArrayLike, contact_point: ArrayLike) -> ArrayLike:
        """Functional max-aggregation update.

        ``contact_point`` may be ``(D,)`` (single) or ``(..., D)`` for batched
        eval (broadcast against ``samples``). Returns the new visit array;
        caller is expected to thread it through their scan / loop carry.
        """
        xp = _xp(self.backend)
        # samples: (N, D);  contact: (..., D) → broadcast diff to (..., N, D).
        diff = self.samples - contact_point[..., None, :]
        d2 = (diff ** 2).sum(axis=-1)
        kernel = xp.exp(-d2 / (2.0 * self.sigma ** 2))
        return xp.maximum(visit_strength, kernel)

    def coverage_mean(self, visit_strength: ArrayLike) -> ArrayLike:
        """Mean visit strength ∈ ``[0, 1]`` — smooth, differentiable metric."""
        return visit_strength.mean(axis=-1)

    def coverage_fraction(self, visit_strength: ArrayLike) -> ArrayLike:
        """Fraction of cells with strength ``> threshold`` ∈ ``[0, 1]``."""
        above = (visit_strength > self.threshold)
        # cast to float in a backend-portable way
        if hasattr(above, "astype"):
            above = above.astype(visit_strength.dtype)
        else:
            above = above.to(visit_strength.dtype)
        return above.mean(axis=-1)


__all__ = ["CoverageMap", "Backend"]
