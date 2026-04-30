"""Pluggable noise samplers for model-based planners.

Defines the :class:`NoiseSampler` protocol and the default
:class:`IsotropicGaussian` implementation. Custom samplers
(e.g. :class:`structured_noise.StructuredNoise`) plug into any solver that
opts into the protocol — see ``solvers.single.<name>.backend_impl`` for how
a backend exposes a ``noise_sampler`` constructor argument.

The protocol is backend-agnostic: ``ArrayLike`` is whatever the caller
passes (numpy / jax / torch). Concrete samplers dispatch internally via the
same ``Backend`` Literal pattern used throughout the package.

Adapter pattern: a single uniform interface here means new samplers ship
with zero changes to solver kernels — solvers only need to call
``self.noise_sampler.sample(state, key, shape)`` once per noise draw.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Protocol, runtime_checkable

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


ArrayLike = Any
PRNGKey = Any
Backend = Literal["numpy", "jax", "torch"]


@runtime_checkable
class NoiseSampler(Protocol):
    """Protocol all noise samplers conform to.

    A sampler returns a noise array of the requested shape. The optional
    ``state`` argument lets state-dependent samplers (e.g. structured /
    geometry-aware noise) inspect the current planning state — for
    isotropic samplers it is unused.
    """

    backend: Backend

    def sample(
        self,
        shape: tuple,
        *,
        key: Optional[PRNGKey] = None,
        state: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Draw noise of shape ``shape`` (last axis = control dim)."""
        ...


class IsotropicGaussian:
    """Drop-in default: ``η ~ N(0, σ² I)`` in raw control space.

    Equivalent to the inline ``randn * sigma`` that solvers use today; this
    class exists so callers can replace it via the same constructor knob
    that custom samplers use, without solver-side branching.
    """

    backend: Backend

    def __init__(self, sigma: float = 1.0, *, backend: Backend = "numpy") -> None:
        if backend == "jax" and jnp is None:
            raise RuntimeError("backend='jax' requested but jax is not installed.")
        if backend == "torch" and torch is None:
            raise RuntimeError("backend='torch' requested but torch is not installed.")
        self.sigma = float(sigma)
        self.backend = backend

    def sample(
        self,
        shape: tuple,
        *,
        key: Optional[PRNGKey] = None,
        state: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        if self.backend == "jax":
            if key is None:
                raise ValueError("jax backend requires a PRNGKey via the `key` arg")
            return jnp.asarray(self.sigma, dtype=jnp.float32) * jax.random.normal(key, shape)
        if self.backend == "torch":
            generator = key  # callers may pass a torch.Generator as `key`
            return self.sigma * torch.randn(*shape, generator=generator)
        # numpy default
        rng = key if isinstance(key, np.random.Generator) else np.random.default_rng(key)
        return (self.sigma * rng.standard_normal(shape)).astype(np.float32)


__all__ = ["NoiseSampler", "IsotropicGaussian", "Backend", "ArrayLike", "PRNGKey"]
