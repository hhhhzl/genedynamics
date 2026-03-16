"""
Protocols for MBD3D (3DGS robust mapping) solver.

Defines abstract interfaces for scene representation, rendering, and
observation likelihood. Implementations can be swapped for different
3DGS backends (differentiable rasterization, neural renderers, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from .types import ObservationBundle, SceneParams


# -----------------------------------------------------------------------------
# Scene representation
# -----------------------------------------------------------------------------


@runtime_checkable
class SceneRepresentation(Protocol):
    """
    Protocol for 3D scene parameterization and prior.

    Implementations: GaussianSplatScene, NeuralScene, etc.
    """

    def flatten(self, params: SceneParams) -> Any:
        """Flatten scene params to 1D optimization variable."""
        ...

    def unflatten(self, flat: Any) -> SceneParams:
        """Reconstruct SceneParams from flattened vector."""
        ...

    def prior_log_prob(self, params: SceneParams) -> float:
        """Log p0(scene). E.g., Gaussian prior on init."""
        ...

    def sample_prior(self, rng: Any, n_gaussians: int, **kwargs: Any) -> SceneParams:
        """Sample initial scene from prior."""
        ...

    def dim(self) -> int:
        """Dimensionality of flattened scene params."""
        ...


# -----------------------------------------------------------------------------
# Observation rendering
# -----------------------------------------------------------------------------


@runtime_checkable
class ObservationRenderer(Protocol):
    """
    Protocol for rendering scene to images given camera poses.

    Implementations: DiffSplatRenderer, MockRenderer (for testing), etc.
    """

    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Render scene at given camera poses.

        Args:
            params: scene parameters
            camera_poses: (N, 7) or list of CameraPose
            intrinsics: optional (3,3) or (N,3,3)

        Returns:
            Rendered images (N, H, W, C)
        """
        ...

    def supports_jax(self) -> bool:
        """Whether renderer is JAX-differentiable."""
        ...


# -----------------------------------------------------------------------------
# Observation likelihood
# -----------------------------------------------------------------------------


@runtime_checkable
class ObservationLikelihood(Protocol):
    """
    Protocol for computing log p(y | θ) with optional low-rank noise.

    Uses core.prob.gaussian_lowrank for correlated observation noise
    (pose drift, exposure variation).
    """

    def log_likelihood(
        self,
        params: SceneParams,
        observations: ObservationBundle,
        camera_trajectory: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Compute log p(observations | params, camera_trajectory).

        Args:
            params: scene parameters
            observations: image bundle
            camera_trajectory: camera poses for each view

        Returns:
            log p(y|θ) (scalar or per-view)
        """
        ...

    def residual(
        self,
        predicted: Any,
        observed: Any,
        **kwargs: Any,
    ) -> Any:
        """Compute residual r = observed - predicted for quad_form."""
        ...


# -----------------------------------------------------------------------------
# Observation noise model (low-rank covariance)
# -----------------------------------------------------------------------------


@runtime_checkable
class ObservationNoiseModel(Protocol):
    """
    Protocol for observation noise covariance Σ = AA' + σ²I.

    Used for correlated noise (pose drift, exposure) in 3DGS robust mapping.
    """

    def lowrank_basis(self, observations: ObservationBundle, **kwargs: Any) -> Any:
        """Return (p, r) low-rank basis A for noise covariance."""
        ...

    def sigma2(self) -> float:
        """IID noise variance σ²."""
        ...

    def observation_dim(self, observations: ObservationBundle) -> int:
        """Dimension p of flattened observation."""
        ...


# -----------------------------------------------------------------------------
# Camera trajectory prior
# -----------------------------------------------------------------------------


@runtime_checkable
class CameraTrajectoryPrior(Protocol):
    """
    Protocol for prior on camera trajectory p0(camera_poses).
    """

    def log_prob(self, camera_poses: Any) -> float:
        """Log p0(camera_poses)."""
        ...

    def sample(self, rng: Any, num_poses: int, initial_pose: Any, **kwargs: Any) -> Any:
        """Sample trajectory from prior."""
        ...


# -----------------------------------------------------------------------------
# Abstract base implementations (for extension)
# -----------------------------------------------------------------------------


class BaseSceneRepresentation(ABC):
    """Abstract base for SceneRepresentation implementations."""

    @abstractmethod
    def flatten(self, params: SceneParams) -> Any:
        pass

    @abstractmethod
    def unflatten(self, flat: Any) -> SceneParams:
        pass

    @abstractmethod
    def prior_log_prob(self, params: SceneParams) -> float:
        pass

    @abstractmethod
    def sample_prior(self, rng: Any, n_gaussians: int, **kwargs: Any) -> SceneParams:
        pass

    @abstractmethod
    def dim(self) -> int:
        pass


class BaseObservationRenderer(ABC):
    """Abstract base for ObservationRenderer implementations."""

    @abstractmethod
    def render(
        self,
        params: SceneParams,
        camera_poses: Any,
        intrinsics: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    def supports_jax(self) -> bool:
        return False


class BaseObservationLikelihood(ABC):
    """Abstract base for ObservationLikelihood implementations."""

    @abstractmethod
    def log_likelihood(
        self,
        params: SceneParams,
        observations: ObservationBundle,
        camera_trajectory: Any,
        **kwargs: Any,
    ) -> Any:
        pass

    def residual(self, predicted: Any, observed: Any, **kwargs: Any) -> Any:
        """Default: element-wise difference."""
        if JAX_AVAILABLE and (hasattr(predicted, "block_until_ready") or hasattr(observed, "block_until_ready")):
            return jnp.ravel(jnp.asarray(observed) - jnp.asarray(predicted))
        return np.ravel(np.asarray(observed) - np.asarray(predicted))
