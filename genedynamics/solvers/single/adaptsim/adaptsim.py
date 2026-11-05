"""AdaptSim solver frontend and source-aligned parameter distribution update.

The task owns rollout collection, policy training in sampled simulations,
reward evaluation, and observation features. The solver owns the learned
branching-Q update and the parameter-distribution state.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import torch

from .backends.branching_q_torch import BranchingQBackendTorch


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    minimum: float
    maximum: float
    step: float
    distribution: str = "uniform"
    min_std: float = 0.0
    max_std: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not np.isfinite([self.minimum, self.maximum, self.step]).all():
            raise ValueError("invalid parameter specification")
        if self.minimum >= self.maximum or self.step <= 0 or self.distribution not in {"uniform", "gaussian"}:
            raise ValueError("invalid parameter range, step, or distribution")
        if self.distribution == "gaussian" and (self.min_std < 0 or self.min_std > self.max_std):
            raise ValueError("invalid Gaussian standard deviation bounds")


@dataclass(frozen=True)
class ParameterDistribution:
    left: np.ndarray
    right: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        left = np.asarray(self.left, dtype=np.float64).copy()
        right = np.asarray(self.right, dtype=np.float64).copy()
        if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("distribution vectors must be matching and finite")
        if self.kind == "uniform" and np.any(right < left):
            raise ValueError("uniform upper bounds must exceed lower bounds")
        if self.kind == "gaussian" and np.any(right < 0):
            raise ValueError("Gaussian standard deviations must be nonnegative")
        if self.kind not in {"uniform", "gaussian"}:
            raise ValueError("unsupported distribution")
        left.setflags(write=False)
        right.setflags(write=False)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("count must be positive")
        shape = (count, len(self.left))
        if self.kind == "uniform":
            return rng.uniform(self.left, self.right, size=shape)
        return rng.normal(self.left, self.right, size=shape)


class AdaptSimSolver:
    """Parameter-distribution adaptation with the authors' discrete action map."""

    def __init__(
        self,
        specs: Sequence[ParameterSpec],
        *,
        feature_dim: int,
        hidden: Sequence[int] = (128, 128),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        single_step: bool = False,
        device: str = "cpu",
    ) -> None:
        self.specs = tuple(specs)
        if not self.specs or len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError("specs must be nonempty with unique names")
        if len({s.distribution for s in self.specs}) != 1 or feature_dim < 0:
            raise ValueError("AdaptSim requires homogeneous distributions and nonnegative feature_dim")
        self.kind = self.specs[0].distribution
        self.feature_dim = feature_dim
        state_dim = (2 if self.kind == "uniform" else 1) * len(self.specs) + feature_dim
        bins = (5 if self.kind == "uniform" else 3,) * len(self.specs)
        self._backend_impl = BranchingQBackendTorch(
            state_dim, bins, hidden=hidden, learning_rate=learning_rate,
            gamma=gamma, tau=tau, single_step=single_step, device=device,
        )

    @property
    def backend(self) -> BranchingQBackendTorch:
        return self._backend_impl

    def state(self, distribution: ParameterDistribution, features: np.ndarray) -> np.ndarray:
        self._check_distribution(distribution)
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.feature_dim,):
            raise ValueError("feature vector size mismatch")
        lower = np.array([spec.minimum for spec in self.specs])
        span = np.array([spec.maximum - spec.minimum for spec in self.specs])
        if self.kind == "uniform":
            parameters = np.stack(((distribution.left - lower) / span, (distribution.right - lower) / span), axis=1).ravel()
        else:
            parameters = (distribution.left - lower) / span
        return np.concatenate((parameters, features)).astype(np.float32)

    def propose_update(self, distribution: ParameterDistribution, features: np.ndarray) -> tuple[np.ndarray, ParameterDistribution]:
        state = self.state(distribution, features)
        action = self.backend.act(torch.from_numpy(state[None])).cpu().numpy()[0]
        return action, self.apply_action(distribution, action)

    def apply_action(self, distribution: ParameterDistribution, action: Sequence[int]) -> ParameterDistribution:
        self._check_distribution(distribution)
        action = np.asarray(action)
        if action.shape != (len(self.specs),):
            raise ValueError("one discrete action is required per parameter")
        left = distribution.left.copy()
        right = distribution.right.copy()
        for index, (spec, choice) in enumerate(zip(self.specs, action)):
            if int(choice) != choice:
                raise ValueError("actions must be integers")
            span = spec.maximum - spec.minimum
            if self.kind == "uniform":
                steps = ((-spec.step, -spec.step), (spec.step, spec.step),
                         (-spec.step / 2, spec.step / 2), (spec.step / 2, -spec.step / 2), (0.0, 0.0))
                if not 0 <= int(choice) < len(steps):
                    raise ValueError("uniform action outside 0..4")
                delta_left, delta_right = steps[int(choice)]
                left[index] = np.clip(left[index] + span * delta_left, spec.minimum, spec.maximum)
                right[index] = max(left[index], np.clip(right[index] + span * delta_right, spec.minimum, spec.maximum))
            else:
                steps = (-spec.step, spec.step, 0.0)
                if not 0 <= int(choice) < len(steps):
                    raise ValueError("Gaussian action outside 0..2")
                left[index] = np.clip(left[index] + span * steps[int(choice)], spec.minimum, spec.maximum)
                right[index] = np.clip(right[index], spec.min_std, spec.max_std)
        return ParameterDistribution(left, right, self.kind)

    def train_batch(self, *args: torch.Tensor) -> float:
        return self.backend.train_batch(*args)

    def clone_particle(
        self, distribution: ParameterDistribution, rng: np.random.Generator,
        *, perturbation: float,
    ) -> ParameterDistribution:
        """Perturb a particle mean by the author's signed half-to-full range rule.

        Gaussian standard deviations and uniform interval widths are retained.
        The bounded rejection loop avoids hanging when no admissible sample is
        drawn; the upstream implementation uses an unbounded loop.
        """
        self._check_distribution(distribution)
        if not np.isfinite(perturbation) or perturbation < 0:
            raise ValueError("particle perturbation must be finite and nonnegative")
        if perturbation == 0:
            return ParameterDistribution(distribution.left, distribution.right, distribution.kind)
        lower = np.array([spec.minimum for spec in self.specs])
        upper = np.array([spec.maximum for spec in self.specs])
        span = upper - lower
        center = (distribution.left + distribution.right) / 2 if self.kind == "uniform" else distribution.left
        half_width = (distribution.right - distribution.left) / 2 if self.kind == "uniform" else np.zeros_like(center)
        for _ in range(1000):
            signed = rng.choice([-1, 1], size=len(self.specs)) * (rng.random(len(self.specs)) * 0.5 + 0.5)
            proposed = center + perturbation * signed * span
            if np.all(proposed - half_width > lower) and np.all(proposed + half_width < upper):
                if self.kind == "uniform":
                    return ParameterDistribution(proposed - half_width, proposed + half_width, self.kind)
                return ParameterDistribution(proposed, distribution.right, self.kind)
        raise ValueError("particle perturbation has no admissible draw after 1000 attempts")

    def _check_distribution(self, distribution: ParameterDistribution) -> None:
        if distribution.kind != self.kind or len(distribution.left) != len(self.specs):
            raise ValueError("distribution does not match solver parameter specification")


__all__ = ["AdaptSimSolver", "ParameterDistribution", "ParameterSpec"]
