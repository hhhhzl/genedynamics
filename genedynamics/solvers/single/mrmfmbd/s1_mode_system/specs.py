"""
S1 Mode Specs: immutable configuration for contact/friction regimes.

Extensible design for mode registration and prior specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ModeSpec:
    """
    Specification for a single contact/friction mode.

    Attributes:
        mode_id: Integer mode index
        name: Human-readable name (e.g. "low_friction", "sticking")
        log_prior: log p(c) for mode prior
        friction: Friction coefficient (for SoftZoo terrain)
        damping: Optional damping
        terrain_variant: Optional terrain override
        extra: Extensible metadata
    """

    mode_id: int
    name: str
    log_prior: float = 0.0
    friction: float = 0.5
    damping: Optional[float] = None
    terrain_variant: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "name": self.name,
            "log_prior": self.log_prior,
            "friction": self.friction,
            "damping": self.damping,
            "terrain_variant": self.terrain_variant,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModeSpec":
        known = {"mode_id", "name", "log_prior", "friction", "damping", "terrain_variant"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs["extra"] = extra
        return cls(**kwargs)


@dataclass
class ModeSystemConfig:
    """
    Configuration for S1 mode marginalization system.

    Attributes:
        modes: List of ModeSpec
        reward_temperature: T for Boltzmann p(R|θ,c) ∝ exp(R_c/T)
        normalize_mode_priors: If True, normalize log_priors to sum to 0
    """

    modes: Tuple[ModeSpec, ...]
    reward_temperature: float = 0.1
    normalize_mode_priors: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    def get_log_priors(self) -> np.ndarray:
        """Return (C,) log p(c) array."""
        log_priors = np.array([m.log_prior for m in self.modes], dtype=np.float32)
        if self.normalize_mode_priors and len(log_priors) > 0:
            log_priors = log_priors - np.max(log_priors)
            log_priors = log_priors - np.log(np.sum(np.exp(log_priors)) + 1e-12)
        return log_priors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modes": [m.to_dict() for m in self.modes],
            "reward_temperature": self.reward_temperature,
            "normalize_mode_priors": self.normalize_mode_priors,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModeSystemConfig":
        modes_data = data.get("modes", [])
        modes = tuple(ModeSpec.from_dict(m) if isinstance(m, dict) else m for m in modes_data)
        return cls(
            modes=modes,
            reward_temperature=data.get("reward_temperature", 0.1),
            normalize_mode_priors=data.get("normalize_mode_priors", True),
        )


def default_mode_system_config(num_modes: int = 4) -> ModeSystemConfig:
    """Factory for default friction regimes."""
    modes = tuple(
        ModeSpec(mode_id=i, name=name, log_prior=0.0, friction=fric)
        for i, (name, fric) in enumerate(
            [
                ("low_friction", 0.2),
                ("medium_friction", 0.5),
                ("high_friction", 0.8),
                ("very_high_friction", 1.1),
            ][:num_modes]
        )
    )
    return ModeSystemConfig(modes=modes, reward_temperature=0.1)
