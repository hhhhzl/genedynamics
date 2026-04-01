"""
Overlay configuration resolver.

Parses nested YAML config into a flat ``OverlayConfig`` dataclass that
all overlay backends consume.  This module has no backend dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OverlayConfig:
    """Immutable overlay configuration.

    All parameters are plain floats/ints, suitable for embedding as
    JAX constants or NumPy scalars.
    """

    # Hardness
    rho_ref: float = 500.0

    # Constraint overlay
    kappa0: float = 1.0
    kappa_rho_gain: float = 0.01
    delta_margin_scale: float = 0.4
    delta_rho_gain: float = 0.02
    delta_min: float = 5e-4

    # Diffusion overlay
    sigma_max: float = 0.25
    sigma_hardness_power: float = 1.0
    theta_min: float = 0.2
    theta_max: float = 0.6
    eta_scale_min: float = 0.5
    eta_scale_max: float = 1.0
    eta_hardness_power: float = 1.0
    gate_every: int = 1

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dict (backward compat with schedule_contract)."""
        return {
            "rho_ref": self.rho_ref,
            "kappa0": self.kappa0,
            "kappa_rho_gain": self.kappa_rho_gain,
            "delta_margin_scale": self.delta_margin_scale,
            "delta_rho_gain": self.delta_rho_gain,
            "delta_min": self.delta_min,
            "sigma_max": self.sigma_max,
            "sigma_hardness_power": self.sigma_hardness_power,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "eta_scale_min": self.eta_scale_min,
            "eta_scale_max": self.eta_scale_max,
            "eta_hardness_power": self.eta_hardness_power,
            "gate_every": self.gate_every,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OverlayConfig":
        """Construct from flat dict (backward compat)."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def resolve_overlay_config(
    config: Optional[Dict[str, Any]] = None,
    *,
    rho_ref_default: float = 500.0,
) -> OverlayConfig:
    """Parse nested YAML overlay config into an ``OverlayConfig``.

    Expected YAML structure::

        twogo_overlay:
          hardness:
            rho_ref: 500.0
          constraint:
            kappa0: 1.0
            ...
          diffusion:
            sigma_max: 0.25
            ...

    Parameters
    ----------
    config : dict or None
        The ``twogo_overlay`` sub-dictionary from solver config.
    rho_ref_default : float
        Fallback rho_ref if not specified in config.

    Returns
    -------
    OverlayConfig
    """
    cfg = dict(config or {})
    h = dict(cfg.get("hardness", {}) or {})
    c = dict(cfg.get("constraint", {}) or {})
    d = dict(cfg.get("diffusion", {}) or {})

    rho_ref = float(h.get("rho_ref", rho_ref_default if rho_ref_default > 0.0 else 500.0))
    rho_ref = max(rho_ref, 1e-6)

    return OverlayConfig(
        rho_ref=rho_ref,
        kappa0=float(c.get("kappa0", 1.0)),
        kappa_rho_gain=float(c.get("kappa_rho_gain", 0.01)),
        delta_margin_scale=float(c.get("delta_margin_scale", 0.4)),
        delta_rho_gain=float(c.get("delta_rho_gain", 0.02)),
        delta_min=float(c.get("delta_min", 5e-4)),
        sigma_max=float(d.get("sigma_max", 0.25)),
        sigma_hardness_power=float(d.get("sigma_hardness_power", 1.0)),
        theta_min=float(d.get("theta_min", 0.2)),
        theta_max=float(d.get("theta_max", 0.6)),
        eta_scale_min=float(d.get("eta_scale_min", 0.5)),
        eta_scale_max=float(d.get("eta_scale_max", 1.0)),
        eta_hardness_power=float(d.get("eta_hardness_power", 1.0)),
        gate_every=int(max(1, int(d.get("gate_every", 1)))),
    )
