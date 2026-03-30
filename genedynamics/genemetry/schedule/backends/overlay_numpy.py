"""
NumPy schedule overlay backend.

Implements hardness, constraint, and diffusion overlays using NumPy
for batch post-hoc computation (e.g. history reconstruction in
``_compute_schedule_series``).

Registered as ``("schedule", "overlay", "numpy")``.
"""

from typing import Any, Tuple

import numpy as np

from genedynamics.genemetry.base import ScheduleOverlayBase
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.schedule.config import OverlayConfig


@register_genemetry("schedule", "overlay", "numpy")
class ScheduleOverlayNumpy(ScheduleOverlayBase):
    """NumPy implementation of the schedule overlay chain.

    Operates on NumPy arrays; suitable for post-hoc batch computation.
    """

    def __init__(self, config: OverlayConfig) -> None:
        self._cfg = config
        self._log_denom = max(float(np.log1p(float(config.rho_ref))), 1e-6)

    def hardness(self, rho: Any) -> Any:
        rho_arr = np.maximum(np.asarray(rho, dtype=np.float32), 0.0)
        return np.clip(
            np.log1p(rho_arr) / self._log_denom, 0.0, 1.0
        ).astype(np.float32)

    def constraint_overlay(self, margin: Any, rho: Any) -> Tuple[Any, Any]:
        c = self._cfg
        rho_arr = np.maximum(np.asarray(rho, dtype=np.float32), 0.0)
        margin_arr = np.maximum(np.asarray(margin, dtype=np.float32), 0.0)

        kappa = c.kappa0 + c.kappa_rho_gain * rho_arr
        delta0 = c.delta_margin_scale * margin_arr
        delta = delta0 / (1.0 + c.delta_rho_gain * rho_arr)
        delta = np.maximum(delta, c.delta_min)

        return (
            np.asarray(kappa, dtype=np.float32),
            np.asarray(delta, dtype=np.float32),
        )

    def diffusion_overlay(
        self, hardness: Any, eta_base: float
    ) -> Tuple[Any, Any, Any]:
        c = self._cfg
        hard = np.clip(np.asarray(hardness, dtype=np.float32), 0.0, 1.0)
        one_minus_h = 1.0 - hard

        sigma = c.sigma_max * (one_minus_h ** c.sigma_hardness_power)
        theta = c.theta_max - (c.theta_max - c.theta_min) * hard
        eta_scale = c.eta_scale_min + (c.eta_scale_max - c.eta_scale_min) * (
            one_minus_h ** c.eta_hardness_power
        )
        eta = float(eta_base) * eta_scale

        return (
            np.asarray(sigma, dtype=np.float32),
            np.asarray(theta, dtype=np.float32),
            np.asarray(eta, dtype=np.float32),
        )
