"""
2GO diffusion scheduler.

Provides coupled diffusion temperature/noise scaling terms driven by
time progress and lambda(t) coupling.
"""

from __future__ import annotations

from typing import Any, Dict

from genedynamics.core.constraints.schedulers.DiffusionScheduler.base import DiffusionScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import get_registry


class TwoGODiffusionScheduler(DiffusionScheduler):
    def __init__(
        self,
        *,
        M_k: int = 64,
        T_k: float = 0.5,
        sigma_max: float = 0.25,
        theta_start: float = 0.6,
        theta_end: float = 0.2,
        q: float = 1.0,
        q_lambda: float = 1.0,
        lambda0: float = 1.0,
        lam_start: float = 0.0,
        lam_end: float = 500.0,
        lam_power: float = 1.0,
        gate_every: int = 1,
        backend: str = "jax",
        _skip_backend_lookup: bool = False,
        **kwargs: Any,
    ):
        self.M_k = int(M_k)
        self.T_k = float(T_k)
        self.sigma_max = float(sigma_max)
        self.theta_start = float(theta_start)
        self.theta_end = float(theta_end)
        self.q = float(q)
        self.q_lambda = float(q_lambda)
        self.lambda0 = float(lambda0)
        self.lam_start = float(lam_start)
        self.lam_end = float(lam_end)
        self.lam_power = float(lam_power)
        self.gate_every = int(max(1, gate_every))
        self.backend = str(backend)
        self.extra = kwargs
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "twogo_diffusion", backend)
            if impl_class is not None and impl_class != TwoGODiffusionScheduler:
                self._backend_impl = impl_class(
                    M_k=M_k,
                    T_k=T_k,
                    sigma_max=sigma_max,
                    theta_start=theta_start,
                    theta_end=theta_end,
                    q=q,
                    q_lambda=q_lambda,
                    lambda0=lambda0,
                    lam_start=lam_start,
                    lam_end=lam_end,
                    lam_power=lam_power,
                    gate_every=gate_every,
                    backend=backend,
                    _skip_backend_lookup=True,
                    **kwargs,
                )
            else:
                self._backend_impl = None
        else:
            self._backend_impl = None

    def _progress(self, state: ScheduleState) -> float:
        K = max(1, int(getattr(state, "K", 1)))
        k = int(getattr(state, "k", 0))
        return float(min(max(k / max(1, K - 1), 0.0), 1.0))

    def diffusion_params(self, state: ScheduleState) -> Dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.diffusion_params(state)
        t = self._progress(state)
        lam = self.lam_start + (self.lam_end - self.lam_start) * (t ** self.lam_power)
        lam_factor = self.lambda0 / (lam + self.lambda0 + 1e-8)
        s_sigma = ((1.0 - t) ** self.q) * (lam_factor ** self.q_lambda)
        theta = self.theta_start + (self.theta_end - self.theta_start) * t

        return {
            "M_k": self.M_k,
            "T_k": self.T_k,
            "sigma_scale": float(self.sigma_max * s_sigma),
            "theta_multi": float(theta),
            "gate_every": self.gate_every,
            "lambda_t": float(lam),
            "lambda_factor": float(lam_factor),
            **self.extra,
        }

