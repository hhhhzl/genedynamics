"""
2GO constraint scheduler.

Time-coupled feasibility schedule for flow/diffusion gating:
- lambda(t) growth
- kappa(t) growth
- delta(t) decay
- constraint-side QP/topK/eps/I_QP adaptation
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from genedynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from genedynamics.core.constraints.core.types import ScheduleState
from genedynamics.core.constraints.core.registry import get_registry


class TwoGOConstraintScheduler(ConstraintScheduler):
    def __init__(
        self,
        *,
        margin: float = 0.05,
        lam_start: float = 0.0,
        lam_end: float = 500.0,
        lam_power: float = 1.0,
        kappa0: float = 1.0,
        kappa_gain: float = 0.01,
        delta0: float = 0.02,
        delta_power: float = 1.0,
        delta_lambda_power: float = 1.0,
        lambda0: float = 1.0,
        cvar_alpha: float = 0.9,
        qp_prob_min: float = 0.2,
        qp_prob_max: float = 1.0,
        topK_min: int = 2,
        topK_max: int = 8,
        I_QP_min: int = 1,
        I_QP_max: int = 2,
        eps_min: float = 1e-5,
        eps_max: float = 1e-3,
        backend: str = "jax",
        _skip_backend_lookup: bool = False,
        **kwargs: Any,
    ):
        self.margin = float(margin)
        self.lam_start = float(lam_start)
        self.lam_end = float(lam_end)
        self.lam_power = float(lam_power)
        self.kappa0 = float(kappa0)
        self.kappa_gain = float(kappa_gain)
        self.delta0 = float(delta0)
        self.delta_power = float(delta_power)
        self.delta_lambda_power = float(delta_lambda_power)
        self.lambda0 = float(lambda0)
        self.cvar_alpha = float(cvar_alpha)
        self.qp_prob_min = float(qp_prob_min)
        self.qp_prob_max = float(qp_prob_max)
        self.topK_min = int(topK_min)
        self.topK_max = int(topK_max)
        self.I_QP_min = int(I_QP_min)
        self.I_QP_max = int(I_QP_max)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.backend = str(backend)
        self.extra = kwargs
        if not _skip_backend_lookup:
            registry = get_registry()
            impl_class = registry.get("scheduler", "twogo_constraint", backend)
            if impl_class is not None and impl_class != TwoGOConstraintScheduler:
                self._backend_impl = impl_class(
                    margin=margin,
                    lam_start=lam_start,
                    lam_end=lam_end,
                    lam_power=lam_power,
                    kappa0=kappa0,
                    kappa_gain=kappa_gain,
                    delta0=delta0,
                    delta_power=delta_power,
                    delta_lambda_power=delta_lambda_power,
                    lambda0=lambda0,
                    cvar_alpha=cvar_alpha,
                    qp_prob_min=qp_prob_min,
                    qp_prob_max=qp_prob_max,
                    topK_min=topK_min,
                    topK_max=topK_max,
                    I_QP_min=I_QP_min,
                    I_QP_max=I_QP_max,
                    eps_min=eps_min,
                    eps_max=eps_max,
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

    def constraint_params(self, state: ScheduleState, record: bool = True) -> Dict[str, Any]:
        if self._backend_impl is not None:
            return self._backend_impl.constraint_params(state, record=record)
        _ = record
        t = self._progress(state)
        lam = self.lam_start + (self.lam_end - self.lam_start) * (t ** self.lam_power)
        lam_factor = self.lambda0 / (lam + self.lambda0 + 1e-8)

        kappa = self.kappa0 + self.kappa_gain * lam
        delta = self.delta0 * ((1.0 - t) ** self.delta_power) * (lam_factor ** self.delta_lambda_power)

        # As schedule hardens, increase qp probability and active set size.
        qp_prob = self.qp_prob_min + (self.qp_prob_max - self.qp_prob_min) * t
        topK = int(round(self.topK_min + (self.topK_max - self.topK_min) * t))
        I_QP = int(round(self.I_QP_min + (self.I_QP_max - self.I_QP_min) * t))
        eps = self.eps_max + (self.eps_min - self.eps_max) * t

        return {
            "rho": max(1e-6, lam),
            "topK": topK,
            "eps": float(eps),
            "I_QP": I_QP,
            "qp_gate": True,
            "qp_prob": float(qp_prob),
            "margin": self.margin,
            "_extra": {
                "aug_lambda": float(lam),
                "kappa": float(kappa),
                "delta": float(delta),
                "lambda_t": float(lam),
                "lambda_factor": float(lam_factor),
                "cvar_alpha": float(self.cvar_alpha),
                **self.extra,
            },
        }

