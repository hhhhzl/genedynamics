"""
ALM adaptive constraint scheduler (JAX).

Residual-driven adaptive scheduling for CFS-MBD:
- ALM dual/penalty updates (lambda, rho) from r_p
- QP effort (p_k, eps, I_QP, topK) from rp_prev, proj_prev
Designed for use inside JAX lax.scan; exposes jax_init_carry, jax_compute_params, jax_update.
"""

from __future__ import annotations

from typing import Any, Dict

from enerdynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from enerdynamics.core.constraints.core.types import ScheduleState

from .backends.alm_adaptive_jax import init_carry as _jax_init_carry
from .backends.alm_adaptive_jax import compute_params as _jax_compute_params
from .backends.alm_adaptive_jax import update as _jax_update


class ALMAdaptiveConstraintScheduler(ConstraintScheduler):
    """
    ALM-based adaptive constraint scheduler for CFS-MBD (JAX scan-internal).

    Uses residual r_p and projection displacement to adapt (lam, rho) and
    QP effort (p_k, eps, I_QP, topK). All update logic runs in JAX; no Python
    update() during diffusion. CFS-MBD calls jax_init_carry, jax_compute_params,
    jax_update from within the reverse_diffuse scan.
    """

    def __init__(
        self,
        lam0: float = 0.0,
        rho0: float = 1.0,
        gamma: float = 2.0,
        rho_max: float = 100.0,
        kappa: float = 0.9,
        r_tol: float = 5e-4,
        margin: float = 0.18,
        robot_radius: float = 0.05,
        p_min: float = 0.2,
        p_max: float = 1.0,
        eps_min: float = 1e-5,
        eps_max: float = 1e-2,
        I_min: int = 1,
        I_max: int = 20,
        topK_min: int = 1,
        topK_max: int = 8,
        r_scale: float = 0.1,
        proj_min: float = 1e-6,
        alpha_smooth: float = 0.2,
        use_stochastic_gate: bool = False,
        backend: str = "jax",
        **kwargs: Any,
    ):
        self.lam0 = float(lam0)
        self.rho0 = float(rho0)
        self.gamma = float(gamma)
        self.rho_max = float(rho_max)
        self.kappa = float(kappa)
        self.r_tol = float(r_tol)
        self.margin_base = float(margin)
        self.robot_radius = float(robot_radius)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.I_min = int(I_min)
        self.I_max = int(I_max)
        self.topK_min = int(topK_min)
        self.topK_max = int(topK_max)
        self.r_scale = float(r_scale)
        self.proj_min = float(proj_min)
        self.alpha_smooth = float(alpha_smooth)
        self.use_stochastic_gate = bool(use_stochastic_gate)
        self.backend = backend
        self.kwargs = kwargs

    def constraint_params(self, state: ScheduleState) -> Dict[str, Any]:
        """Fallback fixed params (e.g. when not used in JAX adaptive path)."""
        return {
            "margin": self.margin_base,
            "rho": self.rho0,
            "topK": self.topK_max,
            "eps": self.eps_max,
            "I_QP": self.I_max,
            "qp_gate": True,
            "qp_prob": 1.0,
            "aug_lambda": self.lam0,
            "aug_rho": self.rho0,
            "_extra": {},
        }

    def jax_init_carry(self, rng):
        """Return initial carry for JAX scan. Must be JAX arrays."""
        return _jax_init_carry(rng, self.lam0, self.rho0, self.p_max)

    def jax_compute_params(self, carry: Dict[str, Any], step_k, K: int):
        """Compute schedule params from carry. Returns (params_dict, rng_next)."""
        return _jax_compute_params(
            carry,
            step_k,
            K,
            margin_base=self.margin_base,
            gamma=self.gamma,
            rho_max=self.rho_max,
            kappa=self.kappa,
            p_min=self.p_min,
            p_max=self.p_max,
            eps_min=self.eps_min,
            eps_max=self.eps_max,
            I_min=self.I_min,
            I_max=self.I_max,
            topK_min=self.topK_min,
            topK_max=self.topK_max,
            r_scale=self.r_scale,
            proj_min=self.proj_min,
            alpha_smooth=self.alpha_smooth,
            use_stochastic_gate=self.use_stochastic_gate,
        )

    def jax_update(self, carry: Dict[str, Any], feedback: Dict[str, Any], rng):
        """Update carry from feedback. Returns new_carry."""
        return _jax_update(
            carry,
            feedback,
            rng,
            gamma=self.gamma,
            rho_max=self.rho_max,
            kappa=self.kappa,
            r_tol=self.r_tol,
            p_min=self.p_min,
            p_max=self.p_max,
            alpha_smooth=self.alpha_smooth,
            r_scale=self.r_scale,
            proj_min=self.proj_min,
        )
