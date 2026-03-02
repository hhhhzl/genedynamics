"""
ALM adaptive constraint scheduler (JAX).

Controller-style adaptive scheduling for CFS-MBD:
- ALM dual/penalty updates (lambda, rho) from r_p
- Non-monotone control of QP effort (p_k, eps, I_QP, topK) from violation statistics v_k
  plus compute dual ν_k under an ergodic compute budget.
Designed for use inside JAX lax.scan; exposes jax_init_carry, jax_compute_params, jax_update.
"""

from __future__ import annotations

from typing import Any, Dict

from genedynamics.core.constraints.schedulers.ConstraintScheduler.base import ConstraintScheduler
from genedynamics.core.constraints.core.types import ScheduleState

from .backends.alm_adaptive_jax import init_carry as _jax_init_carry
from .backends.alm_adaptive_jax import compute_params as _jax_compute_params
from .backends.alm_adaptive_jax import update as _jax_update


class ALMAdaptiveConstraintScheduler(ConstraintScheduler):
    """
    ALM-based adaptive constraint scheduler for CFS-MBD (JAX scan-internal).

    Uses risk residual r_k (=quantile/CVaR over per-trajectory max violation) to adapt (lam, rho),
    and uses violation statistics v_k (rate or mean) + compute dual ν_k to control
    QP effort (p_k, eps, I_QP, topK) in a non-monotone closed loop. All update logic runs
    in JAX; no Python
    update() during diffusion. CFS-MBD calls jax_init_carry, jax_compute_params,
    jax_update from within the reverse_diffuse scan.
    """

    def __init__(
        self,
        lam0: float = 0.0,
        rho0: float = 1.0,
        # Compute dual (ergodic compute budget constraint)
        nu0: float = 0.0,
        compute_cost_a0: float = 0.0,
        compute_cost_aK: float = 1.0,
        compute_cost_aI: float = 1.0,
        compute_cost_mode: str = "product",  # "product" or "linear"
        compute_cost_use_qp_gate: bool = False,
        compute_budget_B: float = 0.0,
        # Scheme A: time-varying compute budget profile B_k (ergodic mean = B)
        compute_budget_time_profile: str = "constant",  # "constant" | "gaussian_bump"
        compute_budget_time_amp: float = 0.0,  # bump amplitude a >= 0
        compute_budget_time_mu: float = 0.5,  # peak location in [0,1]
        compute_budget_time_sigma: float = 0.2,  # bump width
        eta_nu: float = 0.0,
        nu_max: float = 0.0,
        # Non-monotone ALM (λ forgetting + ρ hysteresis)
        eta_lam_forget: float = 0.0,
        rho_min: float = 0.0,
        rho_tau_hi: float = 1e-3,
        rho_tau_lo: float = 1e-4,
        rho_gamma_up: float = 2.0,
        rho_gamma_down: float = 2.0,
        rho_good_steps: int = 5,
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
        # Non-monotone controller config (v_k-based)
        v_signal_mode: str = "rate",  # "rate" or "mean"
        v_star: float = 0.0,
        v_ema_beta: float = 0.3,
        v_tau_hi: float = 0.05,
        v_tau_lo: float = 0.01,
        eta_p: float = 1.0,
        eta_p_plus: float = 1.0,
        eta_p_minus: float = 0.2,
        eta_c_p: float = 0.1,
        eta_topK: float = 5.0,
        eta_topK_plus: float = 5.0,
        eta_topK_minus: float = 1.0,
        eta_c_topK: float = 0.5,
        eta_I: float = 5.0,
        eta_I_plus: float = 5.0,
        eta_I_minus: float = 1.0,
        eta_c_I: float = 0.5,
        eta_eps: float = 1.0,
        eta_eps_plus: float = 1.0,
        eta_eps_minus: float = 0.2,
        eta_c_eps: float = 0.1,
        use_stochastic_gate: bool = False,
        backend: str = "jax",
        **kwargs: Any,
    ):
        self.lam0 = float(lam0)
        self.rho0 = float(rho0)
        self.nu0 = float(nu0)
        self.compute_cost_a0 = float(compute_cost_a0)
        self.compute_cost_aK = float(compute_cost_aK)
        self.compute_cost_aI = float(compute_cost_aI)
        self.compute_cost_mode = str(compute_cost_mode)
        self.compute_cost_use_qp_gate = bool(compute_cost_use_qp_gate)
        self.compute_budget_B = float(compute_budget_B)
        self.compute_budget_time_profile = str(compute_budget_time_profile)
        self.compute_budget_time_amp = float(compute_budget_time_amp)
        self.compute_budget_time_mu = float(compute_budget_time_mu)
        self.compute_budget_time_sigma = float(compute_budget_time_sigma)
        self.eta_nu = float(eta_nu)
        self.nu_max = float(nu_max)
        self.eta_lam_forget = float(eta_lam_forget)
        self.rho_min = float(rho_min)
        self.rho_tau_hi = float(rho_tau_hi)
        self.rho_tau_lo = float(rho_tau_lo)
        self.rho_gamma_up = float(rho_gamma_up)
        self.rho_gamma_down = float(rho_gamma_down)
        self.rho_good_steps = int(rho_good_steps)
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
        self.v_signal_mode = str(v_signal_mode)
        self.v_star = float(v_star)
        self.v_ema_beta = float(v_ema_beta)
        self.v_tau_hi = float(v_tau_hi)
        self.v_tau_lo = float(v_tau_lo)
        self.eta_p = float(eta_p)
        self.eta_p_plus = float(eta_p_plus)
        self.eta_p_minus = float(eta_p_minus)
        self.eta_c_p = float(eta_c_p)
        self.eta_topK = float(eta_topK)
        self.eta_topK_plus = float(eta_topK_plus)
        self.eta_topK_minus = float(eta_topK_minus)
        self.eta_c_topK = float(eta_c_topK)
        self.eta_I = float(eta_I)
        self.eta_I_plus = float(eta_I_plus)
        self.eta_I_minus = float(eta_I_minus)
        self.eta_c_I = float(eta_c_I)
        self.eta_eps = float(eta_eps)
        self.eta_eps_plus = float(eta_eps_plus)
        self.eta_eps_minus = float(eta_eps_minus)
        self.eta_c_eps = float(eta_c_eps)
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
        # Initialize controller states to "max effort" defaults.
        return _jax_init_carry(
            rng,
            self.lam0,
            self.rho0,
            int(self.p_max + self.p_min) / 2,
            self.nu0,
            int(self.topK_max + self.topK_min) / 2,
            int(self.I_max + self.I_min) / 2,
            int(self.eps_max + self.eps_min) / 2,
            0.0,
            0.0,
            0,
        )

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
            use_stochastic_gate=self.use_stochastic_gate,
            v_signal_mode=self.v_signal_mode,
            v_star=self.v_star,
            v_ema_beta=self.v_ema_beta,
            v_tau_hi=self.v_tau_hi,
            v_tau_lo=self.v_tau_lo,
            eta_p=self.eta_p,
            eta_p_plus=self.eta_p_plus,
            eta_p_minus=self.eta_p_minus,
            eta_c_p=self.eta_c_p,
            eta_topK=self.eta_topK,
            eta_topK_plus=self.eta_topK_plus,
            eta_topK_minus=self.eta_topK_minus,
            eta_c_topK=self.eta_c_topK,
            eta_I=self.eta_I,
            eta_I_plus=self.eta_I_plus,
            eta_I_minus=self.eta_I_minus,
            eta_c_I=self.eta_c_I,
            eta_eps=self.eta_eps,
            eta_eps_plus=self.eta_eps_plus,
            eta_eps_minus=self.eta_eps_minus,
            eta_c_eps=self.eta_c_eps,
            compute_budget_B=self.compute_budget_B,
            compute_budget_time_profile=self.compute_budget_time_profile,
            compute_budget_time_amp=self.compute_budget_time_amp,
            compute_budget_time_mu=self.compute_budget_time_mu,
            compute_budget_time_sigma=self.compute_budget_time_sigma,
            compute_cost_mode=self.compute_cost_mode,
            compute_cost_a0=self.compute_cost_a0,
            compute_cost_aK=self.compute_cost_aK,
            compute_cost_aI=self.compute_cost_aI,
            compute_cost_use_qp_gate=self.compute_cost_use_qp_gate,
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
            rho_min=self.rho_min,
            eta_lam_forget=self.eta_lam_forget,
            rho_tau_hi=self.rho_tau_hi,
            rho_tau_lo=self.rho_tau_lo,
            rho_gamma_up=self.rho_gamma_up,
            rho_gamma_down=self.rho_gamma_down,
            rho_good_steps=self.rho_good_steps,
            p_min=self.p_min,
            p_max=self.p_max,
            eps_min=self.eps_min,
            eps_max=self.eps_max,
            I_min=self.I_min,
            I_max=self.I_max,
            topK_min=self.topK_min,
            topK_max=self.topK_max,
            v_signal_mode=self.v_signal_mode,
            v_star=self.v_star,
            v_ema_beta=self.v_ema_beta,
            v_tau_hi=self.v_tau_hi,
            v_tau_lo=self.v_tau_lo,
            eta_p=self.eta_p,
            eta_p_plus=self.eta_p_plus,
            eta_p_minus=self.eta_p_minus,
            eta_c_p=self.eta_c_p,
            eta_topK=self.eta_topK,
            eta_topK_plus=self.eta_topK_plus,
            eta_topK_minus=self.eta_topK_minus,
            eta_c_topK=self.eta_c_topK,
            eta_I=self.eta_I,
            eta_I_plus=self.eta_I_plus,
            eta_I_minus=self.eta_I_minus,
            eta_c_I=self.eta_c_I,
            eta_eps=self.eta_eps,
            eta_eps_plus=self.eta_eps_plus,
            eta_eps_minus=self.eta_eps_minus,
            eta_c_eps=self.eta_c_eps,
            eta_nu=self.eta_nu,
            compute_budget_B=self.compute_budget_B,
            nu_max=self.nu_max,
        )
