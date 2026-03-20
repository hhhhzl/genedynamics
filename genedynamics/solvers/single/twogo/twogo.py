"""
2GO solver (Phase-1 minimal runnable).

2GO currently reuses the high-performance CFS-MBD JAX kernel and adds
geometry/gating diagnostics. This keeps the execution path stable while
providing a dedicated method entry-point for iterative upgrades.
"""

from __future__ import annotations

from typing import Any, Optional

from genedynamics.core.backends import Backend
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.constraints.action_filters import ConstraintFilter
from genedynamics.core.dynamics import DynamicsModel
from genedynamics.core.energy import EnergyFunctional
from genedynamics.solvers.single.cfsmbd.cfsmbd import CFSMBDSolver
from genedynamics.solvers.single.twogo.backend_impl import to_unified_backend

try:
    from genedynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


def _get_twogo_backend(backend_name: str):
    if backend_name == "jax":
        from genedynamics.solvers.single.twogo.backends.twogo_jax import TwoGOBackendJax

        return TwoGOBackendJax
    return None


class TwoGOSolver(CFSMBDSolver):
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        constraint_filter: Optional[ConstraintFilter] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            constraint_filter=constraint_filter,
            **kwargs,
        )
        # 2GO-specific scheduling/gating knobs (Phase-1 proxy diagnostics).
        self.config.update(
            dict(
                twogo_sigma_max=float(kwargs.get("twogo_sigma_max", 0.25)),
                twogo_gate_vrate_threshold=float(kwargs.get("twogo_gate_vrate_threshold", 0.01)),
                twogo_sigma_q=float(kwargs.get("twogo_sigma_q", 1.0)),
                twogo_sigma_q_lambda=float(kwargs.get("twogo_sigma_q_lambda", 1.0)),
                twogo_delta0=float(kwargs.get("twogo_delta0", 0.02)),
                twogo_delta_r=float(kwargs.get("twogo_delta_r", 1.0)),
                twogo_delta_r_lambda=float(kwargs.get("twogo_delta_r_lambda", 1.0)),
                twogo_lambda0=float(kwargs.get("twogo_lambda0", 1.0)),
                twogo_theta_start=float(kwargs.get("twogo_theta_start", 0.6)),
                twogo_theta_end=float(kwargs.get("twogo_theta_end", 0.2)),
                twogo_cvar_alpha=float(kwargs.get("twogo_cvar_alpha", 0.9)),
                twogo_window_size=int(kwargs.get("twogo_window_size", 16)),
                twogo_window_stride=int(kwargs.get("twogo_window_stride", 8)),
                twogo_tail_ratio=float(kwargs.get("twogo_tail_ratio", 0.3)),
                twogo_active_topk=int(kwargs.get("twogo_active_topk", 8)),
                twogo_agp_eta=float(kwargs.get("twogo_agp_eta", 0.08)),
                twogo_cfs_gain=float(kwargs.get("twogo_cfs_gain", 0.35)),
                twogo_multi_scale=float(kwargs.get("twogo_multi_scale", 0.25)),
                twogo_enable_agp_refine=bool(kwargs.get("twogo_enable_agp_refine", False)),
                twogo_enable_local_gating=bool(kwargs.get("twogo_enable_local_gating", True)),
                twogo_enable_sample_tail=bool(kwargs.get("twogo_enable_sample_tail", True)),
                twogo_enable_agp_batch=bool(kwargs.get("twogo_enable_agp_batch", False)),
                twogo_use_jax_scan_core=bool(kwargs.get("twogo_use_jax_scan_core", True)),
                twogo_retract_qp_boost=bool(kwargs.get("twogo_retract_qp_boost", True)),
                twogo_stability_eps=float(kwargs.get("twogo_stability_eps", 1e-6)),
                twogo_gamma_init=float(kwargs.get("twogo_gamma_init", 1.0)),
                twogo_retract_qp_every=int(kwargs.get("twogo_retract_qp_every", 2)),
                twogo_probe_enable=bool(kwargs.get("twogo_probe_enable", True)),
                twogo_probe_tail_mix=float(kwargs.get("twogo_probe_tail_mix", 0.5)),
                twogo_probe_tail_pool_ratio=float(kwargs.get("twogo_probe_tail_pool_ratio", 0.25)),
                twogo_probe_geom_alpha=float(kwargs.get("twogo_probe_geom_alpha", 0.25)),
                twogo_probe_frac=float(kwargs.get("twogo_probe_frac", 0.5)),
                twogo_probe_b=kwargs.get("twogo_probe_b", None),
                twogo_probe_m_cap=kwargs.get("twogo_probe_m_cap", None),
            )
        )

    def _get_backend_impl(self):
        if self._backend_impl is None:
            backend = RuntimeBackendManager.get_backend()
            backend_cls = _get_twogo_backend(backend.name)
            if backend_cls is None:
                raise ValueError(f"2GO backend '{backend.name}' not found")
            self._backend_impl = backend_cls(
                solver=self,
                env_adapter=self._env_adapter,
                legacy_energy=self._legacy_energy,
                horizon=self.horizon,
                dt=self.dt,
                Nsample=self.config["Nsample"],
                Ndiffuse=self.config["Ndiffuse"],
                temp_sample=self.config["temp_sample"],
                beta0=self.config["beta0"],
                betaT=self.config["betaT"],
                action_limit=self.config["action_limit"],
                seed=self.seed,
                scheduler=self.config.get("scheduler"),
                show_tqdm=self.config.get("show_tqdm", False),
                constraint_filter=self.constraint_filter,
                obstacles=self.obstacles,
                aug_lambda=self.config.get("aug_lambda", 0.0),
                aug_rho=self.config.get("aug_rho", 1.0),
                action_extra_sigma=self.config.get("action_extra_sigma", 0.0),
            )
            self._backend_impl = to_unified_backend(self._backend_impl)
        return self._backend_impl


if register_solver is not None:
    try:
        register_solver("2go", TwoGOSolver)
    except Exception:
        pass

