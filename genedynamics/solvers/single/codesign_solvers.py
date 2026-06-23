"""Register the specialized co-design optimizers (MRMFMBD / SHAC / DiffuseBot /
CMA-ES) into the co-design solver registry so the generic experiment runner drives
them by name — no `baselines/` dispatch layer.

These wrap the existing, validated optimizer bodies (e.g. MRMFMBDBackendMBD's full
G2 machinery — A2 decoder, control-variate, CVaR, regime marginalization) so behavior
is bit-identical; the physical relocation of those bodies out of
experiments/.../baselines/ (and deleting that dir) is the final mechanical step.

CEM is registered separately in codesign_runner via the GENERAL CEMSolver + the
(dynamics, energy) env bridge (it is a general algorithm, not co-design specific);
CMA-ES will move to the same general-solver+env pattern once a general CMAESSolver
exists under solvers/single/cmaes/.
"""
from __future__ import annotations

import numpy as np

from genedynamics.experiments.framework.codesign_runner import (
    register_codesign_solver, CoDesignResult,
)


def _wrap_optimizer(optimizer_cls):
    """Adapt an optimizer with run(config, evaluator, task_spec, *, x_dim, phi_dim)
    -> Result into a registered co-design solver run_fn."""
    def run_fn(evaluator, task_spec, config, *, x_dim, phi_dim) -> CoDesignResult:
        from genedynamics.experiments.framework.baseline import BaselineConfig
        opt = optimizer_cls()
        bl_cfg = BaselineConfig(
            task_id=getattr(task_spec, "task_id", ""),
            seed=int(config["seed"]),
            extra=dict(config["method_params"]),
            scheduler=config.get("scheduler"),
        )
        r = opt.run(bl_cfg, evaluator, task_spec, x_dim=x_dim, phi_dim=phi_dim)
        return CoDesignResult(
            theta=np.asarray(r.theta), x=np.asarray(r.x), phi=np.asarray(r.phi),
            return_=float(r.return_), success=bool(r.success),
            num_evaluations=int(getattr(r, "num_evaluations", 0)),
            wall_time=float(getattr(r, "wall_time", 0.0)),
            metadata=getattr(r, "metadata", {}) or {},
            full_dict=r.to_dict(),   # preserve bridge_history/theta_history/... verbatim
        )
    return run_fn


def _register() -> None:
    from genedynamics.solvers.single.mrmfmbd.codesign import MRMFMBDBaseline
    from genedynamics.solvers.single.mrmfmbd_ablation.codesign import MRMFMBDAblationBaseline
    from genedynamics.solvers.single.shac.codesign import SHACBaseline
    from genedynamics.solvers.single.diffusebot.codesign import DiffuseBotBaseline

    register_codesign_solver("mrmfmbd", _wrap_optimizer(MRMFMBDBaseline))
    # Frozen legacy engine for the paper's Q2/Q3 ablations (Stage 1 of the
    # theory-faithful refactor): fixed fidelity ladder / control-variate /
    # reward+cvar regime modes / post-hoc SHAC / mean certification. Selected by
    # `baseline_name: mrmfmbd_ablation` (+ optional method_params.ablation_mode).
    register_codesign_solver("mrmfmbd_ablation", _wrap_optimizer(MRMFMBDAblationBaseline))
    register_codesign_solver("shac", _wrap_optimizer(SHACBaseline))
    register_codesign_solver("diffusebot", _wrap_optimizer(DiffuseBotBaseline))
    # NOTE: CMA-ES is NOT here — it is a GENERAL solver (solvers/single/cmaes/)
    # registered in codesign_runner via _general_solver_run + the env bridge,
    # exactly like CEM.


_register()
