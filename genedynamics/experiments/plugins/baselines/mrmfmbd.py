"""
MRMFMBD as a Baseline: wraps posterior bridge for experiment platform.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...framework.baseline import BaselineConfig, BaselineResult, BaselineProtocol


class MRMFMBDBaseline(BaselineProtocol):
    """
    MRMFMBD posterior bridge as a pluggable baseline.

    Implements BaselineProtocol for the baseline experiment platform.
    Works with any task domain that provides co-design (x, phi) structure.
    """

    @property
    def name(self) -> str:
        return "mrmfmbd"

    def run(
        self,
        config: BaselineConfig,
        evaluator: Any,
        task_spec: Any,
        *,
        x_dim: int,
        phi_dim: int,
        **kwargs: Any,
    ) -> BaselineResult:
        from genedynamics.solvers.single.mrmfmbd import (
            ThetaParametrization,
            ThetaPrior,
            ThetaPriorConfig,
            ModeMarginalizer,
            default_mode_system_config,
            create_fidelity_ladder,
        )
        from genedynamics.solvers.single.mrmfmbd_ablation.backends import (
            MRMFMBDPosteriorBackendJax,
            PosteriorBridgeConfig,
        )

        theta_param = ThetaParametrization(
            x_dim=x_dim,
            phi_dim=phi_dim,
            x_bounds=(0.01, 2.0),
            phi_bounds=(5.0, 150.0),
        )
        theta_prior = ThetaPrior(theta_param, ThetaPriorConfig(x_std=0.5, phi_std=30.0))

        extra = config.extra
        K = extra.get("K", 50)
        M = extra.get("M", 8)
        num_modes = extra.get(
            "num_modes", task_spec.num_modes if hasattr(task_spec, "num_modes") else 4
        )

        mode_marginalizer = ModeMarginalizer(
            default_mode_system_config(num_modes),
            backend="jax",
        )
        fidelity_ladder = create_fidelity_ladder(
            K=K, num_levels=3, ladder_type="geometric", step_ratio=1.5
        )

        backend = MRMFMBDPosteriorBackendJax(
            evaluator=evaluator,
            theta_param=theta_param,
            theta_prior=theta_prior,
            mode_marginalizer=mode_marginalizer,
            fidelity_ladder=fidelity_ladder,
            config=PosteriorBridgeConfig(K=K, M=M),
            task_id=config.task_id,
            num_modes=num_modes,
            seed=config.seed,
            show_tqdm=extra.get("show_tqdm", False),
        )

        result = backend.plan()

        return BaselineResult(
            theta=result["theta"],
            x=result["x"],
            phi=result["phi"],
            return_=float(
                np.mean(result.get("bridge_history", [{}])[-1].get("mean_reward", 0.0))
            ),
            success=True,
            num_evaluations=K * M * num_modes,
            wall_time=result["wall_clock"],
            metadata={
                "bridge_history": result.get("bridge_history", []),
                "mode_responsibilities": result.get("mode_responsibilities", []),
                "fidelity_history": result.get("fidelity_history", []),
            },
        )
