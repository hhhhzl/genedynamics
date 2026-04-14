"""
MRMFMBD as a Baseline: wraps posterior bridge for experiment platform.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..baseline import BaselineConfig, BaselineResult, BaselineProtocol


class MRMFMBDBaseline(BaselineProtocol):
    """
    MRMFMBD posterior bridge as a pluggable baseline.

    Implements BaselineProtocol for the co-design experiment platform.
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
            MRMFMBDPosteriorBackendJax,
            PosteriorBridgeConfig,
            ModeMarginalizerS1,
            default_mode_system_config,
            create_fidelity_ladder,
        )
        from genedynamics.core.inference.annealed_bridge import (
            BridgeScheduleConfig,
            LinearBridgeSchedule,
            GeometricBridgeSchedule,
        )

        theta_param = ThetaParametrization(
            x_dim=x_dim,
            phi_dim=phi_dim,
            x_bounds=(0.01, 2.0),
            phi_bounds=(5.0, 150.0),
        )
        theta_prior = ThetaPrior(theta_param, ThetaPriorConfig(x_std=0.5, phi_std=30.0))

        extra = config.extra
        # Extract diffusion params from scheduler (new-style) or fall back to extra (legacy)
        diff_params = config.get_diffusion_params()

        K = int(diff_params.get("Ndiffuse", extra.get("K", 50)))
        M = int(diff_params.get("M_k", extra.get("M", 8)))
        reward_temperature = float(diff_params.get("T_k", extra.get("reward_temperature", 0.1)))
        beta0 = float(diff_params.get("beta0", extra.get("beta0", 1e-6)))
        betaK = float(diff_params.get("betaT", extra.get("betaT", 1.0)))

        num_modes = extra.get("num_modes", task_spec.num_modes if hasattr(task_spec, "num_modes") else 4)
        num_fidelity_levels = extra.get("num_fidelity_levels", 3)
        fidelity_ladder_type = extra.get("fidelity_ladder_type", "geometric")
        fidelity_step_ratio = float(extra.get("fidelity_step_ratio", 1.5))

        mode_marginalizer = ModeMarginalizerS1(
            default_mode_system_config(num_modes),
            backend="jax",
        )
        fidelity_ladder = create_fidelity_ladder(
            K=K, num_levels=num_fidelity_levels,
            ladder_type=fidelity_ladder_type, step_ratio=fidelity_step_ratio,
        )

        # Create annealing schedule from beta0/betaK
        schedule_type = extra.get("schedule_type", "linear")
        bridge_config = BridgeScheduleConfig(K=K, beta0=beta0, betaK=betaK, schedule_type=schedule_type)
        if schedule_type == "geometric":
            bridge_schedule = GeometricBridgeSchedule(bridge_config)
        else:
            bridge_schedule = LinearBridgeSchedule(bridge_config)

        fine_fidelity = extra.get("fine_fidelity_level", max(0, num_fidelity_levels - 1))
        backend = MRMFMBDPosteriorBackendJax(
            evaluator=evaluator,
            theta_param=theta_param,
            theta_prior=theta_prior,
            bridge_schedule=bridge_schedule,
            mode_marginalizer=mode_marginalizer,
            fidelity_ladder=fidelity_ladder,
            config=PosteriorBridgeConfig(
                K=K, M=M,
                reward_temperature=reward_temperature,
                fine_fidelity_level=fine_fidelity,
            ),
            task_id=config.task_id,
            num_modes=num_modes,
            seed=config.seed,
            show_tqdm=extra.get("show_tqdm", False),
        )

        result = backend.plan()
        theta_star = np.asarray(result["theta"], dtype=np.float32).ravel()
        x_star = theta_star[:x_dim].copy()
        phi_star = theta_star[x_dim:]

        # Clamp geometry (x[0]) to avoid zero-density particles (p_rho_lower_bound_mul=0.1)
        if x_dim >= 1 and x_star[0] < 0.1:
            x_star[0] = 0.1
        # Clamp actuator multiplier (x[2]) so the optimizer can't pick the trivial
        # zero-actuation degenerate minimum that produces a static robot. The
        # `move_forward` reward gives small positive credit for purely passive
        # gravity drift, so without this the optimizer will sometimes choose
        # act_mul ~= 0 to avoid any actuation cost.
        if x_dim >= 3 and x_star[2] < 0.5:
            x_star[2] = 0.5

        # Re-evaluate best theta to report a real rollout return (not bridge proxy).
        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        eval_fidelity = max(0, num_fidelity_levels - 1)
        eval_requests = [
            RolloutRequest(
                morphology_params=x_star,
                controller_params=phi_star,
                mode_id=mode_id,
                fidelity_level=eval_fidelity,
                seed=int(config.seed),
                num_repeats=1,
                record=False,
            )
            for mode_id in range(num_modes)
        ]
        eval_batch = evaluator.evaluate_batch(
            RolloutBatchRequest(task_id=config.task_id, requests=eval_requests),
            parallel=True,
            use_cache=False,
        )
        real_return = float(np.mean(eval_batch.returns)) if eval_batch.returns.size > 0 else 0.0
        real_success = bool(np.any(eval_batch.successes)) if eval_batch.successes.size > 0 else False

        return BaselineResult(
            theta=theta_star,
            x=x_star,
            phi=phi_star,
            return_=real_return,
            success=real_success,
            num_evaluations=K * M * num_modes,
            wall_time=result["wall_clock"],
            metadata={
                "bridge_history": result.get("bridge_history", []),
                "mode_responsibilities": result.get("mode_responsibilities", []),
                "fidelity_history": result.get("fidelity_history", []),
                "final_eval_returns": eval_batch.returns.tolist(),
                "final_eval_successes": eval_batch.successes.tolist(),
                "final_eval_failure_codes": list(eval_batch.failure_codes),
            },
        )
