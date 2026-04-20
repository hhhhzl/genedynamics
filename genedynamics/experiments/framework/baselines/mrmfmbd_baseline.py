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
        from genedynamics.solvers.single.mrmfmbd.backends import MRMFMBDBackendMBD, MBDConfig
        from genedynamics.core.inference.annealed_bridge import (
            BridgeScheduleConfig,
            LinearBridgeSchedule,
            GeometricBridgeSchedule,
        )

        extra = config.extra
        backend_type = str(extra.get("backend", "mcsa")).lower()

        # B1a: phi_dim defaults to 4 (SinWaveOpenLoop uses only the first
        # min(len(phi)//2, 4) as ω; the remaining phi was dead weight). A
        # yaml can still force phi_dim=N via task_spec, but the sane default
        # is the number of frequencies actually plumbed into the controller.
        if backend_type == "mbd" and phi_dim > 4 and not extra.get("phi_dim_override", False):
            phi_dim = 4

        # Z-symmetry: optimizer sees half the voxels along Z, mirrored before rollout.
        # Must be resolved BEFORE ThetaParametrization so x_dim is correct.
        morphology_symmetry = str(extra.get("morphology_symmetry", "")).lower()
        voxel_dims = extra.get("voxel_dims")
        if morphology_symmetry == "z" and voxel_dims is not None:
            vx, vy, vz = (int(v) for v in voxel_dims)
            if vz % 2 != 0:
                raise ValueError(f"Z-symmetry requires even vz, got voxel_dims={voxel_dims}")
            x_dim = vx * vy * (vz // 2)

        x_lo = float(extra.get("x_lo", 0.01))
        x_hi = float(extra.get("x_hi", 2.0))
        x_mean_init = float(extra.get("x_mean", (x_lo + x_hi) / 2.0))
        x_std = float(extra.get("x_std", 0.5))
        phi_lo = float(extra.get("phi_lo", 5.0))
        phi_hi = float(extra.get("phi_hi", 150.0))
        phi_std = float(extra.get("phi_std", 30.0))
        theta_param = ThetaParametrization(
            x_dim=x_dim,
            phi_dim=phi_dim,
            x_bounds=(x_lo, x_hi),
            phi_bounds=(phi_lo, phi_hi),
        )
        theta_prior = ThetaPrior(
            theta_param,
            ThetaPriorConfig(
                x_mean=np.full(x_dim, x_mean_init, dtype=np.float32),
                phi_mean=np.full(phi_dim, (phi_lo + phi_hi) / 2.0, dtype=np.float32),
                x_std=x_std,
                phi_std=phi_std,
            ),
        )

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
        fine_fidelity = extra.get("fine_fidelity_level", max(0, num_fidelity_levels - 1))
        # When num_fidelity_levels==1 the ladder collapses to a single level.
        # `create_fidelity_ladder` otherwise ignores fine_fidelity_level and
        # fills the ladder with level 0 regardless — so we pass explicit_levels
        # to pin the whole ladder at the requested `fine_fidelity_level`.
        if int(num_fidelity_levels) <= 1:
            fidelity_ladder = create_fidelity_ladder(
                K=K,
                num_levels=max(int(fine_fidelity) + 1, 1),  # must cover level `fine_fidelity`
                ladder_type=fidelity_ladder_type, step_ratio=fidelity_step_ratio,
                explicit_levels=[int(fine_fidelity)] * K,
            )
        else:
            fidelity_ladder = create_fidelity_ladder(
                K=K, num_levels=num_fidelity_levels,
                ladder_type=fidelity_ladder_type, step_ratio=fidelity_step_ratio,
            )

        if backend_type == "mbd":
            # Level-3: DDPM reverse-diffusion with S1 marginalization + S3 ladder.
            backend = MRMFMBDBackendMBD(
                evaluator=evaluator,
                theta_param=theta_param,
                theta_prior=theta_prior,
                mode_marginalizer=mode_marginalizer,
                fidelity_ladder=fidelity_ladder,
                config=MBDConfig(
                    K=K, M=M,
                    beta0=beta0, betaT=betaK,
                    reward_temperature=reward_temperature,
                    reward_temperature_min=float(extra.get("reward_temperature_min", max(0.05, reward_temperature * 0.2))),
                    tau_frac=float(extra.get("tau_frac", 0.1)),
                    top_k_fine=int(extra.get("top_k_fine", 3)),
                    fine_fidelity_level=fine_fidelity,
                    extra={
                        "morphology_symmetry": morphology_symmetry,
                        "voxel_dims": voxel_dims,
                    },
                ),
                task_id=config.task_id,
                num_modes=num_modes,
                seed=config.seed,
                show_tqdm=extra.get("show_tqdm", False),
            )
        else:
            schedule_type = extra.get("schedule_type", "linear")
            bridge_config = BridgeScheduleConfig(K=K, beta0=beta0, betaK=betaK, schedule_type=schedule_type)
            if schedule_type == "geometric":
                bridge_schedule = GeometricBridgeSchedule(bridge_config)
            else:
                bridge_schedule = LinearBridgeSchedule(bridge_config)
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

        if x_dim >= 1 and x_star[0] < 0.1:
            x_star[0] = 0.1

        # Expand x_star to full voxel grid if z-symmetric (for eval + GIF).
        x_star_full = x_star
        if morphology_symmetry == "z" and voxel_dims is not None:
            vx, vy, vz = (int(v) for v in voxel_dims)
            h = x_star.reshape(vx, vy, vz // 2)
            x_star_full = np.concatenate([h, np.flip(h, axis=-1)], axis=-1).reshape(-1)

        # Re-evaluate best theta to report a real rollout return (not bridge proxy).
        # Eval at the user-specified fine_fidelity_level — matches what the ladder
        # actually saw. Using (num_fidelity_levels - 1) would pick fid=0 when the
        # ablation has num_fidelity_levels=1 but fine_fidelity=1 or 2, producing
        # a return from a 30-step rollout instead of the intended 100/200.
        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        eval_fidelity = int(fine_fidelity)
        eval_requests = [
            RolloutRequest(
                morphology_params=x_star_full,
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
            theta=np.concatenate([x_star_full, phi_star]),
            x=x_star_full,
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
                # (K, D) theta trajectory used by diffusion-evolution plots.
                # Stored as list-of-lists for JSON serializability.
                "theta_history": np.asarray(
                    result.get("theta_history", []), dtype=np.float32
                ).tolist(),
            },
        )
