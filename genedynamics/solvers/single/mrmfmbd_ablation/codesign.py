"""
MRMFMBD **ablation** baseline (FROZEN legacy engine) — Stage 1 of the
theory-faithful refactor (docs/REFACTOR_PLAN.md). Verbatim copy of the
pre-refactor mrmfmbd/codesign.py, re-pointed at mrmfmbd_ablation.{backends,
fidelity_system}. Drives the OLD mechanisms (fixed ladder, control-variate,
post-hoc SHAC, reward/cvar regime modes, mean certification) for the paper's
Q2/Q3 ablations only. The method baseline lives in mrmfmbd/codesign.py.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.experiments.framework.baseline import BaselineConfig, BaselineResult, BaselineProtocol


_ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    # Q2 — mode-blind: disable the risk-sensitive regime potential (legacy
    # easy-weighted "reward" regime marginalization).
    "mode_blind": {"regime_posterior_mode": "reward"},
    # Q3 — fixed single fidelity (pair with `fine_fidelity_level` in the YAML).
    "fixed_fidelity": {"num_fidelity_levels": 1},
    # Q3 — control-variate multi-fidelity variant.
    "control_variate": {"cv_enabled": True},
    # Legacy mean-return certification is already this engine's default.
    "mean_cert": {},
}


class MRMFMBDAblationBaseline(BaselineProtocol):
    """
    Frozen legacy MRMFMBD engine as a pluggable ablation/baseline.

    Implements BaselineProtocol for the co-design experiment platform. An optional
    ``ablation_mode`` in method_params fills legacy knobs for a named ablation
    (explicit YAML keys always win); see ``_ABLATION_PRESETS``.
    """

    @property
    def name(self) -> str:
        return "mrmfmbd_ablation"

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
        # Reused (unchanged) primitives from the method package; only the fidelity
        # ladder + the MBD backend come from the frozen ablation package, so this
        # baseline is decoupled from the method-path rewrite (Stages 3+).
        from genedynamics.solvers.single.mrmfmbd import (
            ThetaParametrization,
            ThetaPrior,
            ThetaPriorConfig,
            ModeMarginalizer,
            default_mode_system_config,
        )
        from genedynamics.solvers.single.mrmfmbd_ablation.backends import (
            MRMFMBDPosteriorBackendJax,
            PosteriorBridgeConfig,
        )
        from genedynamics.solvers.single.mrmfmbd_ablation.fidelity_system import (
            create_fidelity_ladder,
        )
        from genedynamics.solvers.single.mrmfmbd_ablation.backends import (
            MRMFMBDLegacyBackend,
            LegacyMBDConfig,
        )
        from genedynamics.core.inference.annealed_bridge import (
            BridgeScheduleConfig,
            LinearBridgeSchedule,
            GeometricBridgeSchedule,
        )

        extra = dict(config.extra)
        # Ablation presets: thin convenience that fills legacy knobs for a named
        # ablation. Any key explicitly set in the YAML method_params wins.
        ablation_mode = str(extra.get("ablation_mode", "")).lower()
        for _k, _v in _ABLATION_PRESETS.get(ablation_mode, {}).items():
            extra.setdefault(_k, _v)
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

        # Stage 2 / G2a: A2 shape-latent decoder. If morph_latent_dim>0 and a
        # trained decoder dir is given, the optimizer's x-block becomes a shape
        # latent w ≈ N(0,I); rollout-time occupancy is g(w). Off by default
        # (morph_latent_dim=0) → legacy voxel-occupancy x-block, unchanged.
        morph_latent_dim = int(extra.get("morph_latent_dim", 0))
        morph_decoder_path = str(extra.get("morph_decoder_path", "") or "")
        morph_decoder = None
        if morph_latent_dim > 0 and morph_decoder_path:
            import json
            import os
            from genedynamics.solvers.single.mrmfmbd.morph_system.decoder import MorphDecoder
            from genedynamics.solvers.single.mrmfmbd.morph_system.specs import MorphDecoderConfig

            with open(os.path.join(morph_decoder_path, "decoder_config.json")) as _f:
                _dcfg = json.load(_f)
            if int(_dcfg["latent_dim"]) != morph_latent_dim:
                raise ValueError(
                    f"morph_latent_dim={morph_latent_dim} != decoder latent_dim="
                    f"{_dcfg['latent_dim']} at {morph_decoder_path}"
                )
            _mcfg = MorphDecoderConfig(
                latent_dim=int(_dcfg["latent_dim"]),
                hidden_dim=int(_dcfg["hidden_dim"]),
                n_voxels=int(_dcfg["n_voxels"]),
                x_lo=float(_dcfg["x_lo"]),
                x_hi=float(_dcfg["x_hi"]),
                beta_kl=float(_dcfg.get("beta_kl", 1e-3)),
                # Task-2 multi-head co-design fields (saved by train_morph_ae --multihead).
                decode_actuator=bool(_dcfg.get("decode_actuator", False)),
                decode_stiffness=bool(_dcfg.get("decode_stiffness", False)),
                n_actuators=int(_dcfg.get("n_actuators", 0)),
                e_lo=float(_dcfg.get("e_lo", 0.5)),
                e_hi=float(_dcfg.get("e_hi", 3.0)),
            )
            morph_decoder = MorphDecoder.load(
                os.path.join(morph_decoder_path, "decoder_params.npz"), _mcfg
            )
            # Optimizer x-block is now the latent w ~ N(0, I): override dim,
            # bounds, and prior so the diffusion samples in latent space. The
            # decoder's own [x_lo,x_hi] (from decoder_config.json) governs the
            # produced occupancy and is independent of these latent bounds.
            x_dim = morph_latent_dim
            x_lo = float(extra.get("morph_latent_lo", -2.0))
            x_hi = float(extra.get("morph_latent_hi", 2.0))
            x_mean_init = float(extra.get("morph_latent_mean", 0.0))
            x_std = float(extra.get("morph_latent_std", 1.0))

        theta_param = ThetaParametrization(
            x_dim=x_dim,
            phi_dim=phi_dim,
            x_bounds=(x_lo, x_hi),
            phi_bounds=(phi_lo, phi_hi),
        )
        # Prior-seeded init (contribution 3 → 1): all methods start from the SAME
        # 3D-prior body. morph_init_path is a .npy latent w0 (encoded TripoSG body).
        x_mean_vec = np.full(x_dim, x_mean_init, dtype=np.float32)
        _minit = str(extra.get("morph_init_path", "") or "")
        if _minit:
            _w0 = np.load(_minit).astype(np.float32).ravel()
            if _w0.shape[0] == x_dim:
                x_mean_vec = _w0
        theta_prior = ThetaPrior(
            theta_param,
            ThetaPriorConfig(
                x_mean=x_mean_vec,
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

        mode_marginalizer = ModeMarginalizer(
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
            # Level-3: DDPM reverse-diffusion with mode marginalization + fidelity ladder.
            backend = MRMFMBDLegacyBackend(
                evaluator=evaluator,
                theta_param=theta_param,
                theta_prior=theta_prior,
                mode_marginalizer=mode_marginalizer,
                fidelity_ladder=fidelity_ladder,
                config=LegacyMBDConfig(
                    K=K, M=M,
                    beta0=beta0, betaT=betaK,
                    reward_temperature=reward_temperature,
                    reward_temperature_min=float(extra.get("reward_temperature_min", max(0.05, reward_temperature * 0.2))),
                    tau_frac=float(extra.get("tau_frac", 0.1)),
                    top_k_fine=int(extra.get("top_k_fine", 3)),
                    fine_fidelity_level=fine_fidelity,
                    # Phase 1.3 (JM2D u-step). 0 = identical to baseline.
                    inner_denoise_steps=int(extra.get("inner_denoise_steps", 0)),
                    inner_denoise_shrink=float(extra.get("inner_denoise_shrink", 0.5)),
                    # Phase 2.3: regime-marginalization flavor.
                    # "reward" (default) keeps legacy MBD math; "risk_sensitive"
                    # activates the writeup §5 robust posterior.
                    regime_posterior_mode=str(extra.get("regime_posterior_mode", "reward")),
                    risk_temperature=float(extra.get("risk_temperature", 1.0)),
                    # Stage 4 / G2d: CVaR tail level for regime_posterior_mode="cvar".
                    cvar_alpha=float(extra.get("cvar_alpha", 0.1)),
                    # Stage 2 / G2a: A2 shape-latent decoder knobs (diagnostics).
                    morph_latent_dim=morph_latent_dim,
                    morph_decoder_path=morph_decoder_path,
                    # Task-2: use the multi-head decoder's actuator/stiffness fields
                    # (default on; per-head ablation switches).
                    morph_decoder_actuator=bool(extra.get("morph_decoder_actuator", True)),
                    morph_decoder_stiffness=bool(extra.get("morph_decoder_stiffness", True)),
                    # Shape regularizer weight (TV of occupancy → smoother bodies).
                    shape_weight=float(extra.get("shape_weight", 0.0)),
                    # Stage 2 / G2b: fold model-free prior log p0(θ) into weights.
                    use_prior_weight=bool(extra.get("use_prior_weight", False)),
                    prior_weight=float(extra.get("prior_weight", 1.0)),
                    # Stage 3 / G2c: multi-fidelity control-variate score estimator.
                    cv_enabled=bool(extra.get(
                        "cv_enabled",
                        str(extra.get("method", "")).lower() == "control_variate",
                    )),
                    cv_low_fidelity_level=int(extra.get("cv_low_fidelity_level", 0)),
                    cv_subset_k=int(extra.get("cv_subset_k", 0)),
                    nu_max=float(extra.get("nu_max", 0.0)),
                    cv_low_dt=float(extra.get("cv_low_dt", 0.0)),
                    # G2c-adaptive: budget-dual high-fi subset allocation (contribution 2).
                    cv_adaptive=bool(extra.get("cv_adaptive", False)),
                    cv_budget=float(extra.get("cv_budget", 0.0)),
                    cv_budget_eta=float(extra.get("cv_budget_eta", 0.5)),
                    # Phase 4.2: SHAC top-K refinement (writeup §8.1). Steps=0 (default) → no-op.
                    shac_refine_steps=int(extra.get("shac_refine_steps", 0)),
                    shac_refine_h=int(extra.get("shac_refine_h", 32)),
                    shac_refine_lr=float(extra.get("shac_refine_lr", 5.0e-4)),
                    shac_proximal_lambda=float(extra.get("shac_proximal_lambda", 1.0)),
                    shac_refine_topk=int(extra.get("shac_refine_topk", 3)),
                    # Phase 1.2 plumbing: pull alm_adaptive scalars from the
                    # CompositeScheduler and forward to the backend for
                    # diagnostics. Empty dict when no constraint scheduler is
                    # configured → behavior unchanged.
                    alm_params=config.get_constraint_alm_params(),
                    extra={
                        "morphology_symmetry": morphology_symmetry,
                        "voxel_dims": voxel_dims,
                    },
                ),
                task_id=config.task_id,
                num_modes=num_modes,
                seed=config.seed,
                show_tqdm=extra.get("show_tqdm", False),
                morph_decoder=morph_decoder,
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
                # G2c-adaptive: per-block budget-dual allocation diagnostic (Table 4).
                "cv_adaptive_summary": result.get("cv_adaptive_summary", {}),
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
