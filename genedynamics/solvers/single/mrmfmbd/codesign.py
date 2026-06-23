"""
MRMFMBD as a Baseline: wraps posterior bridge for experiment platform.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.experiments.framework.baseline import BaselineConfig, BaselineResult, BaselineProtocol


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
            ModeMarginalizer,
            default_mode_system_config,
        )
        from genedynamics.solvers.single.mrmfmbd.backends import MRMFMBDBackendMBD, MBDConfig

        extra = config.extra
        backend_type = str(extra.get("backend", "mcsa")).lower()
        if backend_type != "mbd":
            raise ValueError(
                f"The mrmfmbd METHOD solver supports backend='mbd' only (got "
                f"{backend_type!r}); the legacy posterior bridge lives in the "
                f"'mrmfmbd_ablation' solver."
            )

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
        # Stage 5: learned closed-loop controller — the controller block is the
        # latent c (z^MB) of width controller_latent_dim, with prior ~ N(0, 1).
        controller_type = str(extra.get("controller_type", "sinusoid")).lower()
        if controller_type == "learned":
            _cld = int(extra.get("controller_latent_dim", 0))
            phi_dim = _cld if _cld > 0 else 16
            phi_lo, phi_hi, phi_std = -3.0, 3.0, 1.0

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
        # Stage 4: disentangle the model-free prior CENTER from the 3D-prior SEED.
        # The MF prior p_theta^MF(w) stays standard (mean=x_mean_init, ~0 for the
        # latent path) so it actually regularizes; the encoded 3D-prior body w0
        # (morph_init_path) enters as the diffusion INITIALIZATION (theta_init),
        # NOT by shifting the prior center.
        x_mean_vec = np.full(x_dim, x_mean_init, dtype=np.float32)
        _minit = str(extra.get("morph_init_path", "") or "")
        theta_init = None
        if _minit:
            _w0 = np.load(_minit).astype(np.float32).ravel()
            if _w0.shape[0] == x_dim:
                _phi_mid = np.full(phi_dim, (phi_lo + phi_hi) / 2.0, dtype=np.float32)
                theta_init = np.concatenate([_w0, _phi_mid]).astype(np.float32)
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
        mode_marginalizer = ModeMarginalizer(
            default_mode_system_config(num_modes),
            backend="jax",
        )
        # Highest fidelity = the certification (rho_H) level. The method picks
        # fidelity ADAPTIVELY per step inside the scan; no ladder is built here.
        fine_fidelity = extra.get("fine_fidelity_level", max(0, num_fidelity_levels - 1))

        if backend_type == "mbd":
            # Level-3: DDPM reverse-diffusion with mode marginalization + fidelity ladder.
            backend = MRMFMBDBackendMBD(
                evaluator=evaluator,
                theta_param=theta_param,
                theta_prior=theta_prior,
                mode_marginalizer=mode_marginalizer,
                config=MBDConfig(
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
                    # Regime-marginalization flavor. METHOD DEFAULT is the
                    # risk-sensitive robust posterior (new_version.txt §IV-A
                    # eq:regime_risk_objective); "reward"/"cvar" are ablation-only
                    # values, served by the 'mrmfmbd_ablation' solver.
                    regime_posterior_mode=str(extra.get("regime_posterior_mode", "risk_sensitive")),
                    fidelity_adaptive=bool(extra.get("fidelity_adaptive", True)),
                    fidelity_probe=int(extra.get("fidelity_probe", 4)),
                    fidelity_enumerate=bool(extra.get("fidelity_enumerate", False)),
                    rollout_chunk=int(extra.get("rollout_chunk", 0)),
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
                    # Stage 4: unified weight — mandatory block-split MF/MB priors
                    # + hard validity indicator I(z).
                    mf_prior_weight=float(extra.get("mf_prior_weight", 0.2)),
                    mb_prior_weight=float(extra.get("mb_prior_weight", 0.0)),
                    validity_enabled=bool(extra.get("validity_enabled", True)),
                    validity_min_occupied_frac=float(extra.get("validity_min_occupied_frac", 0.03)),
                    validity_connect_iters=int(extra.get("validity_connect_iters", 24)),
                    # Phase 4.2: SHAC top-K refinement (writeup §8.1). Steps=0 (default) → no-op.
                    # Stage 5: learned closed-loop controller + in-loop SHAC.
                    controller_type=controller_type,
                    controller_latent_dim=int(extra.get("controller_latent_dim", 0)),
                    controller_hidden=int(extra.get("controller_hidden", 32)),
                    controller_embed_dim=int(extra.get("controller_embed_dim", 8)),
                    controller_seed=int(extra.get("controller_seed", 0)),
                    controller_path=str(extra.get("controller_path", "") or ""),
                    policy_warmup_save=str(extra.get("policy_warmup_save", "") or ""),
                    eta_c=float(extra.get("eta_c", 0.0)),
                    shac_h=int(extra.get("shac_h", 16)),
                    shac_gamma=float(extra.get("shac_gamma", 0.99)),
                    lambda_prox=float(extra.get("lambda_prox", 1.0)),
                    critic_enabled=bool(extra.get("critic_enabled", False)),
                    critic_hidden=int(extra.get("critic_hidden", 32)),
                    critic_lr=float(extra.get("critic_lr", 1.0e-3)),
                    td_lambda=float(extra.get("td_lambda", 0.95)),
                    policy_warmup_steps=int(extra.get("policy_warmup_steps", 0)),
                    policy_lr=float(extra.get("policy_lr", 3.0e-3)),
                    warmup_batch=int(extra.get("warmup_batch", 16)),
                    warmup_h=int(extra.get("warmup_h", 40)),
                    # Phase 1.2 plumbing: pull alm_adaptive scalars from the
                    # CompositeScheduler and forward to the backend for
                    # diagnostics. Empty dict when no constraint scheduler is
                    # configured → behavior unchanged.
                    alm_params=config.get_constraint_alm_params(),
                    extra={
                        "morphology_symmetry": morphology_symmetry,
                        "voxel_dims": voxel_dims,
                        # Stage 3 budgeted adaptive-fidelity knobs (read by the scan).
                        "vhat_a1": float(extra.get("vhat_a1", 1.0)),
                        "vhat_a2": float(extra.get("vhat_a2", 1.0)),
                        "vhat_a3": float(extra.get("vhat_a3", 0.5)),
                        "vhat_a4": float(extra.get("vhat_a4", 1.0)),
                        "fidelity_eta_nu": float(extra.get("fidelity_eta_nu", 0.05)),
                        "fidelity_cbar": extra.get("fidelity_cbar", None),
                        "fidelity_costs": extra.get("fidelity_costs", None),
                        # Fix C: exploration radius cap (DDPM sigma). Default 0.15 is a
                        # ~5x narrower proposal tube than CEM's init_std=0.8; raise to
                        # ~0.7 so ours searches at a comparable radius.
                        "sigma_max": float(extra.get("sigma_max", 0.15)),
                    },
                ),
                task_id=config.task_id,
                num_modes=num_modes,
                seed=config.seed,
                show_tqdm=extra.get("show_tqdm", False),
                morph_decoder=morph_decoder,
            )
        result = backend.plan(theta_init=theta_init)
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
        if controller_type == "learned":
            # The learned latent c is incompatible with the evaluator's sine
            # rollout; the backend already certified via the closed-loop policy.
            real_return = float(result.get("best_fine_return") or 0.0)
            real_success = result.get("best_fine_return") is not None
            final_eval_returns, final_eval_successes, final_eval_failure_codes = [], [], []
        elif morph_decoder is not None:
            # A2 latent path: decode w -> (occ, actuator field) and re-roll per mode via
            # the JAX-direct rollout_return — IDENTICAL to the trusted dumper /
            # run_baseline_p2 cert. The legacy evaluate_batch (else) treats the raw 32-d
            # latent as occupancy (no decode) -> out-of-bounds gather -> a garbage body
            # certified at rho_H ~ -38 (the cert-path bug). Decoding gives the true value.
            import jax.numpy as _jnp
            from genedynamics.envs.external.jax_mpm.scene import rollout_return as _rr
            from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS as _FS
            from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
                risk_sensitive_marginalize_np as _rsm,
            )
            _steps = int(_FS.get(int(fine_fidelity), _FS[max(_FS)]))
            _occ, _act, _ = morph_decoder.decode_full_batch(_jnp.asarray(x_star_full[None]))
            _occ = _occ[0]
            _aw = _act[0] if _act is not None else None
            _frtab = np.asarray(evaluator._mode_friction, np.float32)[:num_modes]
            _rets = [float(_rr(_occ, _jnp.asarray(phi_star), _jnp.asarray(float(_fr)),
                              evaluator._scene, evaluator._mpm_cfg, _steps,
                              actuator_weight_voxel=_aw)[0]) for _fr in _frtab]
            _regime_mode = str(extra.get("regime_posterior_mode", "risk_sensitive"))
            if _regime_mode == "risk_sensitive":
                real_return = float(_rsm(np.asarray(_rets), np.zeros(num_modes),
                                         float(extra.get("risk_temperature", 1.0))))
            else:
                real_return = float(np.mean(_rets))
            real_success = bool(np.isfinite(real_return))
            final_eval_returns = list(_rets)
            final_eval_successes = [bool(r > 0.0) for r in _rets]
            final_eval_failure_codes = []
        else:
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
            # Report the high-fidelity risk-sensitive certification rho_H over regimes
            # (new_version.txt eq:high_fidelity_risk), matching the in-loop objective —
            # not the cross-regime mean. ("reward"/"cvar" ablations keep the mean.)
            _regime_mode = str(extra.get("regime_posterior_mode", "risk_sensitive"))
            if _regime_mode == "risk_sensitive" and eval_batch.returns.size > 0:
                from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
                    risk_sensitive_marginalize_np,
                )
                real_return = float(risk_sensitive_marginalize_np(
                    np.asarray(eval_batch.returns)[:num_modes],
                    np.zeros(num_modes, dtype=np.float64),
                    float(extra.get("risk_temperature", 1.0)),
                ))
            else:
                real_return = float(np.mean(eval_batch.returns)) if eval_batch.returns.size > 0 else 0.0
            real_success = bool(np.any(eval_batch.successes)) if eval_batch.successes.size > 0 else False
            final_eval_returns = eval_batch.returns.tolist()
            final_eval_successes = eval_batch.successes.tolist()
            final_eval_failure_codes = list(eval_batch.failure_codes)

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
                "fidelity_summary": result.get("fidelity_summary", {}),
                "shac_summary": result.get("shac_summary", {}),
                "final_eval_returns": final_eval_returns,
                "final_eval_successes": final_eval_successes,
                "final_eval_failure_codes": final_eval_failure_codes,
                # (K, D) theta trajectory used by diffusion-evolution plots.
                # Stored as list-of-lists for JSON serializability.
                "theta_history": np.asarray(
                    result.get("theta_history", []), dtype=np.float32
                ).tolist(),
            },
        )
