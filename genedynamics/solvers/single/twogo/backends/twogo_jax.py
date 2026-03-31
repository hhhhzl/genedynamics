"""
JAX backend for 2GO.

- Scan core (predictor–corrector geometry): Geo₀ from mean SDF normals; optional tail-mixture
  B-probe (gather → vmap CFS, I_QP=1) refines geometry to Geo₁; MCSA weights use probed
  trajectories where selected; tangent move uses (G₁,P₁); mean CFS retraction on τ̃.
- Optional NumPy `_refine_candidates` remains for window-local SDF refine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.solvers.single.cfsmbd.backends.cfsmbd_jax import CFSMBDBackendJax
from genedynamics.genemetry import (
    SdfManifold,
    MultimodalGate,
    ProbeModulator,
    CfsRetraction,
    SteppingRetraction,
    SlidingWindow,
    ProbePipeline,
    AgpStep,
    LocalCfsRetraction,
    WindowRefinement,
    ScheduleOverlay,
    OverlayConfig,
    resolve_overlay_config,
)
from genedynamics.genemetry.modulation.goal_direction import GoalDirectionJax
from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
from genedynamics.core.constraints.core.types import ScheduleState


class TwoGOBackendJax:
    def __init__(self, **kwargs: Any):
        self._inner = CFSMBDBackendJax(**kwargs)

        # 2GO knobs (Phase3-5)
        solver = kwargs.get("solver", None)
        cfg = getattr(solver, "config", {}) if solver is not None else {}
        self.twogo_gate_vrate_threshold = float(cfg.get("twogo_gate_vrate_threshold", 0.01))  # fallback
        self.twogo_cvar_alpha = float(cfg.get("twogo_cvar_alpha", 0.9))
        self.twogo_window_size = int(cfg.get("twogo_window_size", 16))
        self.twogo_window_stride = int(cfg.get("twogo_window_stride", 8))
        self.twogo_tail_ratio = float(cfg.get("twogo_tail_ratio", 0.3))
        self.twogo_active_topk = int(cfg.get("twogo_active_topk", 8))
        self.twogo_agp_eta = float(cfg.get("twogo_agp_eta", 0.08))
        self.twogo_cfs_gain = float(cfg.get("twogo_cfs_gain", 0.35))
        self.twogo_multi_scale = float(cfg.get("twogo_multi_scale", 0.25))
        self.twogo_enable_agp_refine = bool(cfg.get("twogo_enable_agp_refine", True))
        self.twogo_enable_local_gating = bool(cfg.get("twogo_enable_local_gating", True))
        self.twogo_enable_sample_tail = bool(cfg.get("twogo_enable_sample_tail", True))
        self.twogo_enable_agp_batch = bool(cfg.get("twogo_enable_agp_batch", False))
        self.twogo_use_jax_scan_core = bool(cfg.get("twogo_use_jax_scan_core", True))
        self.twogo_retract_qp_boost = bool(cfg.get("twogo_retract_qp_boost", True))
        self.twogo_stability_eps = float(cfg.get("twogo_stability_eps", 1e-6))
        self.twogo_gamma_init = float(cfg.get("twogo_gamma_init", 1.0))
        self.twogo_retract_qp_every = int(max(1, cfg.get("twogo_retract_qp_every", 2)))
        self.twogo_probe_enable = bool(cfg.get("twogo_probe_enable", True))
        self.twogo_probe_tail_mix = float(np.clip(float(cfg.get("twogo_probe_tail_mix", 0.5)), 0.0, 1.0))
        self.twogo_probe_tail_pool_ratio = float(np.clip(float(cfg.get("twogo_probe_tail_pool_ratio", 0.25)), 1e-6, 1.0))
        self.twogo_probe_geom_alpha = float(cfg.get("twogo_probe_geom_alpha", 0.25))
        self.twogo_gate_ema_beta = float(np.clip(float(cfg.get("twogo_gate_ema_beta", 0.7)), 0.0, 0.99))
        self.twogo_gate_risk_theta = float(cfg.get("twogo_gate_risk_theta", 0.05))
        self.twogo_cluster_reweight_alpha = float(cfg.get("twogo_cluster_reweight_alpha", 1.0))
        self.twogo_cluster_retract_mu = float(np.clip(float(cfg.get("twogo_cluster_retract_mu", 0.1)), 0.0, 0.5))
        self.twogo_probe_frac = float(np.clip(float(cfg.get("twogo_probe_frac", 0.5)), 0.0, 1.0))
        self.twogo_probe_b = cfg.get("twogo_probe_b", None)
        self.twogo_probe_m_cap = cfg.get("twogo_probe_m_cap", None)
        self.twogo_task_dir_alpha = float(cfg.get("twogo_task_dir_alpha", 0.15))
        rho_ref_default = float(getattr(getattr(self._inner, "_cs", None), "rho_max", 500.0))
        self._overlay_config = resolve_overlay_config(
            cfg.get("twogo_overlay", None), rho_ref_default=rho_ref_default,
        )
        self.twogo_overlay = self._overlay_config.to_dict()  # backward compat
        self._overlay_jax = ScheduleOverlay(config=self._overlay_config, backend="jax")
        self._overlay_numpy = ScheduleOverlay(config=self._overlay_config, backend="numpy")
        self._rng = np.random.default_rng(int(getattr(self._inner, "seed", 0)))
        self._twogo_single_jit = None
        self._twogo_batch_jit = None
        self._M_k_arr = None
        self._gate_every_arr = None
        self._probe_b_int = 0
        self._prepare_twogo_diffusion_arrays()
        _ns = int(self._inner.Nsample)
        if self.twogo_probe_b is not None:
            self._probe_b_int = int(max(0, min(int(self.twogo_probe_b), _ns)))
        else:
            self._probe_b_int = int(max(0, min(_ns, int(round(float(self.twogo_probe_frac) * _ns)))))
        if self.twogo_use_jax_scan_core:
            self._build_twogo_scan_kernels()

        # -- Phase B genemetry: refinement components ----------------------
        from genedynamics.genemetry.window.backends.multimodality_numpy import (
            WindowMultimodalityNumpy,
        )
        _position_dim = int(getattr(self._inner, "position_dim", 2))
        self._refine_window_policy = SlidingWindow(
            self.twogo_window_size, self.twogo_window_stride,
        )
        self._refine_agp_step = AgpStep(
            backend="numpy",
            active_topk=self.twogo_active_topk,
            dt=float(max(getattr(self._inner, "dt", 0.1), 1e-6)),
            action_limit=float(self._inner.action_limit),
            rng=self._rng,
        )
        self._refine_local_cfs = LocalCfsRetraction(
            backend="numpy",
            gain=self.twogo_cfs_gain,
            dt=float(max(getattr(self._inner, "dt", 0.1), 1e-6)),
            action_limit=float(self._inner.action_limit),
        )
        self._refine_multimodality = WindowMultimodalityNumpy(
            multi_scale=self.twogo_multi_scale,
            position_dim=_position_dim,
        )
        self._refinement_pipeline = WindowRefinement(
            backend="numpy",
            window_policy=self._refine_window_policy,
            constrained_step=self._refine_agp_step,
            local_retraction=self._refine_local_cfs,
            multimodality_evaluator=self._refine_multimodality,
            cvar_alpha=self.twogo_cvar_alpha,
            tail_ratio=self.twogo_tail_ratio,
            enable_sample_tail=self.twogo_enable_sample_tail,
            enable_local_gating=self.twogo_enable_local_gating,
        )

    def _resolve_stepping_scene(self):
        env = getattr(self._inner, "env", None)
        if env is not None and hasattr(env, "scene") and getattr(env, "scene") is not None:
            return getattr(env, "scene")
        obs = getattr(self._inner, "obstacles", None)
        if obs is not None and hasattr(obs, "stepping_scene"):
            return getattr(obs, "stepping_scene")
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def num_modes(self) -> int:
        return int(getattr(self._inner, "num_modes", 1))

    @num_modes.setter
    def num_modes(self, value: int) -> None:
        self._inner.num_modes = int(value)

    def _prepare_twogo_diffusion_arrays(self) -> None:
        inner = self._inner
        Ndiffuse = int(inner.Ndiffuse)
        gate_default = int(self.twogo_overlay["gate_every"])
        M_default = int(inner.Nsample)
        M_list = [M_default] * Ndiffuse
        gate_list = [gate_default] * Ndiffuse
        try:
            if inner.scheduler is not None and hasattr(inner.scheduler, "diffusion_schedulers"):
                ds_list = getattr(inner.scheduler, "diffusion_schedulers", [])
                if ds_list:
                    ds = ds_list[0]
                    for k in range(Ndiffuse):
                        st = ScheduleState(k=k, K=max(1, Ndiffuse))
                        d = ds.diffusion_params(st) if hasattr(ds, "diffusion_params") else {}
                        if d is None:
                            d = {}
                        M_list[k] = int(max(1, int(d.get("M_k", M_default))))
                        gate_list[k] = int(max(1, int(d.get("gate_every", gate_default))))
        except Exception:
            pass
        self._M_k_arr = jnp.asarray(np.asarray(M_list, dtype=np.int32), dtype=jnp.int32)
        self._gate_every_arr = jnp.asarray(np.asarray(gate_list, dtype=np.int32), dtype=jnp.int32)

    def _build_twogo_scan_kernels(self) -> None:
        inner = self._inner
        Ndiffuse = int(inner.Ndiffuse)
        horizon = int(inner.horizon)
        act_dim = int(inner.act_dim)
        Nsample = int(inner.Nsample)
        action_limit = float(inner.action_limit)
        total_steps_jnp = jnp.asarray(Ndiffuse, dtype=jnp.int32)

        betas = jnp.linspace(inner.beta0, inner.betaT, Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(Ndiffuse - 1, -1, -1, dtype=jnp.int32)
        denom = jnp.maximum(float(Ndiffuse - 1), 1.0)
        progress_inc = 1.0 - (jnp.arange(Ndiffuse, dtype=jnp.float32) / denom)
        extra_sigmas = inner.action_extra_sigma * (1.0 - progress_inc)
        window_size = int(max(1, min(self.twogo_window_size, horizon)))
        window_stride = int(max(1, self.twogo_window_stride))
        _window_policy = SlidingWindow(window_size, window_stride)
        tail_alpha = float(np.clip(1.0 - self.twogo_cvar_alpha, 0.0, 1.0))
        n_tail_cvar = int(max(1, int(np.ceil(tail_alpha * Nsample))))
        topk_active = int(max(1, min(self.twogo_active_topk, horizon)))
        eps_stab = float(max(self.twogo_stability_eps, 1e-8))
        agp_eta = float(self.twogo_agp_eta)
        enable_local_gating = bool(self.twogo_enable_local_gating)
        retract_boost = bool(self.twogo_retract_qp_boost)
        retract_every_static = int(max(1, self.twogo_retract_qp_every))
        multi_scale = float(max(self.twogo_multi_scale, 1e-6))

        margin_arr = inner._margin_arr
        rho_arr = inner._rho_arr
        qp_gate_arr = inner._qp_gate_arr
        qp_prob_arr = inner._qp_prob_arr
        I_QP_arr = inner._I_QP_arr
        eps_arr = inner._eps_arr
        T_k_arr = inner._T_k_arr
        M_k_arr = self._M_k_arr
        gate_every_arr = self._gate_every_arr
        topK_default = jnp.asarray(inner._topK if inner._topK >= 0 else 8, dtype=jnp.int32)
        aug_lambda_const = jnp.asarray(inner.aug_lambda, dtype=jnp.float32)
        aug_rho_const = jnp.asarray(inner.aug_rho, dtype=jnp.float32)
        _schedule_overlay_jax = self._overlay_jax
        cs = getattr(inner, "_cs", None)
        use_jax_adaptive = bool(
            getattr(inner, "_use_jax_adaptive", False)
            and cs is not None
            and hasattr(cs, "jax_init_carry")
            and hasattr(cs, "jax_compute_params")
            and hasattr(cs, "jax_update")
        )

        _B = int(max(0, min(int(self._probe_b_int), Nsample)))
        _B_vm = _B if _B > 0 else 1
        if self.twogo_probe_m_cap is not None:
            _B_vm = min(_B_vm, int(self.twogo_probe_m_cap))
        M_max = int(max(1, min(Nsample, _B_vm)))
        probe_b_jnp = jnp.asarray(_B, dtype=jnp.int32)
        k_pool_i = int(min(Nsample, max(1, int(np.ceil(float(self.twogo_probe_tail_pool_ratio) * Nsample)))))
        tail_mix_f = float(self.twogo_probe_tail_mix)
        probe_alpha = float(self.twogo_probe_geom_alpha)
        probe_enable = bool(self.twogo_probe_enable)
        stepping_scene = self._resolve_stepping_scene()
        stepping_enabled = bool(stepping_scene is not None and int(act_dim) >= 4)
        if stepping_scene is not None:
            step_centers = jnp.asarray(np.asarray(stepping_scene.stones_centers, dtype=np.float32), dtype=jnp.float32)
            step_radii = jnp.asarray(np.asarray(stepping_scene.stones_radii, dtype=np.float32), dtype=jnp.float32)
            step_lmax = jnp.asarray(float(getattr(stepping_scene, "l_max", 0.35)), dtype=jnp.float32)
        else:
            step_centers = jnp.zeros((1, 2), dtype=jnp.float32)
            step_radii = jnp.ones((1,), dtype=jnp.float32)
            step_lmax = jnp.asarray(0.35, dtype=jnp.float32)

        # -- genemetry components ----------------------------------------
        _constraint_manifold = SdfManifold(backend="jax")
        _task_modulator = ProbeModulator()
        _gate_policy = MultimodalGate(
            backend="jax",
            multi_scale=multi_scale,
            enable_local_gating=enable_local_gating,
        )
        _task_direction = GoalDirectionJax(alpha_task=float(self.twogo_task_dir_alpha))
        _probe_pipeline = ProbePipeline(
            backend="jax",
            tail_mix=tail_mix_f,
            pool_ratio=float(self.twogo_probe_tail_pool_ratio),
            max_probes=M_max,
        )
        if stepping_enabled:
            _retraction_op = SteppingRetraction(
                backend="jax",
                stone_centers=step_centers,
                stone_radii=step_radii,
                l_max=step_lmax,
                action_limit=action_limit,
            )
        else:
            _retraction_op = CfsRetraction(
                backend="jax",
                filter_fn=inner._filter_actions_single_jit,
            )

        def _sched_lookup_fixed(step_k: jnp.ndarray):
            if margin_arr is not None:
                kk = jnp.clip(step_k, 0, margin_arr.shape[0] - 1)
                margin = margin_arr[kk]
                rho = rho_arr[kk]
                qp_gate = qp_gate_arr[kk]
                qp_prob = qp_prob_arr[kk]
                I_qp = I_QP_arr[kk] if I_QP_arr is not None else jnp.asarray(1, dtype=jnp.int32)
                eps = eps_arr[kk] if eps_arr is not None else jnp.asarray(1e-4, dtype=jnp.float32)
            else:
                margin = jnp.asarray(0.0, dtype=jnp.float32)
                rho = jnp.asarray(1.0, dtype=jnp.float32)
                qp_gate = jnp.asarray(True, dtype=jnp.bool_)
                qp_prob = jnp.asarray(1.0, dtype=jnp.float32)
                I_qp = jnp.asarray(1, dtype=jnp.int32)
                eps = jnp.asarray(1e-4, dtype=jnp.float32)
            topK = topK_default
            compute_cost_hat = qp_prob.astype(jnp.float32) * topK.astype(jnp.float32) * I_qp.astype(jnp.float32)
            return {
                "margin": margin,
                "rho": rho,
                "qp_gate": qp_gate,
                "qp_prob": qp_prob,
                "I_QP": I_qp,
                "eps": eps,
                "topK": topK,
                "aug_lambda": aug_lambda_const,
                "aug_rho": aug_rho_const,
                "nu": jnp.asarray(0.0, dtype=jnp.float32),
                "compute_cost_hat": compute_cost_hat,
                "compute_budget_B_eff": jnp.asarray(0.0, dtype=jnp.float32),
            }

        def _cvar_topk(v: jnp.ndarray) -> jnp.ndarray:
            top_vals, _ = jax.lax.top_k(v, n_tail_cvar)
            return jnp.mean(top_vals)

        gate_ema_beta = jnp.asarray(float(self.twogo_gate_ema_beta), dtype=jnp.float32)
        gate_risk_theta = jnp.asarray(float(self.twogo_gate_risk_theta), dtype=jnp.float32)
        cluster_reweight_alpha = jnp.asarray(float(self.twogo_cluster_reweight_alpha), dtype=jnp.float32)
        cluster_retract_mu = jnp.asarray(float(self.twogo_cluster_retract_mu), dtype=jnp.float32)
        # Static flag: only include clustering in trace graph when at
        # least one of the two features is enabled.  Avoids JIT graph
        # changes that alter floating-point behaviour when disabled.
        cluster_enabled = bool(self.twogo_cluster_reweight_alpha > 0 or self.twogo_cluster_retract_mu > 0)

        def _run_single(x0_jnp: jnp.ndarray, rng_key: jnp.ndarray, target: jnp.ndarray):
            rng, rng_sched_seed = jax.random.split(rng_key)
            Ybar_init = jnp.zeros((horizon, act_dim), dtype=jnp.float32)
            gamma_init = jnp.asarray(np.clip(self.twogo_gamma_init, 0.0, 1.0), dtype=jnp.float32)
            pi_ema_init = jnp.asarray(1.0, dtype=jnp.float32)  # start high → diffusion ON
            carry_sched_init = cs.jax_init_carry(rng_sched_seed) if use_jax_adaptive else None

            def body(carry, idx):
                if use_jax_adaptive:
                    rng_curr, Ybar_curr, gamma_prev, pi_ema_prev, carry_sched = carry
                else:
                    rng_curr, Ybar_curr, gamma_prev, pi_ema_prev = carry
                    carry_sched = None
                rng_curr, noise_key, extra_key, key_sched = jax.random.split(rng_curr, 4)
                k_sched, rng_tail, rng_rand, rng_vmap_parent = jax.random.split(key_sched, 4)
                step_k = jnp.asarray(Ndiffuse - 1, dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}

                if use_jax_adaptive:
                    params, rng_sched_next = cs.jax_compute_params(carry_sched, step_k, Ndiffuse)
                else:
                    params = _sched_lookup_fixed(step_k)
                    rng_sched_next = k_sched
                margin = params["margin"]
                rho_k = params["rho"]
                qp_gate = params["qp_gate"]
                qp_prob = params["qp_prob"]
                I_qp = params["I_QP"]
                eps_k = params["eps"]
                topK_k = params["topK"]
                aug_lam = params["aug_lambda"]
                aug_rho = params["aug_rho"]
                _overlay_k = _schedule_overlay_jax.compute(margin, rho_k, agp_eta)
                hardness_k = _overlay_k.hardness
                kappa_k = _overlay_k.kappa
                delta_k = _overlay_k.delta
                sigma_k = _overlay_k.sigma
                theta_k = _overlay_k.theta
                eta_j = _overlay_k.eta
                M_k = M_k_arr[jnp.clip(step_k, 0, M_k_arr.shape[0] - 1)] if M_k_arr is not None else jnp.asarray(Nsample, dtype=jnp.int32)
                M_k = jnp.clip(M_k, 1, Nsample)
                gate_every_k = gate_every_arr[jnp.clip(step_k, 0, gate_every_arr.shape[0] - 1)] if gate_every_arr is not None else jnp.asarray(1, dtype=jnp.int32)
                gate_every_k = jnp.maximum(gate_every_k, 1)
                gate_pass = jnp.equal(jnp.mod(step_k, gate_every_k), 0)
                qp_gate_eff = jnp.logical_and(qp_gate, gate_pass)

                # Probe mini-batch B: independent of M_k and qp_prob (see twogo_probe_b / twogo_probe_frac).
                M_eff = jnp.where(
                    qp_gate_eff,
                    jnp.minimum(probe_b_jnp, jnp.asarray(Nsample, dtype=jnp.int32)),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                do_probe = jnp.logical_and(
                    jnp.asarray(probe_enable, dtype=jnp.bool_),
                    jnp.logical_and(qp_gate_eff, M_eff > 0),
                )

                sched_params = {
                    "margin": margin,
                    "rho": rho_k,
                    "qp_gate": qp_gate_eff,
                    "qp_prob": qp_prob,
                    "I_QP": I_qp,
                    "eps": eps_k,
                    "topK": topK_k,
                    "rng_key": k_sched,
                }

                wmask = _window_policy.mask(step_k, horizon)

                a_geom = inner._constraint_geometry_time_jit(x0_jnp, Ybar_curr, margin)
                a_geom0 = a_geom * wmask[:, None]

                eps_noise = jax.random.normal(noise_key, (Nsample, horizon, act_dim), dtype=jnp.float32)
                Y0s = jnp.clip(eps_noise * sigmas[idx] + Ybar_curr, -action_limit, action_limit)
                rews, v_batch = inner._augmented_and_v_batch_jit(
                    x0_jnp, Y0s, margin, aug_lam, aug_rho, target
                )
                r_p = _cvar_topk(v_batch)
                v_rate = jnp.mean(v_batch > 0.0)
                v_mean = jnp.mean(v_batch)
                cvar = _cvar_topk(jnp.maximum(v_batch, 0.0))
                if use_jax_adaptive:
                    feedback = {
                        "r_p": r_p,
                        "v_k_rate": v_rate,
                        "v_k_mean": v_mean,
                        "compute_cost_hat": params["compute_cost_hat"],
                        "compute_budget_B_eff": params["compute_budget_B_eff"],
                    }
                    carry_sched_new = cs.jax_update(carry_sched, feedback, rng_sched_next)
                else:
                    carry_sched_new = carry_sched

                _probe_sched_params = {
                    "margin": margin,
                    "rho": rho_k,
                    "qp_gate": jnp.asarray(True, dtype=jnp.bool_),
                    "qp_prob": jnp.asarray(1.0, dtype=jnp.float32),
                    "eps": eps_k,
                    "topK": topK_k,
                }
                _probe_retract_params = {
                    "M_eff": M_eff,
                    "sched_state": sched_state,
                    "sched_params": _probe_sched_params,
                }
                _probe_rng_keys = {
                    "rng_tail": rng_tail,
                    "rng_rand": rng_rand,
                    "rng_vmap": rng_vmap_parent,
                }
                def _run_probe(_):
                    pr = _probe_pipeline.sample_and_retract(
                        Y0s, v_batch, _retraction_op, x0_jnp,
                        _probe_retract_params, _probe_rng_keys,
                    )
                    return pr.fixed_trajectories, pr.residual_geometry

                Y_fix, a_probe = jax.lax.cond(
                    do_probe,
                    _run_probe,
                    lambda _: (
                        Y0s,
                        jnp.zeros((horizon, act_dim), dtype=jnp.float32),
                    ),
                    operand=None,
                )
                bmask = (jnp.linalg.norm(Y_fix - Y0s, axis=(1, 2)) > jnp.asarray(1e-12, dtype=jnp.float32)).astype(
                    jnp.float32
                )
                bmask3 = bmask[:, None, None]
                Y_eff = jnp.where(bmask3 > 0.5, Y_fix, Y0s)

                a_geom1 = _task_modulator.modulate(
                    a_geom0,
                    probe_geometry=a_probe,
                    window_mask=wmask,
                    alpha=probe_alpha,
                )
                bundle = _constraint_manifold.geometry(
                    a_geom1, topk_active, eps_stab
                )

                # Phase B/C: cluster-based reweighting + retract blend.
                # Guarded behind a static flag so the trace graph is
                # identical to baseline when both are disabled.
                rew_mean = jnp.mean(rews)
                rew_std = jnp.where(jnp.std(rews) < 1e-4, 1.0, jnp.std(rews))
                T_k = T_k_arr[jnp.clip(step_k, 0, T_k_arr.shape[0] - 1)] if T_k_arr is not None else jnp.asarray(inner.temp_sample, dtype=jnp.float32)
                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y_eff)

                sigma_scale = jnp.maximum(sigmas[idx], jnp.asarray(1e-6, dtype=jnp.float32))
                gate_decision = _gate_policy.evaluate(
                    Y0s, Ybar_curr, wmask, sigma_scale, theta_k,
                )
                pi_route_raw = gate_decision.meta["pi_multi"]

                # A3: EMA smoothing of route proxy to avoid gate jitter.
                pi_ema_k = gate_ema_beta * pi_ema_prev + (1.0 - gate_ema_beta) * pi_route_raw

                # A2: Dual gate — route ambiguity OR feasibility risk.
                # Risk threshold rises with hardness: late stages tolerate
                # more residual violation without reopening diffusion.
                risk_thresh_k = gate_risk_theta * (1.0 + 4.0 * hardness_k)
                route_on = pi_ema_k > theta_k
                risk_on = cvar > risk_thresh_k
                gamma_dual = jnp.logical_or(route_on, risk_on).astype(jnp.float32)

                gamma_raw = jax.lax.cond(
                    jnp.asarray(enable_local_gating),
                    lambda _: gamma_dual,
                    lambda _: jnp.asarray(1.0, dtype=jnp.float32),
                    operand=None,
                )
                pi_multi_k = pi_ema_k
                gamma = jax.lax.cond(gate_pass, lambda _: gamma_raw, lambda _: gamma_prev, operand=None)

                # Task direction bias: decays with hardness.
                task_dir = _task_direction.direction(Ybar_curr, target, step_k, hardness_k)
                score_base = kappa_k * (Ybar_weighted - Ybar_curr) + task_dir
                u_agp_proj = _constraint_manifold.project(score_base, bundle, mode="metric")
                u_agp = jax.lax.cond(bundle.is_valid, lambda _: u_agp_proj, lambda _: score_base, operand=None)

                noise_extra = jax.random.normal(extra_key, (horizon, act_dim), dtype=jnp.float32)
                p_noise_proj = _constraint_manifold.project(noise_extra, bundle, mode="tangent")
                p_noise = jax.lax.cond(bundle.is_valid, lambda _: p_noise_proj, lambda _: noise_extra, operand=None)
                sigma_eff = gamma * (sigma_k + extra_sigmas[idx])
                Ybar_tilde = Ybar_weighted + eta_j * u_agp + sigma_eff * p_noise

                retract_every = jnp.asarray(retract_every_static, dtype=jnp.int32)
                retract_pass = jnp.equal(jnp.mod(step_k, retract_every), 0)
                retract_highrisk = cvar > delta_k
                boost_j = jnp.asarray(retract_boost, dtype=jnp.bool_)
                do_retract_qp = jnp.logical_and(boost_j, jnp.logical_and(qp_gate_eff, jnp.logical_or(retract_pass, retract_highrisk)))

                sched_params_retract = dict(sched_params)
                sched_params_retract["qp_gate"] = jnp.asarray(True, dtype=jnp.bool_)
                sched_params_retract["qp_prob"] = jnp.asarray(1.0, dtype=jnp.float32)
                retract_params = {
                    "sched_state": sched_state,
                    "sched_params": sched_params_retract,
                }
                retract_fn = lambda y: _retraction_op.retract(
                    x0_jnp, y, retract_params
                ).trajectory
                Ybar_next = jax.lax.cond(
                    do_retract_qp,
                    retract_fn,
                    lambda y: y,
                    Ybar_tilde,
                )
                Ybar_next = jnp.clip(Ybar_next, -action_limit, action_limit)

                reward_stage_terminal = jnp.mean(rews)
                m_eff_f = M_eff.astype(jnp.float32)
                qp_call_minibatch = do_probe.astype(jnp.float32) * (m_eff_f / jnp.maximum(jnp.asarray(float(Nsample), dtype=jnp.float32), 1.0))
                qp_call_retract = do_retract_qp.astype(jnp.float32) * jnp.asarray(1.0 if retract_boost else 0.0, dtype=jnp.float32)
                qp_call_geom = jnp.asarray(0.0, dtype=jnp.float32)
                qp_call = qp_call_minibatch + qp_call_retract
                rollout_eval_calls = jnp.asarray(1.0, dtype=jnp.float32)
                outputs = (
                    reward_stage_terminal,
                    Ybar_next,
                    Y_eff,
                    margin,
                    r_p,
                    v_rate,
                    v_mean,
                    gamma,
                    sigma_eff,
                    delta_k,
                    theta_k,
                    eta_j,
                    cvar,
                    qp_call,
                    qp_call_minibatch,
                    qp_call_geom,
                    qp_call_retract,
                    rollout_eval_calls,
                    bundle.active_count.astype(jnp.float32),
                    rho_k.astype(jnp.float32),
                    aug_lam.astype(jnp.float32),
                    qp_prob.astype(jnp.float32),
                    topK_k.astype(jnp.float32),
                    I_qp.astype(jnp.float32),
                    eps_k.astype(jnp.float32),
                    m_eff_f,
                    M_k.astype(jnp.float32),
                    params["compute_cost_hat"].astype(jnp.float32),
                    params["nu"].astype(jnp.float32),
                    pi_multi_k.astype(jnp.float32),
                )
                if use_jax_adaptive:
                    return (rng_curr, Ybar_next, gamma, pi_ema_k, carry_sched_new), outputs
                return (rng_curr, Ybar_next, gamma, pi_ema_k), outputs

            if use_jax_adaptive:
                (_rng_out, Ybar_final, _gamma_final, _pi_ema_final, _carry_final), scan_out = jax.lax.scan(
                    body,
                    (rng, Ybar_init, gamma_init, pi_ema_init, carry_sched_init),
                    diffusion_indices,
                )
            else:
                (_rng_out, Ybar_final, _gamma_final, _pi_ema_final), scan_out = jax.lax.scan(
                    body,
                    (rng, Ybar_init, gamma_init, pi_ema_init),
                    diffusion_indices,
                )
            (
                reward_hist,
                Ybar_hist,
                Y_eff_hist,
                margin_hist,
                r_hist,
                v_rate_hist,
                v_mean_hist,
                gamma_hist,
                sigma_hist,
                delta_hist,
                theta_hist,
                eta_hist,
                cvar_hist,
                qp_call_hist,
                qp_call_minibatch_hist,
                qp_call_geom_hist,
                qp_call_retract_hist,
                rollout_eval_calls_hist,
                activeK_hist,
                rho_hist,
                lambda_hist,
                p_hist,
                topK_hist,
                I_QP_hist,
                eps_hist,
                probe_b_hist,
                m_k_hist,
                compute_cost_hist,
                nu_hist,
                pi_multi_hist,
            ) = scan_out
            return (
                Ybar_final,
                reward_hist[::-1],
                Ybar_hist[::-1],
                Y_eff_hist[::-1],
                margin_hist,
                r_hist,
                v_rate_hist,
                v_mean_hist,
                gamma_hist,
                sigma_hist,
                delta_hist,
                theta_hist,
                eta_hist,
                cvar_hist,
                qp_call_hist,
                qp_call_minibatch_hist,
                qp_call_geom_hist,
                qp_call_retract_hist,
                rollout_eval_calls_hist,
                activeK_hist,
                rho_hist,
                lambda_hist,
                p_hist,
                topK_hist,
                I_QP_hist,
                eps_hist,
                probe_b_hist,
                m_k_hist,
                compute_cost_hist,
                nu_hist,
                pi_multi_hist,
            )

        self._twogo_single_jit = jax.jit(_run_single)
        self._twogo_batch_jit = jax.jit(jax.vmap(_run_single, in_axes=(None, 0, 0)))

    def _targets_per_mode(self, C: int) -> jnp.ndarray:
        if getattr(self._inner, "use_target_line", False) and C > 0:
            target_line = get_d3il_target_line_positions(getattr(self._inner, "num_targets", 4))
            arr = np.asarray([target_line[i % len(target_line)] for i in range(C)], dtype=np.float32)
            return jnp.asarray(arr, dtype=jnp.float32)
        default_tgt = np.asarray(self._inner._default_target, dtype=np.float32).reshape(-1)[: self._inner.position_dim]
        return jnp.tile(jnp.asarray(default_tgt, dtype=jnp.float32), (C, 1))

    def _lambda_hist(self, out: Dict[str, Any], K: int) -> np.ndarray:
        lam = np.asarray(out.get("lambda_hist", []), dtype=np.float32).ravel()
        if lam.size == K:
            return lam
        return np.full(K, float(getattr(self._inner, "aug_lambda", 0.0)), dtype=np.float32)

    def _margin_hist(self, out: Dict[str, Any], K: int) -> np.ndarray:
        margin = np.asarray(out.get("margin_hist", []), dtype=np.float32).ravel()
        if margin.size == K:
            return margin
        return np.full(K, 0.05, dtype=np.float32)

    def _rho_hist(self, out: Dict[str, Any], K: int) -> np.ndarray:
        rho = np.asarray(out.get("rho_hist", []), dtype=np.float32).ravel()
        if rho.size == K:
            return rho
        return np.full(K, float(getattr(self._inner, "aug_rho", 1.0)), dtype=np.float32)

    def _compute_schedule_series(self, out: Dict[str, Any], K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        margin_hist = self._margin_hist(out, K)
        rho_hist = self._rho_hist(out, K)
        params = self._overlay_numpy.compute_series(margin_hist, rho_hist, self.twogo_agp_eta)
        return params.sigma, params.delta, params.theta, params.eta, params.kappa

    def _sample_sdf_grad(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(positions, dtype=np.float32)
        if self._inner.obstacles is None:
            sdf = np.full((pos.shape[0],), 1e3, dtype=np.float32)
            grad = np.zeros((pos.shape[0], pos.shape[1]), dtype=np.float32)
            return sdf, grad

        try:
            if hasattr(self._inner.obstacles, "sample_sdf_and_grad_2d"):
                sdf, grad = self._inner.obstacles.sample_sdf_and_grad_2d(pos, backend="numpy")
                return np.asarray(sdf, dtype=np.float32).reshape(-1), np.asarray(grad, dtype=np.float32).reshape(pos.shape[0], -1)
        except Exception:
            pass

        sdf_vals = []
        grad_vals = []
        for p in pos:
            try:
                sdf_vals.append(float(self._inner.obstacles.sdf(p)))
            except Exception:
                sdf_vals.append(1e3)
            try:
                grad_vals.append(np.asarray(self._inner.obstacles.gradient(p), dtype=np.float32).reshape(-1))
            except Exception:
                grad_vals.append(np.zeros_like(p, dtype=np.float32))
        return np.asarray(sdf_vals, dtype=np.float32), np.asarray(grad_vals, dtype=np.float32)

    def _rollout_states_np(self, x0: Any, actions: np.ndarray) -> np.ndarray:
        st = self._inner._rollout_states_fn(jnp.asarray(x0, dtype=jnp.float32), jnp.asarray(actions, dtype=jnp.float32))
        return np.asarray(st, dtype=np.float32)

    def _rollout_cost_np(self, x0: Any, actions: np.ndarray) -> float:
        rw = self._inner._rollout_rewards_fn(jnp.asarray(x0, dtype=jnp.float32), jnp.asarray(actions, dtype=jnp.float32))
        return float(-np.sum(np.asarray(rw, dtype=np.float32)))

    def _violation_trace(self, states: np.ndarray, clearance: float) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(states[1:, : self._inner.position_dim], dtype=np.float32)
        sdf, grad = self._sample_sdf_grad(pos)
        g_plus = np.maximum(0.0, float(clearance) - sdf).astype(np.float32)
        return g_plus, grad

    def _refine_candidates(self, out: Dict[str, Any], x0: Any) -> Dict[str, Any]:
        if not self.twogo_enable_agp_refine:
            return out

        cand_actions = [np.asarray(a, dtype=np.float32) for a in out.get("candidate_actions", [])]
        cand_states = [np.asarray(s, dtype=np.float32) for s in out.get("candidate_states", [])]
        if not cand_actions:
            cand_actions = [np.asarray(out["actions"], dtype=np.float32)]
            cand_states = [np.asarray(out["states"], dtype=np.float32)]

        K = max(1, len(out.get("r_hist", [])))
        sigma_hist, delta_hist, theta_hist, eta_hist, kappa_hist = self._compute_schedule_series(out, K)
        clearance = float(
            np.asarray(out.get("margin_hist", [0.05]), dtype=np.float32).ravel()[-1]
            if out.get("margin_hist") is not None else 0.05
        )

        # Bind x0 for per-call rollout and cost functions.
        def _violation_fn(states, clr):
            return self._violation_trace(states, clearance=clr)

        def _cost_fn(actions):
            return self._rollout_cost_np(x0, actions)

        def _rollout_fn(actions):
            return self._rollout_states_np(x0, actions)

        schedule_params = {
            "sigma_hist": sigma_hist,
            "delta_hist": delta_hist,
            "theta_hist": theta_hist,
            "eta_hist": eta_hist,
            "kappa_hist": kappa_hist,
            "rollout_fn": _rollout_fn,
        }

        result = self._refinement_pipeline.refine(
            cand_actions, cand_states,
            _violation_fn, _cost_fn, schedule_params,
            clearance=clearance,
        )

        out["candidate_actions"] = [np.asarray(a, dtype=np.float32) for a in result.candidate_actions]
        out["candidate_states"] = [np.asarray(s, dtype=np.float32) for s in result.candidate_states]
        out["candidate_costs"] = result.candidate_costs
        out["best_idx"] = result.best_idx
        out["actions"] = np.asarray(result.candidate_actions[result.best_idx], dtype=np.float32)
        out["states"] = np.asarray(result.candidate_states[result.best_idx], dtype=np.float32)
        out["twogo_window_gamma_hist"] = result.window_gamma_hist
        out["twogo_window_cvar_hist"] = result.window_cvar_hist
        out["twogo_window_delta_hist"] = result.window_delta_hist
        out["twogo_refined_candidate_indices"] = result.refined_indices
        return out

    def _inject_twogo_diagnostics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)

        r_hist = np.asarray(out.get("r_hist", []), dtype=np.float32).ravel()
        v_rate_hist = np.asarray(out.get("v_rate_hist", []), dtype=np.float32).ravel()
        p_hist = np.asarray(out.get("p_hist", []), dtype=np.float32).ravel()
        topK_hist = np.asarray(out.get("topK_hist", []), dtype=np.float32).ravel()

        K = int(r_hist.shape[0])
        if K <= 0:
            out.setdefault("gamma_hist", np.asarray([], dtype=np.float32))
            out.setdefault("sigma_hist", np.asarray([], dtype=np.float32))
            out.setdefault("delta_hist", np.asarray([], dtype=np.float32))
            out.setdefault("theta_hist", np.asarray([], dtype=np.float32))
            out.setdefault("eta_hist", np.asarray([], dtype=np.float32))
            out.setdefault("cvar_hist", np.asarray([], dtype=np.float32))
            out.setdefault("qp_call_hist", np.asarray([], dtype=np.float32))
            out.setdefault("activeK_hist", np.asarray([], dtype=np.float32))
            return out

        sigma_base, delta_hist, theta_hist, eta_hist, _kappa_hist = self._compute_schedule_series(out, K)
        gamma = (v_rate_hist > self.twogo_gate_vrate_threshold).astype(np.float32)
        sigma_hist = gamma * sigma_base

        # cvar proxy uses existing risk residual trace.
        cvar_hist = r_hist.astype(np.float32)

        # qp call proxy from scheduler gate probability and gamma.
        if p_hist.size == K:
            qp_call_hist = ((p_hist > 0.5).astype(np.float32) * gamma).astype(np.float32)
        else:
            qp_call_hist = gamma.astype(np.float32)

        activeK_hist = topK_hist.astype(np.float32) if topK_hist.size == K else np.full(K, np.nan, dtype=np.float32)

        out["gamma_hist"] = gamma
        out["sigma_hist"] = sigma_hist.astype(np.float32)
        out["delta_hist"] = delta_hist.astype(np.float32)
        out["theta_hist"] = theta_hist.astype(np.float32)
        out["eta_hist"] = eta_hist.astype(np.float32)
        out["cvar_hist"] = cvar_hist
        out["qp_call_hist"] = qp_call_hist
        out["activeK_hist"] = activeK_hist
        return out

    def plan(self, x0: Any, rng_key: Any | None = None) -> Dict[str, Any]:
        if not self.twogo_use_jax_scan_core:
            result = self._inner.plan(x0, rng_key)
            result = self._inject_twogo_diagnostics(result)
            result = self._refine_candidates(result, x0)
            return result

        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(getattr(self._inner, "seed", 0)))
        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        target = self._targets_per_mode(1)[0]
        (
            Ybar_final,
            reward_hist,
            actions_traj,
            sampled_traj,
            margin_hist,
            r_hist,
            v_rate_hist,
            v_mean_hist,
            gamma_hist,
            sigma_hist,
            delta_hist,
            theta_hist,
            eta_hist,
            cvar_hist,
            qp_call_hist,
            qp_call_minibatch_hist,
            qp_call_geom_hist,
            qp_call_retract_hist,
            rollout_eval_calls_hist,
            activeK_hist,
            rho_hist,
            lambda_hist,
            p_hist,
            topK_hist,
            I_QP_hist,
            eps_hist,
            probe_b_hist,
            m_k_hist,
            compute_cost_hist,
            nu_hist,
            pi_multi_hist,
        ) = self._twogo_single_jit(x0_jnp, rng_key, target)

        final_actions = jnp.clip(Ybar_final, -self._inner.action_limit, self._inner.action_limit)
        states = self._inner._rollout_states_fn(x0_jnp, final_actions)
        rewards = self._inner._rollout_rewards_with_target_fn(x0_jnp, final_actions, target)
        actions_np = np.asarray(final_actions, dtype=np.float32)
        states_np = np.asarray(states, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        total_cost = float(-np.sum(rewards_np))

        result = {
            "actions": actions_np,
            "states": states_np,
            "rewards": rewards_np,
            "initial_state": states_np[0],
            "reward_history": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_rewards": np.asarray(reward_hist, dtype=np.float32),
            "diffusion_actions_traj": np.asarray(actions_traj, dtype=np.float32),
            "diffusion_sampled_actions": np.asarray(sampled_traj, dtype=np.float32),
            "margin_hist": np.asarray(margin_hist, dtype=np.float32),
            "r_hist": np.asarray(r_hist, dtype=np.float32),
            "v_rate_hist": np.asarray(v_rate_hist, dtype=np.float32),
            "v_mean_hist": np.asarray(v_mean_hist, dtype=np.float32),
            "gamma_hist": np.asarray(gamma_hist, dtype=np.float32),
            "sigma_hist": np.asarray(sigma_hist, dtype=np.float32),
            "delta_hist": np.asarray(delta_hist, dtype=np.float32),
            "theta_hist": np.asarray(theta_hist, dtype=np.float32),
            "eta_hist": np.asarray(eta_hist, dtype=np.float32),
            "cvar_hist": np.asarray(cvar_hist, dtype=np.float32),
            "qp_call_hist": np.asarray(qp_call_hist, dtype=np.float32),
            "qp_call_minibatch_hist": np.asarray(qp_call_minibatch_hist, dtype=np.float32),
            "qp_call_geom_hist": np.asarray(qp_call_geom_hist, dtype=np.float32),
            "qp_call_retract_hist": np.asarray(qp_call_retract_hist, dtype=np.float32),
            "rollout_eval_calls_hist": np.asarray(rollout_eval_calls_hist, dtype=np.float32),
            "activeK_hist": np.asarray(activeK_hist, dtype=np.float32),
            "rho_hist": np.asarray(rho_hist, dtype=np.float32),
            "topK_hist": np.asarray(topK_hist, dtype=np.float32),
            "I_QP_hist": np.asarray(I_QP_hist, dtype=np.float32),
            "eps_hist": np.asarray(eps_hist, dtype=np.float32),
            "m_k_hist": np.asarray(m_k_hist, dtype=np.float32),
            "probe_b_hist": np.asarray(probe_b_hist, dtype=np.float32),
            "lambda_hist": np.asarray(lambda_hist, dtype=np.float32),
            "p_hist": np.asarray(p_hist, dtype=np.float32),
            "nu_hist": np.asarray(nu_hist, dtype=np.float32),
            "compute_cost_hist": np.asarray(compute_cost_hist, dtype=np.float32),
            "pi_multi_hist": np.asarray(pi_multi_hist, dtype=np.float32),
            "candidate_states": [states_np],
            "candidate_actions": [actions_np],
            "candidate_costs": np.asarray([total_cost], dtype=np.float32),
            "best_idx": 0,
            "rng": rng_key,
        }
        return result

    def plan_batch(self, x0: Any, rng_keys: Any) -> List[Dict[str, Any]]:
        if not self.twogo_use_jax_scan_core:
            results = self._inner.plan_batch(x0, rng_keys)
            out = [self._inject_twogo_diagnostics(r) for r in results]
            if self.twogo_enable_agp_batch:
                out = [self._refine_candidates(r, x0) for r in out]
            return out

        x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
        C = int(rng_keys.shape[0])
        targets = self._targets_per_mode(C)
        (
            Ybar_finals,
            reward_hists,
            actions_trajs,
            sampled_trajs,
            margin_hists,
            r_hists,
            v_rate_hists,
            v_mean_hists,
            gamma_hists,
            sigma_hists,
            delta_hists,
            theta_hists,
            eta_hists,
            cvar_hists,
            qp_call_hists,
            qp_call_minibatch_hists,
            qp_call_geom_hists,
            qp_call_retract_hists,
            rollout_eval_calls_hists,
            activeK_hists,
            rho_hists,
            lambda_hists,
            p_hists,
            topK_hists,
            I_QP_hists,
            eps_hists,
            probe_b_hists,
            m_k_hists,
            compute_cost_hists,
            nu_hists,
            pi_multi_hists,
        ) = self._twogo_batch_jit(x0_jnp, rng_keys, targets)

        final_actions_batch = jnp.clip(Ybar_finals, -self._inner.action_limit, self._inner.action_limit)
        states_batch = jax.vmap(self._inner._rollout_states_fn, in_axes=(None, 0))(x0_jnp, final_actions_batch)
        rewards_batch = jax.vmap(self._inner._rollout_rewards_with_target_fn, in_axes=(None, 0, 0))(x0_jnp, final_actions_batch, targets)

        states_np = np.asarray(states_batch, dtype=np.float32)
        actions_np = np.asarray(final_actions_batch, dtype=np.float32)
        rewards_np = np.asarray(rewards_batch, dtype=np.float32)
        costs = -np.sum(rewards_np, axis=-1)

        results: List[Dict[str, Any]] = []
        for i in range(C):
            results.append(
                {
                    "actions": actions_np[i],
                    "states": states_np[i],
                    "rewards": rewards_np[i],
                    "initial_state": states_np[i, 0],
                    "reward_history": np.asarray(reward_hists[i], dtype=np.float32),
                    "diffusion_rewards": np.asarray(reward_hists[i], dtype=np.float32),
                    "diffusion_actions_traj": np.asarray(actions_trajs[i], dtype=np.float32),
                    "diffusion_sampled_actions": np.asarray(sampled_trajs[i], dtype=np.float32),
                    "margin_hist": np.asarray(margin_hists[i], dtype=np.float32),
                    "r_hist": np.asarray(r_hists[i], dtype=np.float32),
                    "v_rate_hist": np.asarray(v_rate_hists[i], dtype=np.float32),
                    "v_mean_hist": np.asarray(v_mean_hists[i], dtype=np.float32),
                    "gamma_hist": np.asarray(gamma_hists[i], dtype=np.float32),
                    "sigma_hist": np.asarray(sigma_hists[i], dtype=np.float32),
                    "delta_hist": np.asarray(delta_hists[i], dtype=np.float32),
                    "theta_hist": np.asarray(theta_hists[i], dtype=np.float32),
                    "eta_hist": np.asarray(eta_hists[i], dtype=np.float32),
                    "cvar_hist": np.asarray(cvar_hists[i], dtype=np.float32),
                    "qp_call_hist": np.asarray(qp_call_hists[i], dtype=np.float32),
                    "qp_call_minibatch_hist": np.asarray(qp_call_minibatch_hists[i], dtype=np.float32),
                    "qp_call_geom_hist": np.asarray(qp_call_geom_hists[i], dtype=np.float32),
                    "qp_call_retract_hist": np.asarray(qp_call_retract_hists[i], dtype=np.float32),
                    "rollout_eval_calls_hist": np.asarray(rollout_eval_calls_hists[i], dtype=np.float32),
                    "activeK_hist": np.asarray(activeK_hists[i], dtype=np.float32),
                    "rho_hist": np.asarray(rho_hists[i], dtype=np.float32),
                    "topK_hist": np.asarray(topK_hists[i], dtype=np.float32),
                    "I_QP_hist": np.asarray(I_QP_hists[i], dtype=np.float32),
                    "eps_hist": np.asarray(eps_hists[i], dtype=np.float32),
                    "m_k_hist": np.asarray(m_k_hists[i], dtype=np.float32),
                    "probe_b_hist": np.asarray(probe_b_hists[i], dtype=np.float32),
                    "lambda_hist": np.asarray(lambda_hists[i], dtype=np.float32),
                    "p_hist": np.asarray(p_hists[i], dtype=np.float32),
                    "nu_hist": np.asarray(nu_hists[i], dtype=np.float32),
                    "compute_cost_hist": np.asarray(compute_cost_hists[i], dtype=np.float32),
                    "pi_multi_hist": np.asarray(pi_multi_hists[i], dtype=np.float32),
                    "candidate_states": [states_np[i]],
                    "candidate_actions": [actions_np[i]],
                    "candidate_costs": np.asarray([float(costs[i])], dtype=np.float32),
                    "best_idx": 0,
                    "rng": rng_keys[i],
                }
            )
        return results

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Any | None = None):
        return self._inner.sample_trajectories(x0, n_samples, rng_key=rng_key)

