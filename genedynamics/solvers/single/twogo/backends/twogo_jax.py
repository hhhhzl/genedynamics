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
from genedynamics.solvers.common.manifold import build_active_rows, project_complement_batch
from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
from genedynamics.core.constraints.core.types import ScheduleState


class TwoGOBackendJax:
    def __init__(self, **kwargs: Any):
        self._inner = CFSMBDBackendJax(**kwargs)

        # 2GO knobs (Phase3-5)
        solver = kwargs.get("solver", None)
        cfg = getattr(solver, "config", {}) if solver is not None else {}
        self.twogo_sigma_max = float(cfg.get("twogo_sigma_max", 0.25))
        self.twogo_gate_vrate_threshold = float(cfg.get("twogo_gate_vrate_threshold", 0.01))  # fallback
        self.twogo_sigma_q = float(cfg.get("twogo_sigma_q", 1.0))
        self.twogo_sigma_q_lambda = float(cfg.get("twogo_sigma_q_lambda", 1.0))
        self.twogo_delta0 = float(cfg.get("twogo_delta0", 0.02))
        self.twogo_delta_r = float(cfg.get("twogo_delta_r", 1.0))
        self.twogo_delta_r_lambda = float(cfg.get("twogo_delta_r_lambda", 1.0))
        self.twogo_lambda0 = float(cfg.get("twogo_lambda0", 1.0))
        self.twogo_theta_start = float(cfg.get("twogo_theta_start", 0.6))
        self.twogo_theta_end = float(cfg.get("twogo_theta_end", 0.2))
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
        self.twogo_probe_frac = float(np.clip(float(cfg.get("twogo_probe_frac", 0.5)), 0.0, 1.0))
        self.twogo_probe_b = cfg.get("twogo_probe_b", None)
        self.twogo_probe_m_cap = cfg.get("twogo_probe_m_cap", None)
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
        gate_default = 1
        Ns = int(inner.Nsample)
        # M_k follows EB-MBD convention: same as full Monte Carlo batch (Nsample), not probe B.
        M_list = [Ns] * Ndiffuse
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

        def _sched_lookup(step_k: jnp.ndarray):
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
            return margin, rho, qp_gate, qp_prob, I_qp, eps

        def _twogo_sched(step_k: jnp.ndarray, rho_k: jnp.ndarray):
            t = step_k.astype(jnp.float32) / jnp.maximum(float(Ndiffuse - 1), 1.0)
            lam = jnp.maximum(rho_k.astype(jnp.float32), 0.0)
            lam_factor = self.twogo_lambda0 / (lam + self.twogo_lambda0 + eps_stab)
            sigma = self.twogo_sigma_max * ((1.0 - t) ** self.twogo_sigma_q) * (lam_factor ** self.twogo_sigma_q_lambda)
            delta = self.twogo_delta0 * ((1.0 - t) ** self.twogo_delta_r) * (lam_factor ** self.twogo_delta_r_lambda)
            theta = self.twogo_theta_start + (self.twogo_theta_end - self.twogo_theta_start) * t
            kappa = 1.0 + 0.01 * lam
            return sigma.astype(jnp.float32), delta.astype(jnp.float32), theta.astype(jnp.float32), kappa.astype(jnp.float32)

        def _cvar_topk(v: jnp.ndarray) -> jnp.ndarray:
            top_vals, _ = jax.lax.top_k(v, n_tail_cvar)
            return jnp.mean(top_vals)

        def _run_single(x0_jnp: jnp.ndarray, rng_key: jnp.ndarray, target: jnp.ndarray):
            rng, _ = jax.random.split(rng_key)
            Ybar_init = jnp.zeros((horizon, act_dim), dtype=jnp.float32)
            gamma_init = jnp.asarray(np.clip(self.twogo_gamma_init, 0.0, 1.0), dtype=jnp.float32)

            def run_probe_pack(
                Y0s_l,
                v_batch_l,
                M_eff_l,
                rng_tail_l,
                rng_rand_l,
                rng_vmap_parent,
                margin_l,
                rho_l,
                eps_l,
                sched_state_l,
                x0_loc,
            ):
                tail_n_l = jnp.clip(
                    jnp.round(M_eff_l.astype(jnp.float32) * jnp.asarray(tail_mix_f, dtype=jnp.float32)).astype(
                        jnp.int32
                    ),
                    0,
                    M_eff_l,
                )
                _, idx_hi_l = jax.lax.top_k(v_batch_l, k_pool_i)
                perm_pool_l = jax.random.permutation(rng_tail_l, k_pool_i)
                shuffled_hi_l = idx_hi_l[perm_pool_l]
                perm_full_l = jax.random.permutation(rng_rand_l, Nsample)
                i_l = jnp.arange(M_max, dtype=jnp.int32)
                valid_l = i_l < M_eff_l
                tail_n_mx = jnp.minimum(tail_n_l, M_max)
                cand_tail_l = shuffled_hi_l[jnp.minimum(i_l, k_pool_i - 1)]
                j_rand_l = i_l - tail_n_mx
                cand_rand_l = perm_full_l[jnp.minimum(jnp.maximum(j_rand_l, 0), Nsample - 1)]
                idx_m_l = jnp.where(i_l < tail_n_mx, cand_tail_l, cand_rand_l)
                Y_g_l = Y0s_l[idx_m_l]
                probe_keys_l = jax.random.split(rng_vmap_parent, M_max)
                I_probe_l = jnp.asarray(1, dtype=jnp.int32)

                def probe_one_l(y, keyp):
                    sp_l = {
                        "margin": margin_l,
                        "rho": rho_l,
                        "qp_gate": jnp.asarray(True, dtype=jnp.bool_),
                        "qp_prob": jnp.asarray(1.0, dtype=jnp.float32),
                        "I_QP": I_probe_l,
                        "eps": eps_l,
                        "topK": topK_default,
                        "rng_key": keyp,
                    }
                    return inner._filter_actions_single_jit(x0_loc, y, sched_state_l, sp_l)

                y_f_l = jax.vmap(probe_one_l)(Y_g_l, probe_keys_l)
                valid3_l = valid_l[:, None, None]
                y_out_l = jnp.where(valid3_l, y_f_l, Y_g_l)
                resid_l = y_out_l - Y_g_l
                wsum_l = jnp.maximum(jnp.sum(valid_l.astype(jnp.float32)), jnp.asarray(1.0, dtype=jnp.float32))
                a_probe_l = jnp.sum(resid_l * valid3_l, axis=0) / wsum_l

                def scatter_step_l(ii, carry_l):
                    Y_c, bm_c = carry_l
                    Y_n = jnp.where(valid_l[ii], Y_c.at[idx_m_l[ii]].set(y_out_l[ii]), Y_c)
                    bm_n = jnp.where(
                        valid_l[ii],
                        bm_c.at[idx_m_l[ii]].set(jnp.asarray(1.0, dtype=jnp.float32)),
                        bm_c,
                    )
                    return (Y_n, bm_n)

                Y_fix_l, _bm_l = jax.lax.fori_loop(
                    0, M_max, scatter_step_l, (Y0s_l, jnp.zeros((Nsample,), dtype=jnp.float32))
                )
                return Y_fix_l, a_probe_l

            def body(carry, idx):
                rng_curr, Ybar_curr, gamma_prev = carry
                rng_curr, noise_key, extra_key, key_sched = jax.random.split(rng_curr, 4)
                k_sched, rng_tail, rng_rand, rng_vmap_parent = jax.random.split(key_sched, 4)
                step_k = jnp.asarray(Ndiffuse - 1, dtype=jnp.int32) - idx
                sched_state = {"k": step_k, "K": total_steps_jnp}

                margin, rho_k, qp_gate, qp_prob, I_qp, eps_k = _sched_lookup(step_k)
                sigma_k, delta_k, theta_k, kappa_k = _twogo_sched(step_k, rho_k)
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
                    "topK": topK_default,
                    "rng_key": k_sched,
                }

                idxs = jnp.arange(horizon, dtype=jnp.int32)
                w_start = (step_k * window_stride) % horizon
                wmask = (((idxs - w_start) % horizon) < window_size).astype(jnp.float32)

                a_geom = inner._constraint_geometry_time_jit(x0_jnp, Ybar_curr, margin)
                a_geom0 = a_geom * wmask[:, None]

                eps_noise = jax.random.normal(noise_key, (Nsample, horizon, act_dim), dtype=jnp.float32)
                Y0s = jnp.clip(eps_noise * sigmas[idx] + Ybar_curr, -action_limit, action_limit)
                rews, v_batch = inner._augmented_and_v_batch_jit(
                    x0_jnp, Y0s, margin, aug_lambda_const, aug_rho_const, target
                )
                r_p = _cvar_topk(v_batch)
                v_rate = jnp.mean(v_batch > 0.0)
                v_mean = jnp.mean(v_batch)
                cvar = _cvar_topk(jnp.maximum(v_batch, 0.0))

                Y_fix, a_probe = jax.lax.cond(
                    do_probe,
                    lambda _: run_probe_pack(
                        Y0s,
                        v_batch,
                        M_eff,
                        rng_tail,
                        rng_rand,
                        rng_vmap_parent,
                        margin,
                        rho_k,
                        eps_k,
                        sched_state,
                        x0_jnp,
                    ),
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

                a_geom1 = a_geom0 + jnp.asarray(probe_alpha, dtype=jnp.float32) * (a_probe * wmask[:, None])
                A_rows, topu, _idx_top, _active_mask_t, _sb = build_active_rows(a_geom1, topk_active, eps_stab)
                K_active = jnp.sum((topu > eps_stab).astype(jnp.float32))
                geom_valid = K_active >= 1.0
                AAT = A_rows @ jnp.transpose(A_rows)
                I_k = jnp.eye(topk_active, dtype=jnp.float32)
                metric_sys = I_k + AAT + eps_stab * I_k
                tan_sys = AAT + eps_stab * I_k

                rew_mean = jnp.mean(rews)
                rew_std = jnp.where(jnp.std(rews) < 1e-4, 1.0, jnp.std(rews))
                T_k = T_k_arr[jnp.clip(step_k, 0, T_k_arr.shape[0] - 1)] if T_k_arr is not None else jnp.asarray(inner.temp_sample, dtype=jnp.float32)
                logp0 = (rews - rew_mean) / (rew_std * T_k)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y_eff)

                sigma_scale = jnp.maximum(sigmas[idx], jnp.asarray(1e-6, dtype=jnp.float32))
                spread = jnp.mean(jnp.std(Y0s - Ybar_curr, axis=0) * wmask[:, None])
                pi_multi = spread / jnp.maximum(jnp.asarray(multi_scale, dtype=jnp.float32) * sigma_scale, jnp.asarray(1e-6, dtype=jnp.float32))
                gamma_geo = (pi_multi > theta_k).astype(jnp.float32)
                gamma_raw = jax.lax.cond(
                    jnp.asarray(enable_local_gating),
                    lambda _: gamma_geo,
                    lambda _: jnp.asarray(1.0, dtype=jnp.float32),
                    operand=None,
                )
                gamma = jax.lax.cond(gate_pass, lambda _: gamma_raw, lambda _: gamma_prev, operand=None)

                score_base = kappa_k * (Ybar_weighted - Ybar_curr)
                u_agp_proj = project_complement_batch(score_base, A_rows, metric_sys)
                u_agp = jax.lax.cond(geom_valid, lambda _: u_agp_proj, lambda _: score_base, operand=None)

                noise_extra = jax.random.normal(extra_key, (horizon, act_dim), dtype=jnp.float32)
                p_noise_proj = project_complement_batch(noise_extra, A_rows, tan_sys)
                p_noise = jax.lax.cond(geom_valid, lambda _: p_noise_proj, lambda _: noise_extra, operand=None)
                sigma_eff = gamma * (sigma_k + extra_sigmas[idx])
                eta_j = jnp.asarray(agp_eta, dtype=jnp.float32)
                Ybar_tilde = Ybar_weighted + eta_j * u_agp + sigma_eff * p_noise

                retract_every = jnp.asarray(retract_every_static, dtype=jnp.int32)
                retract_pass = jnp.equal(jnp.mod(step_k, retract_every), 0)
                retract_highrisk = cvar > delta_k
                boost_j = jnp.asarray(retract_boost, dtype=jnp.bool_)
                do_retract_qp = jnp.logical_and(boost_j, jnp.logical_and(qp_gate_eff, jnp.logical_or(retract_pass, retract_highrisk)))

                sched_params_retract = dict(sched_params)
                sched_params_retract["qp_gate"] = jnp.asarray(True, dtype=jnp.bool_)
                sched_params_retract["qp_prob"] = jnp.asarray(1.0, dtype=jnp.float32)
                Ybar_next = jax.lax.cond(
                    do_retract_qp,
                    lambda y: inner._filter_actions_single_jit(x0_jnp, y, sched_state, sched_params_retract),
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
                return (rng_curr, Ybar_next, gamma), (
                    reward_stage_terminal,
                    Ybar_next,
                    Y_eff,
                    r_p,
                    v_rate,
                    v_mean,
                    gamma,
                    sigma_eff,
                    delta_k,
                    cvar,
                    qp_call,
                    qp_call_minibatch,
                    qp_call_geom,
                    qp_call_retract,
                    rollout_eval_calls,
                    K_active.astype(jnp.float32),
                    rho_k.astype(jnp.float32),
                    qp_prob.astype(jnp.float32),
                    I_qp.astype(jnp.float32),
                    eps_k.astype(jnp.float32),
                    m_eff_f,
                    M_k.astype(jnp.float32),
                )

            (_rng_out, Ybar_final, _gamma_final), scan_out = jax.lax.scan(body, (rng, Ybar_init, gamma_init), diffusion_indices)
            return (Ybar_final,) + tuple(x[::-1] for x in scan_out)

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

    def _compute_schedule_series(self, out: Dict[str, Any], K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 1.0, K, dtype=np.float32)
        lam = self._lambda_hist(out, K)
        lam_factor = (self.twogo_lambda0 / (lam + self.twogo_lambda0 + 1e-8)).astype(np.float32)
        s_sigma = ((1.0 - t) ** self.twogo_sigma_q) * (lam_factor ** self.twogo_sigma_q_lambda)
        sigma_hist = self.twogo_sigma_max * s_sigma
        delta_hist = self.twogo_delta0 * ((1.0 - t) ** self.twogo_delta_r) * (lam_factor ** self.twogo_delta_r_lambda)
        theta_hist = self.twogo_theta_start + (self.twogo_theta_end - self.twogo_theta_start) * t
        return sigma_hist.astype(np.float32), delta_hist.astype(np.float32), theta_hist.astype(np.float32)

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

    def _cvar(self, x: np.ndarray, alpha: float) -> float:
        arr = np.asarray(x, dtype=np.float32).ravel()
        if arr.size == 0:
            return 0.0
        n_tail = max(1, int(np.ceil((1.0 - alpha) * arr.size)))
        part = np.partition(arr, arr.size - n_tail)[arr.size - n_tail :]
        return float(np.mean(part))

    def _window_slices(self, H: int) -> List[Tuple[int, int]]:
        W = max(1, min(self.twogo_window_size, H))
        S = max(1, self.twogo_window_stride)
        slices: List[Tuple[int, int]] = []
        i = 0
        while i < H:
            j = min(H, i + W)
            slices.append((i, j))
            if j == H:
                break
            i += S
        return slices

    def _window_multimodality(self, states_list: List[np.ndarray], a: int, b: int) -> float:
        # Lightweight proxy: normalized spread of window-end positions.
        if len(states_list) <= 1:
            return 0.0
        idx = max(1, min(b, states_list[0].shape[0] - 1))
        pts = []
        for st in states_list:
            p = np.asarray(st[idx, : self._inner.position_dim], dtype=np.float32).reshape(-1)
            pts.append(p)
        P = np.asarray(pts, dtype=np.float32)
        if P.shape[0] <= 1:
            return 0.0
        spread = float(np.mean(np.std(P, axis=0)))
        return float(np.clip(spread / max(self.twogo_multi_scale, 1e-6), 0.0, 1.0))

    def _agp_step(
        self,
        actions: np.ndarray,
        g_plus: np.ndarray,
        grad: np.ndarray,
        window: Tuple[int, int],
        sigma_w: float,
        kappa_w: float,
    ) -> np.ndarray:
        H, U = actions.shape
        d = H * U
        a, b = window
        idx_local = np.argsort(g_plus[a:b])[::-1]
        active_rel = idx_local[: max(1, self.twogo_active_topk)]
        active_t = [a + int(i) for i in active_rel if g_plus[a + int(i)] > 0]
        if not active_t:
            return actions

        K = len(active_t)
        A = np.zeros((K, d), dtype=np.float32)
        w = np.zeros((K,), dtype=np.float32)
        for r, t in enumerate(active_t):
            n = -np.asarray(grad[t], dtype=np.float32).reshape(-1)[:U]
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                n = n / norm
            start = t * U
            A[r, start : start + U] = float(self._inner.dt) * n
            w[r] = float(kappa_w) * float(max(0.0, g_plus[t]))

        x = actions.reshape(-1).astype(np.float32)
        u_raw = -(A.T @ w)

        # Woodbury solve: (I + A^T W A)^-1 u
        W_inv = np.diag(1.0 / np.maximum(w, 1e-6))
        M = W_inv + (A @ A.T)
        Au = A @ u_raw
        y = np.linalg.solve(M + 1e-6 * np.eye(K, dtype=np.float32), Au)
        ginv_u = u_raw - A.T @ y

        # Tangent noise: P xi = xi - A^T (A A^T)^-1 A xi
        xi = self._rng.normal(size=(d,)).astype(np.float32)
        AA = A @ A.T + 1e-6 * np.eye(K, dtype=np.float32)
        z = np.linalg.solve(AA, A @ xi)
        p_xi = xi - A.T @ z

        x_new = x + float(self.twogo_agp_eta) * ginv_u + np.sqrt(max(self.twogo_agp_eta, 1e-8)) * float(sigma_w) * p_xi
        act_new = x_new.reshape(H, U).astype(np.float32)
        act_new = np.clip(act_new, -self._inner.action_limit, self._inner.action_limit)
        return act_new

    def _mini_batch_cfs_local(
        self,
        actions: np.ndarray,
        g_plus: np.ndarray,
        grad: np.ndarray,
        window: Tuple[int, int],
    ) -> np.ndarray:
        a, b = window
        act = np.asarray(actions, dtype=np.float32).copy()
        U = act.shape[1]
        dt = max(float(self._inner.dt), 1e-6)
        for t in range(a, b):
            if g_plus[t] <= 0:
                continue
            n = np.asarray(grad[t], dtype=np.float32).reshape(-1)[:U]
            norm = float(np.linalg.norm(n))
            if norm > 1e-8:
                n = n / norm
            # local retraction proxy in action space
            act[t, :U] += float(self.twogo_cfs_gain) * (float(g_plus[t]) / dt) * n
        return np.clip(act, -self._inner.action_limit, self._inner.action_limit)

    def _refine_candidates(self, out: Dict[str, Any], x0: Any) -> Dict[str, Any]:
        if not self.twogo_enable_agp_refine:
            return out

        cand_actions = [np.asarray(a, dtype=np.float32) for a in out.get("candidate_actions", [])]
        cand_states = [np.asarray(s, dtype=np.float32) for s in out.get("candidate_states", [])]
        if not cand_actions:
            cand_actions = [np.asarray(out["actions"], dtype=np.float32)]
            cand_states = [np.asarray(out["states"], dtype=np.float32)]

        C = len(cand_actions)
        H = int(cand_actions[0].shape[0])
        windows = self._window_slices(H)
        K = max(1, len(out.get("r_hist", [])))
        sigma_hist, delta_hist, theta_hist = self._compute_schedule_series(out, K)
        lam_hist = self._lambda_hist(out, K)
        kappa_hist = 1.0 + 0.01 * lam_hist

        # sample-level gating on tail violations
        viol_tot = []
        traces: List[Tuple[np.ndarray, np.ndarray]] = []
        clearance = float(np.asarray(out.get("margin_hist", [0.05]), dtype=np.float32).ravel()[-1] if out.get("margin_hist") is not None else 0.05)
        for s in cand_states:
            g, grad = self._violation_trace(s, clearance=clearance)
            traces.append((g, grad))
            viol_tot.append(float(np.sum(g)))
        viol_tot_arr = np.asarray(viol_tot, dtype=np.float32)
        if self.twogo_enable_sample_tail:
            n_tail = max(1, int(np.ceil(self.twogo_tail_ratio * C)))
            tail_idx = np.argsort(viol_tot_arr)[::-1][:n_tail]
        else:
            tail_idx = np.arange(C, dtype=np.int32)

        window_gamma = []
        window_cvar = []
        window_delta = []
        refined_indices = []

        for wi, w in enumerate(windows):
            a, b = w
            progress = float((b - 1) / max(1, H - 1))
            sched_idx = int(np.clip(round(progress * (K - 1)), 0, K - 1))
            sigma_w = float(sigma_hist[sched_idx])
            delta_w = float(delta_hist[sched_idx])
            kappa_w = float(kappa_hist[sched_idx])
            theta_w = float(theta_hist[sched_idx])

            pi_multi = self._window_multimodality(cand_states, a, b) if self.twogo_enable_local_gating else 1.0
            gamma_w = 1.0 if pi_multi > theta_w else 0.0

            window_gamma.append(gamma_w)
            window_delta.append(delta_w)

            if gamma_w <= 0:
                window_cvar.append(0.0)
                continue

            cvars = []
            for ci in tail_idx:
                g, grad = traces[int(ci)]
                cvar_w = self._cvar(g[a:b], self.twogo_cvar_alpha)
                cvars.append(cvar_w)
                if cvar_w <= delta_w:
                    continue
                act0 = cand_actions[int(ci)]
                act1 = self._agp_step(act0, g, grad, w, sigma_w=sigma_w, kappa_w=kappa_w)
                act2 = self._mini_batch_cfs_local(act1, g, grad, w)
                st2 = self._rollout_states_np(x0, act2)
                cost2 = self._rollout_cost_np(x0, act2)

                # accept if improves cost or reduces total violation
                g2, grad2 = self._violation_trace(st2, clearance=clearance)
                old_cost = float(out.get("candidate_costs", [np.inf])[int(ci)] if "candidate_costs" in out else np.inf)
                if (cost2 <= old_cost) or (np.sum(g2) < np.sum(g)):
                    cand_actions[int(ci)] = act2
                    cand_states[int(ci)] = st2
                    traces[int(ci)] = (g2, grad2)
                    refined_indices.append(int(ci))
            window_cvar.append(float(np.mean(cvars)) if cvars else 0.0)

        # recompute candidate costs after refinement
        cand_costs = np.asarray([self._rollout_cost_np(x0, a) for a in cand_actions], dtype=np.float32)
        best_idx = int(np.argmin(cand_costs))

        out["candidate_actions"] = [np.asarray(a, dtype=np.float32) for a in cand_actions]
        out["candidate_states"] = [np.asarray(s, dtype=np.float32) for s in cand_states]
        out["candidate_costs"] = cand_costs
        out["best_idx"] = best_idx
        out["actions"] = np.asarray(cand_actions[best_idx], dtype=np.float32)
        out["states"] = np.asarray(cand_states[best_idx], dtype=np.float32)
        out["twogo_window_gamma_hist"] = np.asarray(window_gamma, dtype=np.float32)
        out["twogo_window_cvar_hist"] = np.asarray(window_cvar, dtype=np.float32)
        out["twogo_window_delta_hist"] = np.asarray(window_delta, dtype=np.float32)
        out["twogo_refined_candidate_indices"] = np.asarray(sorted(set(refined_indices)), dtype=np.int32)
        return out

    def _inject_twogo_diagnostics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)

        r_hist = np.asarray(out.get("r_hist", []), dtype=np.float32).ravel()
        v_rate_hist = np.asarray(out.get("v_rate_hist", []), dtype=np.float32).ravel()
        p_hist = np.asarray(out.get("p_hist", []), dtype=np.float32).ravel()
        topK_hist = np.asarray(out.get("topK_hist", []), dtype=np.float32).ravel()
        lambda_hist = np.asarray(out.get("lambda_hist", []), dtype=np.float32).ravel()

        K = int(r_hist.shape[0])
        if K <= 0:
            out.setdefault("gamma_hist", np.asarray([], dtype=np.float32))
            out.setdefault("sigma_hist", np.asarray([], dtype=np.float32))
            out.setdefault("delta_hist", np.asarray([], dtype=np.float32))
            out.setdefault("cvar_hist", np.asarray([], dtype=np.float32))
            out.setdefault("qp_call_hist", np.asarray([], dtype=np.float32))
            out.setdefault("activeK_hist", np.asarray([], dtype=np.float32))
            return out

        sigma_base, delta_hist, _theta_hist = self._compute_schedule_series(out, K)
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
            r_hist,
            v_rate_hist,
            v_mean_hist,
            gamma_hist,
            sigma_hist,
            delta_hist,
            cvar_hist,
            qp_call_hist,
            qp_call_minibatch_hist,
            qp_call_geom_hist,
            qp_call_retract_hist,
            rollout_eval_calls_hist,
            activeK_hist,
            rho_hist,
            p_hist,
            I_QP_hist,
            eps_hist,
            probe_b_hist,
            m_k_hist,
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
            "r_hist": np.asarray(r_hist, dtype=np.float32),
            "v_rate_hist": np.asarray(v_rate_hist, dtype=np.float32),
            "v_mean_hist": np.asarray(v_mean_hist, dtype=np.float32),
            "gamma_hist": np.asarray(gamma_hist, dtype=np.float32),
            "sigma_hist": np.asarray(sigma_hist, dtype=np.float32),
            "delta_hist": np.asarray(delta_hist, dtype=np.float32),
            "cvar_hist": np.asarray(cvar_hist, dtype=np.float32),
            "qp_call_hist": np.asarray(qp_call_hist, dtype=np.float32),
            "qp_call_minibatch_hist": np.asarray(qp_call_minibatch_hist, dtype=np.float32),
            "qp_call_geom_hist": np.asarray(qp_call_geom_hist, dtype=np.float32),
            "qp_call_retract_hist": np.asarray(qp_call_retract_hist, dtype=np.float32),
            "rollout_eval_calls_hist": np.asarray(rollout_eval_calls_hist, dtype=np.float32),
            "activeK_hist": np.asarray(activeK_hist, dtype=np.float32),
            "rho_hist": np.asarray(rho_hist, dtype=np.float32),
            "topK_hist": np.asarray(activeK_hist, dtype=np.float32),
            "I_QP_hist": np.asarray(I_QP_hist, dtype=np.float32),
            "eps_hist": np.asarray(eps_hist, dtype=np.float32),
            "m_k_hist": np.asarray(m_k_hist, dtype=np.float32),
            "probe_b_hist": np.asarray(probe_b_hist, dtype=np.float32),
            "lambda_hist": np.asarray(rho_hist, dtype=np.float32),
            "p_hist": np.asarray(p_hist, dtype=np.float32),
            "nu_hist": np.zeros_like(np.asarray(rho_hist, dtype=np.float32)),
            "compute_cost_hist": (np.asarray(qp_call_hist, dtype=np.float32) * np.asarray(activeK_hist, dtype=np.float32) * np.asarray(I_QP_hist, dtype=np.float32)).astype(np.float32),
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
            r_hists,
            v_rate_hists,
            v_mean_hists,
            gamma_hists,
            sigma_hists,
            delta_hists,
            cvar_hists,
            qp_call_hists,
            qp_call_minibatch_hists,
            qp_call_geom_hists,
            qp_call_retract_hists,
            rollout_eval_calls_hists,
            activeK_hists,
            rho_hists,
            p_hists,
            I_QP_hists,
            eps_hists,
            probe_b_hists,
            m_k_hists,
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
                    "r_hist": np.asarray(r_hists[i], dtype=np.float32),
                    "v_rate_hist": np.asarray(v_rate_hists[i], dtype=np.float32),
                    "v_mean_hist": np.asarray(v_mean_hists[i], dtype=np.float32),
                    "gamma_hist": np.asarray(gamma_hists[i], dtype=np.float32),
                    "sigma_hist": np.asarray(sigma_hists[i], dtype=np.float32),
                    "delta_hist": np.asarray(delta_hists[i], dtype=np.float32),
                    "cvar_hist": np.asarray(cvar_hists[i], dtype=np.float32),
                    "qp_call_hist": np.asarray(qp_call_hists[i], dtype=np.float32),
                    "qp_call_minibatch_hist": np.asarray(qp_call_minibatch_hists[i], dtype=np.float32),
                    "qp_call_geom_hist": np.asarray(qp_call_geom_hists[i], dtype=np.float32),
                    "qp_call_retract_hist": np.asarray(qp_call_retract_hists[i], dtype=np.float32),
                    "rollout_eval_calls_hist": np.asarray(rollout_eval_calls_hists[i], dtype=np.float32),
                    "activeK_hist": np.asarray(activeK_hists[i], dtype=np.float32),
                    "rho_hist": np.asarray(rho_hists[i], dtype=np.float32),
                    "topK_hist": np.asarray(activeK_hists[i], dtype=np.float32),
                    "I_QP_hist": np.asarray(I_QP_hists[i], dtype=np.float32),
                    "eps_hist": np.asarray(eps_hists[i], dtype=np.float32),
                    "m_k_hist": np.asarray(m_k_hists[i], dtype=np.float32),
                    "probe_b_hist": np.asarray(probe_b_hists[i], dtype=np.float32),
                    "lambda_hist": np.asarray(rho_hists[i], dtype=np.float32),
                    "p_hist": np.asarray(p_hists[i], dtype=np.float32),
                    "nu_hist": np.zeros_like(np.asarray(rho_hists[i], dtype=np.float32)),
                    "compute_cost_hist": (np.asarray(qp_call_hists[i], dtype=np.float32) * np.asarray(activeK_hists[i], dtype=np.float32) * np.asarray(I_QP_hists[i], dtype=np.float32)).astype(np.float32),
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

