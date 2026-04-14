"""
JAX backend for MBD3D (3DGS robust mapping).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

import jax
import jax.numpy as jnp
import numpy as np

from genedynamics.core.prob.gaussian_lowrank import woodbury_solve
from genedynamics.core.types import Trajectory
from ..types import ObservationBundle


@jax.jit
def _normalize_quat(q: jnp.ndarray) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


class MBD3DBackendJax:
    def __init__(
        self,
        *,
        scene_repr: Any,
        renderer: Any,
        likelihood: Any,
        bridge_schedule: Any,
        camera_prior: Optional[Any] = None,
        horizon: int = 10,
        n_gaussians: int = 32,
        M: int = 16,
        sigma_mcsa: float = 0.05,
        ess_min: float = 1.0,
        seed: int = 0,
        show_tqdm: bool = False,
        fidelity_ladder: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.scene_repr = scene_repr
        self.renderer = renderer
        self.likelihood = likelihood
        self.bridge_schedule = bridge_schedule
        self.camera_prior = camera_prior
        self.horizon = int(horizon)
        self.n_gaussians = int(n_gaussians)
        self.M = int(M)
        self.sigma_mcsa = float(sigma_mcsa)
        self.ess_min = float(ess_min)
        self.seed = int(seed)
        self.show_tqdm = bool(show_tqdm)
        self.fidelity_ladder = fidelity_ladder
        self.config = kwargs

        self._fix_cameras = bool(kwargs.get("fix_cameras", True))
        self._initialization_mode = str(kwargs.get("initialization_mode", "prior_center"))
        self._init_jitter_scale = float(kwargs.get("init_jitter_scale", 1.0))
        self._enable_subspace = bool(kwargs.get("enable_subspace", True))
        self._subspace_rank = int(kwargs.get("subspace_rank", 64))
        self._subspace_rank_start = int(kwargs.get("subspace_rank_start", self._subspace_rank))
        self._subspace_rank_end = int(kwargs.get("subspace_rank_end", self._subspace_rank))
        self._subspace_power_iters = int(kwargs.get("subspace_power_iters", 2))
        self._subspace_refresh_every = int(kwargs.get("subspace_refresh_every", 1))
        self._subspace_refresh_every_start = int(
            kwargs.get("subspace_refresh_every_start", self._subspace_refresh_every)
        )
        self._subspace_refresh_every_end = int(
            kwargs.get("subspace_refresh_every_end", self._subspace_refresh_every)
        )
        self._proposal_count_start = int(kwargs.get("proposal_count_start", self.M))
        self._proposal_count_end = int(kwargs.get("proposal_count_end", self.M))
        self._profiling = bool(kwargs.get("profiling", True))
        self._subspace_oversample = int(kwargs.get("subspace_oversample", 2))
        self._compile_stable_shapes = bool(kwargs.get("compile_stable_shapes", True))

        self._scene_dim = int(scene_repr.dim())
        self._camera_pose_dim = 7
        self._camera_dim = (self.horizon + 1) * self._camera_pose_dim
        self._theta_dim = self._scene_dim + (0 if self._fix_cameras else self._camera_dim)

    def _downsample_images(self, images: jnp.ndarray, factor: int) -> jnp.ndarray:
        f = int(max(1, factor))
        if f == 1:
            return images
        n, h, w, c = [int(v) for v in images.shape]
        h2 = (h // f) * f
        w2 = (w // f) * f
        x = images[:, :h2, :w2, :]
        x = jnp.reshape(x, (n, h2 // f, f, w2 // f, f, c))
        return jnp.mean(x, axis=(2, 4))

    def _scale_intrinsics(self, intrinsics: Optional[jnp.ndarray], factor: int, n_views: int) -> Optional[jnp.ndarray]:
        if intrinsics is None:
            return None
        f = float(max(1, factor))
        K = jnp.asarray(intrinsics, dtype=jnp.float32)
        if K.ndim == 2:
            K = jnp.tile(K[None, :, :], (n_views, 1, 1))
        K = K.at[:, 0, 0].set(K[:, 0, 0] / f)
        K = K.at[:, 1, 1].set(K[:, 1, 1] / f)
        K = K.at[:, 0, 2].set(K[:, 0, 2] / f)
        K = K.at[:, 1, 2].set(K[:, 1, 2] / f)
        return K

    def _build_fidelity_schedule(self, K: int) -> List[Tuple[float, int]]:
        ladder = self.fidelity_ladder
        if not ladder:
            return [(1.0, 1)]
        if not isinstance(ladder, (list, tuple)):
            return [(1.0, 1)]
        parsed: List[Tuple[float, int]] = []
        for item in ladder:
            if not isinstance(item, dict):
                continue
            end_frac = float(item.get("end_frac", item.get("until_frac", 1.0)))
            downsample = int(item.get("downsample", item.get("render_downsample", 1)))
            parsed.append((min(max(end_frac, 0.0), 1.0), max(1, downsample)))
        if not parsed:
            return [(1.0, 1)]
        parsed = sorted(parsed, key=lambda x: x[0])
        if parsed[-1][0] < 1.0:
            parsed.append((1.0, parsed[-1][1]))
        return parsed

    def _build_log_prior(self):
        def log_prior(theta: jnp.ndarray) -> jnp.ndarray:
            scene_flat = theta[: self._scene_dim]
            scene = self.scene_repr.unflatten(scene_flat)
            lp_scene = jnp.asarray(self.scene_repr.prior_log_prob(scene), dtype=jnp.float32)
            if self._fix_cameras:
                return lp_scene
            cam_flat = theta[self._scene_dim :]
            if self.camera_prior is not None:
                cam_poses = jnp.reshape(cam_flat, (self.horizon + 1, self._camera_pose_dim))
                lp_cam = jnp.asarray(self.camera_prior.log_prob(cam_poses), dtype=jnp.float32)
                return lp_scene + lp_cam
            return lp_scene - 0.5 * jnp.sum(jnp.square(cam_flat)) / 10.0

        return jax.jit(log_prior)

    def _build_model_fns(
        self,
        obs_images: jnp.ndarray,
        cam_fixed: jnp.ndarray,
        intrinsics: Optional[jnp.ndarray],
        render_downsample: int = 1,
    ):
        obs_flat = jnp.ravel(obs_images)
        use_lowrank = bool(getattr(self.likelihood, "use_lowrank", False))
        sigma2 = float(getattr(self.likelihood, "sigma2", 0.01))
        if use_lowrank:
            basis_existing = getattr(self.likelihood, "lowrank_basis", None)
            if basis_existing is not None and hasattr(basis_existing, "shape"):
                p_expected = int(np.asarray(obs_images).size)
                p_basis = int(basis_existing.shape[0])
                if p_basis != p_expected and getattr(self.likelihood, "_basis_key", None) is not None:
                    # Auto-generated basis from another resolution; rebuild for current fidelity.
                    self.likelihood.lowrank_basis = None
                    self.likelihood._cov = None
                    self.likelihood._logdet_cache = None
                    self.likelihood._basis_key = None
            obs_for_basis = ObservationBundle(
                images=np.asarray(obs_images, dtype=np.float32),
                camera_poses=np.asarray(cam_fixed, dtype=np.float32),
                intrinsics=np.asarray(intrinsics, dtype=np.float32) if intrinsics is not None else None,
            )
            self.likelihood.ensure_lowrank_basis(obs_for_basis)
        lowrank_basis = getattr(self.likelihood, "lowrank_basis", None)
        A = (
            jnp.asarray(lowrank_basis, dtype=jnp.float32)
            if (use_lowrank and lowrank_basis is not None)
            else None
        )
        logdet_sigma = (
            float(getattr(self.likelihood, "_logdet_cache", None))
            if use_lowrank and A is not None
            else float(obs_flat.size) * float(np.log(sigma2))
        )
        cam_w2c_fixed = None
        if self._fix_cameras and hasattr(self.renderer, "camera_poses_to_w2c"):
            cam_w2c_fixed = self.renderer.camera_poses_to_w2c(cam_fixed)

        def unpack(theta: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
            scene_flat = theta[: self._scene_dim]
            scene = self.scene_repr.unflatten(scene_flat)
            if self._fix_cameras:
                cam_poses = cam_fixed
            else:
                cam_poses = jnp.reshape(theta[self._scene_dim :], (self.horizon + 1, self._camera_pose_dim))
            return scene, cam_poses

        def render_flat(theta: jnp.ndarray) -> jnp.ndarray:
            scene, cam_poses = unpack(theta)
            pred = self.renderer.render(
                scene,
                cam_poses,
                intrinsics,
                camera_w2c=cam_w2c_fixed if self._fix_cameras else None,
                render_downsample=render_downsample,
            )
            return jnp.ravel(jnp.asarray(pred, dtype=jnp.float32))

        def sigma_inv_apply(v: jnp.ndarray) -> jnp.ndarray:
            if use_lowrank and A is not None:
                return jnp.asarray(woodbury_solve(A, sigma2, v, backend="jax"), dtype=jnp.float32)
            return v / sigma2

        def log_likelihood(theta: jnp.ndarray) -> jnp.ndarray:
            residual = obs_flat - render_flat(theta)
            qf = jnp.dot(residual, sigma_inv_apply(residual))
            return -0.5 * qf - 0.5 * jnp.asarray(logdet_sigma, dtype=jnp.float32)

        return render_flat, sigma_inv_apply, jax.jit(log_likelihood)

    def _estimate_subspace(
        self,
        theta: jnp.ndarray,
        rank: int,
        key: jnp.ndarray,
        render_flat_fn: Any,
        sigma_inv_apply_fn: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = int(theta.shape[0])
        r = int(min(max(1, rank), d))
        l = int(min(d, r + max(0, self._subspace_oversample)))

        if (not self._enable_subspace) or r >= d:
            return jnp.eye(d, dtype=jnp.float32), jnp.ones((d,), dtype=jnp.float32)

        _, vjp_fn = jax.vjp(render_flat_fn, theta)

        def g_matvec(v: jnp.ndarray) -> jnp.ndarray:
            _, jv = jax.jvp(render_flat_fn, (theta,), (v,))
            w = sigma_inv_apply_fn(jv)
            return jnp.asarray(vjp_fn(w)[0], dtype=jnp.float32)

        key, k_omega = jax.random.split(key)
        omega = jax.random.normal(k_omega, (d, l), dtype=jnp.float32)
        Q, _ = jnp.linalg.qr(omega, mode="reduced")
        for _ in range(max(1, self._subspace_power_iters)):
            Y = jax.vmap(g_matvec, in_axes=1, out_axes=1)(Q)
            Q, _ = jnp.linalg.qr(Y, mode="reduced")

        Z = jax.vmap(g_matvec, in_axes=1, out_axes=1)(Q)
        B = 0.5 * (Q.T @ Z + (Q.T @ Z).T)
        evals, evecs = jnp.linalg.eigh(B)
        idx = jnp.argsort(evals)[::-1]
        evals_sorted = evals[idx]
        U = Q @ evecs[:, idx][:, :r]
        return jnp.asarray(U, dtype=jnp.float32), jnp.asarray(evals_sorted[:r], dtype=jnp.float32)

    def _bridge_loop(self, rng: jnp.ndarray, theta_init: jnp.ndarray, observations: Any):
        if not self.renderer.supports_jax():
            raise RuntimeError("MBD3D posterior bridge requires a JAX-differentiable renderer.")

        cam_fixed = jnp.asarray(observations.camera_poses[: self.horizon + 1], dtype=jnp.float32)
        obs_images_full = jnp.asarray(observations.images[: cam_fixed.shape[0]], dtype=jnp.float32)
        intrinsics_full = (
            jnp.asarray(observations.intrinsics, dtype=jnp.float32)
            if observations.intrinsics is not None
            else None
        )
        K = int(self.bridge_schedule.config.K)
        fidelity_schedule = self._build_fidelity_schedule(K)
        fidelity_levels = sorted(set(int(ds) for _, ds in fidelity_schedule))
        model_fns: Dict[int, Tuple[Any, Any, Any, jnp.ndarray]] = {}
        for ds in fidelity_levels:
            obs_ds = self._downsample_images(obs_images_full, ds)
            intrinsics_ds = self._scale_intrinsics(intrinsics_full, ds, int(obs_ds.shape[0]))
            model_fns[ds] = (*self._build_model_fns(obs_ds, cam_fixed, intrinsics_ds, render_downsample=ds), obs_ds)

        active_ds = fidelity_levels[0]
        render_flat_fn, sigma_inv_apply_fn, log_likelihood_fn, obs_images = model_fns[active_ds]
        log_prior_fn = self._build_log_prior()

        # Clean-state target log π_1 = log p_0 + log p(y|θ). No β tempering:
        # Eq 13 fixes the clean posterior; the "annealing" in MBD is encoded
        # in the forward noise schedule ᾱ_k (Eq 17, Eq 19), not in a
        # likelihood exponent. Tempering the likelihood breaks the Tweedie
        # identity that Eq 21 relies on.
        def _make_log_pi(log_likelihood_local: Any):
            def _log_pi(theta: jnp.ndarray) -> jnp.ndarray:
                return log_prior_fn(theta) + log_likelihood_local(theta)
            return _log_pi

        log_pi_fns: Dict[int, Any] = {ds: _make_log_pi(model_fns[ds][2]) for ds in fidelity_levels}
        use_lax_map_logpi = bool(getattr(self.renderer, "requires_non_batched_logpi", False))
        if use_lax_map_logpi:
            def _make_mapped_log_pi(log_pi_single: Any):
                @jax.jit
                def _mapped(theta_batch: jnp.ndarray) -> jnp.ndarray:
                    return jax.lax.map(lambda th: log_pi_single(th), theta_batch)
                return _mapped

            log_pi_batched_fns: Dict[int, Any] = {
                ds: _make_mapped_log_pi(log_pi_fns[ds]) for ds in fidelity_levels
            }
        else:
            log_pi_batched_fns: Dict[int, Any] = {
                ds: jax.jit(jax.vmap(log_pi_fns[ds], in_axes=0)) for ds in fidelity_levels
            }

        # DDPM noise schedule for the forward kernel q(θ_k|θ_1). Indexed 0..K-1
        # from clean-end (large ᾱ) to noisy-end (small ᾱ). Reverse diffusion
        # walks step_idx 0..K-1 with k_noise = (K-1)-step_idx.
        ddpm_beta0 = float(self.config.get("ddpm_beta0", 1e-4))
        ddpm_betaT = float(self.config.get("ddpm_betaT", 1e-2))
        betas_ddpm_np = np.linspace(ddpm_beta0, ddpm_betaT, K, dtype=np.float32)
        alphas_bar_np = np.cumprod(1.0 - betas_ddpm_np, axis=0).astype(np.float32)
        theta = jnp.asarray(theta_init, dtype=jnp.float32)
        history: Dict[str, List[float]] = {
            "ess": [], "score_norm": [], "log_pi": [], "beta": [],
            "sigma": [], "subspace_rank": [], "subspace_energy": [],
            "proposal_count": [],
            "fidelity_downsample": [],
            "step_time_ms": [],
            "subspace_time_ms": [],
            "proposal_eval_time_ms": [],
            "update_time_ms": [],
            "proposals_per_sec": [],
            "views_per_sec": [],
            "alpha_bar": [],
        }

        U = None
        eigvals = None
        prev_ess = None
        ess_min = float(self.ess_min)
        ess_adaptive = ess_min > 1.0
        iter_range = range(K)
        if self.show_tqdm:
            try:
                import tqdm
                iter_range = tqdm.tqdm(iter_range, desc="MBD bridge", unit="step")
            except Exception:
                pass

        n_views = int(obs_images.shape[0])
        m_cap = int(max(2, self.M, self._proposal_count_start, self._proposal_count_end))
        proposal_indices = jnp.arange(m_cap, dtype=jnp.int32)
        rank_cap = int(min(self._theta_dim, max(1, self._subspace_rank, self._subspace_rank_start, self._subspace_rank_end)))
        rank_indices = jnp.arange(rank_cap, dtype=jnp.int32)

        def _linear_int(start: int, end: int, k_step: int, total: int) -> int:
            if total <= 1:
                return int(start)
            alpha = float(k_step) / float(total - 1)
            return int(round((1.0 - alpha) * float(start) + alpha * float(end)))

        # MBD reverse step in clean-state coordinates. theta_k here denotes
        # the current clean-state estimate θ̄ (= θ_{noisy} / √ᾱ_k). Candidates
        # are drawn from the forward kernel N(θ̄, (1/ᾱ_k − 1)·U U^T) as in
        # Eq 19 / Eq 23, then weighted under the fixed clean target p_1
        # (Eq 20). Because the DDPM reverse update
        #     θ_{k−1,noisy} = (θ_{k,noisy} + (1−ᾱ_k)·S) / √α_k
        # followed by dividing by √ᾱ_{k−1} collapses algebraically to the
        # weighted clean-state mean Σ w_m·θ̃_m, we simply return that mean
        # (plus optional subspace exploration noise).
        def _make_proposal_step(log_pi_batched_local: Any):
            @jax.jit
            def _proposal_update_step(
                theta_k: jnp.ndarray,
                U_k: jnp.ndarray,
                scale_prop: float,
                extra_sigma: float,
                temperature: float,
                m_k: int,
                rk_active: int,
                key_eps: jnp.ndarray,
                key_noise: jnp.ndarray,
            ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                eps = jax.random.normal(key_eps, (m_cap, rank_cap), dtype=jnp.float32)
                rank_mask = (rank_indices < rk_active).astype(jnp.float32)
                eps_eff = eps * rank_mask[None, :]
                proposals = theta_k[None, :] + scale_prop * (eps_eff @ U_k.T)
                log_probs = log_pi_batched_local(proposals)
                prop_mask = proposal_indices < m_k
                masked_log_probs = jnp.where(prop_mask, log_probs, -jnp.inf)
                # Softmax temperature T_k (cf. /MBD T_k_arr): scales the
                # spread of weights without changing the target distribution.
                centered = (masked_log_probs - jnp.max(masked_log_probs)) / jnp.maximum(temperature, 1e-8)
                weights = jax.nn.softmax(centered)
                weights = weights * prop_mask.astype(jnp.float32)
                weights = weights / (jnp.sum(weights) + 1e-12)
                theta_bar_next = jnp.sum(weights[:, None] * proposals, axis=0)
                noise = jax.random.normal(key_noise, (rank_cap,), dtype=jnp.float32) * rank_mask
                xi = extra_sigma * (U_k @ noise)
                theta_next = theta_bar_next + xi
                score_dir = theta_bar_next - theta_k
                ess = 1.0 / (jnp.sum(jnp.square(weights)) + 1e-12)
                return theta_next, ess, score_dir

            return _proposal_update_step

        proposal_step_fns: Dict[int, Any] = {
            ds: _make_proposal_step(log_pi_batched_fns[ds]) for ds in fidelity_levels
        }
        estimate_subspace_fns: Dict[int, Any] = {}
        if self._compile_stable_shapes:
            for ds in fidelity_levels:
                render_flat_ds, sigma_inv_ds, _, _ = model_fns[ds]
                estimate_subspace_fns[ds] = jax.jit(
                    lambda theta_k, key_k, rf=render_flat_ds, sf=sigma_inv_ds: self._estimate_subspace(
                        theta_k,
                        rank_cap,
                        key_k,
                        rf,
                        sf,
                    )
                )

        for step_idx in iter_range:
            t0 = time.perf_counter()
            progress = float(step_idx + 1) / float(max(1, K))
            for end_frac, ds in fidelity_schedule:
                if progress <= end_frac:
                    active_ds = ds
                    break
            if model_fns[active_ds][2] is not log_likelihood_fn:
                render_flat_fn, sigma_inv_apply_fn, log_likelihood_fn, obs_images = model_fns[active_ds]
                n_views = int(obs_images.shape[0])

            # Reverse-diffusion noise-level index: early step_idx = high noise
            # (small ᾱ_k); late step_idx = clean (ᾱ_k → 1).
            k_noise = (K - 1) - int(step_idx)
            ab_k = float(max(alphas_bar_np[k_noise], 1e-8))
            one_minus_ab = max(1.0 - ab_k, 1e-8)
            scale_prop = float(np.sqrt(max(1.0 / ab_k - 1.0, 0.0)))
            # τ_k is reused from the bridge schedule as the decayed extra
            # exploration noise; it no longer drives a separate MCSA update.
            tau = float(self.bridge_schedule.tau(step_idx))
            temperature_k = float(self.config.get("temperature", 1.0))

            # ESS-adaptive widening of the proposal scale (biased but stabilizing).
            if ess_adaptive and prev_ess is not None and ess_min > 0 and prev_ess < ess_min:
                scale_boost = 1.0 + 0.4 * max(0.0, 1.0 - float(prev_ess) / ess_min)
                scale_prop = scale_prop * scale_boost

            rank_k = _linear_int(self._subspace_rank_start, self._subspace_rank_end, step_idx, K)
            rank_k = int(min(max(1, rank_k), self._theta_dim))
            m_k = _linear_int(self._proposal_count_start, self._proposal_count_end, step_idx, K)
            m_k = int(max(2, m_k))
            refresh_every_k = _linear_int(
                self._subspace_refresh_every_start,
                self._subspace_refresh_every_end,
                step_idx,
                K,
            )
            refresh_every_k = int(max(1, refresh_every_k))

            t_sub0 = time.perf_counter()
            if (U is None) or (step_idx % refresh_every_k == 0):
                rng, key_u = jax.random.split(rng)
                subspace_rank = rank_cap if self._compile_stable_shapes else rank_k
                if self._compile_stable_shapes:
                    U, eigvals = estimate_subspace_fns[active_ds](theta, key_u)
                else:
                    U, eigvals = self._estimate_subspace(theta, subspace_rank, key_u, render_flat_fn, sigma_inv_apply_fn)
                if int(U.shape[1]) < rank_cap:
                    pad = rank_cap - int(U.shape[1])
                    U = jnp.pad(U, ((0, 0), (0, pad)))
                    eigvals = jnp.pad(eigvals, (0, pad))
                elif int(U.shape[1]) > rank_cap:
                    U = U[:, :rank_cap]
                    eigvals = eigvals[:rank_cap]
            t_sub_ms = (time.perf_counter() - t_sub0) * 1e3

            rk = int(U.shape[1])
            rk_active = int(min(rank_k, rk))
            rng, key_eps, key_noise = jax.random.split(rng, 3)
            t_prop0 = time.perf_counter()
            theta, ess, score_dir = proposal_step_fns[active_ds](
                theta, U, scale_prop, tau, temperature_k, m_k, rk_active, key_eps, key_noise
            )
            t_prop_ms = (time.perf_counter() - t_prop0) * 1e3
            prev_ess = float(np.asarray(ess))

            t_upd_ms = 0.0

            lp = log_pi_fns[active_ds](theta)
            energy = float(jnp.sum(jnp.maximum(eigvals, 0.0))) if eigvals is not None else 0.0
            # Diagnostic score norm via Eq 21 coefficient √ᾱ_k / (1 − ᾱ_k).
            score_coeff = float(np.sqrt(ab_k)) / one_minus_ab
            score_norm_val = float(np.asarray(jnp.linalg.norm(score_dir))) * score_coeff
            step_ms = (time.perf_counter() - t0) * 1e3
            pps = float(m_k / max(t_prop_ms / 1e3, 1e-9))
            vps = float((m_k * n_views) / max(t_prop_ms / 1e3, 1e-9))
            history["ess"].append(float(np.asarray(ess)))
            history["score_norm"].append(score_norm_val)
            history["log_pi"].append(float(np.asarray(lp)))
            history["beta"].append(progress)
            history["sigma"].append(scale_prop)
            history["subspace_rank"].append(float(rk_active))
            history["subspace_energy"].append(energy)
            history["proposal_count"].append(float(m_k))
            history["fidelity_downsample"].append(float(active_ds))
            history["step_time_ms"].append(step_ms)
            history["subspace_time_ms"].append(t_sub_ms)
            history["proposal_eval_time_ms"].append(t_prop_ms)
            history["update_time_ms"].append(t_upd_ms)
            history["proposals_per_sec"].append(pps)
            history["views_per_sec"].append(vps)
            history["alpha_bar"].append(ab_k)
            if self.show_tqdm and self._profiling and hasattr(iter_range, "set_postfix"):
                iter_range.set_postfix(
                    ess=f"{float(np.asarray(ess)):.2f}",
                    ms=f"{step_ms:.1f}",
                    pps=f"{pps:.1f}",
                )

        if self._profiling and history["step_time_ms"]:
            history["perf_summary"] = {
                "step_time_ms_mean": float(np.mean(history["step_time_ms"])),
                "step_time_ms_p95": float(np.percentile(history["step_time_ms"], 95)),
                "subspace_time_ms_mean": float(np.mean(history["subspace_time_ms"])),
                "proposal_eval_time_ms_mean": float(np.mean(history["proposal_eval_time_ms"])),
                "proposals_per_sec_mean": float(np.mean(history["proposals_per_sec"])),
                "views_per_sec_mean": float(np.mean(history["views_per_sec"])),
            }
        return theta, history

    def _render_eval(
        self,
        scene_final: Any,
        cam_final: Any,
        observations: ObservationBundle,
    ) -> Dict[str, Any]:
        """Render predicted images from final scene and compute PSNR/LPIPS."""
        extras: Dict[str, Any] = {}
        try:
            gt_images = np.asarray(observations.images, dtype=np.float32)
            cam_poses = jnp.asarray(cam_final, dtype=jnp.float32)
            intrinsics = getattr(observations, "intrinsics", None)
            pred_jax = self.renderer.render(
                scene_final, cam_poses, intrinsics=intrinsics,
            )
            pred_images = np.asarray(pred_jax, dtype=np.float32)
            pred_images = np.clip(pred_images, 0.0, 1.0)

            # Align shapes — GT may have more views than cam_final
            n = min(gt_images.shape[0], pred_images.shape[0])
            gt_crop = gt_images[:n]
            pred_crop = pred_images[:n]

            # Resize GT to match pred resolution if needed
            if gt_crop.shape[1:3] != pred_crop.shape[1:3]:
                from skimage.transform import resize as sk_resize
                gt_resized = np.stack([
                    sk_resize(gt_crop[i], pred_crop.shape[1:3], anti_aliasing=True).astype(np.float32)
                    for i in range(n)
                ])
            else:
                gt_resized = gt_crop

            # PSNR
            mse = float(np.mean((pred_crop - gt_resized) ** 2))
            psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))

            # Per-view PSNR
            per_view_psnr = []
            for i in range(n):
                v_mse = float(np.mean((pred_crop[i] - gt_resized[i]) ** 2))
                per_view_psnr.append(float(10.0 * np.log10(1.0 / max(v_mse, 1e-10))))

            # LPIPS (optional)
            lpips_val = -1.0
            try:
                import lpips as _lpips
                loss_fn = _lpips.LPIPS(net="alex", verbose=False)
                import torch
                p = torch.from_numpy((pred_crop * 2 - 1).transpose(0, 3, 1, 2))
                g = torch.from_numpy((gt_resized * 2 - 1).transpose(0, 3, 1, 2))
                with torch.no_grad():
                    lpips_val = float(loss_fn(p, g).mean())
            except Exception:
                pass

            # Store for export_3dgs_figures.py
            extras["images"] = gt_resized.tolist()
            extras["predicted_images"] = pred_crop.tolist()
            extras["metrics_3dgs"] = {
                "psnr": float(psnr),
                "lpips": lpips_val,
                "per_view_psnr": per_view_psnr,
            }
        except Exception as e:
            import traceback
            print(f"[mbd3d] Warning: eval rendering failed: {e}")
            traceback.print_exc()
        return extras

    def plan(self, x0: Any, rng_key: Optional[Any] = None, observations: Optional[Any] = None) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        if observations is None:
            raise ValueError("MBD3D requires observations for likelihood.")

        rng_key, key_scene, key_cam = jax.random.split(rng_key, 3)
        initial_scene = getattr(observations, "initial_scene", None)
        if initial_scene is not None and hasattr(self.scene_repr, "set_prior_center"):
            self.scene_repr.set_prior_center(initial_scene)

        init_mode = self._initialization_mode.lower().strip()
        if init_mode not in ("prior_center", "direct", "random"):
            init_mode = "prior_center"

        if init_mode == "direct" and initial_scene is not None:
            scene_flat = self.scene_repr.flatten(initial_scene)
            if int(scene_flat.size) != self._scene_dim:
                scene_flat = self.scene_repr.flatten(
                    self.scene_repr.sample_prior(
                        key_scene, self.n_gaussians, jitter_scale=self._init_jitter_scale
                    )
                )
        else:
            scene_flat = self.scene_repr.flatten(
                self.scene_repr.sample_prior(
                    key_scene,
                    self.n_gaussians,
                    center_params=initial_scene if init_mode == "prior_center" else None,
                    jitter_scale=self._init_jitter_scale,
                )
            )

        if self._fix_cameras:
            theta_init = jnp.asarray(scene_flat, dtype=jnp.float32)
            cam_fixed = jnp.asarray(observations.camera_poses[: self.horizon + 1], dtype=jnp.float32)
        else:
            if x0 is not None and hasattr(x0, "shape"):
                x0_arr = jnp.asarray(x0, dtype=jnp.float32)
                if int(x0_arr.size) >= self._camera_dim:
                    cam_init = jnp.reshape(x0_arr[: self._camera_dim], (self.horizon + 1, self._camera_pose_dim))
                else:
                    cam_init = jnp.tile(jnp.reshape(x0_arr, (-1,))[: self._camera_pose_dim], (self.horizon + 1, 1))
                    if int(cam_init.shape[1]) < self._camera_pose_dim:
                        cam_init = jnp.concatenate([cam_init, jnp.zeros((self.horizon + 1, self._camera_pose_dim - int(cam_init.shape[1])))], axis=-1)
            else:
                cam_init = 0.1 * jax.random.normal(key_cam, (self.horizon + 1, self._camera_pose_dim))
                cam_init = cam_init.at[:, 3:7].set(_normalize_quat(cam_init[:, 3:7]))
            theta_init = jnp.concatenate([jnp.asarray(scene_flat, dtype=jnp.float32), jnp.ravel(cam_init)], axis=0)

        theta_final, history = self._bridge_loop(rng_key, theta_init, observations)
        scene_final = self.scene_repr.unflatten(theta_final[: self._scene_dim])
        cam_final = cam_fixed if self._fix_cameras else jnp.reshape(theta_final[self._scene_dim :], (self.horizon + 1, self._camera_pose_dim))
        states_list = [np.asarray(cam_final[i], dtype=np.float32) for i in range(self.horizon + 1)]
        actions_list = [np.asarray(cam_final[i + 1] - cam_final[i], dtype=np.float32) for i in range(self.horizon)]

        from ..types import MBD3DResult
        result = MBD3DResult(
            scene_params=scene_final,
            camera_trajectory=np.asarray(cam_final, dtype=np.float32),
            states=states_list,
            actions=actions_list,
            diagnostics={"bridge_history": history},
            bridge_history=history,
            total_log_prob=history["log_pi"][-1] if history["log_pi"] else 0.0,
            n_steps=len(history["log_pi"]),
        )

        # --- Render predicted images & compute metrics ---
        eval_extras = self._render_eval(scene_final, cam_final, observations)

        return {
            "states": states_list,
            "actions": actions_list,
            "scene_params": scene_final,
            "camera_trajectory": np.asarray(cam_final, dtype=np.float32),
            "diagnostics": result.diagnostics,
            "bridge_history": history,
            "total_log_prob": result.total_log_prob,
            "initial_state": states_list[0] if states_list else None,
            "candidate_states": [states_list],
            "candidate_actions": [actions_list],
            "candidate_costs": np.asarray([-float(result.total_log_prob)], dtype=np.float32),
            "best_idx": 0,
            **result.to_trajectory_info(),
            **eval_extras,
        }

    def sample_trajectories(self, x0: Any, n_samples: int, rng_key: Optional[Any] = None, observations: Optional[Any] = None) -> List[Trajectory]:
        result = self.plan(x0, rng_key=rng_key, observations=observations)
        traj = Trajectory(states=result["states"], actions=result["actions"], info=result)
        return [traj for _ in range(max(1, int(n_samples)))]

    def plan_batch(self, x0: Any, keys: Any, observations: Optional[Any] = None) -> List[Dict[str, Any]]:
        results = []
        for i in range(int(keys.shape[0])):
            results.append(self.plan(x0, rng_key=keys[i], observations=observations))
        return results
