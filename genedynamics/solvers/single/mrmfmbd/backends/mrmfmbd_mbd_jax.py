"""
Level-3 MBD backend for MR-MF-MBD: a model-based-diffusion posterior sampler.

Algorithm (reverse-time denoising diffusion applied to θ = (x, φ)):

    for idx in reversed(range(K)):
        σ_k  = sqrt(1 - ᾱ_k)                     # DDPM noise schedule
        Y_m  = Ybar + σ_k · scale ⊙ ε_m           # forward-corrupt M proposals
        R_m  = logsumexp_c(log π(c) + R_{m,c}/T)  # mode marginalization
        w_m  = softmax((R_m - R̄) / (σ_R · T_k))  # reward-temp weights
        Ybar = Σ_m w_m · Y_m                      # denoise = reward-weighted mean
        Ybar = Ybar + τ_k · scale ⊙ ξ            # vanishing exploration noise

Two execution paths:

    * _plan_jax_scan  (FAST): When the evaluator exposes the JAX-MPM internals
      (`._scene`, `._mpm_cfg`, `._mode_friction`), the whole reverse loop lives
      inside `jax.lax.scan`, blocked by fidelity level. mode marginalization
      is pure JAX inside the scan body. No numpy round-trips per step.

    * _plan_python_loop  (SLOW, fallback): Original host-side loop that
      dispatches rollouts through evaluator.evaluate_batch. Used for any
      evaluator not exposing the JAX-MPM internals required by _plan_jax_scan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from genedynamics.solvers.single.mrmfmbd.theta_prior import ThetaParametrization, ThetaPrior

try:
    from genedynamics.solvers.single.mrmfmbd.mode_system.marginalizer import (
        ModeMarginalizer,
        default_mode_system_config,
    )
    MODE_SYSTEM_AVAILABLE = True
except ImportError:
    MODE_SYSTEM_AVAILABLE = False
    ModeMarginalizer = None
    default_mode_system_config = None

try:
    from genedynamics.solvers.single.mrmfmbd.fidelity_system.ladder import create_fidelity_ladder
    FIDELITY_SYSTEM_AVAILABLE = True
except ImportError:
    FIDELITY_SYSTEM_AVAILABLE = False
    create_fidelity_ladder = None


@dataclass
class MBDConfig:
    """Configuration for the Level-3 MBD backend."""

    K: int = 16
    M: int = 8
    beta0: float = 1e-4
    betaT: float = 2e-2
    reward_temperature: float = 0.5
    reward_temperature_min: float = 0.1
    tau_frac: float = 0.1
    top_k_fine: int = 3
    fine_fidelity_level: int = 2
    # Phase 1.3 (JM2D Takeaway 1): u-step inner refinement of clean candidates.
    # Each outer reverse-diffusion step runs (inner_denoise_steps + 1) iterations
    # of (rollout → reward-weighted mean → re-propose with shrunk noise). At
    # u=0 we recover the original single-shot Tweedie estimate exactly.
    # JM2D Table 3 reports +50 pp safe-success on Cluttered going u=0 → u=10;
    # cost scales linearly: u=10 means 11x rollouts per outer step.
    inner_denoise_steps: int = 0
    inner_denoise_shrink: float = 0.5
    # Phase 2.3: regime-marginalization flavor.
    #   "reward"          → legacy logsumexp(log p + R/T)        (easy-regime weighted)
    #   "risk_sensitive"  → -tau_r * logsumexp(log p - R/tau_r)  (failure-prone weighted)
    # tau_r controls the soft-min sharpness; tau_r → 0 = max-min, tau_r → ∞ = uniform.
    regime_posterior_mode: str = "reward"
    risk_temperature: float = 1.0
    # Phase 4.2 (writeup §8.1): SHAC local refinement on the top-K candidates
    # AFTER the main MBD reverse loop converges. For each surviving theta
    # (x, phi), runs `shac_refine_steps` of BPTT on the controller portion
    # using rollout_h_from_state. Adds a proximal `λ_prox * ‖phi^+ - phi‖²`
    # penalty so the local update can't override the diffusion sampler.
    # All zero → no refinement (default; identical to Phase 1/2/3 behavior).
    shac_refine_steps: int = 0
    shac_refine_h: int = 32
    shac_refine_lr: float = 5.0e-4
    shac_proximal_lambda: float = 1.0
    shac_refine_topk: int = 3
    # Phase 1.2 plumbing: alm_adaptive constraint scheduler scalars (writeup §6/§7).
    # Backends store these and write them to result metadata for ablation
    # diagnostics; with `nu_max=0.0` (the default in main_v2/crawling_alm.yaml)
    # the budget tracker is INERT and the run is identical to the no-ALM baseline.
    # Phase 2 will consume `nu_k * compute_cost` per candidate-fidelity pair
    # in the importance weight; Phase 1 only wires the scaffold.
    alm_params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def _mirror_z_voxels(
    x_half: jnp.ndarray,  # (..., vx*vy*vz_half)
    vx: int, vy: int, vz_half: int,
) -> jnp.ndarray:
    """Mirror voxel occupancy along Z axis: (vx,vy,vz_half) → (vx,vy,vz_half*2).

    Works on batched (..., D) or single (D,) inputs.
    """
    leading = x_half.shape[:-1]
    h = x_half.reshape(*leading, vx, vy, vz_half)
    full = jnp.concatenate([h, jnp.flip(h, axis=-1)], axis=-1)
    return full.reshape(*leading, vx * vy * vz_half * 2)


def _s1_marginalize_jax(
    rewards_mc: jnp.ndarray,   # (M, C)
    log_prior: jnp.ndarray,    # (C,)
    T: jnp.ndarray,            # scalar
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX mode marginalization.

    Returns (R_m, w_c) where
        R_m[m] = logsumexp_c(log p(c) + R[m,c]/T)
        w_c[m, c] = softmax_c(log p(c) + R[m,c]/T)
    """
    rewards_mc = jnp.nan_to_num(rewards_mc, nan=-1e6, posinf=1e6, neginf=-1e6)
    log_terms = log_prior[None, :] + rewards_mc / jnp.maximum(T, 1e-8)
    R_m = jax.scipy.special.logsumexp(log_terms, axis=-1)
    w_c = jnp.exp(jax.nn.log_softmax(log_terms, axis=-1))
    return R_m, w_c


class MRMFMBDBackendMBD:
    """Level-3 MBD backend: DDPM-style reverse diffusion over θ with mode + fidelity systems hooks."""

    def __init__(
        self,
        *,
        evaluator: Any,
        theta_param: ThetaParametrization,
        theta_prior: ThetaPrior,
        fidelity_ladder: Optional[Any] = None,
        mode_marginalizer: Optional[Any] = None,
        config: Optional[MBDConfig] = None,
        task_id: str = "crawling_ground",
        num_modes: int = 4,
        mode_log_priors: Optional[List[float]] = None,
        seed: int = 0,
        show_tqdm: bool = False,
        **kwargs: Any,
    ):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for MRMFMBDBackendMBD")

        self.evaluator = evaluator
        self.theta_param = theta_param
        self.theta_prior = theta_prior
        self.config = config or MBDConfig(**kwargs)
        self.task_id = task_id
        self.num_modes = num_modes
        self.mode_log_priors = mode_log_priors or [0.0] * num_modes
        self.seed = seed
        self.show_tqdm = show_tqdm

        # Z-symmetry: optimizer sees x_half, mirrored to x_full before rollout.
        # voxel_dims/symmetry are passed via config.extra by the baseline.
        self._z_sym = str(self.config.extra.get("morphology_symmetry", "")).lower() == "z"
        vd = self.config.extra.get("voxel_dims")
        if vd is not None:
            self._voxel_dims = tuple(int(v) for v in vd)
        else:
            self._voxel_dims = None  # will be read from evaluator if needed

        K = int(self.config.K)

        # Mode: mode marginalizer (kept for python-loop path; JAX path uses pure fn).
        self.mode_marginalizer = mode_marginalizer
        if self.mode_marginalizer is None and MODE_SYSTEM_AVAILABLE:
            self.mode_marginalizer = ModeMarginalizer(
                default_mode_system_config(num_modes),
                backend="jax",
            )

        # Fidelity: fidelity ladder (coarse early, fine late on *reverse* index)
        self.fidelity_ladder = fidelity_ladder
        if self.fidelity_ladder is None and FIDELITY_SYSTEM_AVAILABLE and create_fidelity_ladder is not None:
            self.fidelity_ladder = create_fidelity_ladder(
                K=K, num_levels=3, ladder_type="geometric", step_ratio=2.0
            )

        self._D = theta_param.theta_dim

        # Per-dim scale: half the bound-range.
        x_lo, x_hi = theta_param.x_bounds if theta_param.x_bounds else (-1.0, 1.0)
        p_lo, p_hi = theta_param.phi_bounds if theta_param.phi_bounds else (-1.0, 1.0)
        scale_x = np.full(theta_param.x_dim, (float(x_hi) - float(x_lo)) / 2.0, dtype=np.float32)
        scale_p = np.full(theta_param.phi_dim, (float(p_hi) - float(p_lo)) / 2.0, dtype=np.float32)
        self._scale_np = np.concatenate([scale_x, scale_p]).astype(np.float32)
        self._scale = jnp.asarray(self._scale_np)

        # DDPM noise schedule (pre-computed once), capped at sigma_max.
        betas = np.linspace(self.config.beta0, self.config.betaT, K, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(np.maximum(1.0 - alphas_bar, 1e-8)).astype(np.float32)
        sigma_max = float(self.config.extra.get("sigma_max", 0.15))
        sigmas = np.minimum(sigmas, sigma_max).astype(np.float32)
        self._sigmas = jnp.asarray(sigmas)
        self._taus = jnp.asarray(sigmas * float(self.config.tau_frac))

        # Geometric anneal for the reward temperature T_k.
        T0 = float(self.config.reward_temperature)
        TK = float(self.config.reward_temperature_min)
        if TK <= 0:
            TK = T0
        t_ratio = TK / max(T0, 1e-8)
        t_ratio = max(t_ratio, 1e-6)
        T_schedule = T0 * (t_ratio ** (np.arange(K, dtype=np.float32) / max(K - 1, 1)))
        self._T_schedule = jnp.asarray(T_schedule.astype(np.float32))

        # ----- Phase 1.2: alm_adaptive plumbing -----------------------------
        # Build a per-step nu_k schedule and a static cost-per-candidate from
        # the alm_adaptive scalars (writeup §6/§7). Phase 1 uses a deterministic
        # linear ramp from nu0 to nu_max (no feedback yet — soft_robot crawling
        # has no g(x) ≤ 0 to drive the dual). Phase 2 will replace this with
        # the JAX adaptive carry once we have per-candidate fidelity to feed.
        alm = self.config.alm_params or {}
        if alm:
            nu0 = float(alm.get("nu0", 0.0))
            nu_max = float(alm.get("nu_max", 0.0))
            B = float(alm.get("compute_budget_B", 0.0))
            mode = str(alm.get("compute_cost_mode", "product"))
            a0 = float(alm.get("compute_cost_a0", 0.0))
            aK_ = float(alm.get("compute_cost_aK", 1.0))
            aI_ = float(alm.get("compute_cost_aI", 1.0))
            topK_max = float(alm.get("topK_max", 8))
            I_max = float(alm.get("I_max", 5))
            if nu_max <= nu0:
                nu_schedule_np = np.full(K, nu0, dtype=np.float32)
            else:
                nu_schedule_np = np.linspace(nu0, nu_max, K, dtype=np.float32)
            if mode == "linear":
                cost_const = a0 + aK_ * topK_max + aI_ * I_max
            else:  # "product" — match alm_adaptive default
                cost_const = a0 + aK_ * topK_max * aI_ * I_max
            self._alm_active = nu_max > 0.0
            self._alm_nu_schedule_np = nu_schedule_np
            self._alm_cost_const = float(cost_const)
            self._alm_budget_B = B
        else:
            self._alm_active = False
            self._alm_nu_schedule_np = np.zeros(K, dtype=np.float32)
            self._alm_cost_const = 0.0
            self._alm_budget_B = 0.0

        # Bounds as jnp arrays for JIT-compatible clipping.
        self._x_lo = jnp.asarray(float(x_lo), dtype=jnp.float32)
        self._x_hi = jnp.asarray(float(x_hi), dtype=jnp.float32)
        self._phi_lo = jnp.asarray(float(p_lo), dtype=jnp.float32)
        self._phi_hi = jnp.asarray(float(p_hi), dtype=jnp.float32)
        self._x_dim = theta_param.x_dim

        # Mode log prior as jnp.
        self._log_prior_c = jnp.asarray(
            np.asarray(self.mode_log_priors[: self.num_modes], dtype=np.float32)
        )

        # JIT kernels for the python-loop fallback path (unchanged).
        self._propose_jit = jax.jit(self._propose_impl)
        self._denoise_jit = jax.jit(self._denoise_impl)

        # Per-(num_env_steps) block-runner cache for the JAX-direct fast path.
        self._block_runner_cache: Dict[int, Any] = {}

    # --------- JIT kernels (shared) -----------------------------------------

    def _clip_theta_jnp(self, theta: jnp.ndarray) -> jnp.ndarray:
        x_part = jnp.clip(theta[..., : self._x_dim], self._x_lo, self._x_hi)
        phi_part = jnp.clip(theta[..., self._x_dim :], self._phi_lo, self._phi_hi)
        return jnp.concatenate([x_part, phi_part], axis=-1)

    def _propose_impl(
        self, Ybar: jnp.ndarray, sigma_k: jnp.ndarray, eps: jnp.ndarray,
    ) -> jnp.ndarray:
        Y0s = Ybar[None, :] + sigma_k * self._scale[None, :] * eps
        return self._clip_theta_jnp(Y0s)

    def _denoise_impl(
        self,
        Y0s: jnp.ndarray, R_m: jnp.ndarray,
        tau_k: jnp.ndarray, T_k: jnp.ndarray, noise: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        R_mean = jnp.mean(R_m)
        R_std = jnp.maximum(jnp.std(R_m), jnp.asarray(1e-4, dtype=R_m.dtype))
        log_w = (R_m - R_mean) / (R_std * T_k)
        weights = jax.nn.softmax(log_w)
        Ybar_next = jnp.einsum("m,md->d", weights, Y0s)
        Ybar_next = Ybar_next + tau_k * self._scale * noise
        Ybar_next = self._clip_theta_jnp(Ybar_next)
        ess = 1.0 / jnp.sum(weights * weights)
        return Ybar_next, weights, ess

    # --------- Fidelity block partition -------------------------------------

    def _build_fid_blocks(self) -> List[Tuple[int, int, int]]:
        """Return list of (fid, reverse_start_idx, length) for contiguous blocks.

        The reverse loop walks idx = K-1, K-2, ..., 0; step `i` in reverse order
        has kprime = i, and fidelity_ladder(kprime) must be non-decreasing in i
        (coarse early, fine late). Otherwise we raise — blocked scan requires
        a sorted ladder.
        """
        K = int(self.config.K)
        if self.fidelity_ladder is None:
            return [(0, 0, K)]
        fids = [int(self.fidelity_ladder(kprime)) for kprime in range(K)]
        blocks: List[Tuple[int, int, int]] = []
        cur_fid, cur_start, cur_len = fids[0], 0, 1
        for i in range(1, K):
            f = fids[i]
            if f == cur_fid:
                cur_len += 1
            elif f > cur_fid:
                blocks.append((cur_fid, cur_start, cur_len))
                cur_fid, cur_start, cur_len = f, i, 1
            else:
                raise ValueError(
                    f"MBD fidelity ladder must be non-decreasing in reverse-step order; "
                    f"got fids={fids}. Use ladder_type='geometric' or 'linear'."
                )
        blocks.append((cur_fid, cur_start, cur_len))
        return blocks

    # --------- Softzoo-compat path: external rollouts -----------------------

    def _expand_x_np(self, x_opt: np.ndarray) -> np.ndarray:
        """Expand optimizer's x (possibly half) to full voxel occupancy for rollout."""
        if self._z_sym and self._voxel_dims is not None:
            vx, vy, vz = self._voxel_dims
            vz_half = vz // 2
            h = x_opt.reshape(vx, vy, vz_half)
            return np.concatenate([h, np.flip(h, axis=-1)], axis=-1).reshape(-1)
        return x_opt

    def _evaluate_proposals_modes(
        self, proposals: np.ndarray, fidelity_level: int, seed_base: int,
    ) -> np.ndarray:
        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        M = proposals.shape[0]
        flat_requests: List[Any] = []
        index_pairs: List[Tuple[int, int]] = []
        x_dim = self.theta_param.x_dim
        for c in range(self.num_modes):
            for m in range(M):
                flat_requests.append(
                    RolloutRequest(
                        morphology_params=self._expand_x_np(proposals[m, :x_dim]),
                        controller_params=proposals[m, x_dim:],
                        mode_id=c,
                        fidelity_level=fidelity_level,
                        seed=seed_base + m * self.num_modes + c,
                        num_repeats=1,
                        record=False,
                    )
                )
                index_pairs.append((m, c))
        batch_req = RolloutBatchRequest(task_id=self.task_id, requests=flat_requests)
        batch_res = self.evaluator.evaluate_batch(batch_req, parallel=True, use_cache=False)
        rewards = np.zeros((M, self.num_modes), dtype=np.float32)
        for (m, c), r in zip(index_pairs, batch_res.returns):
            rewards[m, c] = r
        return rewards

    def _s1_marginalize(self, rewards_mc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rewards_mc = np.nan_to_num(
            rewards_mc, nan=-1e6, posinf=1e6, neginf=-1e6
        ).astype(np.float32)
        if self.mode_marginalizer is not None:
            result = self.mode_marginalizer(
                rewards_mc, temperature=float(self.config.reward_temperature),
            )
            return (
                np.asarray(result.marginal_log_likelihood, dtype=np.float32),
                np.asarray(result.responsibilities, dtype=np.float32),
            )
        T = max(float(self.config.reward_temperature), 1e-8)
        log_prior_c = np.asarray(self.mode_log_priors[: self.num_modes], dtype=np.float32)
        log_terms = log_prior_c[None, :] + rewards_mc.astype(np.float32) / T
        marginal = np.asarray(
            np.log(np.sum(np.exp(log_terms - log_terms.max(axis=-1, keepdims=True)), axis=-1))
            + log_terms.max(axis=-1),
            dtype=np.float32,
        )
        w = np.exp(
            log_terms
            - np.log(np.sum(np.exp(log_terms - log_terms.max(axis=-1, keepdims=True)), axis=-1, keepdims=True))
            - log_terms.max(axis=-1, keepdims=True)
        )
        return marginal, w.astype(np.float32)

    # --------- JAX-direct fast path: scan block runner ----------------------

    def _is_jax_direct(self) -> bool:
        ev = self.evaluator
        return (
            hasattr(ev, "_scene")
            and hasattr(ev, "_mpm_cfg")
            and hasattr(ev, "_mode_friction")
        )

    def _make_block_runner(self, num_env_steps: int):
        """Build a jitted scan runner for blocks of this num_env_steps.

        Captures scene/cfg/friction_table via closure; returns a function with
        signature:

            run_block(Ybar, rng_key, topk_theta, topk_score,
                      sigmas_b, taus_b, Ts_b)
                -> (Ybar_new, rng_key_new, topk_theta_new, topk_score_new,
                    step_outputs_dict)
        """
        from genedynamics.envs.external.jax_mpm.scene import rollout_return_batch

        scene = self.evaluator._scene
        cfg = self.evaluator._mpm_cfg
        friction_table = jnp.asarray(
            np.asarray(self.evaluator._mode_friction, dtype=np.float32)[: int(self.num_modes)]
        )  # (C,)

        M = int(self.config.M)
        D = int(self._D)
        C = int(self.num_modes)
        x_dim = int(self._x_dim)  # optimizer's x_dim (half if z-sym)
        top_k_cap = max(int(self.config.top_k_fine), 1)
        T_mode = jnp.asarray(
            max(float(self.config.reward_temperature), 1e-8), dtype=jnp.float32
        )
        log_prior_c = self._log_prior_c

        # Phase 2.3: pick the marginalizer once at trace time. Both branches
        # return (R_m (M,), w_c (M, C)) with identical sign convention so the
        # rest of the scan body is agnostic to the choice.
        from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
            risk_sensitive_marginalize_jax,
            validate_mode,
        )
        validate_mode(self.config.regime_posterior_mode)
        risk_mode = (self.config.regime_posterior_mode == "risk_sensitive")
        tau_r = jnp.asarray(
            max(float(self.config.risk_temperature), 1e-8), dtype=jnp.float32
        )

        # Phase 1.3: JM2D-style inner refinement (Alg 2, Table 3 of CoRL'25 paper).
        # Captured as Python ints so JAX unrolls; u=0 → single eval → same as before.
        inner_steps = max(int(self.config.inner_denoise_steps), 0)
        inner_shrink = float(self.config.inner_denoise_shrink)

        # Z-symmetry: optimizer x_dim is half; mirror to full before rollout.
        z_sym = self._z_sym
        if z_sym and self._voxel_dims is not None:
            vx, vy, vz = self._voxel_dims
            vz_half = vz // 2
            x_dim_full = vx * vy * vz
        else:
            z_sym = False
            x_dim_full = x_dim

        # Inline propose+clip (mirrors self._propose_impl). Used by the inner
        # refinement loop with progressively shrunken sigma.
        def _propose_inline(center: jnp.ndarray, sigma_eff: jnp.ndarray, key) -> jnp.ndarray:
            eps = jax.random.normal(key, (M, D), dtype=jnp.float32)
            Y = center[None, :] + sigma_eff * self._scale[None, :] * eps
            return self._clip_theta_jnp(Y)

        # Inline rollout + mode marginalization. Returns (rewards_mc, R_m).
        def _rollout_and_marginalize(Y0s_in: jnp.ndarray):
            x_opt = Y0s_in[:, :x_dim]
            if z_sym:
                x_full = _mirror_z_voxels(x_opt, vx, vy, vz_half)  # (M, x_dim_full)
            else:
                x_full = x_opt
            x_flat = jnp.repeat(x_full, C, axis=0)             # (M*C, x_dim_full)
            phi_flat = jnp.repeat(Y0s_in[:, x_dim:], C, axis=0)  # (M*C, phi_dim)
            fr_flat = jnp.tile(friction_table, M)              # (M*C,)
            rs, _disps = rollout_return_batch(
                x_flat, phi_flat, fr_flat, scene, cfg, num_env_steps
            )
            rewards_mc_local = rs.reshape(M, C)
            if risk_mode:
                R_m_local, w_c_local = risk_sensitive_marginalize_jax(
                    rewards_mc_local, log_prior_c, tau_r
                )
            else:
                R_m_local, w_c_local = _s1_marginalize_jax(
                    rewards_mc_local, log_prior_c, T_mode
                )
            return rewards_mc_local, R_m_local, w_c_local

        # Reward-weighted mean only — no tau noise / no clip yet (those happen
        # after the inner refinement loop completes).
        def _weighted_mean(Y0s_in: jnp.ndarray, R_m_in: jnp.ndarray, T_k):
            R_mean = jnp.mean(R_m_in)
            R_std = jnp.maximum(jnp.std(R_m_in), jnp.asarray(1e-4, dtype=R_m_in.dtype))
            log_w = (R_m_in - R_mean) / (R_std * T_k)
            w = jax.nn.softmax(log_w)
            Yb = jnp.einsum("m,md->d", w, Y0s_in)
            return Yb, w

        def step(carry, xs):
            Ybar, rng_key, topk_theta, topk_score = carry
            sigma_k, tau_k, T_k = xs

            rng_key, k_eps, k_noise = jax.random.split(rng_key, 3)
            Y0s = _propose_inline(Ybar, sigma_k, k_eps)  # (M, D)

            # ---- u-step inner refinement (JM2D Alg 2) -----------------------
            # u=0 → single iteration, identical to original behavior.
            rewards_mc, R_m, w_c = _rollout_and_marginalize(Y0s)
            Ybar_inner, weights = _weighted_mean(Y0s, R_m, T_k)
            for u_iter in range(inner_steps):
                rng_key, k_refine = jax.random.split(rng_key)
                # Shrink the proposal noise on each refinement iter so the
                # candidate cloud collapses around the running weighted mean.
                shrink = jnp.asarray(inner_shrink ** (u_iter + 1), dtype=jnp.float32)
                Y0s = _propose_inline(Ybar_inner, sigma_k * shrink, k_refine)
                rewards_mc, R_m, w_c = _rollout_and_marginalize(Y0s)
                Ybar_inner, weights = _weighted_mean(Y0s, R_m, T_k)
            # -----------------------------------------------------------------

            noise = jax.random.normal(k_noise, (D,), dtype=jnp.float32)
            # Final DDPM exploration noise + clip (was the tail of _denoise_impl).
            Ybar_next = self._clip_theta_jnp(Ybar_inner + tau_k * self._scale * noise)
            ess = 1.0 / jnp.sum(weights * weights)

            # Top-K: concat running top_k with M new candidates, keep top cap.
            cand_scores = jnp.mean(rewards_mc, axis=-1)           # (M,)
            all_theta = jnp.concatenate([topk_theta, Y0s], axis=0)   # (cap+M, D)
            all_scores = jnp.concatenate([topk_score, cand_scores], axis=0)
            _, top_idx = jax.lax.top_k(all_scores, top_k_cap)
            new_topk_theta = all_theta[top_idx]
            new_topk_score = all_scores[top_idx]

            mean_env_return = jnp.mean(rewards_mc)
            mean_R = jnp.mean(R_m)
            w_c_mean = jnp.mean(w_c, axis=0)  # (C,)

            out = {
                "sigma_k": sigma_k,
                "tau_k": tau_k,
                "T_k": T_k,
                "mean_R_s1": mean_R,
                "mean_env_return": mean_env_return,
                "ess": ess,
                "w_c_mean": w_c_mean,
                # Full theta checkpoint after denoise at this step. Scan stacks
                # these into (K, D). Post-hoc visualization uses a sparse subset
                # (e.g. k ∈ {0, K/5, 2K/5, ...}); the cost of keeping all K is
                # K × D × 4 bytes = 100 × 104 × 4 ≈ 42 KB — negligible.
                "Ybar": Ybar_next,
            }
            return (Ybar_next, rng_key, new_topk_theta, new_topk_score), out

        @jax.jit
        def run_block(Ybar, rng_key, topk_theta, topk_score, sigmas_b, taus_b, Ts_b):
            carry = (Ybar, rng_key, topk_theta, topk_score)
            (Ybar_f, rng_f, topk_t_f, topk_s_f), out = jax.lax.scan(
                step, carry, (sigmas_b, taus_b, Ts_b)
            )
            return Ybar_f, rng_f, topk_t_f, topk_s_f, out

        return run_block

    def _get_block_runner(self, num_env_steps: int):
        if num_env_steps not in self._block_runner_cache:
            self._block_runner_cache[num_env_steps] = self._make_block_runner(num_env_steps)
        return self._block_runner_cache[num_env_steps]

    # --------- Ybar init ----------------------------------------------------

    def _bounds_midpoint(self) -> np.ndarray:
        x_lo, x_hi = self.theta_param.x_bounds or (-1.0, 1.0)
        p_lo, p_hi = self.theta_param.phi_bounds or (-1.0, 1.0)
        return np.concatenate([
            np.full(self.theta_param.x_dim, (float(x_lo) + float(x_hi)) / 2.0, dtype=np.float32),
            np.full(self.theta_param.phi_dim, (float(p_lo) + float(p_hi)) / 2.0, dtype=np.float32),
        ])

    def _init_ybar(self, theta_init: Optional[Any]) -> jnp.ndarray:
        if theta_init is not None:
            Ybar = jnp.asarray(theta_init, dtype=jnp.float32)
        else:
            prior_mean = getattr(self.theta_prior, "_mean", None)
            if prior_mean is not None and np.all(np.abs(prior_mean) > 1e-8):
                Ybar = jnp.asarray(prior_mean, dtype=jnp.float32)
            else:
                Ybar = jnp.asarray(self._bounds_midpoint(), dtype=jnp.float32)
        return self._clip_theta_jnp(Ybar)

    # --------- JAX-direct plan path -----------------------------------------

    def _plan_jax_scan(self, theta_init: Optional[Any], rng_key: Any) -> Dict[str, Any]:
        from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS

        K = int(self.config.K)
        M = int(self.config.M)
        D = int(self._D)
        top_k_cap = max(int(self.config.top_k_fine), 1)

        Ybar = self._init_ybar(theta_init)

        # Running top-K buffer, init with -inf so any real candidate wins.
        topk_theta = jnp.zeros((top_k_cap, D), dtype=jnp.float32)
        topk_score = jnp.full((top_k_cap,), -1e30, dtype=jnp.float32)

        blocks = self._build_fid_blocks()

        # Stitched per-step outputs (collected across blocks).
        stitched: Dict[str, List[np.ndarray]] = {
            "sigma_k": [], "tau_k": [], "T_k": [],
            "mean_R_s1": [], "mean_env_return": [], "ess": [], "w_c_mean": [],
            "Ybar": [],
        }
        fidelity_history: List[int] = []
        block_wall_times: List[float] = []

        wall_start = time.perf_counter()

        for (fid, start, length) in blocks:
            num_env_steps = int(FIDELITY_STEPS.get(fid, FIDELITY_STEPS[max(FIDELITY_STEPS)]))
            runner = self._get_block_runner(num_env_steps)

            # Reverse-step index i = start..start+length-1 maps to idx = K-1-i.
            # sigmas/taus/Ts are indexed by `idx` (the original DDPM step idx).
            idxs_block = np.arange(K - 1 - start, K - 1 - start - length, -1)
            sig_b = jnp.asarray(np.asarray(self._sigmas)[idxs_block])
            tau_b = jnp.asarray(np.asarray(self._taus)[idxs_block])
            T_b = jnp.asarray(np.asarray(self._T_schedule)[idxs_block])

            if self.show_tqdm:
                print(f"[MBD] block fid={fid} steps={length} num_env_steps={num_env_steps}",
                      flush=True)

            t0 = time.perf_counter()
            Ybar, rng_key, topk_theta, topk_score, out = runner(
                Ybar, rng_key, topk_theta, topk_score, sig_b, tau_b, T_b,
            )
            Ybar.block_until_ready()
            block_wall = time.perf_counter() - t0
            block_wall_times.append(block_wall)

            for key in stitched:
                stitched[key].append(np.asarray(out[key]))
            fidelity_history.extend([fid] * length)

        # Concatenate per-step arrays in reverse order (matches old bridge_history).
        per_step: Dict[str, np.ndarray] = {
            k: np.concatenate(v, axis=0) for k, v in stitched.items()
        }
        bridge_history: List[Dict[str, Any]] = []
        total = per_step["sigma_k"].shape[0]
        for i in range(total):
            idx = K - 1 - i
            entry = {
                "k_reverse_idx": int(idx),
                "k_forward": int(i),
                "sigma_k": float(per_step["sigma_k"][i]),
                "tau_k": float(per_step["tau_k"][i]),
                "T_k": float(per_step["T_k"][i]),
                "fidelity_level": int(fidelity_history[i]),
                "mean_R_s1": float(per_step["mean_R_s1"][i]),
                "mean_env_return": float(per_step["mean_env_return"][i]),
                "ess": float(per_step["ess"][i]),
            }
            # Phase 1.2: record per-step alm_adaptive state for diagnostics.
            # `nu_k_forward` is indexed in forward-time (i = 0 is the first
            # reverse-diffusion step), matching the rest of bridge_history.
            if self._alm_active or self.config.alm_params:
                entry["alm_nu_k"] = float(self._alm_nu_schedule_np[idx])
                entry["alm_cost"] = float(self._alm_cost_const)
                entry["alm_budget_pen"] = float(
                    self._alm_nu_schedule_np[idx] * self._alm_cost_const
                )
            bridge_history.append(entry)
        mode_resp_history = per_step["w_c_mean"]

        # Extract top-K candidates as Python list (sorted desc).
        topk_theta_np = np.asarray(topk_theta)
        topk_score_np = np.asarray(topk_score)
        order = np.argsort(-topk_score_np)
        _top_k: List[Tuple[float, np.ndarray]] = [
            (float(topk_score_np[i]), topk_theta_np[i])
            for i in order
            if topk_score_np[i] > -1e29
        ]

        # Fine-fidelity re-evaluation of top-K (outside scan; num_env_steps differs).
        best_fine_return, best_fine_theta, fine_calls = self._fine_revalidate(_top_k, Ybar)
        if best_fine_return > -float("inf"):
            Ybar = jnp.asarray(best_fine_theta, dtype=jnp.float32)

        # Phase 4.2: SHAC local refinement (writeup §8.1). One-step short-
        # horizon gradient ascent on the controller portion of the top-K
        # candidates with a proximal penalty on the displacement. Skipped
        # entirely when shac_refine_steps == 0 → behavior unchanged.
        shac_summary = {"active": False, "applied": 0, "improved": 0}
        if int(self.config.shac_refine_steps) > 0 and self._is_jax_direct():
            shac_summary, refined_top, best_refined = self._shac_refine_topk(_top_k)
            if best_refined is not None and best_refined[0] > best_fine_return:
                Ybar = jnp.asarray(best_refined[1], dtype=jnp.float32)
                best_fine_return = float(best_refined[0])
                # Update _top_k so downstream stats use the refined leaderboard.
                _top_k = refined_top

        wall = time.perf_counter() - wall_start
        x_final, phi_final = self.theta_param.unpack(Ybar)

        return {
            "theta": np.asarray(Ybar),
            "x": np.asarray(x_final),
            "phi": np.asarray(phi_final),
            "bridge_history": bridge_history,
            "mode_responsibilities": np.asarray(mode_resp_history),
            "fidelity_history": fidelity_history,
            "fine_calls": fine_calls,
            "best_fine_return": best_fine_return if best_fine_return > -float("inf") else None,
            "top_k_candidates": [
                {"reward": float(s), "theta": np.asarray(t).tolist()} for s, t in _top_k
            ],
            "wall_clock": wall,
            "block_wall_times": block_wall_times,
            # (K, D) trajectory of the denoised mean across all diffusion steps.
            # Row i corresponds to bridge_history[i]["k_forward"] = i.
            "theta_history": per_step["Ybar"],
            # Phase 1.2: top-level alm summary (full per-step values are in bridge_history).
            "alm_summary": {
                "active": bool(self._alm_active),
                "nu_max": float(self._alm_nu_schedule_np.max()) if self._alm_active else 0.0,
                "cost_const": float(self._alm_cost_const),
                "budget_B": float(self._alm_budget_B),
            },
            # Phase 4.2: SHAC refinement diagnostic (writeup §8.1).
            "shac_summary": shac_summary,
        }

    def _shac_refine_topk(
        self,
        top_k: List[Tuple[float, np.ndarray]],
    ) -> Tuple[Dict[str, Any], List[Tuple[float, np.ndarray]], Optional[Tuple[float, np.ndarray]]]:
        """SHAC local refinement on the top-K candidates (writeup §8.1).

        For each candidate `theta = (x, phi)`:
        1. Hold `x` fixed; treat `phi` as the optimization variable.
        2. Run `shac_refine_steps` of gradient ascent on
           `R̂_h(phi) - λ_prox * ‖phi - phi_0‖²`
           where R̂_h is the discounted h-step return from
           `rollout_h_from_state` and `phi_0` is the pre-refinement value.
        3. Re-evaluate at the configured fine fidelity; keep the better of
           the original or refined candidate.

        Returns
        -------
        summary : dict
            {"active": True, "applied": #candidates refined, "improved": # that beat their pre-score}
        refined_top : list[(score, theta)]
            Same length as `top_k`, with refined entries replacing the originals
            ONLY when refinement improved the fine-fidelity score.
        best : (score, theta) | None
            The single best post-refinement candidate.
        """
        n_steps = int(self.config.shac_refine_steps)
        if n_steps <= 0 or not top_k:
            return {"active": False, "applied": 0, "improved": 0}, list(top_k), None

        from genedynamics.envs.external.jax_mpm.scene import (
            _init_carry,
            rollout_h_from_state,
        )

        scene = self.evaluator._scene
        cfg = self.evaluator._mpm_cfg
        friction_table = np.asarray(self.evaluator._mode_friction, dtype=np.float32)
        h = int(self.config.shac_refine_h)
        lr = float(self.config.shac_refine_lr)
        lam_prox = float(self.config.shac_proximal_lambda)
        x_dim = int(self.theta_param.x_dim)

        def loss_fn(phi, x_morph, friction, phi_init):
            """Negative discounted return + proximal penalty (we minimize)."""
            carry = _init_carry(scene)
            _, rewards, _ = rollout_h_from_state(
                carry, jnp.int32(0), phi, h, friction, scene, cfg,
                x_morph=x_morph,
            )
            gammas = jnp.power(jnp.asarray(0.99, dtype=jnp.float32),
                                jnp.arange(h, dtype=jnp.float32))
            R_h = jnp.sum(gammas * rewards)
            prox = lam_prox * jnp.sum((phi - phi_init) ** 2)
            return -(R_h - prox)

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

        refined: List[Tuple[float, np.ndarray]] = []
        best: Optional[Tuple[float, np.ndarray]] = None
        n_applied = 0
        n_improved = 0
        for original_score, theta in top_k:
            theta_np = np.asarray(theta, dtype=np.float32).copy()
            x_part = theta_np[:x_dim]
            phi_part = theta_np[x_dim:].copy()
            phi_init = phi_part.copy()

            x_full = self._expand_x_np(x_part)
            x_full_j = jnp.asarray(x_full, dtype=jnp.float32)
            fr_j = jnp.asarray(float(friction_table[0]), dtype=jnp.float32)
            phi_init_j = jnp.asarray(phi_init, dtype=jnp.float32)
            phi_j = jnp.asarray(phi_part, dtype=jnp.float32)

            # Plain SGD on phi for n_steps. (Adam state across candidates is
            # noise; proximal penalty supplies stability.)
            for _ in range(n_steps):
                g = grad_fn(phi_j, x_full_j, fr_j, phi_init_j)
                phi_j = phi_j - lr * g
                phi_j = jnp.clip(phi_j,
                                  jnp.asarray(self._phi_lo, dtype=jnp.float32),
                                  jnp.asarray(self._phi_hi, dtype=jnp.float32))
            phi_refined = np.asarray(phi_j, dtype=np.float32)
            n_applied += 1

            # Re-evaluate with the existing fine_revalidate machinery so we
            # compare apples to apples (same fidelity, same regime average).
            refined_theta = np.concatenate([x_part, phi_refined]).astype(np.float32)
            re_score, _, _ = self._fine_revalidate(
                [(original_score, refined_theta)], jnp.asarray(refined_theta),
            )
            if re_score > original_score:
                n_improved += 1
                refined.append((float(re_score), refined_theta))
                if best is None or re_score > best[0]:
                    best = (float(re_score), refined_theta)
            else:
                refined.append((original_score, theta_np))

        return (
            {"active": True, "applied": n_applied, "improved": n_improved},
            refined,
            best,
        )

    def _fine_revalidate(
        self,
        top_k: List[Tuple[float, np.ndarray]],
        Ybar_fallback: jnp.ndarray,
    ) -> Tuple[float, np.ndarray, int]:
        best_fine_return = float("-inf")
        best_fine_theta = np.asarray(Ybar_fallback)
        fine_calls = 0
        if self.config.top_k_fine <= 0 or not top_k:
            return best_fine_return, best_fine_theta, fine_calls

        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        x_dim = self.theta_param.x_dim
        fine_level = int(self.config.fine_fidelity_level)
        for _score, cand in top_k:
            cand_np = np.asarray(cand, dtype=np.float32)
            requests = [
                RolloutRequest(
                    morphology_params=self._expand_x_np(cand_np[:x_dim]),
                    controller_params=cand_np[x_dim:],
                    mode_id=c,
                    fidelity_level=fine_level,
                    seed=self.seed + 999 + c,
                    num_repeats=1,
                    record=False,
                )
                for c in range(self.num_modes)
            ]
            req = RolloutBatchRequest(task_id=self.task_id, requests=requests)
            res = self.evaluator.evaluate_batch(req, parallel=True, use_cache=False)
            fine_calls += self.num_modes
            cand_return = float(np.mean(res.returns))
            if cand_return > best_fine_return:
                best_fine_return = cand_return
                best_fine_theta = cand_np
        return best_fine_return, best_fine_theta, fine_calls

    # --------- Softzoo-compat plan path -------------------------------------

    def _plan_python_loop(self, theta_init: Optional[Any], rng_key: Any) -> Dict[str, Any]:
        K = int(self.config.K)
        M = int(self.config.M)

        Ybar = self._init_ybar(theta_init)

        bridge_history: List[Dict[str, Any]] = []
        mode_resp_history: List[np.ndarray] = []
        fidelity_history: List[int] = []
        _top_k: List[Tuple[float, np.ndarray]] = []
        top_k_cap = max(int(self.config.top_k_fine), 1)

        iter_indices: Any = range(K - 1, -1, -1)
        if self.show_tqdm:
            try:
                import tqdm
                iter_indices = tqdm.tqdm(list(iter_indices), desc="MBD reverse")
            except ImportError:
                pass

        wall_start = time.perf_counter()

        for idx in iter_indices:
            t0 = time.perf_counter()

            sigma_k = self._sigmas[idx]
            tau_k = self._taus[idx]
            T_k = self._T_schedule[idx]

            kprime = K - 1 - idx
            fid = int(self.fidelity_ladder(kprime)) if self.fidelity_ladder is not None else 0
            fidelity_history.append(fid)

            rng_key, k_eps, k_noise = jax.random.split(rng_key, 3)
            eps = jax.random.normal(k_eps, (M, self._D), dtype=jnp.float32)

            Y0s = self._propose_jit(Ybar, sigma_k, eps)
            Y0s_np = np.asarray(Y0s)

            rewards_mc = self._evaluate_proposals_modes(
                Y0s_np, fid,
                seed_base=self.seed + idx * M * self.num_modes,
            )
            R_m, w_c = self._s1_marginalize(rewards_mc)
            mode_resp_history.append(w_c.mean(axis=0))

            noise = jax.random.normal(k_noise, (self._D,), dtype=jnp.float32)
            Ybar, weights, ess = self._denoise_jit(
                Y0s, jnp.asarray(R_m, dtype=jnp.float32), tau_k, T_k, noise,
            )

            mean_R = float(np.mean(R_m))
            mean_env_return = float(np.mean(rewards_mc))
            _top_k.append((mean_env_return, np.asarray(Ybar)))
            _top_k.sort(key=lambda t: -t[0])
            _top_k = _top_k[:top_k_cap]

            bridge_history.append({
                "k_reverse_idx": int(idx),
                "k_forward": int(kprime),
                "sigma_k": float(sigma_k),
                "tau_k": float(tau_k),
                "T_k": float(T_k),
                "fidelity_level": fid,
                "mean_R_s1": mean_R,
                "mean_env_return": mean_env_return,
                "ess": float(ess),
                "wall_time_ms": (time.perf_counter() - t0) * 1e3,
            })

        best_fine_return, best_fine_theta, fine_calls = self._fine_revalidate(_top_k, Ybar)
        if best_fine_return > -float("inf"):
            Ybar = jnp.asarray(best_fine_theta, dtype=jnp.float32)

        wall = time.perf_counter() - wall_start
        x_final, phi_final = self.theta_param.unpack(Ybar)

        return {
            "theta": np.asarray(Ybar),
            "x": np.asarray(x_final),
            "phi": np.asarray(phi_final),
            "bridge_history": bridge_history,
            "mode_responsibilities": np.array(mode_resp_history),
            "fidelity_history": fidelity_history,
            "fine_calls": fine_calls,
            "best_fine_return": best_fine_return if best_fine_return > -float("inf") else None,
            "top_k_candidates": [
                {"reward": float(s), "theta": np.asarray(t).tolist()} for s, t in _top_k
            ],
            "wall_clock": wall,
        }

    # --------- Dispatcher ---------------------------------------------------

    def plan(
        self,
        theta_init: Optional[Any] = None,
        rng_key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(self.seed))
        if self._is_jax_direct():
            return self._plan_jax_scan(theta_init, rng_key)
        return self._plan_python_loop(theta_init, rng_key)
