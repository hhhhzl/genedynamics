"""
Level-3 MBD backend for MR-MF-MBD: a model-based-diffusion posterior sampler.

Algorithm (reverse-time denoising diffusion applied to θ = (x, φ)):

    for idx in reversed(range(K)):
        σ_k  = sqrt(1 - ᾱ_k)                     # DDPM noise schedule
        Y_m  = Ybar + σ_k · scale ⊙ ε_m           # forward-corrupt M proposals
        R_m  = logsumexp_c(log π(c) + R_{m,c}/T)  # S1 mode marginalization
        w_m  = softmax((R_m - R̄) / (σ_R · T_k))  # reward-temp weights
        Ybar = Σ_m w_m · Y_m                      # denoise = reward-weighted mean
        Ybar = Ybar + τ_k · scale ⊙ ξ            # vanishing exploration noise

Compared to the old MCSA `mrmfmbd_posterior_jax.MRMFMBDPosteriorBackendJax`:
    - Update rule is MBD weighted-mean (not MCSA score ascent with η decoupled from σ).
    - Iteration is reversed (denoising), not forward annealing.
    - σ is **per-dim scaled** by `scale_d = (hi_d - lo_d) / 2` so every param
      dimension is perturbed in its natural range (fixes the "φ never moves"
      bug where a single scalar σ was applied across x∈(0.01,2) and φ∈(5,150)).
    - No explicit `log p₀(θ)` term: prior enters via `Ybar_init` (= prior mean).
    - S1 stays: `_compute_marginal_log_likelihood_and_responsibilities`.
    - S3 stays: `fidelity_ladder(K-1-idx)` returns coarse early, fine late.

JIT strategy:
    softzoo rollouts are external (Python subprocess with Taichi MPM), so the
    reverse-time loop cannot live inside `jax.lax.scan`. Everything else runs
    inside two JIT-compiled step kernels:
        * `_propose_jit`  — forward-corrupt proposals
        * `_denoise_jit`  — reward-weighted denoise + vanishing noise
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
    from genedynamics.solvers.single.mrmfmbd.s1_mode_system.marginalizer import (
        ModeMarginalizerS1,
        default_mode_system_config,
    )
    S1_AVAILABLE = True
except ImportError:
    S1_AVAILABLE = False
    ModeMarginalizerS1 = None
    default_mode_system_config = None

try:
    from genedynamics.solvers.single.mrmfmbd.s3_fidelity_system.ladder import create_fidelity_ladder
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    create_fidelity_ladder = None


@dataclass
class MBDConfig:
    """Configuration for the Level-3 MBD backend.

    Attributes:
        K: Number of reverse-diffusion steps.
        M: Proposals per step.
        beta0: DDPM forward noise rate at step 0.
        betaT: DDPM forward noise rate at step K-1.
        reward_temperature: Base temperature T0 for weight softmax.
        reward_temperature_min: Final temperature TK (anneal T over K).
        tau_frac: Exploration noise fraction (τ_k = tau_frac · σ_k).
        top_k_fine: How many snapshots to re-validate at fine fidelity.
        fine_fidelity_level: Fidelity level for top-K validation.
    """

    K: int = 16
    M: int = 8
    beta0: float = 1e-4
    betaT: float = 2e-2
    reward_temperature: float = 0.5
    reward_temperature_min: float = 0.1
    tau_frac: float = 0.1
    top_k_fine: int = 3
    fine_fidelity_level: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)


class MRMFMBDBackendMBD:
    """Level-3 MBD backend: DDPM-style reverse diffusion over θ with S1+S3 hooks."""

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

        K = int(self.config.K)

        # S1: mode marginalizer (same module the MCSA backend uses)
        self.mode_marginalizer = mode_marginalizer
        if self.mode_marginalizer is None and S1_AVAILABLE:
            self.mode_marginalizer = ModeMarginalizerS1(
                default_mode_system_config(num_modes),
                backend="jax",
            )

        # S3: fidelity ladder (coarse early, fine late on *reverse* index)
        self.fidelity_ladder = fidelity_ladder
        if self.fidelity_ladder is None and S3_AVAILABLE and create_fidelity_ladder is not None:
            self.fidelity_ladder = create_fidelity_ladder(
                K=K, num_levels=3, ladder_type="geometric", step_ratio=2.0
            )

        self._D = theta_param.theta_dim

        # Per-dim scale: half the bound-range. Used to make σ meaningful for
        # both x (range ≈ 2) and φ (range ≈ 145) simultaneously.
        x_lo, x_hi = theta_param.x_bounds if theta_param.x_bounds else (-1.0, 1.0)
        p_lo, p_hi = theta_param.phi_bounds if theta_param.phi_bounds else (-1.0, 1.0)
        scale_x = np.full(theta_param.x_dim, (float(x_hi) - float(x_lo)) / 2.0, dtype=np.float32)
        scale_p = np.full(theta_param.phi_dim, (float(p_hi) - float(p_lo)) / 2.0, dtype=np.float32)
        self._scale_np = np.concatenate([scale_x, scale_p]).astype(np.float32)
        self._scale = jnp.asarray(self._scale_np)

        # DDPM noise schedule (pre-computed once). Capped at sigma_max so
        # very early (large-σ) proposals can't push per-dim perturbations
        # into numerically explosive territory — we saw phi ± 3σ·scale_phi
        # produce ω=150 rad/s extremes that blow up Taichi MPM particles.
        betas = np.linspace(self.config.beta0, self.config.betaT, K, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)
        sigmas = np.sqrt(np.maximum(1.0 - alphas_bar, 1e-8)).astype(np.float32)
        sigma_max = float(self.config.extra.get("sigma_max", 0.15))
        sigmas = np.minimum(sigmas, sigma_max).astype(np.float32)
        self._sigmas = jnp.asarray(sigmas)
        self._taus = jnp.asarray(sigmas * float(self.config.tau_frac))

        # Geometric anneal for the reward temperature T_k (larger→smaller).
        T0 = float(self.config.reward_temperature)
        TK = float(self.config.reward_temperature_min)
        if TK <= 0:
            TK = T0
        t_ratio = TK / max(T0, 1e-8)
        t_ratio = max(t_ratio, 1e-6)
        T_schedule = T0 * (t_ratio ** (np.arange(K, dtype=np.float32) / max(K - 1, 1)))
        self._T_schedule = jnp.asarray(T_schedule.astype(np.float32))

        # Bounds as jnp arrays for JIT-compatible clipping.
        self._x_lo = jnp.asarray(float(x_lo), dtype=jnp.float32)
        self._x_hi = jnp.asarray(float(x_hi), dtype=jnp.float32)
        self._phi_lo = jnp.asarray(float(p_lo), dtype=jnp.float32)
        self._phi_hi = jnp.asarray(float(p_hi), dtype=jnp.float32)
        self._x_dim = theta_param.x_dim

        # Pre-build JIT'd kernels. Static shapes (M, D) are baked in.
        self._propose_jit = jax.jit(self._propose_impl)
        self._denoise_jit = jax.jit(self._denoise_impl)

    # --------- JIT'd kernels -------------------------------------------------

    def _clip_theta_jnp(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Per-dim clip; works on shapes (D,) or (M, D)."""
        x_part = jnp.clip(theta[..., : self._x_dim], self._x_lo, self._x_hi)
        phi_part = jnp.clip(theta[..., self._x_dim :], self._phi_lo, self._phi_hi)
        return jnp.concatenate([x_part, phi_part], axis=-1)

    def _propose_impl(
        self,
        Ybar: jnp.ndarray,       # (D,)
        sigma_k: jnp.ndarray,    # scalar
        eps: jnp.ndarray,        # (M, D)
    ) -> jnp.ndarray:
        """Y0s = clip(Ybar + σ_k · scale ⊙ ε)."""
        Y0s = Ybar[None, :] + sigma_k * self._scale[None, :] * eps
        return self._clip_theta_jnp(Y0s)

    def _denoise_impl(
        self,
        Y0s: jnp.ndarray,        # (M, D)
        R_m: jnp.ndarray,        # (M,) — S1-marginalized log p(R|θ_m)
        tau_k: jnp.ndarray,      # scalar
        T_k: jnp.ndarray,        # scalar reward temperature at this step
        noise: jnp.ndarray,      # (D,)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Weighted-mean denoising + vanishing exploration.

        Returns (Ybar_next, weights, ess).
        """
        R_mean = jnp.mean(R_m)
        R_std = jnp.maximum(jnp.std(R_m), jnp.asarray(1e-4, dtype=R_m.dtype))
        log_w = (R_m - R_mean) / (R_std * T_k)
        weights = jax.nn.softmax(log_w)                                 # (M,)
        Ybar_next = jnp.einsum("m,md->d", weights, Y0s)                 # (D,)
        Ybar_next = Ybar_next + tau_k * self._scale * noise
        Ybar_next = self._clip_theta_jnp(Ybar_next)
        ess = 1.0 / jnp.sum(weights * weights)
        return Ybar_next, weights, ess

    # --------- External: softzoo rollout batch -------------------------------

    def _evaluate_proposals_modes(
        self,
        proposals: np.ndarray,   # (M, D)
        fidelity_level: int,
        seed_base: int,
    ) -> np.ndarray:
        """Run M × num_modes rollouts, return rewards shape (M, num_modes)."""
        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        M = proposals.shape[0]
        flat_requests: List[Any] = []
        index_pairs: List[Tuple[int, int]] = []
        x_dim = self.theta_param.x_dim
        for c in range(self.num_modes):
            for m in range(M):
                flat_requests.append(
                    RolloutRequest(
                        morphology_params=proposals[m, :x_dim],
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
        """Return (R_m marginal, w_c responsibilities) from (M, num_modes) rewards."""
        # Guard against NaN / ±inf rewards from blown-up rollouts (extreme
        # phi sometimes drives softzoo particles off grid → NaN COM → NaN
        # reward). Replacing with a very negative finite value lets MBD
        # softmax assign near-zero weight instead of propagating NaN into Ybar.
        rewards_mc = np.nan_to_num(
            rewards_mc, nan=-1e6, posinf=1e6, neginf=-1e6
        ).astype(np.float32)
        if self.mode_marginalizer is not None:
            result = self.mode_marginalizer(
                rewards_mc,
                temperature=float(self.config.reward_temperature),
            )
            return (
                np.asarray(result.marginal_log_likelihood, dtype=np.float32),
                np.asarray(result.responsibilities, dtype=np.float32),
            )
        # Inline fallback (same formula as THEORY.md).
        T = max(float(self.config.reward_temperature), 1e-8)
        log_prior_c = np.asarray(self.mode_log_priors[: self.num_modes], dtype=np.float32)
        log_terms = log_prior_c[None, :] + rewards_mc.astype(np.float32) / T
        marginal = np.asarray(
            np.log(np.sum(np.exp(log_terms - log_terms.max(axis=-1, keepdims=True)), axis=-1))
            + log_terms.max(axis=-1),
            dtype=np.float32,
        )
        w = np.exp(log_terms - np.log(np.sum(np.exp(log_terms - log_terms.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - log_terms.max(axis=-1, keepdims=True))
        return marginal, w.astype(np.float32)

    # --------- Main driver ---------------------------------------------------

    def _bounds_midpoint(self) -> np.ndarray:
        x_lo, x_hi = self.theta_param.x_bounds or (-1.0, 1.0)
        p_lo, p_hi = self.theta_param.phi_bounds or (-1.0, 1.0)
        mid = np.concatenate([
            np.full(self.theta_param.x_dim, (float(x_lo) + float(x_hi)) / 2.0, dtype=np.float32),
            np.full(self.theta_param.phi_dim, (float(p_lo) + float(p_hi)) / 2.0, dtype=np.float32),
        ])
        return mid

    def plan(
        self,
        theta_init: Optional[Any] = None,
        rng_key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        K = int(self.config.K)
        M = int(self.config.M)

        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(self.seed))

        # Ybar init: theta_init > prior mean > bounds midpoint.
        if theta_init is not None:
            Ybar = jnp.asarray(theta_init, dtype=jnp.float32)
        else:
            prior_mean = getattr(self.theta_prior, "_mean", None)
            if prior_mean is not None and np.all(np.abs(prior_mean) > 1e-8):
                Ybar = jnp.asarray(prior_mean, dtype=jnp.float32)
            else:
                Ybar = jnp.asarray(self._bounds_midpoint(), dtype=jnp.float32)
        Ybar = self._clip_theta_jnp(Ybar)

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

            # S3: reverse-time index k' = K-1-idx — coarse early (large idx),
            # fine late (small idx). Equivalent to "cheap when noise large".
            kprime = K - 1 - idx
            fid = int(self.fidelity_ladder(kprime)) if self.fidelity_ladder is not None else 0
            fidelity_history.append(fid)

            rng_key, k_eps, k_noise = jax.random.split(rng_key, 3)
            eps = jax.random.normal(k_eps, (M, self._D), dtype=jnp.float32)

            # --- JIT block 1: propose ---
            Y0s = self._propose_jit(Ybar, sigma_k, eps)
            Y0s_np = np.asarray(Y0s)

            # --- External: softzoo rollouts + S1 marginalization ---
            rewards_mc = self._evaluate_proposals_modes(
                Y0s_np, fid,
                seed_base=self.seed + idx * M * self.num_modes,
            )
            R_m, w_c = self._s1_marginalize(rewards_mc)
            mode_resp_history.append(w_c.mean(axis=0))

            # --- JIT block 2: denoise ---
            noise = jax.random.normal(k_noise, (self._D,), dtype=jnp.float32)
            Ybar, weights, ess = self._denoise_jit(
                Y0s, jnp.asarray(R_m, dtype=jnp.float32), tau_k, T_k, noise,
            )

            # Track top-K by this step's mean S1 marginal reward.
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

        # Top-K fine validation (same pattern as the old backend).
        best_fine_return = float("-inf")
        best_fine_theta = np.asarray(Ybar)
        fine_calls = 0
        if self.config.top_k_fine > 0 and _top_k:
            from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
            x_dim = self.theta_param.x_dim
            fine_level = int(self.config.fine_fidelity_level)
            for _score, cand in _top_k:
                cand_np = np.asarray(cand, dtype=np.float32)
                requests = [
                    RolloutRequest(
                        morphology_params=cand_np[:x_dim],
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
