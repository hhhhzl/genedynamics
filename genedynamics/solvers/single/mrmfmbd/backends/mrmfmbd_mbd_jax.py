"""
MR-MF-MBD METHOD backend: theory-faithful model-based-diffusion co-design sampler
(new_version.txt "New Co-Design"). The legacy fixed-ladder / control-variate
engine is frozen separately in the `mrmfmbd_ablation` package.

One reverse-diffusion step (mean-tracking practical form; new_version.txt
special-case (3)), now with RISK-SENSITIVE regime marginalization (Stage 2) and
per-step BUDGETED ADAPTIVE FIDELITY SELECTION (Stage 3):

    for idx in reversed(range(K)):              # single jax.lax.scan, carries nu
        Y_m   = Ybar + σ_k · scale ⊙ ε_m         # propose M clean candidates
        for ℓ in levels:                         # roll out ALL fidelity levels
            ρ_ℓ, q_ℓ = risk_marginalize(R_{m,c}@ℓ)   # eq:regime_risk_objective
        ℓ*    = argmax_ℓ [ V̂_k(ℓ) − ν · C_ℓ ]    # eq:fidelity_choice (state-coupled)
        w_m   = softmax((ρ_{ℓ*} − ρ̄)/(σ_ρ·T_k) − ν·C_{ℓ*})   # −νC is softmax-inert
        Ybar  = Σ_m w_m · Y_m + τ_k · scale ⊙ ξ  # mean-tracking update
        ν     ← [ ν + η·(C_{ℓ*} − C̄) ]_+         # eq:dual_update

Final design is certified by the high-fidelity risk objective ρ_H
(eq:high_fidelity_risk), not the cross-regime mean. Execution requires a
JAX-direct evaluator (`._scene`/`._mpm_cfg`/`._mode_friction`); `plan()` asserts
this (the python-loop fallback is legacy-only — use `mrmfmbd_ablation`).
"""

from __future__ import annotations

import dataclasses
import os
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

# Stage 3: the method path selects fidelity ADAPTIVELY per step inside the scan
# (no fixed ladder). The legacy create_fidelity_ladder / BlockFidelityLadder now
# live only in the frozen mrmfmbd_ablation package.


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
    # Regime-marginalization flavor. METHOD DEFAULT (Stage 2 of the theory-faithful
    # refactor) is the risk-sensitive objective
    #   rho = -tau_r * logsumexp_m(log p(m) - R/tau_r)   (new_version.txt eq:regime_risk_objective)
    # which weights up failure-prone regimes. The legacy "reward" (easy-regime
    # logsumexp(log p + R/T)) and "cvar" modes are kept ONLY for the frozen
    # ablation engine (mrmfmbd_ablation) and must not be used on the method path.
    # tau_r controls soft-min sharpness; tau_r → 0 = worst-mode (max-min), tau_r → ∞ = mean.
    regime_posterior_mode: str = "risk_sensitive"
    risk_temperature: float = 1.0
    # Phase 2.3 / G2d (Stage 4): CVaR_α tail level for regime_posterior_mode="cvar".
    # Smaller α → more worst-case (α→0 = max-min over regimes); α=1 = prior-mean.
    cvar_alpha: float = 0.1
    # Phase 2 (Stage 2 / G2a): A2 shape-latent decoder. When morph_latent_dim>0
    # AND morph_decoder_path is set, the optimizer's x-block is a shape latent
    # w ∈ R^morph_latent_dim (≈ N(0,I)); rollout-time occupancy is x_morph = g(w)
    # from a trained MorphDecoder (loaded by the baseline, passed in as
    # `morph_decoder`). morph_latent_dim=0 (default) → legacy voxel-occupancy
    # x-block; the decode branch is skipped and behavior is byte-identical.
    morph_latent_dim: int = 0
    morph_decoder_path: str = ""
    # Task-2 (DiffuseBot-aligned co-design fields): when the loaded decoder is
    # multi-head, also feed its decoded continuous actuator-placement field and
    # stiffness field into the rollout (DiffuseBot Ψ = {geo, actuator, stiffness},
    # but optimized GRADIENT-FREE via the shape latent w). Default True → a
    # multi-head decoder uses both heads; set False to ablate a single head while
    # keeping the geometry decode. Inert for a legacy occ-only decoder.
    morph_decoder_actuator: bool = True
    morph_decoder_stiffness: bool = True
    # Stage 4: unified importance weight. The model-free morphology prior
    # p_theta^MF(w) over the x-block is MANDATORY (block-split, per-dim normalized);
    # mf_prior_weight scales it. p0^MB(phi) over the controller block is scaled by
    # mb_prior_weight (default 0 for the sinusoidal controller). The hard validity
    # indicator I(z) (eq:validity_indicator) drives non-connected / too-sparse
    # decoded morphologies out of the softmax; active only with a decoder + grid.
    mf_prior_weight: float = 0.2
    mb_prior_weight: float = 0.0
    validity_enabled: bool = True
    validity_min_occupied_frac: float = 0.03
    validity_connect_iters: int = 24
    # Budgeted adaptive multi-fidelity. True = per-step V-hat over ALL levels ->
    # argmax[V-hat - nu*C] (enumerate-all; rolls every level each step). False = roll
    # ONE fixed FINE fidelity per step (no V-hat, ~1.65x faster, cleanest search signal).
    # The adaptive-fidelity THEORY is untouched — this only disables it for a run; a
    # faithful cheap-V-hat (control-variate/bandit) impl is the proper way to keep it on.
    fidelity_adaptive: bool = True
    # Faithful budgeted adaptive fidelity (eq:fidelity_value_estimator): estimate V-hat
    # from a small PROBE (fidelity_probe candidates rolled at all L levels), pick
    # ell*=argmax[V-hat-nu*C], then roll ALL M candidates at ONLY ell* via lax.switch.
    # Cost ~ fidelity_probe*L + M (vs enumerate-all's M*L), so adaptive fidelity now
    # actually SAVES compute (it picks cheap levels). fidelity_enumerate=True restores
    # the old roll-all-levels path (ablation / exactness check).
    fidelity_probe: int = 8       # rank-fidelity V-hat needs enough probe pairs (npb=8 -> 28 pairs)
    fidelity_enumerate: bool = False
    # Memory-bounded large-M: roll candidates in chunks of rollout_chunk (via lax.map) and
    # accumulate, so the MBD weighted-mean (a SUM over candidates) uses an arbitrarily large M
    # at peak memory ~ rollout_chunk rollouts. 0 = off (single-shot). Required for M>=128 and
    # the [13,8,13] decoder / model-free prior where a full-M vmap OOMs the GPU.
    rollout_chunk: int = 0
    # (Stage 3) Control-variate multi-fidelity is REMOVED from the method path;
    # the method selects fidelity adaptively per step inside the scan. The
    # adaptive-fidelity knobs (vhat_a1..a4, fidelity_eta_nu, fidelity_cbar,
    # fidelity_costs) are read from `extra`. The legacy cv_* knobs live only in
    # mrmfmbd_ablation.LegacyMBDConfig.
    # Stage 5: learned closed-loop controller (new_version.txt §IV-A). With
    # controller_type='learned' the controller block IS the latent c (z^MB, width
    # = phi_dim or controller_latent_dim); a fixed MLP policy pi conditioned on
    # [proprio, psi(t), e_x, c] drives the actuators. 'sinusoid' (default) keeps
    # the SinWaveOpenLoop controller (the new-theory special case / ablation).
    controller_type: str = "sinusoid"
    controller_latent_dim: int = 0      # d_c when learned (0 -> use phi_dim, set by codesign)
    controller_hidden: int = 32
    controller_embed_dim: int = 8
    controller_seed: int = 0
    controller_path: str = ""        # load a pretrained policy (beta/E_chi/V_psi); skips re-init
    policy_warmup_save: str = ""     # after warmup, persist the trained controller here
    # In-loop per-candidate SHAC refinement of c (eq:controller_latent_refinement):
    # c+ = c + eta_c * grad_c Qhat_h ; the proximal lambda_prox*||c+ - c||^2 enters
    # the importance weight (eq:shac_regime_weight). eta_c=0 -> pure MCSA (default).
    eta_c: float = 0.0
    shac_h: int = 16
    shac_gamma: float = 0.99
    lambda_prox: float = 1.0
    # Stage 5b in-loop critic (eq:shac_q V_psi bootstrap), trained ONLINE by
    # TD(lambda) inside the scan. critic_enabled=False -> V_psi unused (Qhat =
    # pure h-step return), i.e. exact no-critic behavior.
    critic_enabled: bool = False
    critic_hidden: int = 32
    critic_lr: float = 1.0e-3
    td_lambda: float = 0.95
    # Differentiable-physics (SHAC) policy WARMUP: before diffusion, pretrain the
    # policy beta + embedding E_chi + critic V_psi on sampled morphologies + random
    # controller latents, by backprop of the short-horizon return through the
    # differentiable MPM. No dataset; data = on-policy sim rollouts. 0 -> skip.
    policy_warmup_steps: int = 0
    policy_lr: float = 3.0e-3
    warmup_batch: int = 16
    warmup_h: int = 40
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


def compute_vhat(rho_L, q_L, T_k, a1, a2, a3, a4, logC):
    """Per-fidelity value-of-information V_hat_k(ell) (new_version.txt
    eq:fidelity_value_estimator). One permitted instantiation — the theory
    explicitly allows others; the binding requirement is that the choice is
    COUPLED to the current candidate batch, not a fixed coarse->fine schedule.

    rho_L : (L, M) per-level risk-sensitive candidate scores.
    q_L   : (L, M, C) per-level regime posteriors.
    Returns V_hat (L,). Terms (candidate-disagreement variance, regime entropy,
    low/high mismatch, inverse-ESS degeneracy) are normalized to comparable O(1)
    scales so the argmax is not dominated by a single term.
    """
    R_std = jnp.maximum(jnp.std(rho_L), jnp.asarray(1e-4, jnp.float32))
    var_l = jnp.var(rho_L, axis=1)                                 # (L,)
    qbar = jnp.mean(q_L, axis=1)                                   # (L, C)
    ent_l = -jnp.sum(qbar * jnp.log(qbar + 1e-12), axis=1)         # (L,)
    mean_l = jnp.mean(rho_L, axis=1)                               # (L,)
    mis_l = jnp.abs(mean_l - mean_l[-1])                           # (L,) vs highest level

    def _ess(row):
        lw = (row - jnp.mean(row)) / (jnp.maximum(jnp.std(row), 1e-4) * T_k)
        w = jax.nn.softmax(lw)
        return 1.0 / jnp.sum(w * w)
    ess_l = jax.vmap(_ess)(rho_L)                                  # (L,) in [1, M]
    return (a1 * var_l / (R_std ** 2)
            + a2 * ent_l / logC
            + a3 * mis_l / R_std
            - a4 * (1.0 / ess_l))


def compute_vhat_rank(rho_L, q_L, T_k, a1, a2, a3, a4, logC, beta, rank_thresh, kappa):
    """Rank-fidelity V_hat_k(ell) (eq:fidelity_value_estimator, rank-faithful instantiation).

    The denoise is a softmax over candidate scores -> it consumes ONLY the candidate
    *ranking* (a per-level constant mean shift is provably inert). So the value of a level
    is how well it ORDERS candidates like the fine level, NOT how its mean differs (the old
    `mis_l` term: softmax-inert AND structurally inflated for coarse levels -> it selected
    cheap-but-rank-WRONG levels and denoised on a corrupted ordering).

    This replaces `mis_l` with a soft-Kendall agreement of level-ell's candidate ordering
    vs fine's, WEIGHTED by the fine softmax weights so the top-of-leaderboard pairs the
    denoise actually mixes count most. A soft gate (kappa*relu(thresh - rankfid)) disqualifies
    levels that scramble the fine ranking -> argmax[V_hat - nu*C] returns the CHEAPEST
    rank-faithful level, and FALLS BACK TO FINE when no cheap level is faithful (rankfid[H]=1
    always passes) -> adaptive is never worse than single-fine on quality.

    rho_L : (L, npb) per-level scores for the SAME npb probe candidates (paired across levels).
    Returns V_hat (L,).
    """
    R_std = jnp.maximum(jnp.std(rho_L), jnp.asarray(1e-4, jnp.float32))
    var_l = jnp.var(rho_L, axis=1)                                 # (L,)
    qbar = jnp.mean(q_L, axis=1)                                   # (L, C)
    ent_l = -jnp.sum(qbar * jnp.log(qbar + 1e-12), axis=1)         # (L,)

    def _ess(row):
        lw = (row - jnp.mean(row)) / (jnp.maximum(jnp.std(row), 1e-4) * T_k)
        w = jax.nn.softmax(lw)
        return 1.0 / jnp.sum(w * w)
    ess_l = jax.vmap(_ess)(rho_L)                                  # (L,)

    # Rank fidelity vs the fine level (last index = highest fidelity).
    rho_fine = rho_L[-1]                                           # (npb,)
    w_fine = jax.nn.softmax((rho_fine - jnp.mean(rho_fine))
                            / (jnp.maximum(jnp.std(rho_fine), 1e-4) * T_k))   # (npb,)
    P = jnp.tanh(beta * (rho_L[:, :, None] - rho_L[:, None, :]))   # (L, npb, npb) soft pairwise sign
    Pf = P[-1]                                                     # (npb, npb) fine ordering
    npb = rho_L.shape[1]
    wpair = (w_fine[:, None] + w_fine[None, :]) * jnp.triu(jnp.ones((npb, npb), jnp.float32), k=1)
    num = jnp.sum(jnp.sign(Pf)[None] * P * wpair[None], axis=(1, 2))          # (L,)
    den = jnp.sum(jnp.abs(Pf) * wpair) + 1e-8
    rankfid = num / den                                           # (L,) in [-1,1], fine -> ~1
    gate_pen = a3 * kappa * jax.nn.relu(rank_thresh - rankfid)    # disqualify rank-scramblers
    return (a1 * var_l / (R_std ** 2)
            + a2 * ent_l / logC
            + a3 * rankfid
            - a4 * (1.0 / ess_l)
            - gate_pen)


def dual_update_nu(nu, C_star, Cbar, eta):
    """Compute-budget dual update (eq:dual_update): nu <- [nu + eta*(C_star - Cbar)]_+."""
    return jnp.maximum(0.0, nu + eta * (C_star - Cbar))

def validity_logmask(occ01, vx, vy, vz, min_frac, connect_iters):
    """Hard validity indicator I(z) (new_version.txt eq:validity_indicator) ->
    log I in {0, -1e4}. occ01: (M, vx*vy*vz) occupancy in [0,1]. A morphology is
    valid iff its occupied fraction >= min_frac AND it is 6-connected (every
    occupied voxel is reachable from the first occupied voxel by `connect_iters`
    6-neighbour dilations). Invalid candidates are driven out of the softmax."""
    M = occ01.shape[0]
    g = (occ01.reshape(M, vx, vy, vz) > 0.5).astype(jnp.float32)
    occ_frac = jnp.mean(g, axis=(1, 2, 3))
    flat = g.reshape(M, -1)
    ramp = jnp.arange(flat.shape[1], 0, -1, dtype=jnp.float32)
    seed_idx = jnp.argmax(flat * ramp[None, :], axis=-1)              # first occupied voxel
    reached = jax.nn.one_hot(seed_idx, flat.shape[1]).reshape(M, vx, vy, vz) * g

    def _neigh_or(r):
        o = r
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (1, 0), (0, 0), (0, 0)))[:, :-1])
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (0, 1), (0, 0), (0, 0)))[:, 1:])
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (0, 0), (1, 0), (0, 0)))[:, :, :-1])
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (0, 0), (0, 1), (0, 0)))[:, :, 1:])
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (0, 0), (0, 0), (1, 0)))[..., :-1])
        o = jnp.maximum(o, jnp.pad(r, ((0, 0), (0, 0), (0, 0), (0, 1)))[..., 1:])
        return o

    for _ in range(int(connect_iters)):
        reached = _neigh_or(reached) * g
    n_reached = jnp.sum(reached, axis=(1, 2, 3))
    n_occ = jnp.maximum(jnp.sum(g, axis=(1, 2, 3)), 1.0)
    connected = n_reached / n_occ
    valid = (occ_frac >= min_frac) & (connected >= 0.999)
    return jnp.where(valid, 0.0, -1.0e4)


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
        morph_decoder: Optional[Any] = None,
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
        # Stage 2 / G2a: trained A2 shape-latent decoder (or None for the legacy
        # voxel-occupancy x-block). When set, the optimizer x-block is a latent w
        # and rollout-time occupancy is decoder.decode(w).
        self._morph_decoder = morph_decoder

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

        # Fidelity is chosen adaptively per step inside the scan (Stage 3); the
        # fixed coarse->fine ladder is gone from the method path. `fidelity_ladder`
        # is accepted for signature compat but unused.
        self.fidelity_ladder = fidelity_ladder

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

        # Single adaptive-fidelity scan runner, built + cached on first plan().
        self._scan_runner_cached = None
        self._controller_cache = None

    # --------- JIT kernels (shared) -----------------------------------------

    def _clip_theta_jnp(self, theta: jnp.ndarray) -> jnp.ndarray:
        x_part = jnp.clip(theta[..., : self._x_dim], self._x_lo, self._x_hi)
        phi_part = jnp.clip(theta[..., self._x_dim :], self._phi_lo, self._phi_hi)
        return jnp.concatenate([x_part, phi_part], axis=-1)

    # --------- Softzoo-compat path: external rollouts -----------------------

    def _expand_x_np(self, x_opt: np.ndarray) -> np.ndarray:
        """Expand optimizer's x to full voxel occupancy for rollout.

        With an A2 decoder (G2a) the x-block is a shape latent w → decode to
        occupancy. Otherwise (legacy) it is occupancy, possibly the z-symmetric
        half that gets mirrored to the full grid.
        """
        if self._morph_decoder is not None:
            w = jnp.asarray(np.asarray(x_opt, dtype=np.float32))
            return np.asarray(self._morph_decoder.decode(w), dtype=np.float32)
        if self._z_sym and self._voxel_dims is not None:
            vx, vy, vz = self._voxel_dims
            vz_half = vz // 2
            h = x_opt.reshape(vx, vy, vz_half)
            return np.concatenate([h, np.flip(h, axis=-1)], axis=-1).reshape(-1)
        return x_opt

    # --------- JAX-direct fast path: scan block runner ----------------------

    def _is_jax_direct(self) -> bool:
        ev = self.evaluator
        return (
            hasattr(ev, "_scene")
            and hasattr(ev, "_mpm_cfg")
            and hasattr(ev, "_mode_friction")
        )

    def _learned_controller(self):
        """Build (policy_params, E_proj, d_c, occ_width) ONCE — shared by the scan
        and the closed-loop certification so both use the IDENTICAL controller."""
        if self._controller_cache is not None:
            return self._controller_cache
        from genedynamics.solvers.single.mrmfmbd.controller_system import (
            init_policy, init_embed, obs_dim, init_critic, state_dim)
        cfg = self.evaluator._mpm_cfg
        x_dim = int(self._x_dim)
        if self._morph_decoder is not None:
            occ_width = int(self._morph_decoder.cfg.n_voxels)
        elif self._z_sym and self._voxel_dims is not None:
            vx, vy, vz = self._voxel_dims
            occ_width = int(vx * vy * vz)
        else:
            occ_width = x_dim
        d_c = int(self._D - x_dim)
        d_e = int(self.config.controller_embed_dim)
        pp = init_policy(int(self.config.controller_seed), obs_dim(d_c, d_e),
                         int(self.config.controller_hidden), int(cfg.n_actuators))
        E = init_embed(int(self.config.controller_seed), occ_width, d_e)
        crit = init_critic(int(self.config.controller_seed), state_dim(d_e),
                           int(self.config.critic_hidden))
        cpath = str(self.config.controller_path)
        if cpath and os.path.exists(cpath):
            from genedynamics.solvers.single.mrmfmbd.controller_system.io import load_controller
            pp_l, E_l, crit_l = load_controller(cpath)
            pp, E = pp_l, E_l            # reuse a pretrained policy + embedding
            if crit_l:
                crit = crit_l
        self._controller_cache = (pp, E, d_c, occ_width, crit)
        return self._controller_cache

    def _warmup_policy(self, n_steps, batch, warmup_h, policy_lr):
        """Differentiable-physics (SHAC) policy warmup. Pretrain the policy beta,
        the morphology embedding E_chi and the critic V_psi by backprop of the
        short-horizon closed-loop return through the differentiable MPM over a batch
        of SAMPLED morphologies + random controller latents (+ TD(lambda) for V_psi).
        No external dataset; the 'data' is the on-policy simulator rollouts. The
        trained params replace the controller cache used by the scan + certification."""
        from genedynamics.envs.external.jax_mpm.scene import (
            rollout_h_closed_from_state as _rh, _init_carry as _ic)
        from genedynamics.solvers.single.mrmfmbd.controller_system import critic_sgd_step
        scene = self.evaluator._scene
        cfg = self.evaluator._mpm_cfg
        fr0 = float(np.asarray(self.evaluator._mode_friction, np.float32)[0])
        pp, E, d_c, occ_width, crit = self._learned_controller()
        gamma = float(self.config.shac_gamma); lam = float(self.config.td_lambda)
        clr = float(self.config.critic_lr); h = int(warmup_h)
        gammas = gamma ** jnp.arange(h, dtype=jnp.float32)
        dec = self._morph_decoder

        @jax.jit
        def _step(pp, E, crit, occ_b, c_b):
            def _ret(pp, E, occ, c):
                _, cv, cx = _rh(_ic(scene), jnp.int32(0), c, h, jnp.float32(fr0),
                                scene, cfg, occ, pp, E)
                # Net forward displacement over the window (NOT discounted velocity):
                # gamma<1 on a velocity integral lets a lunge-then-recoil score positive
                # while netting ~0 travel (the bobbing exploit). Net displacement = the
                # certified objective, so front-load+recoil nets exactly zero.
                return (cx[-1, 0] - cx[0, 0]), (cv, cx)

            def _loss(pp, E):
                rets, (cv, cx) = jax.vmap(lambda o, c: _ret(pp, E, o, c))(occ_b, c_b)
                return -jnp.mean(rets), (cv, cx)

            (nl, (cv, cx)), (gp, gE) = jax.value_and_grad(
                _loss, argnums=(0, 1), has_aux=True)(pp, E)
            pp = {k: pp[k] - policy_lr * jnp.nan_to_num(gp[k]) for k in pp}
            E = E - policy_lr * jnp.nan_to_num(gE)
            e_x = jnp.tanh(occ_b @ E)
            e_x_h = jnp.broadcast_to(e_x[:, None, :], (occ_b.shape[0], h, e_x.shape[-1]))
            states = jnp.concatenate([cx, cv, e_x_h], axis=-1)
            crit = critic_sgd_step(crit, states, cv[:, :, 0], gamma, lam, clr)
            return pp, E, crit, -nl

        key = jax.random.PRNGKey(int(self.config.controller_seed) + 7)
        every = max(1, int(n_steps) // 10)
        for it in range(int(n_steps)):
            key, kw, kc = jax.random.split(key, 3)
            if dec is not None:
                w = jax.random.normal(kw, (int(batch), int(dec.cfg.latent_dim)), jnp.float32)
                occ_b = dec.decode_batch(w)
            else:
                occ_b = jnp.clip(jax.random.uniform(kw, (int(batch), int(occ_width)),
                                                    dtype=jnp.float32), 0.2, 1.0)
            c_b = jax.random.normal(kc, (int(batch), int(d_c)), jnp.float32)
            pp, E, crit, ret = _step(pp, E, crit, occ_b, c_b)
            if self.show_tqdm and (it % every == 0 or it == int(n_steps) - 1):
                print(f"[policy-warmup {it+1}/{n_steps}] mean_return={float(ret):.4f}", flush=True)
        self._controller_cache = (pp, E, d_c, occ_width, crit)
        return self._controller_cache

    def _make_scan_runner(self):
        """Single jitted reverse-diffusion scan over ALL K steps with per-step
        BUDGETED ADAPTIVE FIDELITY SELECTION (Stage 3; new_version.txt §IV-B).

        Replaces the fixed coarse->fine block ladder. At each step the M clean
        candidates are rolled out at EVERY fidelity level; a value-of-information
        V_hat_k(ell) (eq:fidelity_value_estimator) coupled to the current batch is
        estimated; the level ell* = argmax_ell [V_hat_k(ell) - nu_k * C_ell]
        (eq:fidelity_choice) is selected; candidates are weighted by the
        risk-sensitive rho_{ell*} (with a -nu_k*C_{ell*} budget term that is
        within-step softmax-inert and only acts across steps via nu); the
        mean-tracking update is applied; and the compute dual nu is updated
        (eq:dual_update). nu and the per-step ell* live in the scan carry/outputs.
        """
        from genedynamics.envs.external.jax_mpm.scene import rollout_return_batch
        from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS

        scene = self.evaluator._scene
        cfg = self.evaluator._mpm_cfg
        friction_table = jnp.asarray(
            np.asarray(self.evaluator._mode_friction, dtype=np.float32)[: int(self.num_modes)]
        )  # (C,)

        M = int(self.config.M)
        D = int(self._D)
        C = int(self.num_modes)
        x_dim = int(self._x_dim)
        # Per-mode regime banks beyond friction (slope=gravity tilt / terrain / mass /
        # init). None unless the evaluator exposes them -> friction-only legacy path.
        def _mode_tbl(attr):
            v = getattr(self.evaluator, attr, None)
            return None if v is None else jnp.asarray(np.asarray(v, np.float32)[:C])
        mode_terrain    = _mode_tbl("_mode_terrain")      # (C,n,n) | None
        mode_gravity    = _mode_tbl("_mode_gravity")      # (C,3)   | None
        mode_mass_scale = _mode_tbl("_mode_mass_scale")   # (C,)    | None
        mode_init_vel   = _mode_tbl("_mode_init_vel")     # (C,3)   | None
        top_k_cap = max(int(self.config.top_k_fine), 1)
        T_mode = jnp.asarray(max(float(self.config.reward_temperature), 1e-8), dtype=jnp.float32)
        log_prior_c = self._log_prior_c

        from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
            risk_sensitive_marginalize_jax, cvar_marginalize_jax, validate_mode,
        )
        validate_mode(self.config.regime_posterior_mode)
        regime_mode = str(self.config.regime_posterior_mode)
        tau_r = jnp.asarray(max(float(self.config.risk_temperature), 1e-8), dtype=jnp.float32)
        cvar_alpha = jnp.asarray(min(max(float(self.config.cvar_alpha), 1e-6), 1.0), dtype=jnp.float32)

        morph_decoder = self._morph_decoder

        # Stage 4: MANDATORY block-split priors for the unified weight. p_theta^MF
        # on the morphology latent (x-block; with the A2 VAE decoder it IS w~N(0,I)),
        # p0^MB on the controller block (phi). Per-dim normalized so each is O(1).
        mf_prior_weight = jnp.asarray(float(self.config.mf_prior_weight), dtype=jnp.float32)
        mb_prior_weight = jnp.asarray(float(self.config.mb_prior_weight), dtype=jnp.float32)
        prior_mean = jnp.asarray(self.theta_prior._mean, dtype=jnp.float32)
        prior_log_std = jnp.asarray(self.theta_prior._log_std, dtype=jnp.float32)
        _log2pi = jnp.asarray(float(np.log(2.0 * np.pi)), dtype=jnp.float32)
        _xm, _xls = prior_mean[:x_dim], prior_log_std[:x_dim]
        _pm, _pls = prior_mean[x_dim:], prior_log_std[x_dim:]

        def _log_pmf(Y0s_in):
            z = (Y0s_in[:, :x_dim] - _xm) / jnp.exp(_xls)
            return -0.5 * jnp.mean(z * z + 2.0 * _xls + _log2pi, axis=-1)

        def _log_pmb(Y0s_in):
            z = (Y0s_in[:, x_dim:] - _pm) / jnp.exp(_pls)
            return -0.5 * jnp.mean(z * z + 2.0 * _pls + _log2pi, axis=-1)

        # ----- Budgeted adaptive-fidelity setup (static at trace time) ---------
        FID_LEVELS = sorted(int(l) for l in FIDELITY_STEPS)        # e.g. [0,1,2]
        ENV_STEPS = [int(FIDELITY_STEPS[l]) for l in FID_LEVELS]   # e.g. [30,100,200]
        L = len(FID_LEVELS)
        H_idx = L - 1                                              # highest-fidelity index
        FID_LEVEL_IDS = jnp.asarray(FID_LEVELS, dtype=jnp.int32)
        ex = self.config.extra or {}
        # Per-level cost C_ell, normalized to [0,1] (proportional to env-steps ==
        # rollout cost). Cbar = target average per-step cost for the dual.
        _costs = ex.get("fidelity_costs") or ENV_STEPS
        _costs = [float(c) for c in _costs][:L]
        _cmax = max(_costs) if max(_costs) > 0 else 1.0
        C_vec = jnp.asarray([c / _cmax for c in _costs], dtype=jnp.float32)        # (L,)
        _cbar = ex.get("fidelity_cbar", None)
        Cbar = jnp.asarray(float(_cbar) if _cbar is not None else float(np.mean(_costs) / _cmax),
                           dtype=jnp.float32)
        eta_nu = jnp.asarray(float(ex.get("fidelity_eta_nu", 0.05)), dtype=jnp.float32)
        a1 = float(ex.get("vhat_a1", 1.0)); a2 = float(ex.get("vhat_a2", 1.0))
        a3 = float(ex.get("vhat_a3", 0.5)); a4 = float(ex.get("vhat_a4", 1.0))
        logC = float(np.log(max(C, 2)))
        # Rank-fidelity V-hat (default): the value term measures how well a level ORDERS
        # candidates like fine (the only thing the denoise consumes), with a soft gate that
        # disqualifies rank-scramblers -> picks the cheapest rank-faithful level, else fine.
        _vhat_rank = bool(ex.get("vhat_rank", True))
        _rank_beta = float(ex.get("vhat_rank_beta", 8.0))
        _rank_thresh = float(ex.get("vhat_rank_thresh", 0.85))  # conservative: only a near-perfect
        _rank_kappa = float(ex.get("vhat_rank_kappa", 10.0))    # rank match unlocks a cheap level
        _fid_adaptive = bool(self.config.fidelity_adaptive)   # False -> single fine level/step
        _fid_enumerate = bool(self.config.fidelity_enumerate) # True -> old roll-all-levels (ablation)
        n_probe = max(1, int(self.config.fidelity_probe))     # candidates for the cheap V-hat probe
        rollout_chunk = int(self.config.rollout_chunk)        # 0=single-shot; >0=chunked large-M (mem-bounded)

        # Z-symmetry: optimizer x_dim is half; mirror to full before rollout.
        z_sym = self._z_sym and (morph_decoder is None)
        if z_sym and self._voxel_dims is not None:
            vx, vy, vz = self._voxel_dims
            vz_half = vz // 2
            x_dim_full = vx * vy * vz
        else:
            z_sym = False
            x_dim_full = x_dim

        def _propose_inline(center, sigma_eff, key):
            eps = jax.random.normal(key, (M, D), dtype=jnp.float32)
            Y = center[None, :] + sigma_eff * self._scale[None, :] * eps
            return self._clip_theta_jnp(Y)

        morph_use_act = (morph_decoder is not None and getattr(morph_decoder, "has_actuator", False)
                         and bool(self.config.morph_decoder_actuator))
        morph_use_stiff = (morph_decoder is not None and getattr(morph_decoder, "has_stiffness", False)
                           and bool(self.config.morph_decoder_stiffness))
        morph_codesign = morph_use_act or morph_use_stiff

        def _decode_morph(Y0s_in):
            x_opt = Y0s_in[:, :x_dim]
            if morph_codesign:
                occ, act, stiff = morph_decoder.decode_full_batch(x_opt)
                return occ, (act if morph_use_act else None), (stiff if morph_use_stiff else None)
            if morph_decoder is not None:
                return morph_decoder.decode_batch(x_opt), None, None
            elif z_sym:
                return _mirror_z_voxels(x_opt, vx, vy, vz_half), None, None
            return x_opt, None, None

        def _marginalize(rewards_mc_local):
            if regime_mode == "risk_sensitive":
                return risk_sensitive_marginalize_jax(rewards_mc_local, log_prior_c, tau_r)
            if regime_mode == "cvar":
                return cvar_marginalize_jax(rewards_mc_local, log_prior_c, cvar_alpha)
            return _s1_marginalize_jax(rewards_mc_local, log_prior_c, T_mode)

        def _rollout_at(x_full, phi_block, env_steps, aw_voxel=None, E_voxel=None):
            B = int(x_full.shape[0])                             # actual batch (M, or n_probe for the V-hat probe)

            # Roll a chunk of `b` candidates x C modes; returns (b, C). The MBD weighted mean
            # is a SUM over candidates, so chunking + concat is EXACT (only peak memory changes).
            def _roll_chunk(xb, pb, awb, evb):
                b = int(xb.shape[0])
                x_flat = jnp.repeat(xb, C, axis=0)               # (b*C, x_dim_full)
                ph_flat = jnp.repeat(pb, C, axis=0)              # (b*C, phi_dim | d_c)
                fr_flat = jnp.tile(friction_table, b)            # (b*C,)
                aw_flat = jnp.repeat(awb, C, axis=0) if awb is not None else None
                ev_flat = jnp.repeat(evb, C, axis=0) if evb is not None else None
                # Per-mode regime tables, tiled MODE-MINOR (row j*C+c = candidate j, mode c).
                th_flat = None if mode_terrain    is None else jnp.tile(mode_terrain,    (b, 1, 1))
                gv_flat = None if mode_gravity    is None else jnp.tile(mode_gravity,    (b, 1))
                ms_flat = None if mode_mass_scale is None else jnp.tile(mode_mass_scale, b)
                iv_flat = None if mode_init_vel   is None else jnp.tile(mode_init_vel,   (b, 1))
                if learned:                                      # Stage 5: closed-loop policy
                    rs, _ = _roll_closed_batch(
                        x_flat, ph_flat, fr_flat, scene, cfg, env_steps, policy_params, E_proj,
                        actuator_weight_voxel_batch=aw_flat, E_voxel_batch=ev_flat,
                        terrain_height_batch=th_flat, gravity_vec_batch=gv_flat,
                        mass_scale_batch=ms_flat, init_vel_batch=iv_flat)
                else:                                            # sinusoid (default / ablation)
                    rs, _ = rollout_return_batch(
                        x_flat, ph_flat, fr_flat, scene, cfg, env_steps,
                        actuator_weight_voxel_batch=aw_flat, E_voxel_batch=ev_flat,
                        terrain_height_batch=th_flat, gravity_vec_batch=gv_flat,
                        mass_scale_batch=ms_flat, init_vel_batch=iv_flat)
                return rs.reshape(b, C)

            # Memory-bounded large-M: roll in chunks of rollout_chunk via lax.map (sequential
            # over chunks, vmapped within) so peak memory ~ rollout_chunk*C rollouts, not B*C.
            rc = rollout_chunk if (rollout_chunk and 0 < rollout_chunk < B and B % rollout_chunk == 0) else 0
            if rc == 0:
                return _roll_chunk(x_full, phi_block, aw_voxel, E_voxel)
            nchunk = B // rc
            def _resh(a):
                return a.reshape((nchunk, rc) + a.shape[1:])
            if aw_voxel is not None:
                out = jax.lax.map(lambda a: _roll_chunk(a[0], a[1], a[2], a[3]),
                                  (_resh(x_full), _resh(phi_block), _resh(aw_voxel), _resh(E_voxel)))
            else:
                out = jax.lax.map(lambda a: _roll_chunk(a[0], a[1], None, None),
                                  (_resh(x_full), _resh(phi_block)))
            return out.reshape(B, C)

        # Stage 4: hard validity indicator I(z) (eq:validity_indicator). occ -> occ01
        # in [0,1] (decoder: via its [x_lo,x_hi]; legacy occ: via theta bounds).
        _sg = self._voxel_dims
        occ_width = int(morph_decoder.cfg.n_voxels) if morph_decoder is not None else int(x_dim_full)
        validity_enabled = (bool(self.config.validity_enabled) and _sg is not None
                            and (occ_width == int(np.prod(_sg))))
        if morph_decoder is not None:
            _olo, _ohi = float(morph_decoder.cfg.x_lo), float(morph_decoder.cfg.x_hi)
        else:
            _olo, _ohi = float(self._x_lo), float(self._x_hi)
        _min_occ = float(self.config.validity_min_occupied_frac)
        _connect_iters = int(self.config.validity_connect_iters)

        def _to_occ01(x_full):
            return jnp.clip((x_full - _olo) / max(_ohi - _olo, 1e-6), 0.0, 1.0)

        # ----- Stage 5: learned closed-loop controller + in-loop SHAC ----------
        controller_type = str(self.config.controller_type)
        eta_c = float(self.config.eta_c)
        shac_h = int(self.config.shac_h)
        shac_gamma = float(self.config.shac_gamma)
        lambda_prox = jnp.asarray(float(self.config.lambda_prox), dtype=jnp.float32)
        learned = (controller_type == "learned")
        shac_active = learned and (eta_c > 0.0)
        critic_enabled = bool(self.config.critic_enabled)
        critic_active = shac_active and critic_enabled
        critic_lr = float(self.config.critic_lr)
        td_lambda = float(self.config.td_lambda)
        if learned:
            from genedynamics.envs.external.jax_mpm.scene import (
                rollout_return_closed_batch as _roll_closed_batch,
                rollout_h_closed_from_state as _roll_h_closed, _init_carry as _ic)
            from genedynamics.solvers.single.mrmfmbd.controller_system import (
                critic_value, critic_sgd_step)
            policy_params, E_proj, _d_c_unused, _ow_unused, critic_init = self._learned_controller()
        else:
            _roll_closed_batch = _roll_h_closed = _ic = None
            policy_params = E_proj = critic_init = None

        def _rollout_levels_only(Y0s_in):
            """rho_L (L,M) for a (refined) candidate set — to score c+ at all levels."""
            x_full, aw_v, ev_v = _decode_morph(Y0s_in)
            phi_block = Y0s_in[:, x_dim:]
            rows = []
            for steps in ENV_STEPS:
                rho_l, _ = _marginalize(_rollout_at(x_full, phi_block, steps, aw_v, ev_v))
                rows.append(rho_l)
            return jnp.stack(rows, 0)

        def _shac_refine(Y0s_in, critic_params):
            """In-loop per-candidate SHAC (eq:controller_latent_refinement): refine c
            by one grad step of the short-horizon return Qhat_h (eq:shac_q, with the
            V_psi bootstrap when critic_active); also do one online TD(lambda) critic
            update. Returns (refined theta, proximal ||c+ - c||^2, new critic params)."""
            x_full, aw_v, ev_v = _decode_morph(Y0s_in)
            c0 = Y0s_in[:, x_dim:]
            fr0 = friction_table[0]
            gammas = shac_gamma ** jnp.arange(shac_h, dtype=jnp.float32)
            gh = shac_gamma ** shac_h

            def _Q(c_m, xm, aw_m, ev_m):
                _, com_v_h, com_x_h = _roll_h_closed(
                    _ic(scene), jnp.int32(0), c_m, shac_h, fr0, scene, cfg, xm,
                    policy_params, E_proj, actuator_weight_voxel=aw_m, E_voxel=ev_m)
                Q = com_x_h[-1, 0] - com_x_h[0, 0]   # net forward travel, not discounted velocity
                if critic_active:
                    e_x = jnp.tanh(xm @ E_proj)
                    s_h = jnp.concatenate([com_x_h[-1], com_v_h[-1], e_x])
                    Q = Q + gh * critic_value(jax.lax.stop_gradient(critic_params), s_h)
                return Q, (com_v_h, com_x_h)

            vg = jax.value_and_grad(_Q, argnums=0, has_aux=True)
            if morph_codesign:
                (_Qv, (cv, cx)), grad_c = jax.vmap(vg)(c0, x_full, aw_v, ev_v)
            else:
                (_Qv, (cv, cx)), grad_c = jax.vmap(
                    lambda cm, xm: vg(cm, xm, None, None))(c0, x_full)
            grad_c = jnp.nan_to_num(grad_c, nan=0.0, posinf=0.0, neginf=0.0)
            c_plus = jnp.clip(c0 + eta_c * grad_c, self._phi_lo, self._phi_hi)
            prox = jnp.sum((c_plus - c0) ** 2, axis=-1)
            if critic_active:                                  # online TD(lambda) value update
                e_x_all = jnp.tanh(x_full @ E_proj)            # (M, d_e)
                e_x_h = jnp.broadcast_to(e_x_all[:, None, :], (M, shac_h, e_x_all.shape[-1]))
                states = jnp.concatenate([cx, cv, e_x_h], axis=-1)   # (M, h, d_state)
                critic_params = critic_sgd_step(
                    critic_params, states, cv[:, :, 0], shac_gamma, td_lambda, critic_lr)
            return jnp.concatenate([Y0s_in[:, :x_dim], c_plus], axis=-1), prox, critic_params

        def _rollout_all_levels(Y0s_in):
            """Roll out all M candidates at EVERY fidelity level (decode once).
            Returns rho_L (L,M), q_L (L,M,C), rewards_L (L,M,C), occ01 (M,n_vox)|None."""
            x_full, aw_v, ev_v = _decode_morph(Y0s_in)
            phi_block = Y0s_in[:, x_dim:]
            occ01 = _to_occ01(x_full) if validity_enabled else None
            rho_rows, q_rows, rew_rows = [], [], []
            for steps in ENV_STEPS:
                rew = _rollout_at(x_full, phi_block, steps, aw_v, ev_v)   # (M,C)
                rho_l, q_l = _marginalize(rew)                            # (M,), (M,C)
                rho_rows.append(rho_l); q_rows.append(q_l); rew_rows.append(rew)
            return jnp.stack(rho_rows, 0), jnp.stack(q_rows, 0), jnp.stack(rew_rows, 0), occ01

        def _vhat(rho_L, q_L, T_k):
            # Coupled to the current candidate batch (eq:fidelity_value_estimator);
            # module-level compute_vhat[_rank] is unit-tested in test_adaptive_fidelity.
            if _vhat_rank:
                return compute_vhat_rank(rho_L, q_L, T_k, a1, a2, a3, a4, logC,
                                         _rank_beta, _rank_thresh, _rank_kappa)
            return compute_vhat(rho_L, q_L, T_k, a1, a2, a3, a4, logC)

        def _weighted_mean(Y0s_in, rho_in, T_k, nu, C_star, occ01, prox):
            R_mean = jnp.mean(rho_in)
            R_std = jnp.maximum(jnp.std(rho_in), jnp.asarray(1e-4, dtype=rho_in.dtype))
            # Unified importance weight (eq:final_importance_weight): standardized
            # risk reward + MF prior(x) + MB prior(phi) + hard validity I(z)
            # - lambda_prox*||c+ - c||^2 (in-loop SHAC) - nu*C (budget, softmax-inert).
            log_w = (rho_in - R_mean) / (R_std * T_k) - nu * C_star - lambda_prox * prox
            log_w = log_w + mf_prior_weight * _log_pmf(Y0s_in) + mb_prior_weight * _log_pmb(Y0s_in)
            if occ01 is not None:
                log_I = validity_logmask(occ01, _sg[0], _sg[1], _sg[2], _min_occ, _connect_iters)
                # all-invalid fallback: keep reward-only ranking instead of a
                # degenerate uniform softmax over -1e4.
                any_valid = jnp.any(log_I > -1.0e3)
                log_w = log_w + jnp.where(any_valid, log_I, 0.0)
            w = jax.nn.softmax(log_w)
            Yb = jnp.einsum("m,md->d", w, Y0s_in)
            return Yb, w

        def step(carry, xs):
            Ybar, rng_key, topk_theta, topk_score, nu, critic_params = carry
            sigma_k, tau_k, T_k = xs
            rng_key, k_eps, k_noise = jax.random.split(rng_key, 3)
            Y0s = _propose_inline(Ybar, sigma_k, k_eps)            # (M, D)

            if _fid_adaptive and _fid_enumerate:
                # Ablation: roll ALL levels every step (the old, costly enumerate-all).
                rho_L, q_L, rew_L, occ01 = _rollout_all_levels(Y0s)   # (L,M),(L,M,C),(L,M,C),(M,nvox)
                vhat = _vhat(rho_L, q_L, T_k)                          # (L,)
                ell_star = jnp.argmax(vhat - nu * C_vec)
                q_star = q_L[ell_star]; rew_star = rew_L[ell_star]
                C_star = C_vec[ell_star]; rho_at_ell = rho_L[ell_star]
            elif _fid_adaptive:
                # FAITHFUL cheap adaptive fidelity: estimate V-hat from a small PROBE
                # (n_probe candidates rolled at all L levels), pick ell*=argmax[V-hat-nu*C],
                # then roll ALL M candidates at ONLY ell* via lax.switch (only one static
                # branch executes). Cost ~ n_probe*L + M  vs enumerate's M*L. Theory intact.
                x_full, aw_v, ev_v = _decode_morph(Y0s)
                occ01 = _to_occ01(x_full) if validity_enabled else None
                phi_block = Y0s[:, x_dim:]
                npb = min(n_probe, M)
                xp = x_full[:npb]; php = phi_block[:npb]
                awp = aw_v[:npb] if aw_v is not None else None
                evp = ev_v[:npb] if ev_v is not None else None
                _rp, _qp = [], []
                for steps in ENV_STEPS:
                    _rw = _rollout_at(xp, php, steps, awp, evp)        # (npb, C) cheap probe
                    _rl, _ql = _marginalize(_rw)
                    _rp.append(_rl); _qp.append(_ql)
                vhat = _vhat(jnp.stack(_rp, 0), jnp.stack(_qp, 0), T_k)  # (L,)
                ell_star = jnp.argmax(vhat - nu * C_vec)
                C_star = C_vec[ell_star]
                def _mk_branch(steps):
                    return lambda xfph: _rollout_at(xfph[0], xfph[1], steps, aw_v, ev_v)
                rew_star = jax.lax.switch(ell_star, [_mk_branch(s) for s in ENV_STEPS],
                                          (x_full, phi_block))          # (M, C) at ell* only
                rho_at_ell, q_star = _marginalize(rew_star)
            else:
                # Non-adaptive: ONE fixed FINE fidelity per step (no enumerate-all, no
                # V-hat). ~1.65x faster + cleanest signal; adaptive theory untouched.
                x_full_s, aw_s, ev_s = _decode_morph(Y0s)
                occ01 = _to_occ01(x_full_s) if validity_enabled else None
                rew_star = _rollout_at(x_full_s, Y0s[:, x_dim:], ENV_STEPS[H_idx], aw_s, ev_s)  # (M,C)
                rho_at_ell, q_star = _marginalize(rew_star)
                ell_star = jnp.int32(H_idx); C_star = C_vec[H_idx]
                vhat = jnp.zeros((L,), dtype=jnp.float32)

            # Stage 5b in-loop SHAC (static branch). sinusoid / eta_c=0 ->
            # Y0s_plus=Y0s, prox=0, rho_star=rho[ell*] => exact pure MCSA.
            if shac_active:
                Y0s_plus, prox, critic_params = _shac_refine(Y0s, critic_params)
                rho_star = _rollout_levels_only(Y0s_plus)[ell_star]   # reward at z+ (selected level)
            else:
                Y0s_plus, prox, rho_star = Y0s, jnp.zeros((M,), jnp.float32), rho_at_ell

            Ybar_inner, weights = _weighted_mean(Y0s_plus, rho_star, T_k, nu, C_star, occ01, prox)
            noise = jax.random.normal(k_noise, (D,), dtype=jnp.float32)
            Ybar_next = self._clip_theta_jnp(Ybar_inner + tau_k * self._scale * noise)
            ess = 1.0 / jnp.sum(weights * weights)

            # Compute dual update (eq:dual_update): nu <- [nu + eta*(C_{ell*} - Cbar)]_+
            nu_next = dual_update_nu(nu, C_star, Cbar, eta_nu)

            cand_scores = rho_star                                 # risk-sensitive leaderboard
            all_theta = jnp.concatenate([topk_theta, Y0s_plus], axis=0)
            all_scores = jnp.concatenate([topk_score, cand_scores], axis=0)
            _, top_idx = jax.lax.top_k(all_scores, top_k_cap)
            new_topk_theta = all_theta[top_idx]
            new_topk_score = all_scores[top_idx]

            out = {
                "sigma_k": sigma_k, "tau_k": tau_k, "T_k": T_k,
                "mean_R_s1": jnp.mean(rho_star),
                "mean_env_return": jnp.mean(rew_star),
                "ess": ess,
                "w_c_mean": jnp.mean(q_star, axis=0),              # (C,)
                "Ybar": Ybar_next,
                "fidelity_level": FID_LEVEL_IDS[ell_star],         # actual level id
                "nu_k": nu,
                "vhat": vhat,                                      # (L,)
            }
            return (Ybar_next, rng_key, new_topk_theta, new_topk_score, nu_next, critic_params), out

        @jax.jit
        def run_scan(Ybar, rng_key, topk_theta, topk_score, nu0, critic0, sigmas, taus, Ts):
            carry = (Ybar, rng_key, topk_theta, topk_score, nu0, critic0)
            (Ybar_f, rng_f, tk_t, tk_s, nu_f, _crit_f), out = jax.lax.scan(
                step, carry, (sigmas, taus, Ts))
            return Ybar_f, rng_f, tk_t, tk_s, nu_f, out

        return run_scan

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
        FID_LEVELS = sorted(int(l) for l in FIDELITY_STEPS)

        Ybar = self._init_ybar(theta_init)
        topk_theta = jnp.zeros((top_k_cap, D), dtype=jnp.float32)
        topk_score = jnp.full((top_k_cap,), -1e30, dtype=jnp.float32)

        # ONE jitted scan over all K reverse steps (idx = K-1 .. 0); per-step
        # fidelity is chosen adaptively inside the scan, so there is no host-side
        # block loop and no fidelity-keyed recompilation.
        # Differentiable-physics SHAC policy warmup (trains beta/E_chi/V_psi BEFORE
        # the scan captures them) — only for the learned controller.
        if (str(self.config.controller_type) == "learned"
                and int(self.config.policy_warmup_steps) > 0):
            self._warmup_policy(int(self.config.policy_warmup_steps),
                                int(self.config.warmup_batch),
                                int(self.config.warmup_h), float(self.config.policy_lr))
            spath = str(self.config.policy_warmup_save)
            if spath:
                from genedynamics.solvers.single.mrmfmbd.controller_system.io import save_controller
                _pp, _E, _dc, _ow, _cr = self._controller_cache
                save_controller(spath, _pp, _E, _cr)
        if getattr(self, "_scan_runner_cached", None) is None:
            self._scan_runner_cached = self._make_scan_runner()
        runner = self._scan_runner_cached

        idxs = np.arange(K - 1, -1, -1)
        sigmas = jnp.asarray(np.asarray(self._sigmas)[idxs])
        taus = jnp.asarray(np.asarray(self._taus)[idxs])
        Ts = jnp.asarray(np.asarray(self._T_schedule)[idxs])
        nu0 = jnp.asarray(0.0, dtype=jnp.float32)
        # In-loop critic params for the scan carry (real only when learned +
        # eta_c>0 + critic_enabled; else an empty pytree placeholder).
        if (str(self.config.controller_type) == "learned" and float(self.config.eta_c) > 0.0
                and bool(self.config.critic_enabled)):
            critic0 = self._learned_controller()[4]
        else:
            critic0 = {}

        wall_start = time.perf_counter()
        Ybar, rng_key, topk_theta, topk_score, nu_f, out = runner(
            Ybar, rng_key, topk_theta, topk_score, nu0, critic0, sigmas, taus, Ts)
        Ybar.block_until_ready()

        per_step = {k: np.asarray(out[k]) for k in (
            "sigma_k", "tau_k", "T_k", "mean_R_s1", "mean_env_return",
            "ess", "w_c_mean", "Ybar", "fidelity_level", "nu_k", "vhat")}
        fidelity_history = [int(x) for x in per_step["fidelity_level"]]

        bridge_history: List[Dict[str, Any]] = []
        total = per_step["sigma_k"].shape[0]
        for i in range(total):
            idx = K - 1 - i
            bridge_history.append({
                "k_reverse_idx": int(idx), "k_forward": int(i),
                "sigma_k": float(per_step["sigma_k"][i]),
                "tau_k": float(per_step["tau_k"][i]),
                "T_k": float(per_step["T_k"][i]),
                "fidelity_level": int(per_step["fidelity_level"][i]),
                "mean_R_s1": float(per_step["mean_R_s1"][i]),
                "mean_env_return": float(per_step["mean_env_return"][i]),
                "ess": float(per_step["ess"][i]),
                # Stage 3 budgeted-adaptive-fidelity diagnostics.
                "nu_k": float(per_step["nu_k"][i]),
                "vhat": [float(v) for v in np.asarray(per_step["vhat"][i]).ravel()],
            })
        mode_resp_history = per_step["w_c_mean"]

        # Extract top-K candidates as Python list (sorted desc).
        topk_theta_np = np.asarray(topk_theta)
        topk_score_np = np.asarray(topk_score)
        order = np.argsort(-topk_score_np)
        _top_k: List[Tuple[float, np.ndarray]] = [
            (float(topk_score_np[i]), topk_theta_np[i])
            for i in order if topk_score_np[i] > -1e29
        ]

        # High-fidelity risk-sensitive certification (rho_H) of the top-K.
        best_fine_return, best_fine_theta, fine_calls = self._fine_revalidate(_top_k, Ybar)
        if best_fine_return > -float("inf"):
            Ybar = jnp.asarray(best_fine_theta, dtype=jnp.float32)

        # Stage 5b: SHAC is now IN-LOOP (per-candidate, inside the scan); the old
        # post-hoc top-K refinement is gone.
        shac_summary = {
            "in_loop": str(self.config.controller_type) == "learned" and float(self.config.eta_c) > 0.0,
            "controller_type": str(self.config.controller_type),
            "eta_c": float(self.config.eta_c),
        }

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
            "theta_history": per_step["Ybar"],
            "shac_summary": shac_summary,
            # Budgeted adaptive-fidelity diagnostic (Stage 3) — replaces the old
            # fixed-ladder / cv_adaptive summaries.
            "fidelity_summary": {
                "levels": list(FID_LEVELS),
                "per_step_level": fidelity_history,
                "per_step_nu": [float(x) for x in per_step["nu_k"]],
                "nu_final": float(nu_f),
                "level_counts": {int(l): int(sum(1 for f in fidelity_history if f == l))
                                 for l in FID_LEVELS},
            },
        }

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

        # Learned controller: certify via the closed-loop policy (the sine
        # evaluator path expects 80-dim phi, not the d_c-dim latent c).
        if str(self.config.controller_type) == "learned" and self._is_jax_direct():
            return self._fine_revalidate_closed(top_k, best_fine_theta)

        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest
        from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
            risk_sensitive_marginalize_np,
        )
        x_dim = self.theta_param.x_dim
        fine_level = int(self.config.fine_fidelity_level)
        # High-fidelity certification (eq:high_fidelity_risk): aggregate the
        # per-regime fine returns by the SAME risk-sensitive rho_H used in the loop.
        regime_mode = str(self.config.regime_posterior_mode)
        cert_tau_r = max(float(self.config.risk_temperature), 1e-8)
        cert_log_prior = np.asarray(self.mode_log_priors[: self.num_modes], dtype=np.float64)
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
            if regime_mode == "risk_sensitive":
                cand_return = float(risk_sensitive_marginalize_np(
                    np.asarray(res.returns)[: self.num_modes], cert_log_prior, cert_tau_r,
                ))
            else:
                cand_return = float(np.mean(res.returns))
            if cand_return > best_fine_return:
                best_fine_return = cand_return
                best_fine_theta = cand_np
        return best_fine_return, best_fine_theta, fine_calls

    def _fine_revalidate_closed(self, top_k, best_fine_theta):
        """rho_H certification via the LEARNED closed-loop policy (jax-direct)."""
        from genedynamics.envs.external.jax_mpm.scene import rollout_return_closed
        from genedynamics.envs.external.jax_mpm.adapters import FIDELITY_STEPS
        from genedynamics.solvers.single.mrmfmbd.mode_system.regime_posterior import (
            risk_sensitive_marginalize_np)
        scene = self.evaluator._scene
        cfg = self.evaluator._mpm_cfg
        fr_table = np.asarray(self.evaluator._mode_friction, dtype=np.float32)[: self.num_modes]
        def _ct(attr):
            v = getattr(self.evaluator, attr, None)
            return None if v is None else np.asarray(v, np.float32)[: self.num_modes]
        t_tab = _ct("_mode_terrain"); g_tab = _ct("_mode_gravity")
        m_tab = _ct("_mode_mass_scale"); iv_tab = _ct("_mode_init_vel")
        pp, E_proj, _d_c, _ow, _crit = self._learned_controller()
        x_dim = int(self._x_dim)
        fine_steps = int(FIDELITY_STEPS.get(int(self.config.fine_fidelity_level),
                                            FIDELITY_STEPS[max(FIDELITY_STEPS)]))
        tau_r = max(float(self.config.risk_temperature), 1e-8)
        log_prior = np.asarray(self.mode_log_priors[: self.num_modes], dtype=np.float64)
        best_ret = float("-inf"); best_theta = np.asarray(best_fine_theta); calls = 0
        for _s, cand in top_k:
            cand = np.asarray(cand, dtype=np.float32)
            occ = jnp.asarray(self._expand_x_np(cand[:x_dim]), dtype=jnp.float32)
            c = jnp.asarray(cand[x_dim:], dtype=jnp.float32)
            rets = []
            for ci, fr in enumerate(fr_table):
                r, _, _ = rollout_return_closed(
                    occ, c, jnp.asarray(float(fr), jnp.float32), scene, cfg, fine_steps, pp, E_proj,
                    terrain_height=(None if t_tab is None else jnp.asarray(t_tab[ci])),
                    gravity_vec=(None if g_tab is None else jnp.asarray(g_tab[ci])),
                    mass_scale=(None if m_tab is None else float(m_tab[ci])),
                    init_vel=(None if iv_tab is None else jnp.asarray(iv_tab[ci])))
                rets.append(float(r)); calls += 1
            cand_ret = float(risk_sensitive_marginalize_np(np.asarray(rets), log_prior, tau_r))
            if cand_ret > best_ret:
                best_ret = cand_ret; best_theta = cand
        return best_ret, best_theta, calls

    # --------- Softzoo-compat plan path -------------------------------------

    # --------- Dispatcher ---------------------------------------------------

    def plan(
        self,
        theta_init: Optional[Any] = None,
        rng_key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(self.seed))
        # The method path requires a JAX-direct evaluator (._scene/._mpm_cfg/
        # ._mode_friction). The python-loop fallback uses legacy reward-mean
        # marginalization (_s1_marginalize) and cannot honor the risk-sensitive /
        # rho_H method objective, so it is not a valid method execution. Run the
        # frozen engine via the 'mrmfmbd_ablation' solver if a non-JAX evaluator
        # is genuinely required.
        if not self._is_jax_direct():
            raise RuntimeError(
                "MR-MF-MBD method path requires a JAX-direct evaluator "
                "(exposing ._scene/._mpm_cfg/._mode_friction); got "
                f"{type(self.evaluator).__name__}. The python-loop fallback is "
                "legacy-only — use the 'mrmfmbd_ablation' solver for it."
            )
        return self._plan_jax_scan(theta_init, rng_key)
