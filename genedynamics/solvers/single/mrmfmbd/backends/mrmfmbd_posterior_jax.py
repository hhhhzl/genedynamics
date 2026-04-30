"""
JAX backend for MRMFMBD posterior bridge (theta = x, phi co-design).

True posterior bridge: π_k(θ) ∝ p0(θ) * p(R|θ)^β_k with MCSA score ascent.
- Mode marginalization (ModeMarginalizer) - log p(R|θ)=logsumexp(log p(c)+R_c/T)
- Multi-fidelity ladder (BlockFidelityLadder) - coarse→fine, cost-optimal
- Output: bridge_history, mode_responsibilities, fidelity_history, fine_calls
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

from genedynamics.core.inference.annealed_bridge import (
    BridgeSchedule,
    create_linear_bridge_schedule,
)
from genedynamics.core.inference.mcsa import ImportanceWeighter, MCSADiagnostics
from genedynamics.solvers.single.mrmfmbd.theta_prior import ThetaParametrization, ThetaPrior

# Mode + Fidelity systems (optional - backward compat with core.inference)
try:
    from genedynamics.solvers.single.mrmfmbd.mode_system import (
        ModeMarginalizer,
        ModeSystemConfig,
        default_mode_system_config,
    )
    MODE_SYSTEM_AVAILABLE = True
except ImportError:
    ModeMarginalizer = None
    ModeSystemConfig = None
    default_mode_system_config = None
    MODE_SYSTEM_AVAILABLE = False

try:
    from genedynamics.solvers.single.mrmfmbd.fidelity_system import (
        BlockFidelityLadder,
        create_fidelity_ladder,
        FidelitySystemConfig,
        default_fidelity_system_config,
    )
    FIDELITY_SYSTEM_AVAILABLE = True
except ImportError:
    BlockFidelityLadder = None
    create_fidelity_ladder = None
    FidelitySystemConfig = None
    default_fidelity_system_config = None
    FIDELITY_SYSTEM_AVAILABLE = False

# Fallback to core.inference
try:
    from genedynamics.core.inference.fidelity import FidelityConfig, FidelityLadder
    CORE_FIDELITY_AVAILABLE = True
except ImportError:
    FidelityConfig = None
    FidelityLadder = None
    CORE_FIDELITY_AVAILABLE = False


@dataclass
class PosteriorBridgeConfig:
    """
    Configuration for MRMFMBD posterior bridge.

    Attributes:
        K: Number of bridge steps
        M: Proposals per step (MCSA batch size)
        reward_temperature: Temperature for reward-to-log-likelihood
        ess_min: Min ESS for adaptive step scaling
        top_k_fine: Top-K candidates to validate at fine fidelity
        fine_fidelity_level: Fidelity level for final validation
    """

    K: int = 100
    M: int = 16
    reward_temperature: float = 0.1
    ess_min: float = 2.0
    top_k_fine: int = 5
    fine_fidelity_level: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)


def _linear_int(start: int, end: int, k: int, K: int) -> int:
    """Linear interpolation of int from start to end over k in [0, K-1]."""
    if K <= 1:
        return start
    alpha = float(k) / float(K - 1)
    return int(round((1.0 - alpha) * start + alpha * end))


class MRMFMBDPosteriorBackendJax:
    """
    Posterior bridge for soft-robot co-design: theta = (x, phi).

    Uses annealed bridge π_k(θ) ∝ p0(θ) * p(R|θ)^β_k with MCSA score ascent.
    Mode marginalization. Multi-fidelity ladder.
    """

    def __init__(
        self,
        *,
        evaluator: Any,
        theta_param: ThetaParametrization,
        theta_prior: ThetaPrior,
        bridge_schedule: Optional[BridgeSchedule] = None,
        fidelity_ladder: Optional[Any] = None,
        mode_marginalizer: Optional[Any] = None,
        config: Optional[PosteriorBridgeConfig] = None,
        task_id: str = "crawling_ground",
        num_modes: int = 4,
        mode_log_priors: Optional[List[float]] = None,
        seed: int = 0,
        show_tqdm: bool = False,
        **kwargs: Any,
    ):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is required for MRMFMBDPosteriorBackendJax")

        self.evaluator = evaluator
        self.theta_param = theta_param
        self.theta_prior = theta_prior
        self.config = config or PosteriorBridgeConfig(**kwargs)
        self.task_id = task_id
        self.num_modes = num_modes
        self.mode_log_priors = mode_log_priors or [0.0] * num_modes
        self.seed = seed
        self.show_tqdm = show_tqdm

        K = self.config.K
        self.bridge_schedule = bridge_schedule or create_linear_bridge_schedule(K=K)

        # Mode marginalization (theory-correct)
        self.mode_marginalizer = mode_marginalizer
        if self.mode_marginalizer is None and MODE_SYSTEM_AVAILABLE and default_mode_system_config:
            self.mode_marginalizer = ModeMarginalizer(
                default_mode_system_config(num_modes),
                backend="jax",
            )

        # Fidelity ladder (cost-optimal coarse→fine)
        self.fidelity_ladder = fidelity_ladder
        if self.fidelity_ladder is None and FIDELITY_SYSTEM_AVAILABLE and create_fidelity_ladder:
            self.fidelity_ladder = create_fidelity_ladder(
                K=K, num_levels=3, ladder_type="geometric", step_ratio=1.5
            )
        elif self.fidelity_ladder is None and CORE_FIDELITY_AVAILABLE and FidelityLadder:
            self.fidelity_ladder = FidelityLadder(FidelityConfig(num_levels=3), K=K)

        self._theta_dim = theta_param.theta_dim
        self._weighter = ImportanceWeighter(temperature=1.0, backend="jax")

    def _sample_proposals(
        self,
        theta: jnp.ndarray,
        sigma: float,
        M: int,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample M proposals: z_m = theta + sigma * eps_m, return (proposals, deltas)."""
        eps = jax.random.normal(rng, (M, self._theta_dim), dtype=jnp.float32)
        proposals = theta[None, :] + sigma * eps
        proposals = self.theta_param.clip(proposals)
        return proposals, eps

    def _evaluate_proposals_modes(
        self,
        proposals: np.ndarray,
        fidelity_level: int,
        seed_base: int,
    ) -> np.ndarray:
        """
        Evaluate proposals at each mode. Returns (M, num_modes) rewards.

        Uses evaluator.evaluate_batch. External (Python) call.
        """
        from genedynamics.envs.evaluators import (
            RolloutBatchRequest,
            RolloutRequest,
        )

        M, _ = proposals.shape
        rewards = np.zeros((M, self.num_modes), dtype=np.float32)

        # Pack all M*num_modes requests into one batch so the evaluator's
        # process pool can saturate (up to max_workers). The previous
        # per-mode loop only exposed M parallelism, leaving idle workers.
        flat_requests = []
        index_pairs = []
        for c in range(self.num_modes):
            for m in range(M):
                flat_requests.append(
                    RolloutRequest(
                        morphology_params=proposals[m, : self.theta_param.x_dim],
                        controller_params=proposals[m, self.theta_param.x_dim :],
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
        for (m, c), r in zip(index_pairs, batch_res.returns):
            rewards[m, c] = r
        return rewards

    def _compute_marginal_log_likelihood_and_responsibilities(
        self,
        rewards: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Mode theory-correct: log p(R|θ) = logsumexp(log p(c) + R_c/T)
        w_c = softmax(log p(c) + R_c/T).  Note: β NOT inside mixture.
        """
        if self.mode_marginalizer is not None:
            result = self.mode_marginalizer(
                rewards,
                temperature=self.config.reward_temperature,
            )
            return (
                jnp.asarray(result.marginal_log_likelihood, dtype=jnp.float32),
                jnp.asarray(result.responsibilities, dtype=jnp.float32),
            )
        # Fallback: inline (no beta in log_terms)
        T = max(self.config.reward_temperature, 1e-8)
        rewards_jnp = jnp.asarray(rewards, dtype=jnp.float32)
        log_prior_c = jnp.asarray(self.mode_log_priors[: self.num_modes], dtype=jnp.float32)
        log_terms = log_prior_c[None, :] + rewards_jnp / T
        marginal_log = jax.scipy.special.logsumexp(log_terms, axis=-1)
        w_c = jnp.exp(jax.nn.log_softmax(log_terms, axis=-1))
        return marginal_log, w_c

    def _compute_weighted_score(
        self,
        proposals: jnp.ndarray,
        deltas: jnp.ndarray,
        log_probs: jnp.ndarray,
        sigma: float,
    ) -> Tuple[jnp.ndarray, MCSADiagnostics]:
        """
        MCSA score with baseline subtraction (Eq. 169 in tex):
            score = (1/σ) Σ_m w_m * (δ_m - δ̄)
        where δ̄ = Σ_m w_m * δ_m is the weighted mean (baseline).
        The subtraction reduces variance without changing the expectation.
        """
        weights = self._weighter(log_probs, axis=0)
        weights = jnp.reshape(weights, (-1, 1))

        # Baseline subtraction: subtract weighted mean perturbation
        delta_baseline = jnp.sum(weights * deltas, axis=0, keepdims=True)
        centered_deltas = deltas - delta_baseline

        score = jnp.sum(weights * centered_deltas, axis=0) / max(sigma, 1e-8)

        from genedynamics.core.inference.diagnostics import effective_sample_size, degeneracy_flags

        ess = effective_sample_size(weights, axis=0, backend="jax")
        flags = degeneracy_flags(weights, ess_min=self.config.ess_min, axis=0, backend="jax")
        diag = MCSADiagnostics(
            ess=float(np.asarray(ess)),
            degeneracy=flags.get("degenerate", False),
        )
        return score, diag

    def plan(
        self,
        theta_init: Optional[jnp.ndarray] = None,
        rng_key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run posterior bridge. Returns theta*, bridge_history, diagnostics.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)

        theta = theta_init
        if theta is None:
            theta = self.theta_prior.sample(rng_key)
            rng_key, _ = jax.random.split(rng_key)
        theta = jnp.asarray(theta, dtype=jnp.float32)
        theta = self.theta_param.clip(theta)

        K = self.config.K
        M = self.config.M

        bridge_history: List[Dict[str, Any]] = []
        mode_responsibilities_history: List[np.ndarray] = []
        fidelity_history: List[int] = []
        fine_calls = 0
        wall_clock_start = time.perf_counter()

        # Top-K candidate tracking: keep best candidates seen during optimization
        top_k = max(self.config.top_k_fine, 1)
        # Each entry: (mean_reward, theta_snapshot)
        _top_k_candidates: List[Tuple[float, np.ndarray]] = []

        iter_range = range(K)
        if self.show_tqdm:
            try:
                import tqdm
                iter_range = tqdm.tqdm(iter_range, desc="MRMFMBD bridge")
            except ImportError:
                pass

        prev_ess: Optional[float] = None

        for k in iter_range:
            t0 = time.perf_counter()
            beta_k = float(self.bridge_schedule.beta(k))
            sigma_k = float(self.bridge_schedule.sigma(k))
            eta_k = float(self.bridge_schedule.eta(k))
            tau_k = float(self.bridge_schedule.tau(k))
            fidelity_level = self.fidelity_ladder(k)
            fidelity_history.append(fidelity_level)

            # ESS-adaptive scaling
            if prev_ess is not None and self.config.ess_min > 0 and prev_ess < self.config.ess_min:
                eta_scale = min(1.0, max(0.3, float(prev_ess) / self.config.ess_min))
                sigma_scale = 1.0 + 0.4 * max(0.0, 1.0 - float(prev_ess) / self.config.ess_min)
                eta_k *= eta_scale
                sigma_k *= sigma_scale

            # Sample proposals
            rng_key, key_eps, key_noise = jax.random.split(rng_key, 3)
            proposals, deltas = self._sample_proposals(theta, sigma_k, M, key_eps)
            proposals_np = np.asarray(proposals)

            # Evaluate (external Python call)
            rewards = self._evaluate_proposals_modes(
                proposals_np, fidelity_level, seed_base=self.seed + k * M * self.num_modes
            )

            # Log prior for each proposal
            log_prior_vals = self.theta_prior.log_prob_batch(proposals)

            # Mode: log p(R|θ) = logsumexp(log p(c) + R_c/T), w_c = responsibilities
            # Theory: β_k applied externally: log π_k(θ) = log p0(θ) + β_k * log p(R|θ)
            marginal_log, w_c = self._compute_marginal_log_likelihood_and_responsibilities(
                rewards
            )
            log_probs = log_prior_vals + beta_k * marginal_log  # full log π_k

            # Responsibilities (mean over proposals for diagnostics)
            w_c_mean = np.mean(np.asarray(w_c), axis=0)
            mode_responsibilities_history.append(w_c_mean)

            # MCSA score
            score, diag = self._compute_weighted_score(proposals, deltas, log_probs, sigma_k)
            prev_ess = diag.ess

            # Score ascent + exploration noise
            noise = jax.random.normal(key_noise, (self._theta_dim,), dtype=jnp.float32)
            theta = theta + eta_k * score + tau_k * noise
            theta = self.theta_param.clip(theta)

            # Track top-K candidates by mean reward
            step_mean_reward = float(np.mean(rewards))
            theta_snapshot = np.asarray(theta).copy()
            _top_k_candidates.append((step_mean_reward, theta_snapshot))
            _top_k_candidates.sort(key=lambda t: -t[0])
            _top_k_candidates = _top_k_candidates[:top_k]

            step_time = (time.perf_counter() - t0) * 1e3
            bridge_history.append({
                "k": k,
                "beta_k": beta_k,
                "sigma_k": sigma_k,
                "eta_k": eta_k,
                "tau_k": tau_k,
                "fidelity_level": fidelity_level,
                "mean_reward": float(np.mean(rewards)),
                "ess": diag.ess,
                "wall_time_ms": step_time,
            })

        # Top-K fine validation using tracked candidates
        best_fine_return = -np.inf
        best_fine_theta = np.asarray(theta)
        if self.config.top_k_fine > 0 and _top_k_candidates:
            from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest

            fine_level = self.config.fine_fidelity_level
            for _score, cand in _top_k_candidates:
                cand_np = np.asarray(cand)
                requests = [
                    RolloutRequest(
                        morphology_params=cand_np[: self.theta_param.x_dim],
                        controller_params=cand_np[self.theta_param.x_dim :],
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

            # Use the best fine-validated candidate
            theta = jnp.asarray(best_fine_theta, dtype=jnp.float32)

        wall_clock_total = time.perf_counter() - wall_clock_start

        x_final, phi_final = self.theta_param.unpack(theta)

        return {
            "theta": np.asarray(theta),
            "x": np.asarray(x_final),
            "phi": np.asarray(phi_final),
            "bridge_history": bridge_history,
            "mode_responsibilities": np.array(mode_responsibilities_history),
            "fidelity_history": fidelity_history,
            "fine_calls": fine_calls,
            "best_fine_return": best_fine_return if best_fine_return > -np.inf else None,
            "top_k_candidates": [
                {"reward": s, "theta": t.tolist()} for s, t in _top_k_candidates
            ],
            "wall_clock": wall_clock_total,
            "diagnostics": {
                "K": K,
                "M": M,
                "num_modes": self.num_modes,
            },
        }

    def sample_trajectories(
        self,
        x0: Any,
        n_samples: int,
        rng_key: Optional[Any] = None,
    ) -> List[Any]:
        """Not used for co-design; plan returns theta."""
        result = self.plan(theta_init=None, rng_key=rng_key)
        return [result]

    def plan_batch(self, x0: Any, keys: Any) -> List[Dict[str, Any]]:
        """Run plan for multiple seeds."""
        return [self.plan(rng_key=keys[i]) for i in range(keys.shape[0])]
