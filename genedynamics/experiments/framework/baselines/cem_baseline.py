"""
CEM (Cross-Entropy Method) baseline for co-design experiments.

Diagonal-Gaussian CEM over θ = (x, φ). Each generation samples P=`popsize`
candidates, evaluates mode-averaged return, selects the top `elite_frac`, and
refits (mean, std) to the elite set. Structure mirrors CMAESBaseline so the
two can be compared on equal footing (same budget, same bounds, same eval).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..baseline import BaselineConfig, BaselineResult, BaselineProtocol


class CEMBaseline(BaselineProtocol):
    """Cross-Entropy Method as a pluggable co-design baseline.

    Hyperparameters (via method_params):
        popsize       — samples per generation  (default 32)
        elite_frac    — fraction kept as elites (default 0.25)
        init_std      — initial std relative to (hi-lo)/2  (default 0.3)
        min_std       — floor on std to avoid collapse  (default 0.02)
        noise_decay   — per-generation multiplicative shrink of std (default 0.97)
        extra_noise   — additive std floor that shrinks with decay (default 0.05)
        x_lo/x_hi/phi_lo/phi_hi — search bounds (required for jax_mpm).
        x_mean/phi_mean        — init-mean overrides; default to (lo+hi)/2.
    """

    @property
    def name(self) -> str:
        return "cem"

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
        from genedynamics.envs.evaluators import RolloutBatchRequest, RolloutRequest

        extra = config.extra
        diff_params = config.get_diffusion_params()

        # Budget: match MRMFMBD's total rollout count  (K × M × num_modes).
        K = int(diff_params.get("Ndiffuse", extra.get("K", 50)))
        M = int(diff_params.get("M_k", extra.get("M", 8)))
        num_modes = int(extra.get("num_modes", 4))
        total_budget = K * M * num_modes

        # CEM params
        popsize = int(extra.get("popsize", max(M, 32)))
        elite_frac = float(extra.get("elite_frac", 0.25))
        num_elite = max(int(popsize * elite_frac), 2)
        init_std_rel = float(extra.get("init_std", 0.3))
        min_std = float(extra.get("min_std", 0.02))
        noise_decay = float(extra.get("noise_decay", 0.97))
        extra_noise_floor = float(extra.get("extra_noise", 0.05))

        max_generations = max(total_budget // (popsize * num_modes), 1)

        # Bounds (explicit — jax_mpm needs φ ∈ [-0.5, 0.5], x ∈ [0.2, 1.0]).
        x_lo_v = float(extra.get("x_lo", 0.01))
        x_hi_v = float(extra.get("x_hi", 2.0))
        phi_lo_v = float(extra.get("phi_lo", 5.0))
        phi_hi_v = float(extra.get("phi_hi", 150.0))
        lo = np.concatenate([
            np.full(x_dim, x_lo_v, dtype=np.float32),
            np.full(phi_dim, phi_lo_v, dtype=np.float32),
        ])
        hi = np.concatenate([
            np.full(x_dim, x_hi_v, dtype=np.float32),
            np.full(phi_dim, phi_hi_v, dtype=np.float32),
        ])

        x_mean_v = float(extra.get("x_mean", (x_lo_v + x_hi_v) / 2.0))
        phi_mean_v = float(extra.get("phi_mean", (phi_lo_v + phi_hi_v) / 2.0))
        mean = np.concatenate([
            np.full(x_dim, x_mean_v, dtype=np.float32),
            np.full(phi_dim, phi_mean_v, dtype=np.float32),
        ]).astype(np.float64)

        # Init std: init_std_rel × (hi - lo) / 2 per dim.
        std = (init_std_rel * (hi - lo) / 2.0).astype(np.float64)
        std = np.maximum(std, min_std)

        rng = np.random.RandomState(int(extra.get("seed", 42)) + int(config.seed))
        fidelity_level = int(extra.get("fine_fidelity_level", 0))

        num_evals = 0
        wall_start = time.perf_counter()

        best_theta = mean.astype(np.float32).copy()
        best_return = -np.inf
        history: list = []

        def evaluate_population(thetas: np.ndarray) -> np.ndarray:
            """Mode-averaged return for a batch of θ, mode-major for env reuse."""
            nonlocal num_evals
            thetas = np.clip(thetas, lo[None, :], hi[None, :]).astype(np.float32)
            P = thetas.shape[0]
            flat_requests = []
            index_pairs = []
            for c in range(num_modes):
                for p in range(P):
                    flat_requests.append(
                        RolloutRequest(
                            morphology_params=thetas[p, :x_dim],
                            controller_params=thetas[p, x_dim:],
                            mode_id=c,
                            fidelity_level=fidelity_level,
                            seed=int(config.seed + num_evals + p * num_modes + c),
                            num_repeats=1,
                            record=False,
                        )
                    )
                    index_pairs.append((p, c))
            batch = evaluator.evaluate_batch(
                RolloutBatchRequest(task_id=config.task_id, requests=flat_requests),
                parallel=True,
                use_cache=False,
            )
            rewards_pc = np.zeros((P, num_modes), dtype=np.float32)
            for (p, c), r in zip(index_pairs, batch.returns):
                rewards_pc[p, c] = r
            num_evals += P * num_modes
            return rewards_pc.mean(axis=1)

        show = bool(extra.get("show_tqdm", False))
        iterator = range(max_generations)
        if show:
            try:
                import tqdm
                iterator = tqdm.tqdm(iterator, desc="CEM")
            except ImportError:
                pass

        for gen in iterator:
            z = rng.randn(popsize, lo.size).astype(np.float64)
            population = mean[None, :] + std[None, :] * z
            population = np.clip(population, lo, hi)

            returns = evaluate_population(population.astype(np.float32))
            gen_best_idx = int(np.argmax(returns))
            if returns[gen_best_idx] > best_return:
                best_return = float(returns[gen_best_idx])
                best_theta = population[gen_best_idx].astype(np.float32).copy()

            # Refit to top-elite
            elite_idx = np.argsort(-returns)[:num_elite]
            elite = population[elite_idx]
            mean = elite.mean(axis=0)
            new_std = elite.std(axis=0)
            # Decayed extra noise prevents premature collapse.
            noise_floor = extra_noise_floor * (noise_decay ** gen) * (hi - lo) / 2.0
            std = np.maximum(new_std, np.maximum(min_std, noise_floor))

            history.append({
                "generation": gen,
                "mean_return": float(returns.mean()),
                "elite_mean_return": float(returns[elite_idx].mean()),
                "best_return": float(returns.max()),
                "std_mean": float(std.mean()),
            })

            if num_evals >= total_budget:
                break

        wall_time = time.perf_counter() - wall_start

        best_theta = np.clip(best_theta, lo, hi).astype(np.float32)
        x_star = best_theta[:x_dim].copy()
        phi_star = best_theta[x_dim:].copy()
        final_return = float(evaluate_population(best_theta[None, :])[0])

        return BaselineResult(
            theta=best_theta,
            x=x_star,
            phi=phi_star,
            return_=final_return,
            success=final_return > 0,
            num_evaluations=num_evals,
            wall_time=wall_time,
            metadata={
                "method": "cem",
                "popsize": popsize,
                "elite_frac": elite_frac,
                "num_elite": num_elite,
                "init_std_rel": init_std_rel,
                "max_generations": max_generations,
                "total_budget": total_budget,
                "generation_history": history,
            },
        )
