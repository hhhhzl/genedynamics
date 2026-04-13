"""
CMA-ES baseline for co-design experiments.

Black-box evolutionary optimization over θ = (x, φ).
Uses the `cma` package (Hansen 2006) if available, falls back to
a simple (μ, λ)-ES with covariance adaptation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ..baseline import BaselineConfig, BaselineResult, BaselineProtocol


class CMAESBaseline(BaselineProtocol):
    """
    CMA-ES as a pluggable co-design baseline.

    Optimizes θ = (x, φ) by treating the mode-averaged return as the objective.
    Fair comparison: uses the same total rollout budget as MRMFMBD.
    """

    @property
    def name(self) -> str:
        return "cmaes"

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

        # Budget: match MRMFMBD's total evaluations = K * M * num_modes
        K = int(diff_params.get("Ndiffuse", extra.get("K", 50)))
        M = int(diff_params.get("M_k", extra.get("M", 8)))
        num_modes = extra.get("num_modes", 4)
        total_budget = K * M * num_modes

        # CMA-ES parameters
        popsize = extra.get("popsize", max(M, 8))
        sigma0 = extra.get("sigma0", 0.3)
        max_generations = total_budget // (popsize * num_modes)
        max_generations = max(max_generations, 1)

        theta_dim = x_dim + phi_dim
        x_lo = np.full(x_dim, 0.01, dtype=np.float32)
        x_hi = np.full(x_dim, 2.0, dtype=np.float32)
        phi_lo = np.full(phi_dim, 5.0, dtype=np.float32)
        phi_hi = np.full(phi_dim, 150.0, dtype=np.float32)
        lo = np.concatenate([x_lo, phi_lo])
        hi = np.concatenate([x_hi, phi_hi])

        # Initial mean: center of bounds
        x0 = (lo + hi) / 2.0

        num_evals = 0
        wall_start = time.perf_counter()

        def evaluate_theta(theta: np.ndarray) -> float:
            """Evaluate mode-averaged return for a single θ."""
            nonlocal num_evals
            theta = np.clip(theta, lo, hi).astype(np.float32)
            x_part = theta[:x_dim]
            phi_part = theta[x_dim:]

            requests = [
                RolloutRequest(
                    morphology_params=x_part,
                    controller_params=phi_part,
                    mode_id=c,
                    fidelity_level=extra.get("fine_fidelity_level", 0),
                    seed=int(config.seed + num_evals + c),
                    num_repeats=1,
                    record=False,
                )
                for c in range(num_modes)
            ]
            batch = evaluator.evaluate_batch(
                RolloutBatchRequest(task_id=config.task_id, requests=requests),
                parallel=False,
                use_cache=False,
            )
            num_evals += num_modes
            return float(np.mean(batch.returns))

        # Try cma package first
        best_theta, best_return = self._run_cma(
            evaluate_theta, x0, sigma0, lo, hi,
            popsize=popsize, max_generations=max_generations,
        )

        wall_time = time.perf_counter() - wall_start

        best_theta = np.clip(best_theta, lo, hi).astype(np.float32)
        x_star = best_theta[:x_dim].copy()
        phi_star = best_theta[x_dim:].copy()

        # Final evaluation at fine fidelity
        final_return = evaluate_theta(best_theta)

        return BaselineResult(
            theta=best_theta,
            x=x_star,
            phi=phi_star,
            return_=final_return,
            success=final_return > 0,
            num_evaluations=num_evals,
            wall_time=wall_time,
            metadata={
                "method": "cmaes",
                "popsize": popsize,
                "sigma0": sigma0,
                "max_generations": max_generations,
                "total_budget": total_budget,
            },
        )

    def _run_cma(
        self,
        objective,
        x0: np.ndarray,
        sigma0: float,
        lo: np.ndarray,
        hi: np.ndarray,
        popsize: int,
        max_generations: int,
    ):
        """Run CMA-ES. Try `cma` package, fall back to simple ES."""
        try:
            import cma
            return self._run_cma_package(
                objective, x0, sigma0, lo, hi, popsize, max_generations,
            )
        except ImportError:
            return self._run_simple_es(
                objective, x0, sigma0, lo, hi, popsize, max_generations,
            )

    def _run_cma_package(
        self, objective, x0, sigma0, lo, hi, popsize, max_generations,
    ):
        """CMA-ES via Hansen's `cma` package."""
        import cma

        opts = {
            "popsize": popsize,
            "maxiter": max_generations,
            "bounds": [lo.tolist(), hi.tolist()],
            "seed": 42,
            "verbose": -9,  # suppress output
        }
        # CMA-ES minimizes; we maximize return, so negate
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

        best_theta = x0.copy()
        best_return = -np.inf

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for sol in solutions:
                ret = objective(np.array(sol, dtype=np.float32))
                fitnesses.append(-ret)  # negate for minimization
                if ret > best_return:
                    best_return = ret
                    best_theta = np.array(sol, dtype=np.float32)
            es.tell(solutions, fitnesses)

        return best_theta, best_return

    def _run_simple_es(
        self, objective, x0, sigma0, lo, hi, popsize, max_generations,
    ):
        """
        Simple (μ, λ)-ES fallback when `cma` package is not installed.

        Uses diagonal covariance (separable ES) with rank-μ update.
        """
        rng = np.random.RandomState(42)
        dim = len(x0)
        mean = x0.astype(np.float64).copy()
        sigma = sigma0
        # Diagonal covariance
        C_diag = np.ones(dim, dtype=np.float64)

        mu = max(popsize // 2, 1)  # parent count
        # Log weights (CMA-style)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)

        best_theta = x0.copy()
        best_return = -np.inf

        for _gen in range(max_generations):
            # Sample population
            z = rng.randn(popsize, dim)
            population = mean[None, :] + sigma * np.sqrt(C_diag)[None, :] * z
            population = np.clip(population, lo, hi)

            # Evaluate
            returns = np.array([
                objective(population[i].astype(np.float32))
                for i in range(popsize)
            ])

            # Track best
            gen_best_idx = np.argmax(returns)
            if returns[gen_best_idx] > best_return:
                best_return = returns[gen_best_idx]
                best_theta = population[gen_best_idx].astype(np.float32).copy()

            # Select top-μ
            rank = np.argsort(-returns)[:mu]
            selected_z = z[rank]

            # Update mean
            mean = mean + sigma * np.sqrt(C_diag) * np.dot(weights, selected_z)
            mean = np.clip(mean, lo, hi)

            # Update diagonal covariance (simplified)
            z_weighted = np.dot(weights, selected_z ** 2)
            c_c = 2.0 / (dim + 2.0)
            C_diag = (1 - c_c) * C_diag + c_c * z_weighted

            # Step-size adaptation (1/5 success rule approximation)
            success_rate = np.mean(returns[rank] > np.median(returns))
            if success_rate > 0.2:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 1e-8, 2.0)

        return best_theta, best_return
