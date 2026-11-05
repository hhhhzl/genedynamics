"""Sampling comparison uses the same problem and real deadline as repair."""

import math
import time

from genedynamics.solvers.single.budgeted_repair.contracts import (
    RepairBudget,
    RepairResult,
)


class ModelMPPISolver:
    def __init__(
        self,
        runtime,
        *,
        samples=64,
        temperature=0.5,
        noise_scale=0.2,
        seed=0,
        budget=None,
    ):
        if samples < 1 or temperature <= 0 or noise_scale <= 0:
            raise ValueError("Invalid MPPI configuration")
        self.runtime, self.samples, self.temperature = runtime, samples, temperature
        self.noise_scale, self.seed, self.budget = (
            noise_scale,
            seed,
            budget or RepairBudget(),
        )

    def solve(self, problem, initial):
        started, runtime, xp = time.perf_counter(), self.runtime, self.runtime.xp
        current, best, best_rank = runtime.copy(initial), runtime.copy(initial), None
        evaluations, completed, reason = 0, 0, "iteration_limit"
        for iteration in range(max(1, self.budget.iterations)):
            perturbations = (
                runtime.normal((self.samples,) + initial.shape, self.seed + iteration)
                * self.noise_scale
            )
            candidates, costs = [], []
            for sample in range(self.samples):
                if (
                    evaluations
                    and time.perf_counter() - started >= self.budget.deadline_seconds
                ):
                    reason = "deadline"
                    break
                candidate = current if sample == 0 else current + perturbations[sample]
                if problem.lower is not None:
                    candidate = runtime.clip(candidate, problem.lower, problem.upper)
                objective = problem.objective(candidate)
                residual = runtime.clip(
                    problem.constraints(candidate).reshape(-1), 0.0, float("inf")
                )
                violation = (
                    runtime.scalar(xp.max(residual)) if residual.shape[0] else 0.0
                )
                score = objective + self.budget.penalty * xp.sum(residual * residual)
                evaluations += 1
                if not math.isfinite(runtime.scalar(score)) or not math.isfinite(
                    violation
                ):
                    continue
                rank = (
                    violation > self.budget.tolerance,
                    violation if violation > self.budget.tolerance else 0.0,
                    runtime.scalar(objective),
                )
                if best_rank is None or rank < best_rank:
                    best, best_rank = runtime.copy(candidate), rank
                candidates.append(candidate)
                costs.append(score)
            if not costs:
                reason = "nonfinite_candidates"
                break
            values = xp.stack(costs)
            weights = xp.exp(-(values - xp.min(values)) / self.temperature)
            weights = weights / xp.sum(weights)
            current = xp.sum(
                xp.stack(candidates) * weights.reshape((-1,) + (1,) * initial.ndim),
                axis=0,
            )
            completed = iteration + 1
            if reason == "deadline":
                break
        residual = runtime.clip(
            problem.constraints(best).reshape(-1), 0.0, float("inf")
        )
        violation = runtime.scalar(xp.max(residual)) if residual.shape[0] else 0.0
        objective = runtime.scalar(problem.objective(best))
        if not math.isfinite(violation) or not math.isfinite(objective):
            violation = float("inf")
        return RepairResult(
            best,
            objective,
            violation,
            violation <= self.budget.tolerance,
            completed,
            time.perf_counter() - started,
            reason,
            evaluations,
        )
