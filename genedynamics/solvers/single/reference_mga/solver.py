"""Prior-centered annealing with AL weights and constraint pullback geometry.

This adapts GD MGA's sampling/soft-feasibility/geometry seams to a caller-owned tensor
control sequence. It has no Brax substrate, spline nodes or impedance-stiffness variables;
it is not a byte-compatible replacement for the original JAX MGA backend.
"""

import math
import time

from genedynamics.solvers.single.budgeted_repair.contracts import (
    RepairBudget,
    RepairResult,
)


class ReferenceMGASolver:
    def __init__(
        self,
        runtime,
        *,
        samples=32,
        noise_scale=0.2,
        temperature=0.5,
        annealing=0.8,
        seed=0,
        budget=None,
    ):
        if samples < 1 or min(noise_scale, temperature) <= 0 or not 0 < annealing <= 1:
            raise ValueError("Invalid reference MGA settings")
        self.runtime, self.samples = runtime, samples
        self.noise_scale, self.temperature, self.annealing = (
            noise_scale,
            temperature,
            annealing,
        )
        self.seed, self.budget = seed, budget or RepairBudget()

    def solve(self, problem, initial):
        runtime, xp, budget = self.runtime, self.runtime.xp, self.budget
        started = time.perf_counter()
        current, best = runtime.copy(initial), runtime.copy(initial)
        best_rank, evaluations, completed, reason = None, 0, 0, "iteration_limit"
        multipliers = None
        for iteration in range(max(1, budget.iterations)):
            if evaluations and time.perf_counter() - started >= budget.deadline_seconds:
                reason = "deadline"
                break
            residual = problem.constraints(current).reshape(-1)
            if not runtime.scalar(xp.all(xp.isfinite(residual))):
                reason = "nonfinite_initial"
                break
            jacobian = runtime.jacobian(
                lambda control: problem.constraints(control).reshape(-1), current
            ).reshape(len(residual), -1)
            dimension = initial.numel() if runtime.name == "torch" else initial.size
            active = runtime.array(residual > -budget.tolerance)
            weighted = jacobian * active[:, None]
            metric = runtime.eye(dimension, current) + weighted.T @ weighted
            factor = xp.linalg.cholesky(metric)
            noise = runtime.normal((self.samples, dimension), self.seed + iteration)
            noise = xp.linalg.solve(factor.T, noise.T).T.reshape(
                (self.samples,) + initial.shape
            )
            scale = self.noise_scale * self.annealing**iteration
            candidates, scores = [], []
            if multipliers is None:
                multipliers = residual * 0
            for index in range(self.samples):
                if (
                    evaluations
                    and time.perf_counter() - started >= budget.deadline_seconds
                ):
                    reason = "deadline"
                    break
                candidate = current if index == 0 else current + scale * noise[index]
                if problem.lower is not None:
                    candidate = runtime.clip(candidate, problem.lower, problem.upper)
                objective = problem.objective(candidate)
                values = problem.constraints(candidate).reshape(-1)
                positive = runtime.clip(values, 0.0, float("inf"))
                violation = runtime.scalar(xp.max(positive)) if len(positive) else 0.0
                score = objective + xp.sum(
                    multipliers * positive + 0.5 * budget.penalty * positive**2
                )
                evaluations += 1
                if not math.isfinite(runtime.scalar(score)) or not math.isfinite(
                    violation
                ):
                    continue
                rank = (
                    violation > budget.tolerance,
                    violation if violation > budget.tolerance else 0.0,
                    runtime.scalar(objective),
                )
                if best_rank is None or rank < best_rank:
                    best, best_rank = runtime.copy(candidate), rank
                candidates.append(candidate)
                scores.append(score)
            if not scores:
                reason = "nonfinite_candidates"
                break
            scores = xp.stack(scores)
            weights = xp.exp(-(scores - xp.min(scores)) / self.temperature)
            weights = weights / xp.sum(weights)
            current = xp.sum(
                xp.stack(candidates) * weights.reshape((-1,) + (1,) * initial.ndim),
                axis=0,
            )
            multipliers = runtime.clip(
                multipliers + budget.penalty * problem.constraints(current).reshape(-1),
                0.0,
                float("inf"),
            )
            completed = iteration + 1
            if reason == "deadline":
                break
        objective = runtime.scalar(problem.objective(best))
        residual = problem.constraints(best).reshape(-1)
        violation = max(0.0, runtime.scalar(xp.max(residual))) if len(residual) else 0.0
        if not math.isfinite(objective) or (
            len(residual) and not math.isfinite(runtime.scalar(xp.max(residual)))
        ):
            violation = float("inf")
        return RepairResult(
            best,
            objective,
            violation,
            violation <= budget.tolerance,
            completed,
            time.perf_counter() - started,
            reason,
            evaluations,
        )
