"""Constraint pullback metric and trust-region halfspace corrections."""

import math
import time

from .contracts import RepairBudget, RepairResult


class BudgetedRepairSolver:
    def __init__(self, runtime, budget=RepairBudget()):
        self.runtime, self.budget = runtime, budget

    def solve(self, problem, initial):
        runtime, budget, xp = self.runtime, self.budget, self.runtime.xp
        started = time.perf_counter()
        evaluations = 0
        shape = initial.shape

        def bounded(control):
            if problem.lower is not None and problem.upper is not None:
                return runtime.clip(control, problem.lower, problem.upper)
            return control

        def assess(control):
            nonlocal evaluations
            evaluations += 1
            objective = runtime.scalar(problem.objective(control))
            residual = problem.constraints(control).reshape(-1)
            raw_maximum = runtime.scalar(xp.max(residual)) if residual.shape[0] else 0.0
            if not math.isfinite(objective) or not math.isfinite(raw_maximum):
                return objective, float("inf"), float("inf")
            maximum = max(0.0, raw_maximum)
            positive = runtime.clip(residual, 0.0, float("inf"))
            merit = objective + budget.penalty * runtime.scalar(xp.sum(positive * positive))
            return objective, maximum, merit

        current = bounded(runtime.copy(initial))
        objective, violation, merit = assess(current)
        best, best_objective, best_violation = runtime.copy(current), objective, violation
        reason, completed = "iteration_limit", 0
        for iteration in range(budget.iterations):
            if time.perf_counter() - started >= budget.deadline_seconds:
                reason = "deadline"
                break
            _, gradient = runtime.value_and_grad(problem.objective, current)
            residual = problem.constraints(current).reshape(-1)
            jacobian = runtime.jacobian(lambda control: problem.constraints(control).reshape(-1), current).reshape(residual.shape[0], -1)
            identity = runtime.eye(current.size if runtime.name != "torch" else current.numel(), current)
            active = runtime.array(residual > -budget.tolerance)
            weighted = jacobian * active.reshape(-1, 1)
            metric = identity + weighted.T @ weighted
            direction = -budget.learning_rate * xp.linalg.solve(metric, gradient.reshape(-1))
            for row in range(residual.shape[0]):
                normal = jacobian[row]
                error = residual[row] + xp.sum(normal * direction)
                if runtime.scalar(error) > 0:
                    inverse_normal = xp.linalg.solve(metric, normal)
                    denominator = xp.sum(normal * inverse_normal) + 1e-8
                    direction = direction - error / denominator * inverse_normal
            norm = runtime.scalar(xp.linalg.norm(direction))
            if norm > budget.trust_radius:
                direction = direction * (budget.trust_radius / norm)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                if time.perf_counter() - started >= budget.deadline_seconds:
                    reason = "deadline"
                    break
                proposal = bounded(current + fraction * direction.reshape(shape))
                candidate_objective, candidate_violation, candidate_merit = assess(proposal)
                if candidate_merit < merit or candidate_violation < violation - budget.tolerance:
                    current, objective, violation, merit = runtime.copy(proposal), candidate_objective, candidate_violation, candidate_merit
                    accepted = True
                    if (violation > budget.tolerance, violation if violation > budget.tolerance else 0.0, objective) < (
                        best_violation > budget.tolerance, best_violation if best_violation > budget.tolerance else 0.0, best_objective
                    ):
                        best, best_objective, best_violation = runtime.copy(current), objective, violation
                    break
            completed = iteration + 1
            if reason == "deadline":
                break
            if not accepted:
                reason = "stationary" if violation <= budget.tolerance else "no_improvement"
                break
        elapsed = time.perf_counter() - started
        if elapsed >= budget.deadline_seconds:
            reason = "deadline"
        return RepairResult(best, best_objective, best_violation, best_violation <= budget.tolerance,
                            completed, elapsed, reason, evaluations)
