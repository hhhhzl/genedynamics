"""Budget allocation + dual variable for the multi-fidelity estimator.

Two pieces:
  - `optimal_subset_size`: at a per-step budget B with M low-fidelity evals
    already committed (cost M·c_lo), how many high-fidelity corrections K can
    we afford? (budget-feasible split; the variance-optimal MFMC split given
    correlation is a theory refinement, see writeup §T.)
  - `BudgetDual`: the compute-budget dual variable ν (report Eq 23), updated
    from realized per-step cost so the schedule self-regulates toward B̄.
"""

from __future__ import annotations

from dataclasses import dataclass


def realized_cost(M: int, K: int, c_lo: float, c_hi: float) -> float:
    """Cost of M low-fidelity evals + K high-fidelity corrections."""
    return float(M) * float(c_lo) + float(K) * float(c_hi)


def optimal_subset_size(budget: float, M: int, c_lo: float, c_hi: float) -> int:
    """Largest K ∈ [0, M] with M·c_lo + K·c_hi ≤ budget.

    Returns 0 if even the low-fidelity sweep over M exceeds the budget (caller
    should then shrink M). The variance-optimal K≤this given lo/hi correlation
    is a refinement; this is the budget-feasibility ceiling.
    """
    c_hi = float(c_hi)
    if c_hi <= 0:
        return int(M)
    remaining = float(budget) - float(M) * float(c_lo)
    if remaining <= 0:
        return 0
    K = int(remaining // c_hi)
    return max(0, min(int(M), K))


def single_fidelity_pool(budget: float, c_hi: float) -> int:
    """#proposals a pure high-fidelity step affords at the same budget."""
    c_hi = float(c_hi)
    if c_hi <= 0:
        return 0
    return max(0, int(float(budget) // c_hi))


@dataclass
class BudgetDual:
    """Compute-budget dual variable ν (report Eq 23):

        ν_{k+1} = max(0, ν_k + η · (realized_cost − B̄))

    ν rises when steps overspend (pushing toward cheaper fidelity) and decays
    when underspending (allowing more high-fidelity). Penalizes a candidate's
    high-fidelity use by ν·C in the importance weight when wired into the loop.
    """

    nu: float = 0.0
    eta: float = 0.1
    target: float = 200.0

    def update(self, realized: float) -> float:
        self.nu = max(0.0, self.nu + self.eta * (float(realized) - self.target))
        return self.nu

    def penalty(self, cost: float) -> float:
        """ν·cost — the budget penalty subtracted from a candidate's log-weight."""
        return self.nu * float(cost)
