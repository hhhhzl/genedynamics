"""Task callbacks supply normalized residuals; positive means violation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControlProblem:
    objective: Callable
    constraints: Callable
    lower: Any = None
    upper: Any = None


@dataclass(frozen=True)
class RepairBudget:
    iterations: int = 12
    deadline_seconds: float = 1.0
    trust_radius: float = 0.2
    learning_rate: float = 0.05
    penalty: float = 10.0
    tolerance: float = 1e-4

    def __post_init__(self):
        if (
            self.iterations < 0
            or self.tolerance < 0
            or min(
                self.deadline_seconds,
                self.trust_radius,
                self.learning_rate,
                self.penalty,
            )
            <= 0
        ):
            raise ValueError("Invalid repair budget")


@dataclass(frozen=True)
class RepairResult:
    controls: Any
    objective: float
    maximum_violation: float
    predicted_feasible: bool
    iterations: int
    elapsed_seconds: float
    termination_reason: str
    # Explicit candidate assessments. Instrument callbacks for actual model-forward counts;
    # derivatives can perform additional callback evaluations, especially finite differences.
    model_evaluations: int
