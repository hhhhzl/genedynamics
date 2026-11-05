"""Tensor-native local constraint repair with explicit online budgets."""

from .contracts import ControlProblem, RepairBudget, RepairResult
from .solver import BudgetedRepairSolver

__all__ = ["BudgetedRepairSolver", "ControlProblem", "RepairBudget", "RepairResult"]
