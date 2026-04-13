"""Standalone tests for the WBC QP solver adapter.

The adapter is constructed for the WBC's specific QP shape but its
``solve(H, f, C, d, G, h)`` interface is generic, so we can hit it with
small canonical QPs (no MuJoCo needed) and verify the fallback chain
behaves sensibly.
"""

import numpy as np
import pytest

from genedynamics.deploy.controllers.wbc.config import SolverConfig
from genedynamics.deploy.controllers.wbc.qp_solver_adapter import (
    QPSolverAdapter,
    SolveResult,
)


def _adapter(**overrides) -> QPSolverAdapter:
    cfg = SolverConfig(**overrides)
    return QPSolverAdapter(cfg)


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_unconstrained_qp_returns_minimum():
    """min ½‖x‖² → x = 0."""
    H = np.eye(3)
    f = np.zeros(3)
    empty = (np.zeros((0, 3)), np.zeros(0))
    result = _adapter().solve(H, f, *empty, *empty)
    assert isinstance(result, SolveResult)
    assert np.allclose(result.x, 0.0, atol=1e-6)
    assert result.eq_residual == 0.0
    assert result.ineq_violation == 0.0


def test_unconstrained_qp_with_linear_term():
    """min ½xᵀIx + bᵀx → x = -b."""
    H = np.eye(2)
    f = np.array([1.0, -2.0])
    empty = (np.zeros((0, 2)), np.zeros(0))
    result = _adapter().solve(H, f, *empty, *empty)
    assert np.allclose(result.x, -f, atol=1e-5)


# ---------------------------------------------------------------------------
# Equality constraints
# ---------------------------------------------------------------------------


def test_equality_constrained_qp_lies_on_plane():
    """min ½‖x‖² s.t. x[0] + x[1] + x[2] = 3 → x = [1, 1, 1]."""
    H = np.eye(3)
    f = np.zeros(3)
    C = np.array([[1.0, 1.0, 1.0]])
    d = np.array([3.0])
    G_empty, h_empty = np.zeros((0, 3)), np.zeros(0)
    result = _adapter().solve(H, f, C, d, G_empty, h_empty)
    assert np.allclose(result.x, [1.0, 1.0, 1.0], atol=1e-5)
    assert result.eq_residual < 1e-6


# ---------------------------------------------------------------------------
# Inequality constraints
# ---------------------------------------------------------------------------


def test_single_active_inequality():
    """min ½‖x‖² s.t. x[0] ≥ 1 → x = [1, 0, 0]."""
    H = np.eye(3)
    f = np.zeros(3)
    C, d = np.zeros((0, 3)), np.zeros(0)
    G = np.array([[-1.0, 0.0, 0.0]])
    h = np.array([-1.0])
    result = _adapter().solve(H, f, C, d, G, h)
    assert result.x[0] >= 0.999
    assert abs(result.x[1]) < 1e-3
    assert abs(result.x[2]) < 1e-3
    assert result.ineq_violation < 1e-6


def test_box_constrained_qp():
    """min ½‖x − [2, 2]‖² s.t. 0 ≤ x ≤ 1 → x = [1, 1]."""
    H = np.eye(2)
    f = np.array([-2.0, -2.0])
    C, d = np.zeros((0, 2)), np.zeros(0)
    # x ≤ 1 and -x ≤ 0
    G = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    h = np.array([1.0, 1.0, 0.0, 0.0])
    result = _adapter().solve(H, f, C, d, G, h)
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-4)
    assert result.ineq_violation < 1e-5


# ---------------------------------------------------------------------------
# Mixed equality + inequality
# ---------------------------------------------------------------------------


def test_equality_and_inequality_combined():
    """min ½‖x‖² s.t. x[0]+x[1]=1, x[1] ≥ 0.25 → x = [0.75, 0.25].

    The constrained minimum lies in the corner where the equality plane
    meets the inequality boundary."""
    H = np.eye(2)
    f = np.zeros(2)
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    G = np.array([[0.0, -1.0]])
    h = np.array([-0.25])
    result = _adapter().solve(H, f, C, d, G, h)
    assert abs(result.x.sum() - 1.0) < 1e-4
    assert result.x[1] >= 0.249
    assert result.eq_residual < 1e-5


# ---------------------------------------------------------------------------
# Fallback chain behavior
# ---------------------------------------------------------------------------


def test_fallback_chain_when_osqp_disabled():
    """When OSQP is disabled, SLSQP / trust-constr take over and still solve."""
    H = np.eye(3)
    f = np.zeros(3)
    C, d = np.zeros((0, 3)), np.zeros(0)
    G = np.array([[-1.0, 0.0, 0.0]])
    h = np.array([-1.0])
    adapter = _adapter(use_osqp=False, use_slsqp=True, use_trust_constr=False)
    result = adapter.solve(H, f, C, d, G, h)
    assert result.x[0] >= 0.999
    # Should not have used OSQP
    assert "osqp" not in result.method


def test_warm_start_preserves_solution_across_calls():
    """Calling solve twice with the same problem should give the same answer."""
    H = np.eye(4)
    f = np.array([0.1, 0.2, 0.3, 0.4])
    C, d = np.zeros((0, 4)), np.zeros(0)
    G, h = np.zeros((0, 4)), np.zeros(0)
    adapter = _adapter()
    r1 = adapter.solve(H, f, C, d, G, h)
    r2 = adapter.solve(H, f, C, d, G, h)
    assert np.allclose(r1.x, r2.x, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_acceptable_predicate_classifies_results():
    res = SolveResult(x=np.zeros(3), method="x", eq_residual=1e-9, ineq_violation=1e-9)
    assert res.is_acceptable(tol=1e-6)
    bad = SolveResult(x=np.zeros(3), method="x", eq_residual=1e-3, ineq_violation=0.0)
    assert not bad.is_acceptable(tol=1e-6)
