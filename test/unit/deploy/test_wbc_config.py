"""Static checks on the grouped WBCConfig.

These tests don't need MuJoCo; they just verify that the new nested config
covers every legacy parameter and that the defaults haven't drifted.
"""

import dataclasses

from genedynamics.deploy.controllers.wbc.config import (
    LimitsConfig,
    SolverConfig,
    TaskGainsConfig,
    TaskWeightsConfig,
    WBCConfig,
)


def test_wbc_config_default_construction():
    cfg = WBCConfig()
    assert isinstance(cfg.weights, TaskWeightsConfig)
    assert isinstance(cfg.gains, TaskGainsConfig)
    assert isinstance(cfg.limits, LimitsConfig)
    assert isinstance(cfg.solver, SolverConfig)


def test_wbc_config_field_counts_match_legacy_92_param_total():
    """The legacy G1WBCTaskStackConfig had ~92 flat fields. We split them
    into four sub-configs. Field counts should sum to roughly the same total
    (allowing for organizational additions/removals)."""
    total = (
        len(dataclasses.fields(TaskWeightsConfig))
        + len(dataclasses.fields(TaskGainsConfig))
        + len(dataclasses.fields(LimitsConfig))
        + len(dataclasses.fields(SolverConfig))
    )
    assert 80 <= total <= 100, f"unexpected total field count: {total}"


def test_wbc_config_legacy_default_parity():
    """Spot-check that critical defaults match the legacy config exactly.

    These are the values the legacy controller was tuned against; if any
    drifts during refactor cleanup the WBC behavior changes silently.
    """
    cfg = WBCConfig()
    assert cfg.weights.contact == 80.0
    assert cfg.weights.com == 8.0
    assert cfg.weights.pelvis == 7.0
    assert cfg.weights.swing_foot == 8.0
    assert cfg.gains.com_kp == 45.0
    assert cfg.gains.com_kd == 14.0
    assert cfg.gains.pelvis_orientation_kp == 40.0
    assert cfg.gains.swing_foot_position_kp == 20.0
    assert cfg.limits.friction_coeff == 0.60
    assert cfg.limits.lambda_min_normal == 20.0
    assert cfg.limits.lambda_max_normal == 550.0
    assert cfg.limits.torque_limit_scale == 0.85
    assert cfg.limits.qp_regularization == 1e-8
    assert cfg.solver.use_osqp is True
    assert cfg.solver.constraint_tol == 1e-6


def test_wbc_config_independent_subgroups():
    """Mutating one sub-config must not affect the defaults of another."""
    cfg = WBCConfig()
    cfg.weights.com = 999.0
    cfg.gains.com_kp = 1.0
    cfg.limits.friction_coeff = 0.1
    cfg.solver.use_osqp = False

    fresh = WBCConfig()
    assert fresh.weights.com == 8.0
    assert fresh.gains.com_kp == 45.0
    assert fresh.limits.friction_coeff == 0.60
    assert fresh.solver.use_osqp is True
