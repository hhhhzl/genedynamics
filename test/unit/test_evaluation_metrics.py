"""Gates for the general evaluation-metrics library (fedguide, pure numpy).

Verifies each metric's math, the registry/by-name path, aggregation, and that the
library has ZERO task/algorithm coupling (every metric is a pure array fn)."""

import numpy as np
import pytest

from genedynamics import evaluation as ev
from genedynamics.evaluation import metrics as m
from genedynamics.evaluation.aggregate import aggregate, aggregate_by


# --- core stats ---

def test_cvar_high_and_low_tail():
    x = np.arange(100.0)                      # 0..99
    # worst 5% high tail = mean(95..99) = 97
    assert abs(m.cvar(x, 0.95, "high") - 97.0) < 1e-9
    # worst 5% low tail = mean(0..4) = 2
    assert abs(m.cvar(x, 0.95, "low") - 2.0) < 1e-9
    assert m.cvar([], 0.95) == 0.0


def test_rate():
    assert m.rate([1, 0, 1, 1]) == 0.75
    assert m.rate([False, False]) == 0.0


def test_episode_event_and_series_reducers():
    assert m.event_occurred([0, 0, 1, 0]) == 1.0
    assert m.event_occurred([0, 0]) == 0.0
    assert m.maximum([1, 4, 2]) == 4.0
    assert m.terminal_value([1, 4, 2]) == 2.0
    assert m.time_integral([1, 2, 3], dt=0.1) == pytest.approx(0.6)
    assert m.first_event_time([0, 0, 1], dt=0.02) == pytest.approx(0.04)
    assert m.first_event_time([0, 0], dt=0.02) == pytest.approx(0.04)
    assert m.transition_count([0, 0, 1, 2, 2]) == 2.0
    assert m.conditional_event_success([0, 1, 0], [0, 0, 1]) == 1.0
    assert np.isnan(m.conditional_event_success([0, 0], [0, 1]))


def test_pearson_correlation():
    assert m.pearson_correlation([0, 1, 2], [0, 2, 4]) == pytest.approx(1.0)
    assert m.pearson_correlation([0, 1, 2], [4, 2, 0]) == pytest.approx(-1.0)
    assert m.pearson_correlation([1, 1, 1], [0, 1, 2]) == 0.0


# --- success / progress ---

def test_goal_and_success():
    assert abs(m.goal_error([1.0, 0.0], [0.0, 0.0]) - 1.0) < 1e-9
    assert m.success(0.1, margin=0.2, violation=0.0) == 1.0
    assert m.success(0.3, margin=0.2) == 0.0            # too far
    assert m.success(0.1, margin=0.2, violation=0.5) == 0.0  # violated


def test_progress_and_completion():
    assert abs(m.progress_ratio([0, 0], [0.5, 0], [1.0, 0]) - 0.5) < 1e-9
    assert m.progress_ratio([0, 0], [1, 0], [1, 0]) == 1.0
    pos = np.array([[0, 0], [0.5, 0], [0.95, 0], [1.0, 0]])
    assert m.completion_step(pos, [1.0, 0.0], margin=0.1) == 2
    assert m.completion_step(pos, [9.0, 0.0], margin=0.1) == 4   # never reached -> T


def test_coverage():
    assert m.coverage_ratio(7, 10) == 0.7
    assert m.coverage_ratio(15, 10) == 1.0


# --- constraint / safety ---

def test_violation_metrics():
    g = np.array([-1.0, -0.5, 0.0, 0.5, 2.0])   # two violations (0.5, 2.0)
    assert m.violation_rate(g) == 0.4
    assert abs(m.violation_mean(g) - (0.5 + 2.0) / 5) < 1e-9
    assert m.max_violation(g) == 2.0
    assert m.violation_cvar(g, 0.8) > 0.0
    assert abs(m.equality_residual_rms([3.0, 4.0]) - np.sqrt(12.5)) < 1e-9


# --- contact / force ---

def test_force_and_contact():
    assert abs(m.force_tracking_error([5, 5], [5, 4]) - np.sqrt(0.5)) < 1e-9
    assert m.force_tracking_mae([5, 5], [5, 4]) == 0.5
    assert m.normalized_force_tracking_mae([5, 5], [5, 4]) == 0.1
    mask = [0, 1, 1]
    assert m.force_tracking_error_tracked([0, 8, 10], [0, 10, 10], mask) == pytest.approx(np.sqrt(2.0))
    assert m.force_tracking_mae_tracked([0, 8, 10], [0, 10, 10], mask) == 1.0
    assert m.normalized_force_tracking_mae_tracked(
        [0, 8, 10], [0, 10, 10], mask,
    ) == 0.1
    assert m.force_peak([0, 7, 3]) == 7.0
    assert m.force_overshoot_ratio([0, 7, 5], [0, 5, 5]) == 0.4
    assert m.force_excess_impulse([0, 7, 5], [0, 5, 5], dt=0.1) == 0.2
    assert m.force_settling_time(
        [0, 5, 10, 10, 10], [0, 5, 10, 10, 10], dt=0.1,
        hold_steps=2,
    ) == pytest.approx(0.3)
    assert m.force_violation_rate([1, 5, 25], f_min=0.0, f_max=20.0) == pytest.approx(1 / 3)
    assert m.contact_loss_rate([1, 1, 0, 1]) == 0.25
    assert m.tangential_slip([-2, 2, 0, 4]) == 2.0
    # |f_t| > mu f_n : ft=[3,1], fn=[4,4], mu=0.5 -> 3>2 viol, 1>2 ok -> 0.5
    assert m.friction_cone_violation_rate([3, 1], [4, 4], mu=0.5) == 0.5
    # signed tangential uses MAGNITUDE: -3 -> |−3|=3 > 2 -> still a violation
    assert m.friction_cone_violation_rate([-3, 1], [4, 4], mu=0.5) == 0.5


# --- balance ---

def test_balance():
    assert m.fall_rate([0, 0, 1, 0]) == 0.25
    com = np.array([[0.0, 0.0], [0.2, 0.0]])    # dist 0, 0.2 from center; radius 0.25
    assert abs(m.balance_margin(com, [0.0, 0.0], 0.25) - 0.05) < 1e-9   # worst = 0.25-0.2
    assert m.tip_margin([0.3, 0.1, 0.2]) == pytest.approx(0.1)


# --- control quality ---

def test_control_quality():
    u = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # steps: 1, 1
    assert abs(m.smoothness(u) - 1.0) < 1e-9
    assert abs(m.stiffness_smoothness(u) - 1.0) < 1e-9
    assert abs(m.energy([[3.0, 4.0]]) - 25.0) < 1e-9
    p = np.array([[0, 0], [3, 0], [3, 4]])
    assert abs(m.path_length(p) - 7.0) < 1e-9
    assert abs(m.path_efficiency(p, [3.0, 4.0]) - 5.0 / 7.0) < 1e-9
    # path_efficiency clamps to [0,1] when the target is not reached
    assert m.path_efficiency([[0.0, 0.0], [0.1, 0.0]], [10.0, 0.0]) == 1.0
    assert m.runtime(1.5) == 1.5


def test_pose_error_and_path_completion():
    # SE(2): translation 1m, angle 3.14 vs -3.14 wraps to ~0 (not 6.28)
    pe = m.pose_error([1.0, 0.0, 3.14], [0.0, 0.0, -3.14], rot_dims=(2,))
    assert abs(pe - 1.0) < 1e-2
    # vs goal_error which wrongly mixes the un-wrapped angle into the norm
    assert m.goal_error([1.0, 0.0, 3.14], [0.0, 0.0, -3.14]) > 6.0
    # path_completion: final at 60% along a straight reference
    assert abs(m.path_completion([[0.0, 0.0], [0.6, 0.0]],
                                 [[0.0, 0.0], [1.0, 0.0]]) - 0.6) < 1e-6


def test_acquisition_aware_path_progress_and_coverage():
    ref = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11)], axis=1)
    pos = np.array([[0.0, 0.0], [0.2, 0.02], [0.4, 0.0], [0.6, 0.2]])
    valid = np.array([0, 1, 1, 1])
    # Max progress is independent of tracking tolerance and therefore retains
    # the farthest valid closest-point projection.
    assert m.max_path_progress(pos, ref, valid) == pytest.approx(0.6)
    # Valid points cover neighborhoods around x=.2 and x=.4; the off-path .6
    # point and acquisition x=0 point do not count.
    assert m.trajectory_path_coverage(
        pos, ref, valid, tolerance=0.051
    ) == pytest.approx(2 / 11)


def test_cvar_no_negative_zero():
    import math
    v = m.cvar([0.0, 0.0, 0.0], 0.95, "low")     # low tail of all-zeros -> +0.0, not -0.0
    assert v == 0.0 and math.copysign(1.0, v) > 0


# --- registry / by-name ---

def test_registry_and_signals():
    names = m.list_metrics()
    for n in [
        "cvar", "pearson_correlation", "goal_error", "violation_cvar",
        "smoothness", "energy", "fall_rate",
    ]:
        assert n in names
    assert m.metric_signals("goal_error") == ("final_pos", "target")
    assert m.get_metric("goal_error") is m.goal_error


def test_compute_metrics_by_name_pulls_signals_and_config():
    signals = {
        "final_pos": [1.0, 0.0], "target": [0.0, 0.0],
        "controls": np.array([[0.0], [1.0], [2.0]]),
        "g": np.array([-1.0, 0.5, 0.0]),
    }
    out = m.compute_metrics(
        ["goal_error", "control_smoothness", "violation_rate", "violation_cvar"],
        signals, config={"alpha": 0.9},
    )
    assert set(out) == {"goal_error", "control_smoothness", "violation_rate", "violation_cvar"}
    assert abs(out["goal_error"] - 1.0) < 1e-9
    # missing-signal metric is skipped, not errored
    out2 = m.compute_metrics(["goal_error", "balance_margin"], signals)
    assert "balance_margin" not in out2 and "goal_error" in out2
    with pytest.raises(KeyError):
        m.compute_metrics(["balance_margin"], signals, skip_missing=False)


# --- aggregation ---

def test_aggregate_over_runs():
    records = [
        {"success": True, "goal_error": 0.1, "level": "a"},
        {"success": False, "goal_error": 0.5, "level": "a"},
        {"success": True, "goal_error": 0.2, "level": "b"},
    ]
    agg = aggregate(records)
    assert abs(agg["success"]["mean"] - 2 / 3) < 1e-9      # bool -> success RATE
    assert agg["goal_error"]["n"] == 3
    assert "cvar95" in agg["goal_error"]
    by = aggregate_by(records, "level")
    assert abs(by["a"]["success"]["mean"] - 0.5) < 1e-9
    assert by["b"]["success"]["mean"] == 1.0


def test_aggregate_cvar_tail_per_metric():
    # higher-is-better (balance_margin) -> worst case is the SMALLEST (low tail);
    # cost (goal_error) -> worst case is the LARGEST (high tail).
    recs = [{"balance_margin": v, "goal_error": v} for v in (0.1, 0.5, 0.9)]
    agg = aggregate(recs)
    assert agg["balance_margin"]["cvar95"] == 0.1     # low tail -> min
    assert agg["goal_error"]["cvar95"] == 0.9         # high tail -> max


def test_compute_metrics_aliasing_binding_shorthand():
    signals = {"g_fric": [-1.0, 0.5, 2.0], "g_bal": [-1.0, -1.0, 0.1]}
    out = m.compute_metrics([
        {"name": "violation_cvar", "as": "fric_cvar", "bind": {"g": "g_fric"}, "config": {"alpha": 0.5}},
        ("max_violation", "g_bal"),                   # shorthand binding
    ], signals)
    assert out["fric_cvar"] > 0.0                     # bound to g_fric
    assert abs(out["max_violation:g_bal"] - 0.1) < 1e-9


def test_compute_metrics_chained_composite_and_per_metric_config():
    signals = {"final_pos": [0.1, 0.0], "target": [0.0, 0.0], "g": [-1.0, 0.5]}
    # success composes off the goal_error metric output + max_violation (bound)
    out = m.compute_metrics([
        "goal_error", "max_violation",
        {"name": "success", "bind": {"violation": "max_violation"}, "config": {"margin": 0.2}},
    ], signals)
    assert out["success"] == 0.0                       # reached but violating (g=0.5)
    # feasible variant
    out2 = m.compute_metrics([
        "goal_error", "max_violation",
        {"name": "success", "bind": {"violation": "max_violation"}},
    ], {"final_pos": [0.1, 0.0], "target": [0.0, 0.0], "g": [-1.0, -0.5]})
    assert out2["success"] == 1.0
    # per-metric config: tight margin fails, loose passes (nested config by label)
    base = {"goal_error": 0.1}
    assert m.compute_metrics([{"name": "success", "config": {"margin": 0.05}}], base)["success"] == 0.0
    assert m.compute_metrics(["success"], base, config={"success": {"margin": 0.5}})["success"] == 1.0


# --- framework adapter is general (task-specificity only in the extractor) ---

def test_general_plugin_adapter():
    from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin

    class _Traj:
        states = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        actions = [np.array([0.0]), np.array([0.5])]

    class _Env:
        target = np.array([1.0, 0.0])

    def extractor(traj, env, obstacles, constraints, **kw):
        st = np.stack([np.ravel(s) for s in traj.states])
        return {"final_pos": st[-1], "target": env.target,
                "controls": np.stack([np.ravel(a) for a in traj.actions]),
                "g": kw.get("violations", np.zeros(len(st)))}

    plug = GeneralMetricsPlugin(["goal_error", "control_smoothness", "violation_rate"],
                                extractor, name="demo")
    assert plug.name == "demo"
    out = plug.compute(_Traj(), _Env(), None, None, violations=np.array([0.0, 1.0]))
    assert abs(out["goal_error"]) < 1e-9 and out["violation_rate"] == 0.5
