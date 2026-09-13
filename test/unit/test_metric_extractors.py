"""Gates for task signal extractors (fedguide — corridor extractor is pure numpy).

Shows a task reporting the SAME general metrics via a tiny extractor + the shared
library, with task-specificity isolated to the extractor."""

import numpy as np
import pytest
from types import SimpleNamespace

from genedynamics.experiments.plugins.metrics.extractors import (
    CORRIDOR_METRICS,
    HUMANOID_METRICS,
    _HUMANOID_EVALUATION_CONTRACT,
    _humanoid_evaluation_view,
    _contact_step_signals,
    _humanoid_goal_signals,
    _humanoid_walk_diagnostics,
    corridor_metrics_plugin,
    corridor_signals,
)


class _Traj:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions


class _CorridorEnv:
    target = np.array([1.0, 0.0])

    def __init__(self, clearance=0.3):
        self._c = clearance

    def get_min_clearance(self, state):
        return self._c


def _make_traj(final_xy=(0.95, 0.0)):
    # 14-D corridor states; only [:2] (position) matters to the extractor
    s0 = np.zeros(14); s1 = np.zeros(14); s1[:2] = final_xy
    a = [np.zeros(9), 0.1 * np.ones(9)]
    return _Traj([s0, s1], a)


def test_corridor_signals_shapes_and_values():
    env = _CorridorEnv(clearance=0.3)
    sig = corridor_signals(_make_traj(), env, None, None, planning_time=0.05)
    assert sig["positions"].shape == (2, 2)
    assert np.allclose(sig["final_pos"], [0.95, 0.0])
    assert np.allclose(sig["target"], [1.0, 0.0])
    assert np.allclose(sig["g"], [-0.3, -0.3])      # clearance 0.3 -> no violation
    assert sig["controls"].shape == (2, 9)
    assert sig["seconds"] == 0.05


def test_corridor_plugin_reports_general_metrics():
    plug = corridor_metrics_plugin(margin=0.2)
    out = plug.compute(_make_traj((0.95, 0.0)), _CorridorEnv(0.3), None, None, planning_time=0.05)
    # all metrics are GENERAL functions, produced via the extractor
    assert abs(out["goal_error"] - 0.05) < 1e-9        # reached within margin
    assert out["success"] == 1.0                        # reached + feasible
    assert out["violation_rate"] == 0.0
    assert out["runtime"] == 0.05
    assert "violation_cvar" in out and "control_smoothness" in out
    assert "path_efficiency" in out and "completion_step" in out


def test_corridor_plugin_flags_collision_and_miss():
    # collision (clearance -0.1 -> g=+0.1 violation) and goal missed
    plug = corridor_metrics_plugin(margin=0.2)
    out = plug.compute(_make_traj((0.4, 0.0)), _CorridorEnv(-0.1), None, None)
    assert out["violation_rate"] == 1.0
    assert out["max_violation"] > 0.0
    assert out["success"] == 0.0                        # violating AND too far
    assert out["goal_error"] > 0.2


def test_contact_signals_use_collected_states_without_replay(monkeypatch):
    import jax.numpy as jnp

    class State:
        pipeline_state = object()

        def __init__(self, value):
            self.value = value

    trajectory = _Traj(
        [State(0.0), State(1.0), State(2.0)],
        [jnp.asarray([0.25]), jnp.asarray([0.5])],
    )

    def fail_replay(*args, **kwargs):
        raise AssertionError("structured contact trajectories must not be replayed")

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.metrics.extractors._roll_brax",
        fail_replay,
    )
    signals = _contact_step_signals(
        trajectory,
        object(),
        jnp.asarray(trajectory.actions),
        lambda env, state, action: {
            "value": jnp.asarray([state.value]),
            "action": action,
        },
        {},
    )
    assert np.allclose(signals["value"].reshape(-1), [1.0, 2.0])
    assert np.allclose(signals["action"].reshape(-1), [0.25, 0.5])


def test_humanoid_locomotion_metrics_reject_box_only_success():
    from genedynamics.evaluation.metrics import completion_step

    cfg = SimpleNamespace(level="push_walk", walk_success_mode="locomotion", goal_eps=0.01)
    signals = _humanoid_goal_signals([0.1, 0.3, 0.3], [0, 0, 0], [0, 0, 0], 0.3, cfg)
    assert signals["task_goal_error"] > cfg.goal_eps
    assert completion_step(signals["completion_residual"], signals["completion_target"], cfg.goal_eps) == 3
    signals = _humanoid_goal_signals([0.1, 0.3, 0.3], [0, 0, 0], [0, 0, 1], 0.3, cfg)
    assert signals["task_goal_error"] == 0.0
    assert completion_step(signals["completion_residual"], signals["completion_target"], cfg.goal_eps) == 2


def test_humanoid_legacy_completion_and_yaw_metrics_unchanged():
    from genedynamics.evaluation.metrics import completion_step

    positions = np.array([[0.1], [0.29], [0.299]])
    for level in ("push_to_line", "heavy_dr", "unjam", "push_walk"):
        cfg = SimpleNamespace(level=level, goal_eps=0.005, unjam_yaw_eps=0.03)
        signals = _humanoid_goal_signals(positions, [0.1, 0.1, 0.06], [0, 0, 0], 0.3, cfg)
        assert completion_step(signals["completion_residual"], signals["completion_target"], cfg.goal_eps) == completion_step(positions, [0.3], cfg.goal_eps)
        expected = 0.01 if level == "unjam" else abs(0.299 - 0.3)
        assert np.isclose(signals["task_goal_error"], expected)


@pytest.mark.parametrize("level", ["push_to_line", "heavy_dr", "unjam", "push_walk"])
def test_humanoid_completion_uses_task_event_not_box_entry(level):
    from genedynamics.evaluation.metrics import compute_metrics

    cfg = SimpleNamespace(level=level, goal_eps=0.005, unjam_yaw_eps=0.03,
                          walk_success_mode="locomotion" if level == "push_walk" else "legacy")
    positions = np.array([[0.1], [0.3], [0.3]])
    goal = _humanoid_goal_signals(positions, [0.08, 0.06, 0.02], [0, 0, 1], 0.3, cfg)
    requests = [item for item in HUMANOID_METRICS if isinstance(item, dict)
                and item.get("as") in {"completion_step", "box_completion_step"}]
    signals = {**goal, "positions": positions, "target": [0.3], "goal_tolerance": 0.005}
    result = compute_metrics(requests, signals)
    assert result == {"completion_step": 2, "box_completion_step": 1}
    signals.update(_humanoid_goal_signals(positions, [0.08, 0.06, 0.02], [0, 0, 0], 0.3, cfg))
    assert compute_metrics(requests, signals)["completion_step"] == 3


def _h1_evaluation_signals(padding=0):
    force = np.array([10., 20., 70.] + [70.] * padding)
    physics = np.stack([force, force + 1., force + 2., force + 1., force], axis=1)
    margins = np.zeros((*physics.shape, 4))
    margins[:, :, 0] = (physics - 60.) / 60.
    return {
        "force": force, "force_des": np.array([15., 25., 40.] + [0.] * padding),
        "task_force_reference": np.array([15., 25., 40.] + [0.] * padding),
        "success_padding": np.array([0., 0., 0.] + [1.] * padding),
        "task_success": np.array([0., 0., 1.] + [1.] * padding),
        "g_safety": (force - 60.) / 60.,
        "f_min": 0., "f_max": 60., "dt": 0.02, "seconds": 12.0,
        "reliability_features": np.arange(len(force) * 24).reshape(len(force), 24),
        "reliability_risk": np.ones((len(force), 4)),
        "physics_hand_force": physics,
        "physics_nonhand_force": np.zeros_like(physics),
        "physics_safety_margins": margins,
        "physics_samples_valid": np.array([True] * 3 + [False] * padding),
        "physics_initial_safety_margins": np.array([-1., -0.5 / 60., -1., -1.]),
        "physics_dt": 0.004,
        "task_metadata": {"evaluation_contract": dict(_HUMANOID_EVALUATION_CONTRACT)},
    }


def test_humanoid_view_preserves_completion_risk_raw_artifacts_and_padding_invariance():
    from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin

    requests = [{"name": "cvar", "as": "force_cvar95", "bind": {"x": "force"}},
                "force_tracking_mae", "force_violation_rate",
                {"name": "max_violation", "as": "safety_violation", "bind": {"g": "g_safety"}}]
    results = []
    for padding in (0, 1, 54):
        signals = _h1_evaluation_signals(padding)
        view = _humanoid_evaluation_view(signals)
        assert len(view["force"]) == 3
        assert view["force_des"][-1] == 40.0  # Real completion target remains scored.
        assert view["reliability_features"] is signals["reliability_features"]
        assert view["reliability_risk"] is signals["reliability_risk"]
        assert view["task_metadata"] is signals["task_metadata"]
        assert view["seconds"] == 12.0
        plugin = GeneralMetricsPlugin(requests, extractor=lambda *args, **kw: signals,
                                      persist_signals=True, evaluation_view=_humanoid_evaluation_view)
        results.append(plugin.compute(None, None, None, None))
        raw = plugin.pop_artifacts()["task_signals"]
        assert raw is signals
        assert len(raw["force"]) == 3 + padding
        assert len(raw["reliability_risk"]) == 3 + padding
    assert results[0] == results[1] == results[2]
    assert results[0]["force_violation_rate"] == pytest.approx(1 / 3)
    assert results[0]["force_tracking_mae"] == pytest.approx(40 / 3)
    assert results[0]["force_cvar95"] == 70.0
    assert results[0]["safety_violation"] > 0  # Completion never hides a violation.


def test_humanoid_view_does_not_infer_new_contract_for_old_or_invalid_padding():
    signals = _h1_evaluation_signals(1)
    del signals["success_padding"]
    with pytest.raises(ValueError, match="audit-only"):
        _humanoid_evaluation_view(signals)
    signals = _h1_evaluation_signals(1)
    del signals["task_metadata"]
    with pytest.raises(ValueError, match="audit-only"):
        _humanoid_evaluation_view(signals)
    for mask in ([0., 1., 0., 1.], [1., 1., 1., 1.], [0., 0., 0., float("nan")]):
        signals = _h1_evaluation_signals(1)
        signals["success_padding"] = mask
        with pytest.raises(ValueError):
            _humanoid_evaluation_view(signals)
    signals = _h1_evaluation_signals(1)
    signals["task_success"] = np.zeros(4)
    with pytest.raises(ValueError, match="successful transition"):
        _humanoid_evaluation_view(signals)
    signals = _h1_evaluation_signals(0)
    signals["task_success"] = np.zeros(3)
    assert len(_humanoid_evaluation_view(signals)["force"]) == 3


def test_humanoid_view_retains_first_fall_and_excludes_post_fall_fake_walk():
    signals = _h1_evaluation_signals(0)
    # A fixed-length diagnostic continues for two calls after the true
    # terminal transition and appears to complete two steps while lying down.
    def extend(values, tail):
        return np.concatenate([np.asarray(values), np.asarray(tail)], axis=0)

    signals["force"] = extend(signals["force"], [50., 55.])
    signals["force_des"] = extend(signals["force_des"], [40., 40.])
    signals["task_force_reference"] = extend(signals["task_force_reference"], [40., 40.])
    signals["g_safety"] = extend(signals["g_safety"], [0., 0.])
    signals["success_padding"] = np.zeros(5)
    signals["task_success"] = np.zeros(5)
    signals["task_fallen"] = np.array([0., 0., 1., 1., 1.])
    signals["positions"] = np.array([[0.1], [0.2], [0.25], [0.5], [0.8]])
    signals["box_yaw"] = np.array([0., .01, .02, .4, .8])
    signals["walk_left_steps"] = np.array([0., 0., 0., 1., 1.])
    signals["walk_right_steps"] = np.array([0., 0., 0., 0., 1.])
    signals["walk_body_progress"] = np.array([0., .02, .04, .3, .6])
    signals["walk_support_progress"] = np.array([0., 0., .01, .2, .5])
    signals["physics_hand_force"] = np.concatenate([
        signals["physics_hand_force"], np.full((2, 5), 50.)
    ])
    signals["physics_nonhand_force"] = np.zeros((5, 5))
    signals["physics_safety_margins"] = np.concatenate([
        signals["physics_safety_margins"], np.zeros((2, 5, 4))
    ])
    signals["physics_samples_valid"] = np.ones(5)
    signals["reliability_features"] = np.zeros((5, 24))
    signals["reliability_risk"] = np.zeros((5, 4))

    view = _humanoid_evaluation_view(signals)
    assert len(view["force"]) == 3
    assert view["walk_left_steps"][-1] == view["walk_right_steps"][-1] == 0.
    assert view["walk_support_progress"][-1] == pytest.approx(.01)
    np.testing.assert_allclose(view["final_pos"], [.25])
    np.testing.assert_allclose(view["final_yaw"], [.02])
    # Learned-reliability evidence retains the raw fixed-horizon collection;
    # only task/physics reporting is an episode-prefix view.
    assert view["reliability_features"] is signals["reliability_features"]


def _h1_physics_signals(physics, padding=0):
    physics = np.asarray(physics, dtype=float)
    n = len(physics)
    force = physics[:, -1].copy()
    # Invalid padding storage deliberately contains NaNs.  It must be removed
    # before any sample statistic or claimed-valid finite check is evaluated.
    physics = np.concatenate([physics, np.full((padding, physics.shape[1]), np.nan)])
    margins = np.zeros((*physics.shape, 4))
    margins[:, :, 0] = (physics - 60.) / 60.
    return {
        "force": np.r_[force, np.full(padding, force[-1])],
        "force_des": np.r_[np.full(n, 45.), np.zeros(padding)],
        "task_goal_error": 0., "goal_tolerance": 0.005,
        "g_safety": np.r_[(force - 60.) / 60., np.zeros(padding)],
        "task_success": np.r_[np.zeros(n - 1), np.ones(1 + padding)],
        "success_padding": np.r_[np.zeros(n), np.ones(padding)],
        "f_min": 0., "f_max": 60., "dt": 0.02, "physics_dt": 0.004,
        "physics_hand_force": physics,
        "physics_nonhand_force": np.zeros_like(physics),
        "physics_safety_margins": margins,
        "physics_samples_valid": np.r_[np.ones(n), np.zeros(padding)],
        "physics_initial_safety_margins": np.array([-1., -0.5 / 60., -1., -1.]),
        "reliability_features": np.zeros((n + padding, 24)),
        "reliability_risk": np.ones((n + padding, 4)),
        "task_metadata": {"robot": "h1", "evaluation_contract": dict(_HUMANOID_EVALUATION_CONTRACT)},
    }


def test_humanoid_physics_actual_samples_peak_tail_rate_and_impulse():
    from genedynamics.evaluation.metrics import compute_metrics

    # Saved same-state P3 diagnostic, outer transitions 48--51.  This is a
    # regression fixture, not a claim that four windows are independent runs.
    physics = np.array([
        [30.550533294677734, 24.78330421447754, 55.53253173828125, 9.015192031860352, 156.3263397216797],
        [1.5643279552459717, 331.50250244140625, 0., 184.62794494628906, 4.387762546539307],
        [103.33590698242188, 2.3505561351776123, 210.02981567382812, 0., 166.8761749267578],
        [4.558783531188965, 92.83455657958984, 3.9115443229675293, 200.64463806152344, 0.],
    ])
    signals = _h1_physics_signals(physics)
    result = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert result["physics_force_peak"] == result["physics_force_cvar95"] == pytest.approx(331.50250244140625)
    assert result["physics_force_normalized_cvar95"] == pytest.approx(result["physics_force_cvar95"] / 60.)
    assert result["force_peak"] == pytest.approx(166.8761749267578)
    assert result["physics_force_violation_rate"] == 8 / 20
    assert result["force_violation_rate"] == 2 / 4
    assert result["physics_force_impulse"] == pytest.approx(6.3313296604156495)
    assert result["physics_force_limit_excess_impulse"] == pytest.approx(3.8647115173339843)
    endpoint_limit_excess = np.maximum(physics[:, -1] - 60., 0.).sum() * 0.02
    assert endpoint_limit_excess == pytest.approx(4.06405029296875)
    assert result["physics_force_limit_excess_impulse"] < endpoint_limit_excess
    assert result["physics_sample_coverage"] == 1.
    assert result["safe_success"] == 0.
    assert _HUMANOID_EVALUATION_CONTRACT["primary_force_tail"] == "physics_force_normalized_cvar95"


@pytest.mark.parametrize("padding", [0, 1, 54])
def test_humanoid_substep_tail_uses_real_samples_and_keeps_completion(padding):
    from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin

    physics = np.zeros((20, 5))
    physics[-1, 2] = 120.  # Only the middle of the successful transition is unsafe.
    signals = _h1_physics_signals(physics, padding)
    plugin = GeneralMetricsPlugin(HUMANOID_METRICS, extractor=lambda *args, **kw: signals,
                                  persist_signals=True, evaluation_view=_humanoid_evaluation_view)
    result = plugin.compute(None, None, None, None)
    assert result["force_peak"] == result["force_violation_rate"] == 0.
    assert result["physics_force_peak"] == 120.
    assert result["physics_force_cvar95"] == 24.  # Worst five of 100, not repeated row maxima.
    assert result["physics_force_violation_rate"] == 0.01
    assert result["physics_force_impulse"] == 0.48
    assert result["safe_success"] == 0.
    raw = plugin.pop_artifacts()["task_signals"]
    assert raw is signals and raw["physics_hand_force"].shape == (20 + padding, 5)
    view = _humanoid_evaluation_view(signals)
    assert view["reliability_features"] is raw["reliability_features"]
    assert view["reliability_risk"] is raw["reliability_risk"]
    assert len(view["force"]) == 20 and len(view["physics_hand_force_samples"]) == 100


def test_humanoid_initial_safety_is_not_a_duration_weighted_sample():
    from genedynamics.evaluation.metrics import compute_metrics

    signals = _h1_physics_signals(np.full((2, 5), 10.))
    safe = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert safe["safe_success"] == 1.
    signals["physics_initial_safety_margins"][0] = 1.
    unsafe = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert unsafe["safe_success"] == 0. and unsafe["physics_safety_violation"] == 1.
    for key in ("physics_force_peak", "physics_force_cvar95", "physics_force_impulse",
                "physics_force_violation_rate"):
        assert unsafe[key] == safe[key]


def test_humanoid_p1_force_step_has_dedicated_physics_pass_and_no_box_ssr():
    from genedynamics.evaluation.metrics import compute_metrics

    physics = np.r_[np.zeros((10, 5)), np.full((10, 5), 15.0)]
    signals = _h1_physics_signals(physics)
    signals.update({
        "force_des": np.full(20, 15.0),
        "task_force_reference": np.full(20, 15.0),
        "force_expected": np.ones(20),
        "force_step_applicable": True,
        "force_step_target": 15.0,
        "force_step_rise_fraction": 0.9,
        "force_step_band_fraction": 0.1,
        "force_step_band_absolute": 1.0,
        "force_step_hold_time": 0.2,
    })
    view = _humanoid_evaluation_view(signals)
    result = compute_metrics(HUMANOID_METRICS, view)
    assert "safe_success_goal_error" not in view
    assert "safe_success" not in result
    assert result["force_step_success"] == 1.0
    assert result["physics_force_rise_time"] == pytest.approx(0.2)
    assert result["physics_force_settling_time"] == pytest.approx(0.204)
    assert result["physics_steady_force_tracking_mae"] == pytest.approx(7.5)

    # A last-window tracking miss fails the P1 pass even though the fixed box
    # happens to satisfy the unrelated geometric goal fixture.
    signals["physics_hand_force"][-1, -1] = 20.0
    signals["physics_safety_margins"][-1, -1, 0] = 0.0
    failed = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert failed["force_step_success"] == 0.0
    assert "safe_success" not in failed


@pytest.mark.parametrize("missing", ["none", "one", "all"])
def test_humanoid_unknown_physics_retains_diagnostics_not_safe_certification(missing):
    from genedynamics.evaluation.metrics import compute_metrics

    signals = _h1_physics_signals(np.zeros((2, 5)))
    if missing == "none":  # No sampler fields, e.g. a historical diagnostic.
        for key in list(signals):
            if key.startswith("physics_"):
                del signals[key]
    else:
        signals["physics_samples_valid"][:] = [1., 0.] if missing == "one" else 0.
    result = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert result["force_peak"] == 0. and result["safe_success"] == 0.
    assert result["physics_sample_coverage"] == (0.5 if missing == "one" else 0.)
    assert "physics_force_peak" not in result and "physics_force_normalized_cvar95" not in result
    assert "physics_safety_violation" not in result  # Unknown is not an observed violation.
    assert "safe_success" not in compute_metrics(HUMANOID_METRICS, signals)


@pytest.mark.parametrize("bad", ["force_shape", "margin_shape", "dt", "flag", "nan", "negative", "initial"])
def test_humanoid_claimed_valid_physics_rejects_corrupt_contract(bad):
    signals = _h1_physics_signals(np.zeros((2, 5)))
    if bad == "force_shape":
        signals["physics_hand_force"] = np.zeros((2, 4))
    elif bad == "margin_shape":
        signals["physics_safety_margins"] = np.zeros((2, 5, 3))
    elif bad == "dt":
        signals["physics_dt"] = 0.02
    elif bad == "flag":
        signals["physics_samples_valid"][0] = np.nan
    elif bad == "nan":
        signals["physics_hand_force"][0, 2] = np.nan
    elif bad == "negative":
        signals["physics_nonhand_force"][0, 1] = -1.
    else:
        del signals["physics_initial_safety_margins"]
    with pytest.raises(ValueError, match="H1"):
        _humanoid_evaluation_view(signals)


def test_humanoid_old_h1_contract_rejected_but_g1_endpoint_contract_preserved():
    from genedynamics.evaluation.metrics import compute_metrics
    from genedynamics.experiments.plugins.metrics.extractors import _HUMANOID_LEGACY_EVALUATION_CONTRACT

    signals = _h1_physics_signals(np.zeros((2, 5)))
    signals["task_metadata"]["evaluation_contract"] = dict(_HUMANOID_LEGACY_EVALUATION_CONTRACT)
    with pytest.raises(ValueError, match="audit-only"):
        _humanoid_evaluation_view(signals)
    for key in list(signals):
        if key.startswith("physics_"):
            del signals[key]
    signals["task_metadata"]["robot"] = "g1"
    signals["g_safety"] = np.array([0., 0.5e-6])  # Original endpoint tolerance preserved.
    result = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert result["safe_success"] == 1. and result["force_peak"] == 0.
    assert not any(key.startswith("physics_") for key in result)


def test_general_evaluation_view_is_opt_in_for_existing_arm_and_peg_plugins():
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_metrics_plugin, peg_insert_metrics_plugin,
    )
    from genedynamics.experiments.plugins.metrics.general import GeneralMetricsPlugin

    signals = _h1_evaluation_signals(54)
    signals["force"][3:] = 2.0
    signals["axial_force"] = signals["force"]
    for plugin in (arm_surface_scan_metrics_plugin(), peg_insert_metrics_plugin()):
        assert plugin._evaluation_view is None
        plugin._extractor = lambda *args, **kw: signals
        result = plugin.compute(None, None, None, None)
        assert plugin.pop_artifacts()["task_signals"] is signals
        assert len(signals["force"]) == 57
        if plugin.name == "arm_surface_scan_metrics":
            assert result["force_tracking_error"] == pytest.approx(
                np.sqrt(np.mean((signals["force"] - signals["force_des"]) ** 2))
            )
        else:
            assert result["axial_force_cvar95"] == pytest.approx(100.0 / 3.0)
    plugin = GeneralMetricsPlugin(["force_tracking_mae"], extractor=lambda *args, **kw: signals)
    assert plugin.compute(None, None, None, None)["force_tracking_mae"] == pytest.approx(
        np.mean(np.abs(signals["force"] - signals["force_des"]))
    )


@pytest.mark.parametrize("sampling", ["recorded", "unknown"])
@pytest.mark.parametrize("mode", [0, 1, None])
@pytest.mark.parametrize("startup_mode", ["legacy", "synchronized"])
def test_humanoid_extractor_keeps_actual_completion_force_reference_without_physics(
        sampling, mode, startup_mode):
    import jax.numpy as jnp
    from genedynamics.experiments.plugins.metrics.extractors import humanoid_box_push_signals

    cfg = SimpleNamespace(level="unjam", goal_eps=0.005, unjam_yaw_eps=0.03,
                          approach_time=0.5, force_ramp_time=0.5, f_target=45.,
                          f_max=60., support_radius=0.25, timestep=0.004,
                          walk_force_startup_mode=startup_mode)
    if startup_mode == "synchronized":
        cfg.level, cfg.robot = "push_walk", "h1"
        cfg.walk_leg_control, cfg.walk_success_mode = "joint_target", "locomotion"
    taper = 0.5 if startup_mode == "synchronized" else 1.0
    expected_benchmark = 37.8 * taper
    expected_execution = expected_benchmark if startup_mode == "synchronized" else 45.
    ps = SimpleNamespace(
        x=SimpleNamespace(pos=jnp.array([[1., 0., 0.55], [0., 0., 1.]]),
                          rot=jnp.array([[1., 0., 0., 0.]] * 2)),
        site_xpos=jnp.array([[0., -0.1, 0.], [0., 0.1, 0.]]),
    )
    def state(step, success, padding):
        return SimpleNamespace(pipeline_state=ps, info={
            "step": jnp.int32(step), "task_success": jnp.float32(success),
            "success_padding": jnp.bool_(padding), "box_goal_x": jnp.float32(1.),
            "box_x0": jnp.float32(0.9), "contact_acquired": jnp.float32(1.),
            "contact_step": jnp.int32(1),
            "physics_hand_force": jnp.array([70., 75., 80., 75., 70.]),
            "physics_nonhand_force": jnp.zeros(5),
            "physics_safety_margins": jnp.array([
                [(force - 60.) / 60., 0., 0., 0.] for force in [70., 75., 80., 75., 70.]
            ]),
            "physics_samples_valid": jnp.bool_(not padding and sampling == "recorded"),
            **({"mga_execution_mode": jnp.int32(mode)} if mode is not None else {}),
            "mga_unload_stiffness_matrix": jnp.eye(3) * 7.,
        })
    initial, completion, padding = state(45, 0, False), state(46, 1, False), state(47, 1, True)
    contact = {"f_t": jnp.zeros(3), "f_n": jnp.float32(70.),
               "slip": jnp.float32(0.), "F_eff": jnp.float32(30.)}
    env = SimpleNamespace(
        _bcfg=cfg, dt=0.02, _box_idx=1, _pelvis_idx=2, _feet_site_id=jnp.array([0, 1]),
        _mu=jnp.float32(0.6),
        _hand_contact=lambda *args: contact,
        _manifold=lambda *args: (jnp.zeros(14), jnp.zeros(3)),
        _corridor_clearance=lambda *args: jnp.float32(0.01),
        _has_fallen=lambda *args: jnp.bool_(False),
        _box_contact_forces=lambda *args: {"hand": jnp.float32(70.),
                                         "wall": jnp.float32(0.), "nonhand": jnp.float32(0.)},
        _safety_margins=lambda *args: jnp.array([1. / 6., 0., 0., 0.]),
        _transition_safety_margins=lambda s, endpoint: jnp.where(
            s.info["success_padding"], endpoint,
            jnp.max(s.info["physics_safety_margins"], axis=0)),
        _unpack=lambda *args: (None, None, None, None, jnp.eye(3), None),
        realized_hand_stiffness=lambda s, u: jnp.where(
            s.info.get("mga_execution_mode", -1) == 1, s.info["mga_unload_stiffness_matrix"], jnp.eye(3)),
        walk_force_scale=lambda *args: jnp.float32(taper),
        requested_force_reference=lambda ps, info: 45. * jnp.clip(
            (info["step"] * 0.02 - 0.5) / 0.5, 0., 1.) * taper,
        _walk_contact_diagnostics=lambda ps: {},
        reliability_features=lambda *args: jnp.zeros(24),
        reliability_contract=lambda: {"probe": "array_only"},
        reliability_source_sha256=lambda: {"probe": "array_only"},
    )
    if startup_mode == "synchronized":
        from types import MethodType
        from genedynamics.envs.domains.humanoid.box_push_brax import HumanoidBoxPushEnv
        for name in ("_global_force_startup_scale", "_normal_force_startup_scale"):
            setattr(env, name, MethodType(getattr(HumanoidBoxPushEnv, name), env))
    # The force action encodes 60 N; execution_force_reference must still use
    # nominal f_target=45 N, not the selected command or measured 70 N.
    action = (jnp.zeros(23).at[11].set(1.) if startup_mode == "synchronized" else jnp.zeros(12))
    trajectory = _Traj([initial, completion, padding], [action, action])
    signals = humanoid_box_push_signals(trajectory, env, None, None, x0=initial)
    np.testing.assert_allclose(signals["task_force_reference"], [expected_benchmark, 0.], atol=1e-5)
    np.testing.assert_allclose(signals["execution_force_reference"],
                               [0. if mode == 1 else expected_execution, 0.], atol=1e-5)
    np.testing.assert_array_equal(signals["force"], [70., 70.])
    np.testing.assert_array_equal(signals["physics_hand_force"],
                                   np.tile([70., 75., 80., 75., 70.], (2, 1)))
    if startup_mode == "synchronized" and mode == 0:
        selected_post_scale = float(env._normal_force_startup_scale(completion.info))
        assert signals["execution_force_reference"][0] == pytest.approx(
            45. * selected_post_scale * taper, abs=1e-5)
        assert signals["execution_force_reference"][0] != pytest.approx(
            60. * selected_post_scale * taper, abs=1e-5)
    np.testing.assert_array_equal(signals["mga_execution_mode"], [-1 if mode is None else mode] * 2)
    np.testing.assert_array_equal(signals["reliability_execution_applicable"], [mode == 0] * 2)
    np.testing.assert_array_equal(signals["stiffness"], np.tile(np.eye(3).reshape(1, -1) * (7. if mode else 1.), (2, 1)))
    assert len(_humanoid_evaluation_view(signals)["force"]) == 1
    assert signals["reliability_risk"].shape == (2, 4)
    assert signals["physics_hand_force"].shape == (2, 5)
    assert signals["physics_safety_margins"].shape == (2, 5, 4)
    np.testing.assert_array_equal(signals["physics_initial_safety_margins"], np.asarray([1. / 6., 0., 0., 0.], np.float32))
    if mode != 0:
        assert np.isnan(signals["reliability_risk"]).all()
        np.testing.assert_array_equal(signals["reliability_risk_valid"], [0., 0.])
        np.testing.assert_array_equal(signals["physics_samples_valid"], [sampling == "recorded", False])
        if sampling == "recorded":
            # N/A/unknown learned labels must not erase known physical violations.
            view = _humanoid_evaluation_view(signals)
            assert np.max(view["physics_hand_force_samples"]) == 80.
            assert view["safe_success_violation"] > 0.
        if mode is None:
            from scripts.tasks.robot.humanoid.train_box_push_reliability import _sequence_rows

            with pytest.raises(ValueError, match="invalid H1 reliability execution flags"):
                _sequence_rows({"actions": np.zeros((2, 12)), "task_signals": signals},
                               {"env_params": {"level": "unjam"}})
    elif sampling == "recorded":
        assert np.all(signals["reliability_risk"][:, 0] == 1.)
        np.testing.assert_array_equal(signals["reliability_risk_valid"], [1., 1.])
        assert signals["reliability_risk"][0, 3] == pytest.approx(
            (70. - expected_benchmark) / 45., abs=1e-6)
    else:
        from scripts.tasks.robot.humanoid.train_box_push_reliability import _sequence_rows

        assert np.isnan(signals["reliability_risk"][0]).all()
        assert signals["reliability_risk"][1, 0] == 1.  # Known frozen risk is retained.
        np.testing.assert_array_equal(signals["reliability_risk_valid"], [0., 1.])
        with pytest.raises(ValueError, match="non-finite H1 reliability|unknown NORMAL"):
            _sequence_rows({"actions": np.zeros((2, 12)), "task_signals": signals},
                           {"env_params": {"level": "unjam"}})


def test_physical_walk_diagnostics_are_new_mode_only_and_task_owned():
    state = object()
    expected = {"foot_normal_loads": np.array([100.0, 0.0]),
                "foot_floor_clearance": np.array([-0.001, 0.06])}
    calls = []

    def diagnostics(value):
        calls.append(value)
        return expected

    cfg = SimpleNamespace(level="push_walk", walk_success_mode="locomotion",
                          walk_leg_control="joint_target")
    env = SimpleNamespace(_bcfg=cfg, _walk_contact_diagnostics=diagnostics)
    assert _humanoid_walk_diagnostics(env, state) is expected
    assert calls == [state]
    del env._walk_contact_diagnostics
    for level, success, control in (("push_to_line", "locomotion", "joint_target"),
                                    ("push_walk", "legacy", "joint_target"),
                                    ("push_walk", "locomotion", "legacy")):
        cfg.level, cfg.walk_success_mode, cfg.walk_leg_control = level, success, control
        assert _humanoid_walk_diagnostics(env, state) == {}


def test_humanoid_unknown_execution_mode_does_not_discard_known_physical_statistics():
    from genedynamics.evaluation.metrics import compute_metrics

    signals = _h1_physics_signals(np.full((2, 5), 10.))
    signals["mga_execution_mode"] = np.zeros(2, np.int32)
    signals["reliability_execution_applicable"] = np.ones(2, bool)
    normal = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    signals["mga_execution_mode"][:] = -1
    signals["reliability_execution_applicable"][:] = False
    signals["reliability_risk"][:] = np.nan
    unknown = compute_metrics(HUMANOID_METRICS, _humanoid_evaluation_view(signals))
    assert normal["safe_success"] == unknown["safe_success"] == 1.
    for key in ("physics_sample_coverage", "physics_force_peak", "physics_force_cvar95",
                "physics_force_impulse", "physics_force_violation_rate",
                "force_peak", "force_violation_rate"):
        assert unknown[key] == normal[key]
    np.testing.assert_array_equal(signals["physics_hand_force"], np.full((2, 5), 10.))
    np.testing.assert_array_equal(signals["physics_samples_valid"], 1.)
