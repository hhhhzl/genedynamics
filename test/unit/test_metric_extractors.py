"""Gates for task signal extractors (fedguide — corridor extractor is pure numpy).

Shows a task reporting the SAME general metrics via a tiny extractor + the shared
library, with task-specificity isolated to the extractor."""

import numpy as np

from genedynamics.experiments.plugins.metrics.extractors import (
    corridor_signals, corridor_metrics_plugin, CORRIDOR_METRICS,
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
