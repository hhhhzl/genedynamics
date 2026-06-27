"""Unit tests for the receding-horizon bridge (DIAL-MPC execution layer).

Pure-Python / numpy: no jax, no mjx, no solver internals. Verifies the bridge
logic (warm-start threading, step-count annealing, shift, first-action, real
stepping) with a mock planner + mock dynamics, so it runs on fedguide x86.
"""

import numpy as np
import pytest

from genedynamics.solvers.common.receding_horizon import (
    RecedingHorizonController,
    WarmStartPlanner,
)


class MockPlanner:
    """Minimal WarmStartPlanner that records how the bridge drives it.

    plan_var = dense control of shape (H, du). replan does a trivial one-shot
    "refinement" (nudge toward zero by a factor that depends on the schedule
    length) and records the warm_start it received + the schedule length.
    """

    def __init__(self, H=5, du=2):
        self.H, self.du = H, du
        self.replan_calls = []  # list of (schedule_len, warm_start_copy)

    def init_plan_var(self):
        return np.zeros((self.H, self.du), dtype=np.float32)

    def make_schedule(self, n_diffuse):
        # geometric-style factor prefix; the bridge only cares about its length
        return [0.5 ** k for k in range(n_diffuse)]

    def replan(self, state, warm_start, schedule, rng, t0=0.0):
        self.replan_calls.append((len(schedule), np.array(warm_start)))
        # deterministic "improvement": each diffusion step shrinks toward a
        # constant target action of 1.0, so more steps => closer to target.
        out = np.array(warm_start, dtype=np.float32)
        for _ in schedule:
            out = out + 0.5 * (1.0 - out)
        return out

    def first_action(self, plan_var):
        return np.array(plan_var[0])

    def shift(self, plan_var):
        rolled = np.roll(plan_var, -1, axis=0)
        rolled[-1] = 0.0
        return rolled


def _double_integrator_step(state, action):
    # state = [pos(du), vel(du)]; action = accel(du). dt = 1.
    du = action.shape[0]
    pos, vel = state[:du], state[du:]
    vel = vel + action
    pos = pos + vel
    return np.concatenate([pos, vel]).astype(np.float32)


def _counter_split(rng):
    # deterministic numpy-friendly splitter: (next, sub)
    return rng + 1, rng


def _make_controller(planner, n_steps=8, n_diffuse_init=10, n_diffuse=2):
    return RecedingHorizonController(
        planner,
        _double_integrator_step,
        n_steps=n_steps,
        n_diffuse_init=n_diffuse_init,
        n_diffuse=n_diffuse,
        split_rng=_counter_split,
        collect_plan_vars=True,
    )


def test_protocol_satisfied():
    assert isinstance(MockPlanner(), WarmStartPlanner)


def test_runs_n_steps_and_shapes():
    p = MockPlanner(H=5, du=2)
    ctrl = _make_controller(p, n_steps=8)
    res = ctrl.run(np.zeros(4, dtype=np.float32), rng=0)
    assert res.horizon == 8
    assert len(res.actions) == 8
    assert len(res.states) == 9  # states = actions + 1
    assert all(a.shape == (2,) for a in res.actions)


def test_step_count_annealing_init_then_steady():
    # First replan must use n_diffuse_init; all later ones n_diffuse.
    p = MockPlanner()
    ctrl = _make_controller(p, n_steps=6, n_diffuse_init=10, n_diffuse=2)
    ctrl.run(np.zeros(4, dtype=np.float32), rng=0)
    sched_lens = [c[0] for c in p.replan_calls]
    assert sched_lens[0] == 10
    assert sched_lens[1:] == [2, 2, 2, 2, 2]
    assert ctrl.n_diffuse_at(0) == 10 and ctrl.n_diffuse_at(3) == 2


def test_warm_start_threaded_across_steps():
    # The warm_start handed to replan at step t must equal shift(replan_output)
    # of step t-1 -- i.e. the plan is genuinely carried, not reset to zeros.
    p = MockPlanner()
    ctrl = _make_controller(p, n_steps=4)
    res = ctrl.run(np.zeros(4, dtype=np.float32), rng=0)

    # step 0 warm_start is the cold init (all zeros)
    assert np.allclose(p.replan_calls[0][1], 0.0)
    # step t>=1 warm_start == shift(plan_var from step t-1)
    for t in range(1, 4):
        expected = p.shift(res.plan_vars[t - 1])
        got = p.replan_calls[t][1]
        assert np.allclose(got, expected), f"warm-start break at step {t}"
    # and it is NOT the cold zeros after the first step (warm-start really used)
    assert not np.allclose(p.replan_calls[1][1], 0.0)


def test_first_action_executed_on_real_dynamics():
    # The executed control is plan_var[0]; the state must advance by the
    # double-integrator using exactly that action.
    p = MockPlanner()
    ctrl = _make_controller(p, n_steps=3)
    res = ctrl.run(np.zeros(4, dtype=np.float32), rng=0)
    s = res.states[0]
    for t in range(3):
        s_next = _double_integrator_step(s, res.actions[t])
        assert np.allclose(s_next, res.states[t + 1])
        s = s_next


def test_more_init_steps_refine_more():
    # Sanity: with more diffusion steps the first action is closer to target 1.0.
    p_few = MockPlanner()
    RecedingHorizonController(
        p_few, _double_integrator_step, n_steps=1, n_diffuse_init=1, n_diffuse=1,
        split_rng=_counter_split,
    ).run(np.zeros(4, dtype=np.float32), rng=0)
    a_few = p_few.replan_calls[0]

    p_many = MockPlanner()
    res_many = RecedingHorizonController(
        p_many, _double_integrator_step, n_steps=1, n_diffuse_init=12, n_diffuse=1,
        split_rng=_counter_split, collect_plan_vars=True,
    ).run(np.zeros(4, dtype=np.float32), rng=0)
    first_action_many = res_many.actions[0]
    # 12 steps gets very close to 1.0; 1 step only reaches 0.5
    assert first_action_many[0] > 0.9
    assert a_few[0] == 1  # schedule length recorded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
