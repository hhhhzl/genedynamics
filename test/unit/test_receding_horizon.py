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


def test_can_keep_only_endpoint_states_for_long_runs():
    p = MockPlanner(H=5, du=2)
    ctrl = RecedingHorizonController(
        p,
        _double_integrator_step,
        n_steps=8,
        n_diffuse_init=10,
        n_diffuse=2,
        split_rng=_counter_split,
        collect_states=False,
        synchronize_steps=True,
    )
    res = ctrl.run(np.zeros(4, dtype=np.float32), rng=0)
    assert res.horizon == 8
    assert len(res.states) == 2
    assert np.allclose(res.states[0], np.zeros(4))


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


def test_task_initialization_runs_once_before_refinement_and_preserves_shifting():
    planner = MockPlanner(H=5, du=2)
    initial_state = np.array([0.2, -0.3, 0.0, 0.0], dtype=np.float32)
    calls = []

    def initialize(state, default_plan):
        calls.append((state.copy(), default_plan.copy()))
        return default_plan + state[:2]

    controller = RecedingHorizonController(
        planner, _double_integrator_step, n_steps=3,
        n_diffuse_init=2, n_diffuse=1, split_rng=_counter_split,
        collect_plan_vars=True, initialize_plan=initialize,
    )
    result = controller.run(initial_state, rng=0)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], initial_state)
    np.testing.assert_array_equal(calls[0][1], np.zeros((5, 2), np.float32))
    np.testing.assert_array_equal(
        planner.replan_calls[0][1], np.broadcast_to(initial_state[:2], (5, 2)),
    )
    for step in (1, 2):
        np.testing.assert_array_equal(
            planner.replan_calls[step][1], planner.shift(result.plan_vars[step - 1]),
        )
    # A reused controller initializes from the new episode, not stale state.
    controller.run(initial_state + 0.1, rng=1)
    assert len(calls) == 2


def test_no_initializer_retains_legacy_plan_and_noncallables_fail():
    kwargs = dict(n_steps=2, n_diffuse_init=3, n_diffuse=1, split_rng=_counter_split)
    first = RecedingHorizonController(MockPlanner(), _double_integrator_step, **kwargs)
    second = RecedingHorizonController(
        MockPlanner(), _double_integrator_step, initialize_plan=None, **kwargs,
    )
    x0 = np.zeros(4, np.float32)
    a, b = first.run(x0, 0), second.run(x0, 0)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.states, b.states)
    with pytest.raises(TypeError, match="initialize_plan"):
        RecedingHorizonController(
            MockPlanner(), _double_integrator_step, initialize_plan=0.0, **kwargs,
        )


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


def test_optional_replan_diagnostics_are_collected():
    class DiagnosticPlanner(MockPlanner):
        def replan_with_info(self, state, warm_start, schedule, rng, t0=0.0):
            plan = super().replan(
                state, warm_start, schedule, rng, t0=t0
            )
            return plan, {
                "prior_accepted": float(t0 % 2 == 0),
                "prior_predicted_improvement": float(t0),
            }

    planner = DiagnosticPlanner()
    result = _make_controller(planner, n_steps=3).run(
        np.zeros(4, dtype=np.float32), rng=0
    )
    assert len(result.infos) == 3
    assert [x["prior_accepted"] for x in result.infos] == [1.0, 0.0, 1.0]
    assert [x["prior_predicted_improvement"] for x in result.infos] == [0.0, 1.0, 2.0]


def test_replan_diagnostics_can_be_consumed_online_without_retention():
    seen = []

    class DiagnosticPlanner(MockPlanner):
        def replan_with_info(self, state, warm_start, schedule, rng, t0=0.0):
            plan = super().replan(state, warm_start, schedule, rng, t0=t0)
            return plan, {"real_step": int(t0), "large_candidate_tensor": np.ones(100)}

    def execute(state, action, info):
        seen.append(info["real_step"])
        return _double_integrator_step(state, action)

    result = RecedingHorizonController(
        DiagnosticPlanner(), _double_integrator_step,
        n_steps=3, n_diffuse_init=2, n_diffuse=1,
        split_rng=_counter_split, execution_step=execute,
        collect_states=False, collect_infos=False,
    ).run(np.zeros(4, dtype=np.float32), rng=0)
    assert seen == [0, 1, 2]
    assert result.infos == []
    assert len(result.states) == 2


@pytest.mark.parametrize("collect_states", [False, True])
def test_execution_step_none_and_identity_preserve_ordinary_controller(collect_states):
    kwargs = dict(n_steps=4, n_diffuse_init=3, n_diffuse=1,
                  split_rng=_counter_split, collect_states=collect_states,
                  collect_plan_vars=True)
    initial = np.zeros(4, np.float32)
    reference = RecedingHorizonController(
        MockPlanner(), _double_integrator_step, **kwargs
    ).run(initial, 0)
    seen = []
    def identity(state, action, info):
        seen.append(info)
        return _double_integrator_step(state, action)
    actual = RecedingHorizonController(
        MockPlanner(), _double_integrator_step, execution_step=identity, **kwargs
    ).run(initial, 0)
    assert seen == [None] * 4
    for key in ("states", "actions", "plan_vars"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(reference, key))
    assert actual.infos == reference.infos == []


@pytest.mark.parametrize("callback", [False, {}])
def test_execution_step_rejects_noncallable(callback):
    with pytest.raises(TypeError, match="execution_step"):
        RecedingHorizonController(
            MockPlanner(), _double_integrator_step, n_steps=1,
            n_diffuse_init=1, n_diffuse=1, execution_step=callback,
        )


def test_execution_step_receives_final_info_and_preserves_commit_shift_order():
    events, selected_infos = [], []
    class Planner(MockPlanner):
        def replan_with_info(self, state, warm_start, schedule, rng, t0=0.0):
            events.append("replan")
            selected_infos.append({"execution_mode": int(t0 % 2)})
            return warm_start + 1.0, selected_infos[-1]
        def first_action(self, nodes):
            events.append("first")
            return super().first_action(nodes)
        def after_step(self, state, action, next_state):
            events.append("after")
            assert next_state[0] == state[0] + 10 + selected_infos[-1]["execution_mode"]
        def shift(self, nodes):
            events.append("shift")
            return super().shift(nodes)
    def execute(state, action, info):
        events.append("execute")
        assert info is selected_infos[-1]
        return state + 10 + info["execution_mode"]
    result = RecedingHorizonController(
        Planner(), lambda *_: pytest.fail("ordinary transition executed"),
        n_steps=2, n_diffuse_init=1, n_diffuse=1, split_rng=_counter_split,
        execution_step=execute,
    ).run(np.zeros(4, np.float32), 0)
    assert events == ["replan", "first", "execute", "after", "shift"] * 2
    assert all(a is b for a, b in zip(result.infos, selected_infos))


def test_execution_step_error_never_falls_back_or_commits():
    class Planner(MockPlanner):
        def after_step(self, *_):
            pytest.fail("failed execution must not commit")
        def shift(self, *_):
            pytest.fail("failed execution must not shift")
    def execute(*_):
        raise RuntimeError("rejected context")
    with pytest.raises(RuntimeError, match="rejected context"):
        RecedingHorizonController(
            Planner(), lambda *_: pytest.fail("implicit ordinary fallback"),
            n_steps=1, n_diffuse_init=1, n_diffuse=1,
            execution_step=execute, split_rng=_counter_split,
        ).run(np.zeros(4, np.float32), 0)


def test_execution_step_preserves_fixed_length_success_padding():
    executed = []
    def execute(state, action, info):
        executed.append((int(state[0]), info))
        return np.array([min(int(state[0]) + 1, 2)], np.float32)
    result = RecedingHorizonController(
        MockPlanner(du=1), lambda *_: pytest.fail("ordinary step"),
        n_steps=5, n_diffuse_init=1, n_diffuse=1,
        execution_step=execute, split_rng=_counter_split,
    ).run(np.array([0], np.float32), 0)
    np.testing.assert_array_equal(np.array(result.states).ravel(), [0, 1, 2, 2, 2, 2])
    assert len(executed) == 5 and result.horizon == 5



@pytest.mark.parametrize(
    "reject_after,collect_states,diagnostics",
    [(0, True, True), (2, True, True), (2, False, True), (2, True, False)],
)
def test_execution_rejection_carries_only_committed_prefix(
    reject_after, collect_states, diagnostics
):
    from genedynamics.core.types import ExecutionRejected
    events = []
    class Planner(MockPlanner):
        def after_step(self, *_):
            events.append("after")
        def shift(self, nodes):
            events.append("shift")
            return super().shift(nodes)
    class DiagnosticPlanner(Planner):
        def replan_with_info(self, state, warm_start, schedule, rng, t0=0):
            return self.replan(state, warm_start, schedule, rng, t0=t0), {
                "execution_mode": 1, "real_step": t0,
            }
    failure = ExecutionRejected("invalid_geometry", {"geometry_valid": False})
    committed = []
    def execute(state, action, info):
        if len(committed) == reject_after:
            raise failure
        next_state = _double_integrator_step(state, action)
        committed.append(next_state)
        return next_state
    initial = np.zeros(4, np.float32)
    controller = RecedingHorizonController(
        DiagnosticPlanner() if diagnostics else Planner(),
        lambda *_: pytest.fail("rejection must not execute an ordinary fallback"),
        n_steps=4, n_diffuse_init=1, n_diffuse=1,
        execution_step=execute, split_rng=_counter_split,
        collect_states=collect_states, collect_plan_vars=True,
    )
    with pytest.raises(ExecutionRejected) as caught:
        controller.run(initial, 0)
    assert caught.value is failure
    prefix = failure.partial_result
    assert len(prefix.actions) == len(prefix.infos) == reject_after
    assert len(prefix.plan_vars) == reject_after
    assert events == ["after", "shift"] * reject_after
    np.testing.assert_array_equal(prefix.states[0], initial)
    if collect_states:
        assert len(prefix.states) == reject_after + 1
        for actual, expected in zip(prefix.states[1:], committed):
            np.testing.assert_array_equal(actual, expected)
        assert failure.partial_states_complete is True
    else:
        assert len(prefix.states) == 2
        np.testing.assert_array_equal(prefix.states[-1], committed[-1])
        assert failure.partial_states_complete is False
    assert failure.rejected_step == reject_after and failure.requested_steps == 4
    assert np.asarray(failure.rejected_action).shape == (2,)
    if diagnostics:
        assert [info["real_step"] for info in prefix.infos] == list(range(reject_after))
        assert failure.rejected_info["real_step"] == reject_after
    else:
        assert prefix.infos == [None] * reject_after
        assert failure.rejected_info is None

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
