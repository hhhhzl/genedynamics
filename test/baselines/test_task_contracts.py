"""Real scene execution, privacy, histories and independent task accounting."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from genedynamics.experiments.baselines import instantiate_task, task_spec
from genedynamics.experiments.baselines.evaluation import evaluate_records
from genedynamics.learning.cross_embodiment.contact import ContactMechanismSpec


def test_four_tasks_describe_actual_profiles_without_claiming_native_qualification():
    specs = [task_spec(name) for name in ("T1", "T2", "T3", "T4")]
    assert specs[0].config.timeout_seconds == 10.
    assert specs[1].config.guide_profile == "spatial_dual_channel"
    assert specs[2].config.socket_profile == "matched_keyway"
    assert specs[3].config.robot == "go2_z1" and specs[3].config.reset_task_noise == 0
    assert "body qualification pending" in specs[3].body_scope
    assert "planar mobile proxy" in task_spec("T4", robot="xarm7").body_scope
    for spec in specs:
        assert len(spec.skill_units) == 13
        assert "physics" in spec.describe()["excluded_policy_fields"]
    with pytest.raises(ValueError, match="identity/profile"):
        task_spec("T3", config_overrides={"wrench_semantics": "legacy_aggregate"})


def test_public_sensors_own_arrays_and_never_reveal_hidden_delay_or_model():
    task = instantiate_task("T3", robot="panda", mechanism_spec=ContactMechanismSpec(delay_steps=2))
    observation = task.reset((10,), seed=17)
    before = task.state.physics[0].qpos.copy()
    assert not set(observation) & {"physics", "mechanism", "command_buffer", "delay_steps", "friction", "axial_support"}
    observation["joint_position"][:] = 999
    np.testing.assert_array_equal(task.state.physics[0].qpos, before)
    action = np.zeros((1, 13), dtype=np.float32); action[:, 0] = .01
    old_reference = task.observe()["reference_position"].copy()
    result = task.step(action)
    # Physical delay is hidden. Publicly issued commands are known and need
    # not equal the plant's latent, delayed command retained in the evaluator.
    assert result.executed_action[0, 0] == 0
    assert result.observation["previous_action"][0, 0] > 0
    np.testing.assert_allclose(task.fada_observation()["history_act"][0, -1], result.observation["previous_action"][0])
    assert result.observation["reference_position"][0, 0] > old_reference[0, 0]


def test_reorder_partial_reset_histories_and_full_denominator():
    task = instantiate_task("T3", robot="panda")
    task.reset((101, 202), seed=18)
    actions = np.zeros((2, 13)); actions[:, 0] = [.005, -.005]
    task.step(actions)
    before = task.observe()
    history = task.fada_observation()
    task.reorder([1, 0])
    np.testing.assert_array_equal(task.observe()["joint_position"], before["joint_position"][[1, 0]])
    np.testing.assert_array_equal(task.fada_observation()["history_act"], history["history_act"][[1, 0]])
    unchanged = task.observe()["joint_position"][1].copy()
    task.reset((202, 101), seed=19, mask=[True, False])
    np.testing.assert_array_equal(task.observe()["joint_position"][1], unchanged)
    np.testing.assert_array_equal(task.fada_observation()["history_valid_mask"].sum(axis=1), [1, 2])
    task.reject("source unavailable", mask=[True, False])
    report = task.evaluate()
    assert report["requested_episodes"] == 3
    assert report["outcomes"] == {"incomplete": 2, "rejected": 1}
    assert report["mean_task_cost"] == 1
    assert task.accounting["reset_worlds"] == 3
    with pytest.raises(ValueError, match="slot identities"):
        task.reset((101, 202), mask=[True, False])


def test_full_checkpoint_restores_physics_history_delay_and_ledger_without_reset():
    task = instantiate_task("T3", robot="panda", mechanism_spec=ContactMechanismSpec(servo_gain=.8, delay_steps=2))
    task.reset((303, 404), seed=20)
    action = np.zeros((2, 13)); action[:, 0] = [.004, -.004]
    task.step(action)
    payload = task.get_checkpoint()
    first = task.step(action)
    first_history = task.fada_observation()
    first_report = task.evaluate()
    task.restore_checkpoint(payload)
    second = task.step(action)
    for name in first.observation:
        np.testing.assert_array_equal(first.observation[name], second.observation[name])
    for name in first_history:
        np.testing.assert_array_equal(first_history[name], task.fada_observation()[name])
    second_report = task.evaluate()
    assert second_report["episodes"] == first_report["episodes"]
    assert second_report["accounting"] == first_report["accounting"]
    assert second_report["lifetime_accounting"]["control_steps"] == first_report["lifetime_accounting"]["control_steps"] + 2
    assert task.accounting["reset_calls"] == 1


@pytest.mark.parametrize("name", ["T1", "T2", "T3"])
def test_each_fixed_task_executes_actual_dynamics_to_a_terminal_timeout_and_cpu_replay(name):
    task = instantiate_task(name, robot="panda", config_overrides={"timeout_seconds": .08})
    task.reset((501,), seed=21)
    old_time = task.state.physics[0].time
    while task.state.status[0] == 0:
        task.step(np.zeros(13))
    assert task.state.physics[0].time > old_time
    report = task.evaluate()
    assert report["requested_episodes"] == 1 and report["status_mismatches"] == 0
    assert report["episodes"][0]["outcome"] == "timeout"
    assert report["episodes"][0]["physical_control_steps"] == 4
    assert task.accounting["physics_steps"] == 40
    replay = task.replay_episode(task.records[0])
    assert replay["matches_status"] and replay["max_pose_error"] == 0. and replay["trace_exact"]
    assert replay["accounting"]["control_steps"] == 4
    corrupted = task.records
    corrupted[0].traces[-1]["backend_status"] = 1
    independent = evaluate_records(corrupted, task.spec)
    assert independent["successes"] == 0 and independent["status_mismatches"] == 1


def test_mobile_proxy_scope_is_explicit_and_steps_actual_base_dynamics():
    task = instantiate_task("T4", robot="xarm7", config_overrides={"timeout_seconds": .08})
    task.reset((601,), seed=22)
    task.step(np.zeros(13))
    assert "planar mobile proxy" in task.spec.body_scope
    assert task.state.physics[0].time == pytest.approx(.02)
    assert task.observe()["body_position"].shape == (1, 3)
    assert task.evaluate()["status_mismatches"] == 0


def _acp_command(task):
    observation = task.observe()
    pose = np.tile(np.eye(4), (task.num_envs, 1, 1))
    pose[:, :3, :3] = observation["tcp_rotation"]
    pose[:, :3, 3] = observation["tcp_position"]
    virtual = pose.copy(); virtual[:, 2, 3] -= .0005
    basis = np.array([[1., 1., 0.], [-1., 1., 0.], [0., 0., 1.]]) / np.sqrt(2)
    basis[2, 2] = 1.
    stiffness = np.tile(np.diag([800., 800., 200., 60., 60., 60.]), (task.num_envs, 1, 1))
    stiffness[:, :3, :3] = basis @ np.diag([200., 1200., 3000.]) @ basis.T
    return dict(reference_pose_world=pose, virtual_target_pose_world=virtual, stiffness_tool=stiffness)


def test_acp_preserves_structured_targets_anisotropic_matrix_delay_and_checkpoint():
    task = instantiate_task("T3", robot="panda", mechanism_spec=ContactMechanismSpec(delay_steps=1))
    task.reset((701,), seed=23)
    command = _acp_command(task)
    step = task.step_acp(command)
    assert isinstance(step.executed_action, dict)
    assert step.executed_action["stiffness_tool"].shape == (1, 6, 6)
    assert task.state.physics[0].time == pytest.approx(.02)
    payload = task.get_checkpoint()
    first = task.step_acp(command)
    np.testing.assert_array_equal(first.executed_action["stiffness_tool"], command["stiffness_tool"])
    task.restore_checkpoint(payload)
    second = task.step_acp(command)
    np.testing.assert_array_equal(first.observation["tcp_position"], second.observation["tcp_position"])
    assert task.evaluate()["status_mismatches"] == 0
    replay = task.replay_episode(task.records[0])
    assert replay["matches_status"] and replay["max_pose_error"] == 0 and replay["trace_exact"]
    with pytest.raises(ValueError, match="family"):
        task.step(np.zeros(13))


def test_acp_prefix_uses_real_control_clock_and_rejects_invalid_stiffness_before_physics():
    task = instantiate_task("T3", robot="panda")
    task.reset((801,), seed=24)
    command = _acp_command(task)
    bad = deepcopy(command); bad["stiffness_tool"][0, 0, 0] = -1000
    with pytest.raises(ValueError, match="positive definite"):
        task.step_acp(bad)
    assert task.accounting["control_steps"] == 0
    prediction = {
        "reference_pose_world": np.repeat(command["reference_pose_world"][:, None, None], 2, axis=1),
        "virtual_target_pose_world": np.repeat(command["virtual_target_pose_world"][:, None, None], 2, axis=1),
        "stiffness_tool": np.repeat(command["stiffness_tool"][:, None, None], 2, axis=1),
        "waypoint_times_s": np.array([0., .04]), "execution_horizon": 2,
    }
    result = task.execute_acp_chunk(prediction)
    assert result["executed_control_steps"] == 4
    assert task.accounting["control_steps"] == 4
    assert task.state.physics[0].time == pytest.approx(.08)


@pytest.mark.slow
def test_floating_go2_actual_stance_reset_preparation_is_charged():
    pytest.importorskip("osqp")
    task = instantiate_task("T4")
    task.reset((901,), seed=25)
    assert task.accounting["preparation_control_steps"] == 50
    task.step(np.zeros(13))
    assert task.observe()["body_twist"].shape == (1, 6)
    assert task.evaluate()["status_mismatches"] == 0
    assert task.records[0].preparation_control_steps == 50
