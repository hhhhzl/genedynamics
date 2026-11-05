"""Physical and public-contract checks for the opt-in passive T3 extension."""
from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from genedynamics.envs.adapters.transfer_task import instantiate_task, FLAT_FIELDS
from genedynamics.tasks.transfer.compliant_socket import (
    CompliantSocketMechanismSpec, ENVIRONMENT_NAME, FIXTURE_FIELDS,
    FIXTURE_REST_POSITION, FixtureRelativePolicy, compliant_socket_spec,
    fixture_relative_observation,
)
from genedynamics.tasks.transfer.episode import (
    EpisodeRecord, episode_evidence, audit_episode_evidence,
)
from genedynamics.tasks.transfer.evaluation import evaluate_records
from genedynamics.tasks.transfer.compliant_socket_qualification import _public_contact_probe


def test_public_fixture_probe_triggers_once_and_uses_only_measured_wrench():
    spec = {"version": "public-contact-x-pulse-v1", "wrench_trigger_N": 1.,
            "x_delta_mps": .004, "positive_controls": 2, "negative_controls": 2}
    state = {"trigger_tick": None, "applied_controls": 0}
    action = np.zeros((1, 13), dtype=np.float32)
    public = {"measured_wrench": np.zeros((1, 6), dtype=np.float32)}
    np.testing.assert_array_equal(_public_contact_probe(action, public, 0, state, spec), action)
    public["measured_wrench"][0, 2] = 1.2
    issued = [_public_contact_probe(action, public, tick, state, spec)[0, 0] for tick in range(1, 6)]
    np.testing.assert_allclose(issued, [.004, .004, -.004, -.004, 0.])
    assert state == {"trigger_tick": 1, "applied_controls": 4}
    np.testing.assert_array_equal(action, np.zeros((1, 13), dtype=np.float32))


def test_dynamic_observation_keeps_fixture_out_of_robot_encoders():
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7",
        config_overrides={"fixture_initial_offset_m": .004, "reset_task_noise": 0.})
    public = task.reset((11,), seed=10, initial="clear")
    assert task._model.nq == 8
    assert public["joint_position"].shape == public["joint_velocity"].shape == (1, 7)
    assert set(public) == set(FLAT_FIELDS) | set(FIXTURE_FIELDS)
    np.testing.assert_array_equal(public["joint_position"][0],
        task.state.physics[0].qpos[task._provider.robot_qpos_indices].astype(np.float32))
    np.testing.assert_allclose(public["fixture_position"], [[.444, -.2, 0.]], atol=1e-7)
    np.testing.assert_allclose(public["goal_position"], [[.444, -.2, .58]], atol=1e-7)
    assert task.flat_observation().shape == (1, task.observation_size)
    assert task.public_state_fields()[-1]["stop"] == task.observation_size
    assert task.describe()["contract_version"] == "compliant-keyed-insertion-public-v1"


def test_fixture_has_real_passive_spring_and_damping_without_an_actuator():
    mechanism = CompliantSocketMechanismSpec(fixture_stiffness_Npm=700., fixture_damping_Nspm=11.)
    task = instantiate_task(ENVIRONMENT_NAME, robot="panda", mechanism_spec=mechanism,
        config_overrides={"reset_task_noise": 0.})
    task.reset((5,), seed=5, initial="clear")
    provider, data = task._provider, task.state.physics[0]
    data.qpos[provider.fixture_qpos_index] = .003
    data.qvel[provider.fixture_dof_index] = .02
    mujoco.mj_forward(task._model, data)
    assert provider.fixture_dof_index not in provider.actuator_dofs
    np.testing.assert_allclose(data.qfrc_passive[provider.fixture_dof_index], -700.*.003 - 11.*.02)
    np.testing.assert_allclose(task.observe()["fixture_twist"], [[.02, 0., 0., 0., 0., 0.]])
    np.testing.assert_array_equal(task.observe()["joint_velocity"], np.zeros((1, 7)))
    assert task._model.jnt_limited[provider.fixture_joint_id]
    np.testing.assert_allclose(task._model.jnt_range[provider.fixture_joint_id], [-.012, .012])


def test_shared_task_frame_converts_current_pose_and_velocity_not_historical_actions():
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7")
    public = task.reset((3,), seed=3, initial="clear")
    shifted = {key: value.copy() for key, value in public.items()}
    for key in ("tcp_position", "reference_position", "goal_position", "fixture_position"):
        shifted[key][:, 0] += .004
    shifted["fixture_twist"][:, 0] = .015
    shifted["tcp_twist"][:, 0] += .015
    shifted["previous_action"][:, 0] = .011
    relative = fixture_relative_observation(shifted)
    np.testing.assert_allclose(relative["tcp_position"], public["tcp_position"], atol=1e-7)
    np.testing.assert_allclose(relative["tcp_twist"], public["tcp_twist"], atol=1e-7)
    np.testing.assert_array_equal(relative["previous_action"], shifted["previous_action"])
    class Actor:
        def act(self, observation):
            assert set(observation) == set(FLAT_FIELDS)
            return np.zeros((1, 13), dtype=np.float32)
    action = FixtureRelativePolicy(Actor()).act(shifted)
    np.testing.assert_allclose(action[0, :3], [.015, 0., 0.])


def test_independent_seating_evaluator_uses_current_fixture_not_world_rest():
    spec = compliant_socket_spec(robot="xarm7")
    rows = []
    for index in range(15):
        rows.append(dict(valid=True, tcp_position=np.asarray(spec.config.goal) + [.004, 0., 0.],
            tcp_rotation=np.diag([1., -1., -1.]), fixture_position=FIXTURE_REST_POSITION + [.004, 0., 0.],
            fixture_rotation=np.eye(3), fixture_twist=np.zeros(6), normal_force=5., force_peak=5.,
            lateral_force_peak=0., loading=True, axial_support=True, forbidden_collision=False,
            finite=True, joint_limit=False, joint_speed_peak=0., backend_status=1 if index == 14 else 0))
    record = EpisodeRecord(1, 1, 0, "clear", traces=rows, requested_mode=0, state="ended")
    result = evaluate_records([record], spec)
    assert result["episodes"][0]["physical_success"]
    assert result["episodes"][0]["final_position_error_m"] < 1e-12
    assert result["status_mismatches"] == 0
    # A stationary-world evaluator would be wrong by 4 mm, above the 2 mm band.
    static = evaluate_records([record], replace(spec, name="keyed_insertion_recovery"))
    assert not static["episodes"][0]["physical_success"]


def test_dynamic_cpu_evidence_roundtrip_and_exact_charged_replay(tmp_path):
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7")
    task.reset((8,), seed=8, initial="clear")
    for _ in range(4):
        task.step(np.zeros((1, 13), dtype=np.float32))
    evidence = tmp_path / "evidence.json.gz"
    evidence.write_bytes(gzip.compress(json.dumps(episode_evidence(task.records, task.spec, task.evaluate())).encode()))
    assert audit_episode_evidence(evidence)["passed"]
    replay = task.replay_episode(task.records[0])
    assert replay["trace_exact"] and replay["matches_status"]
    assert replay["accounting"]["control_steps"] == 4


def test_static_t3_default_sensor_contract_and_fixture_are_unchanged():
    task = instantiate_task("T3", robot="xarm7")
    public = task.reset((3,), seed=3, initial="clear")
    assert set(public) == set(FLAT_FIELDS)
    assert task._model.nq == 7
    assert "compliant_socket" not in [task._model.body(i).name for i in range(task._model.nbody)]
    assert task.spec.describe()["contract_version"] == "cross-embodiment-public-cpu-v2"


def test_dynamic_socket_rejects_unqualified_gpu_provider():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    with pytest.raises(NotImplementedError, match="CPU-only"):
        ContactRunner(config=compliant_socket_spec().config, backend="jax", device="cpu",
                      mechanism=CompliantSocketMechanismSpec())


def test_normal_actor_binding_requires_dynamic_interface_and_rejects_old_weights():
    from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7")
    config = dict(actor_binding=dict(version="transfer-actor-binding-v1", retreat_release_version="safe_settled_v1",
                                    feedback={"compensation_gain": 2.0}),
                  task_actor_interface=FixtureRelativePolicy.VERSION)
    actor = create_transfer_actor(task, "public_geometry", config)
    assert isinstance(actor, FixtureRelativePolicy)
    public = task.reset((4,), seed=4)
    assert np.isfinite(actor.act(public)).all()
    with pytest.raises(ValueError, match="fixture-observation checkpoint"):
        create_transfer_actor(task, "cross_ebdm", config)
    with pytest.raises(ValueError, match="explicit shared task-frame"):
        create_transfer_actor(task, "public_geometry", {**config, "task_actor_interface": None})


@pytest.mark.parametrize(("travel", "offset"), [(.002, .001), (.003, -.0015)])
@pytest.mark.parametrize("initial", ["clear", "jammed"])
def test_robot_only_reset_ik_preserves_fixture_coordinate_and_public_pose(travel, offset, initial):
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7", config_overrides={
        "fixture_travel_m": travel, "fixture_initial_offset_m": offset,
        "reset_task_noise": .0005,
    })
    public = task.reset((19,), seed=21, initial=initial)
    fixture_index = task._provider.fixture_qpos_index
    assert task.state.physics[0].qpos[fixture_index] == offset
    assert task.records[0].reset_evidence["physical_state"]["qpos"][fixture_index] == offset
    np.testing.assert_allclose(public["fixture_position"][0],
                               FIXTURE_REST_POSITION + [offset, 0., 0.], atol=2e-8)


def test_six_robot_joint_ik_keeps_twelve_restarts_with_passive_scene_joint(monkeypatch):
    from genedynamics.envs.domains.manipulation.transfer_workcell import workcell
    links = "".join(f'<body pos="0 0 .04"><joint name="robot_{i}" type="hinge" axis="0 1 0" range="-2 2"/><geom type="sphere" size=".01" mass=".1"/>' for i in range(6))
    xml = '<mujoco><worldbody>' + links + '<site name="tcp"/>' + '</body>' * 6
    xml += '<body><joint name="fixture" type="slide" axis="1 0 0" range="-.002 .002"/><geom type="sphere" size=".01" mass=".1"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    robot_ids = np.asarray([model.joint(f"robot_{i}").id for i in range(6)])
    fixture_q = model.jnt_qposadr[model.joint("fixture").id]
    q0 = model.qpos0.copy(); q0[fixture_q] = .001
    real_rng = np.random.default_rng(1701); sizes = []; passive_positions = []
    class TrackedRng:
        def uniform(self, low, high, size):
            sizes.append(size)
            return real_rng.uniform(low, high, size)
    monkeypatch.setattr(workcell.np.random, "default_rng", lambda seed: TrackedRng())
    forward = workcell.mujoco.mj_forward
    def tracked_forward(model, data):
        passive_positions.append(float(data.qpos[fixture_q]))
        return forward(model, data)
    monkeypatch.setattr(workcell.mujoco, "mj_forward", tracked_forward)
    with pytest.raises(ValueError, match="pose unreachable"):
        workcell.solve_ik(model, model.site("tcp").id, q0, np.asarray([100., 0., 0.]), np.eye(3), joint_ids=robot_ids)
    assert model.nq == 7 and sizes == [6] * 11
    assert len(passive_positions) == 350 * 12
    assert np.array_equal(passive_positions, np.full(350 * 12, .001))


@pytest.mark.parametrize("damage", ["nonfinite_position", "off_axis", "rotation", "angular_velocity"])
def test_independent_evaluator_rejects_undeclared_fixture_sensor_values(damage):
    task = instantiate_task(ENVIRONMENT_NAME, robot="xarm7")
    task.reset((3,), seed=3, initial="clear")
    task.step(np.zeros((1, 13), dtype=np.float32))
    record = task.records[0]; row = record.traces[0]
    if damage == "nonfinite_position":
        row["fixture_position"][0] = np.nan
    elif damage == "off_axis":
        row["fixture_position"][1] += .01
    elif damage == "rotation":
        row["fixture_rotation"] = np.diag([-1., -1., 1.])
    else:
        row["fixture_twist"][5] = .1
    with pytest.raises(ValueError, match="passive_x_v1"):
        evaluate_records([record], task.spec)
