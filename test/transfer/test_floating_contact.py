"""Real Go2+Z1 stance dynamics, explicit QP carry and failure bookkeeping."""
import numpy as np
import pytest


def test_joint_viability_bounds_keep_preview_inside_and_brake_outward_motion():
    from genedynamics.envs.domains.manipulation.transfer_workcell.backends.floating_contact import joint_acceleration_bounds
    q=np.array([.95,-.95,.0]);v=np.array([.5,-.5,.0]);limits=np.tile([-1.,1.],(3,1))
    lower,upper=joint_acceleration_bounds(q,v,limits)
    assert upper[0]<0 and lower[1]>0
    acceleration=np.clip(np.zeros(3),lower,upper)
    predicted=q+.12*v+.5*.12**2*acceleration
    assert (predicted<=.98+1e-12).all() and (predicted>=-.98-1e-12).all()
    next_velocity=v+.12*acceleration
    distance=np.minimum(q+.98,.98-q)
    assert (np.abs(next_velocity)<=np.sqrt(8*distance)+1e-12).all()
    # Away from limits zero acceleration is feasible; violating positions
    # require inward acceleration rather than silently enlarged tolerances.
    assert lower[2]<0<upper[2]
    lower,upper=joint_acceleration_bounds(np.array([1.01]),np.array([0.]),limits[:1])
    assert upper[0]<0


def test_joint_reference_governor_does_not_mutate_measured_pose():
    from genedynamics.envs.domains.manipulation.transfer_workcell.backends.floating_contact import FloatingContact
    from types import SimpleNamespace
    from genedynamics.core.control.transfer import orientation_error,integrate_rotation
    provider=object.__new__(FloatingContact);provider.sid=0
    measured=np.array([[.4,-.2,.6]]);rotation=np.eye(3)
    data=SimpleNamespace(site_xpos=measured.copy(),site_xmat=rotation.reshape(1,9).copy())
    request=measured[0]+[.1,.1,.1]
    target_rotation=integrate_rotation(np,rotation,np.array([0.,0.,.3]),1.)
    reference,rot=provider._govern_reference(data,request,target_rotation)
    np.testing.assert_array_equal(data.site_xpos,measured)
    np.testing.assert_array_equal(data.site_xmat,rotation.reshape(1,9))
    assert np.isclose(np.linalg.norm(reference-measured[0]),.02)
    assert np.isclose(np.linalg.norm(orientation_error(np,rotation,rot)),np.sin(.1))


def test_t4_isolation_profiles_separate_bounds_from_hidden_reference_governor():
    from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
    from genedynamics.tasks.transfer.specs import task_spec

    providers = {}
    for profile in ("legacy", "joint_margin", "joint_margin_bounds",
                    "joint_margin_public_reference"):
        spec = task_spec("T4", config_overrides={"execution_profile": profile})
        providers[profile] = CpuTaskAdapter(spec)._provider
    assert not providers["legacy"].joint_margin_execution
    assert providers["joint_margin"].joint_margin_execution
    assert providers["joint_margin"].hidden_reference_governor
    assert providers["joint_margin_bounds"].joint_margin_execution
    assert not providers["joint_margin_bounds"].hidden_reference_governor
    assert providers["joint_margin_public_reference"].public_reference_governor
    assert not providers["joint_margin_public_reference"].hidden_reference_governor


def test_t4_post_preparation_receipt_clones_exact_state_only_across_matching_physics():
    from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
    from genedynamics.tasks.transfer.specs import mechanism_spec, task_spec

    overrides = {"reset_joint_noise": .002, "reset_profile": "folded_stance"}
    mechanism = mechanism_spec("T4", dict(
        servo_gain=.7, delay_steps=3, payload_mass=.5, body_servo_gain=.9,
    ))
    source = CpuTaskAdapter(
        task_spec("T4", config_overrides={**overrides, "execution_profile": "legacy"}),
        mechanism_spec=mechanism, seed=931000,
    )
    source.reset((931000,), seed=931000, modes=(0,))
    receipt = source.get_checkpoint()
    target = CpuTaskAdapter(
        task_spec("T4", config_overrides={
            **overrides, "execution_profile": "joint_margin_bounds",
        }), mechanism_spec=mechanism, seed=931000,
    )
    target.restore_runtime_profile_checkpoint(receipt)
    cloned = target.get_checkpoint()
    np.testing.assert_array_equal(cloned["integration"], receipt["integration"])
    for key in source.observe():
        np.testing.assert_array_equal(target.observe()[key], source.observe()[key])
    assert target._contract_hash() != source._contract_hash()
    assert target._runtime_profile_neutral_hash() == source._runtime_profile_neutral_hash()

    different = CpuTaskAdapter(
        task_spec("T4", config_overrides={
            **overrides, "execution_profile": "joint_margin_bounds",
        }), mechanism_spec=mechanism_spec("T4", dict(
            servo_gain=.71, delay_steps=3, payload_mass=.5, body_servo_gain=.9,
        )), seed=931000,
    )
    with pytest.raises(ValueError, match="physical/reset contract differs"):
        different.restore_runtime_profile_checkpoint(receipt)


def test_cpu_adapter_records_public_planar_body_intent():
    from genedynamics.envs.adapters.transfer_task import CpuTaskAdapter
    from genedynamics.tasks.transfer.specs import task_spec

    task = CpuTaskAdapter(task_spec("T4"), seed=81)
    task.reset((81,), seed=81, modes=(0,))
    initial = np.asarray(task.state.body_reference[0]).copy()
    step = task.step_contact(
        np.zeros((1, 13), np.float32),
        body_twist=np.array([[.005, .005, 0.]], np.float32),
    )
    assert np.all(np.asarray(task.state.body_reference[0, :2]) > initial[:2])
    np.testing.assert_allclose(step.executed_action["body_twist"], [[.005, .005, 0.]])
    request = task.records[-1].requested_actions[-1]
    np.testing.assert_array_equal(request["skill"], np.zeros(13))
    np.testing.assert_allclose(request["body_twist"], [.005, .005, 0.])


def test_public_planar_body_intent_is_finite_and_time_bounded():
    from genedynamics.tasks.transfer.controllers.mating_feedback import (
        PublicPlanarBodyIntentSpec,
        PublicPlanarBodyMatingFeedbackPolicy,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    spec = task_spec("T4")
    actor = PublicPlanarBodyMatingFeedbackPolicy(
        spec, body_intent=PublicPlanarBodyIntentSpec(
            velocity_world=(.005, .005, 0.), start_seconds=1., duration_seconds=2.,
        ), modes=(0,), seed=0,
    )
    public = {"elapsed_seconds": np.array([[.5]], np.float32)}
    np.testing.assert_array_equal(actor.body_twist(public), [[0., 0., 0.]])
    public["elapsed_seconds"][:] = 1.
    np.testing.assert_allclose(actor.body_twist(public), [[.005, .005, 0.]])
    public["elapsed_seconds"][:] = 3.
    np.testing.assert_array_equal(actor.body_twist(public), [[0., 0., 0.]])
    with pytest.raises(ValueError, match="velocity support"):
        PublicPlanarBodyIntentSpec(velocity_world=(.02, 0., 0.))


def test_floating_stance_wbc_carry_and_partial_reset():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import ContactConfig, ContactCommands
    from genedynamics.solvers.single.cross_ebdm.learning.contact_data import contact_carry
    from genedynamics.tasks.transfer import RandomRequest
    r=ContactRunner(config=ContactConfig(environment='mobile_keyed_insertion_recovery',robot='go2_z1',mobile=True))
    request=RandomRequest(6,(60,61)); state=r.reset_batch(request)
    assert r._impl.model.nq==25 and r._impl.model.nv==24
    command=np.zeros((2,30,13),np.float32); body=np.zeros((2,30,3),np.float32);body[...,0]=.005
    result=r.collect_chunk(state,ContactCommands(command,body))
    assert np.asarray(result.state.status).tolist()==[0,0]
    assert (result.trace.wbc_residual<.01).all()
    assert (result.trace.support_count[:,-1]>=3).all()
    assert (result.trace.base_position[:,-1,2]>.2).all()
    carry=contact_carry(result.state,r);assert carry.shape==(2,263)
    assert np.linalg.norm(carry[0,210:262])>0  # explicit QP warm start, not fresh zeros
    reset=r.reset_batch(request,previous=result.state,mask=np.array([True,False]))
    np.testing.assert_array_equal(reset.wbc_warm[1],result.state.wbc_warm[1])
    np.testing.assert_array_equal(reset.foot_anchors[1],result.state.foot_anchors[1])
    assert not reset.wbc_warm[0].any()
    failed=r.collect_chunk(result.state,np.full((2,2,13),np.nan,np.float32))
    assert (failed.state.status==6).all() and failed.trace.terminal.sum()==2
    assert not failed.trace.valid[:,1].any()
    # A failed optimization is a physical/control failure, never a silent zero
    # command rollout counted as success; subsequent padding freezes the body.
    r._impl._qp=lambda data,state,ref,rot,command,body,warm:(np.zeros(r._impl.model.nu),warm,1.)
    qp_failed=r.collect_chunk(result.state,np.zeros((2,2,13),np.float32))
    assert (qp_failed.state.status==11).all() and qp_failed.trace.terminal.sum()==2
    assert not qp_failed.trace.valid[:,1].any()
