"""Contact support, fresh source carry and movable-body reset contracts."""
import numpy as np
import pytest


def test_source_query_contact_phase_and_controller_isolation():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import ContactConfig
    from genedynamics.envs.domains.manipulation.transfer_workcell.queries.contact import ContactSourceQuery
    from genedynamics.solvers.single.cross_ebdm.learning.contact_data import observe_contact
    from genedynamics.tasks.transfer import RandomRequest
    config=ContactConfig(robot="xarm7"); runner=ContactRunner(config=config)
    request=RandomRequest(4,(41,42)); state=runner.reset_batch(request)
    semantic=observe_contact(state,runner,np); query=ContactSourceQuery(config)
    result=query.build(semantic,request)
    assert result.support.all()
    np.testing.assert_allclose(result.state.reference_position,semantic.tcp_position,atol=1e-4)
    assert not result.state.last_command.any() and not result.state.command_buffer.any()
    # Force labels alone cannot manufacture a source contact in free space.
    unsupported=query.build(semantic._replace(phase=np.ones(2),normal_force=np.full(2,5.)),request,previous=result.state)
    assert not unsupported.support.any() and (unsupported.state.status==7).all()
    malformed=query.build(semantic._replace(tcp_rotation=np.zeros((2,3,3))),request)
    assert not malformed.support.any()
    assert np.isfinite(malformed.state.reference_position).all()
    for d in malformed.state.physics: assert np.isfinite(d.qpos).all()


def test_contact_query_jax_parity_and_warm_replay():
    import jax
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import ContactConfig
    from genedynamics.envs.domains.manipulation.transfer_workcell.queries.contact import ContactSourceQuery
    from genedynamics.solvers.single.cross_ebdm.learning.contact_data import observe_contact
    from genedynamics.tasks.transfer import RandomRequest
    config=ContactConfig(robot="ur5e"); runner=ContactRunner(config=config)
    request=RandomRequest(6,(61,)); semantic=observe_contact(runner.reset_batch(request),runner,np)
    reference=ContactSourceQuery(config).build(semantic,request)
    query=ContactSourceQuery(config,backend="jax"); result=query.build(semantic,request); query.synchronize(result)
    np.testing.assert_array_equal(result.support,reference.support)
    np.testing.assert_allclose(result.state.reference_position,reference.state.reference_position,atol=1e-4)
    again=query.build(semantic,request,previous=result.state); query.synchronize(again)
    assert np.asarray(again.support).all()
    np.testing.assert_allclose(result.state.reference_position,again.state.reference_position,atol=1e-4)


def test_mobile_body_actuator_mapping_and_partial_reset():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import ContactConfig, ContactCommands
    from genedynamics.tasks.transfer import RandomRequest
    runner=ContactRunner(config=ContactConfig(environment="mobile_keyed_insertion_recovery",robot="xarm7",mobile=True))
    request=RandomRequest(8,(81,82)); state=runner.reset_batch(request)
    skill=np.zeros((2,100,13),np.float32); body=np.zeros((2,100,3),np.float32); body[...,0]=.02
    result=runner.collect_chunk(state,ContactCommands(skill,body))
    assert np.max(result.state.body_reference[:,0])>0
    assert any(abs(d.qpos[0])>1e-5 for d in result.state.physics)
    reset=runner.reset_batch(request,previous=result.state,mask=np.array([True,False]))
    np.testing.assert_array_equal(reset.physics[1].qpos,result.state.physics[1].qpos)
    np.testing.assert_array_equal(reset.command_buffer[1],result.state.command_buffer[1])
    np.testing.assert_array_equal(reset.body_reference[1],result.state.body_reference[1])
    assert not reset.body_reference[0].any() and not reset.command_buffer[0].any()
