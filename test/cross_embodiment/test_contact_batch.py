"""Same-topology contact mechanics are full numeric models, not mass-only edits."""
import numpy as np


def test_contact_heterogeneous_microbatch_and_guard():
    import jax
    import jax.numpy as jp
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactConfig
    from genedynamics.learning.cross_embodiment import RandomRequest,MechanismSpec
    mechanisms=(MechanismSpec(1.,0,.5),MechanismSpec(.8,2,.8))
    config=ContactConfig()
    reference=ContactRunner(config=config,mechanism=mechanisms)
    compiled=ContactRunner(config=config,mechanism=mechanisms,backend='jax',microbatch_size=1)
    request=RandomRequest(5,(51,52)); a=reference.reset_batch(request);b=compiled.reset_batch(request)
    commands=np.zeros((2,2,13),np.float32);commands[...,:3]=[.002,0.,-.002];commands[...,12]=5.
    x=reference.collect_chunk(a,commands);y=compiled.collect_chunk(b,commands);compiled.synchronize(y)
    np.testing.assert_allclose(x.trace.tcp_position,y.trace.tcp_position,atol=1e-4)
    np.testing.assert_allclose(np.stack([d.qpos for d in x.state.physics]),y.state.physics.qpos,atol=1e-4)
    np.testing.assert_allclose(y.trace.command[1],0.,atol=1e-7) # two controls of delay
    assert compiled.describe()['heterogeneous']
    params=compiled._impl.models
    assert not np.array_equal(np.asarray(params.body_mass)[0],np.asarray(params.body_mass)[1])
    assert not np.array_equal(np.asarray(params.dof_M0)[0],np.asarray(params.dof_M0)[1])
    device_commands=jp.asarray(commands)
    with jax.transfer_guard('disallow'):
        again=compiled.collect_chunk(b,device_commands);compiled.synchronize(again)
    np.testing.assert_array_equal(y.state.physics.qpos,again.state.physics.qpos)
