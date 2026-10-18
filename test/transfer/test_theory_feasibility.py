"""Physical and theoretical failure boundaries for the matched-keyway pilot."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_matching_keyway_rejects_wrong_key_orientation_without_reset_penetration():
    import mujoco
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import mating_contact_config
    from genedynamics.tasks.transfer import RandomRequest
    from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import solve_ik,rotation
    runner=ContactRunner(config=mating_contact_config())
    p=runner._impl;state=runner.reset_batch(RandomRequest(280,(560,)),initial="jammed")
    assert min((c.dist for c in state.physics[0].contact),default=0)>-1e-4
    def penetration(yaw):
        data=mujoco.MjData(p.model)
        data.qpos[:]=solve_ik(p.model,p.sid,p.scene.goal_qpos,np.asarray(runner.config.goal),rotation(yaw))
        mujoco.mj_forward(p.model,data)
        relevant=[c.dist for c in data.contact if tuple(sorted((c.geom1,c.geom2))) in p.allowed and
                  "socket_floor" not in {p.model.geom(c.geom1).name,p.model.geom(c.geom2).name}]
        return min(relevant,default=0.)
    assert penetration(0.)>-1e-4
    assert penetration(np.pi)<-.002


def test_tcp_wrench_matches_independent_generalized_constraint_force():
    import mujoco
    from genedynamics.envs.domains.manipulation.transfer_workcell.backends.contact import MujocoContact
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><geom name="support" type="plane" size="1 1 .1"/>
      <body pos="0 0 .045"><freejoint/>
      <geom name="contact_peg" type="box" pos=".02 0 0" size=".03 .03 .04" mass="1"/>
      <site name="tcp" pos="0 0 .12"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    for _ in range(800):mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    p=MujocoContact.__new__(MujocoContact);p.model=model;p.sid=model.site("tcp").id
    p.config=SimpleNamespace(wrench_semantics="tool_at_tcp_world")
    p.allowed=frozenset({tuple(sorted((model.geom("support").id,model.geom("contact_peg").id)))})
    wrench,_,touch,bad=p._measure(data)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,jp,jr,p.sid)
    np.testing.assert_allclose(jp.T@wrench[:3]+jr.T@wrench[3:],data.qfrc_constraint,atol=2e-7)
    assert touch and not bad and wrench[2]>9. and wrench[4]<-.1


def test_real_lip_stall_and_intentional_withdrawal():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import mating_contact_config
    from genedynamics.tasks.transfer import RandomRequest
    runner=ContactRunner(config=mating_contact_config());state=runner.reset_batch(RandomRequest(281,(562,)),initial="jammed")
    commands=np.zeros((1,150,13),np.float32);commands[...,12]=5.
    loaded=runner.collect_chunk(state,commands)
    tail=loaded.trace.tcp_position[0,-25:,2]
    assert loaded.state.status[0]==0 and loaded.trace.normal_force[0,-25:].min()>.5
    assert np.ptp(tail)<.0002 and tail.mean()>runner.config.goal[2]+.012
    commands=np.zeros((1,60,13),np.float32);commands[...,2]=.02
    released=runner.collect_chunk(loaded.state,commands)
    assert released.state.status[0]==0 and released.trace.tcp_position[0,-1,2]>tail.mean()+.008


def test_transient_side_contact_is_not_latched_seated_support():
    from genedynamics.tasks.transfer.contact import mating_contact_config,contact_termination
    from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import rotation
    cfg=mating_contact_config();s=SimpleNamespace(last_command=np.r_[np.zeros(12),5.],contact_seen=True,loss_count=14,hold_count=0,steps=50)
    evaluate=lambda z:contact_termination(np,s,np.array([.44,-.2,z]),0.,1.,False,True,False,cfg,rotation=rotation(),axial_contact=False)
    assert evaluate(.59)[0]==0 and not evaluate(.59)[1]
    assert evaluate(.580)[0]==8


def test_geometry_alignment_includes_tip_and_is_jittable():
    import jax
    import jax.numpy as jp
    from genedynamics.tasks.transfer.behavior import keyway_alignment
    from genedynamics.envs.domains.manipulation.transfer_workcell.workcell import rotation
    p=np.array([[.4405,-.2,.614],[.44,-.2,.614]])
    r=np.stack((rotation(),rotation(np.pi)))
    expected=keyway_alignment(p,r,.0008)
    assert expected.tolist()==[True,False]
    np.testing.assert_array_equal(jax.jit(lambda a,b:keyway_alignment(a,b,.0008,xp=jp))(jp.asarray(p),jp.asarray(r)),expected)


def test_finite_theory_audit_rejects_unidentifiability_and_small_sample_claims():
    from genedynamics.solvers.single.cross_ebdm.learning.research_audit import ResearchAudit
    audit=ResearchAudit()
    assert not audit.separation([[0.,0.],[0.,0.]],noise_bound=0.,model_bias_bound=0.)["identified"]
    assert not audit.separation([[0.],[.2]],noise_bound=.1,model_bias_bound=0.)["identified"]
    assert audit.separation([[0.],[1.]],noise_bound=.1,model_bias_bound=.1)["identified"]
    small=np.tile([1.,0.],(20,1));large=np.tile([1.,0.],(10000,1))
    assert not audit.cost_certificate(small,small)["certified_vs_column0"][1]
    assert audit.cost_certificate(large,large)["certified_vs_column0"][1]
    with pytest.raises(ValueError):audit.cost_certificate([[2.]],[[1.]])
    paired=audit.paired_improvement(np.tile([.9,.1],(25,1)))
    assert paired["certified_vs_column0"]==[True] and paired["improvement_lower"][0]>.3
    assert not audit.paired_improvement(np.tile([.1,.9],(25,1)))["certified_vs_column0"][0]
    law=audit.mode_law([0.,2.],[.5,.5],[True,False])
    assert law["probabilities"]==[1.,0.]
    assert audit.mode_law([0.,2.],[.5,.5],[False,False])["stop_mass"]==1.


def test_stage_transport_jax_torch_and_unsupported_physics(tmp_path):
    from genedynamics.solvers.single.cross_ebdm.learning.reference_transport import ReferenceTransport
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import mating_contact_config
    k=(1.,1.,0.,1.,1.,1.)
    t=ReferenceTransport.from_stage_parameters(2.,(-.004,.002),alignment_stiffness=k,insertion_stiffness=k)
    t.save(tmp_path/"stage")
    u=np.zeros((3,10,13),np.float32);u[...,12]=5.;phase=np.array([0,2,3])
    expected=np.asarray(t.apply(u,phase))
    assert expected[0,0,6]==0 and expected[1,0,6]==1 and expected[1,0,12]==5 and expected[2,0,12]==10
    actual=ReferenceTransport.load(tmp_path/"stage",backend="torch").apply(u,phase)
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    constructed=ReferenceTransport.from_stage_parameters(2.,(-.004,.002),alignment_stiffness=k,insertion_stiffness=k,backend="torch")
    np.testing.assert_allclose(constructed.apply(u,phase),expected,atol=1e-6)
    with pytest.raises(ValueError):ReferenceTransport.from_stage_parameters(2.,(float("nan"),0.))
    with pytest.raises(NotImplementedError,match="fixed Panda/xArm7"):
        ContactRunner(config=mating_contact_config("ur5e"),backend="jax")


def test_candidate_mpc_restarts_observed_state_without_hidden_model_or_mutation():
    from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
    from genedynamics.tasks.transfer.contact import mating_contact_config, ContactMechanismSpec
    from genedynamics.solvers.single.cross_ebdm.contact_planner import ContactPlanner
    from genedynamics.tasks.transfer.behavior import behavior_reset
    from genedynamics.tasks.transfer import RandomRequest
    class Identity:
        @staticmethod
        def apply(command,phase):return command
    config=mating_contact_config()
    target=ContactRunner(config=config,mechanism=ContactMechanismSpec(.6,3,.5,.7,.04))
    observed=target.reset_batch(RandomRequest(290,(580,)),initial="clear")
    observed.physics[0].qacc_warmstart[:]=123.
    qpos=observed.physics[0].qpos.copy();qvel=observed.physics[0].qvel.copy()
    planner=ContactPlanner(config,ContactMechanismSpec(),[Identity(),Identity()],horizon_chunks=1)
    restarted=planner._impl.restart(observed)
    assert planner._impl.runner._impl.model is not target._impl.model
    np.testing.assert_array_equal(restarted.physics[0].qacc_warmstart,0.)
    np.testing.assert_array_equal(restarted.physics[0].qpos,qpos)
    def policy(runner,state,memory):
        command=np.zeros((1,10,13),np.float32)
        reference=np.r_[state.physics[0].site_xpos[runner._impl.sid],0.,0.][None]
        return command,memory,reference
    winner,detail=planner.select(observed,behavior_reset(1,[0]),policy)
    assert winner==0 and len(detail["scores"])==2 and detail["valid_control_steps"]==20
    np.testing.assert_array_equal(observed.physics[0].qpos,qpos)
    np.testing.assert_array_equal(observed.physics[0].qvel,qvel)
    assert planner.statistics()["candidate_rollouts"]==2
    with pytest.raises(NotImplementedError):ContactPlanner(config,ContactMechanismSpec(),[Identity()],backend="torch")


def test_stage_transport_native_uses_same_phase_features(tmp_path,monkeypatch):
    import shutil
    import subprocess
    from pathlib import Path
    from genedynamics.solvers.single.cross_ebdm.learning.reference_transport import ReferenceTransport
    compiler=shutil.which("g++") or shutil.which("clang++")
    if compiler is None:pytest.skip("native compiler unavailable")
    source=Path(__file__).resolve().parents[2]/"genedynamics/solvers/single/cross_ebdm/learning/backends/native/ensemble.cpp"
    library=tmp_path/"stage.so"
    subprocess.run([compiler,"-std=c++17","-O2","-fPIC","-shared",str(source),"-o",str(library)],check=True)
    monkeypatch.setenv("CROSS_EBDM_NATIVE_LIBRARY",str(library))
    t=ReferenceTransport.from_stage_parameters(2.5,(-.004,.002),alignment_stiffness=(1.,)*6)
    t.save(tmp_path/"stage")
    command=np.zeros((3,10,13),np.float32);command[...,12]=4.
    phase=np.array([0,2,3])
    native=ReferenceTransport.load(tmp_path/"stage",backend="native")
    np.testing.assert_allclose(native.apply(command,phase),t.apply(command,phase),atol=1e-6)
