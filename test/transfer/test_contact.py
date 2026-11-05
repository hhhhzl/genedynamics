import numpy as np
import pytest
from genedynamics.envs.domains.manipulation.transfer_workcell.contact_runtime import ContactRunner
from genedynamics.tasks.transfer.contact import ContactConfig
from genedynamics.tasks.transfer import RandomRequest


def test_force_channel_carry_failure_and_partial_reset():
    runner=ContactRunner(config=ContactConfig())
    request=RandomRequest(1,(31,32)); initial=runner.reset_batch(request)
    commands=np.zeros((2,3,13),np.float32); commands[...,12]=5.
    result=runner.collect_chunk(initial,commands)
    assert result.trace.command[0,-1,12]>0
    assert np.max(result.trace.normal_force)==0  # still above plane, not command force
    assert np.max(result.state.force_integral)==0  # no force windup before contact
    reset=runner.reset_batch(request,previous=result.state,mask=np.array([True,False]))
    assert reset.steps.tolist()==[0,3]
    np.testing.assert_array_equal(reset.physics[1].qpos,result.state.physics[1].qpos)
    # Declare a previously established contact which has just been lost in air.
    loss=result.state._replace(contact_seen=np.ones(2,bool),loss_count=np.ones(2,int)*14)
    failed=runner.collect_chunk(loss,commands[:,:2])
    assert failed.state.status.tolist()==[8,8]
    assert failed.trace.terminal.sum()==2
    assert not failed.trace.valid[:,1].any()
    np.testing.assert_allclose(failed.state.physics[0].qpos,runner.collect_chunk(loss,commands[:,:1]).state.physics[0].qpos)


def test_contact_jax_reference_short_parity():
    reference=ContactRunner(); compiled=ContactRunner(backend="jax")
    request=RandomRequest(2,(41,)); a=reference.reset_batch(request); b=compiled.reset_batch(request)
    commands=np.zeros((1,10,13),np.float32); commands[...,12]=5.
    # Warm/compile shape once, then run identical chunks; includes load impact.
    for _ in range(30):
        x=reference.collect_chunk(a,commands); y=compiled.collect_chunk(b,commands); compiled.synchronize(y)
        a,b=x.state,y.state
        np.testing.assert_allclose(x.trace.tcp_position,y.trace.tcp_position,atol=.001)
        np.testing.assert_allclose(x.trace.normal_force,y.trace.normal_force,atol=2.,rtol=.1)
    assert x.trace.allowed_contact.any()
    assert np.isfinite(np.asarray(y.trace.force_peak)).all()


def test_normal_projection_and_antiwindup_math():
    from genedynamics.tasks.transfer.contact import hybrid_wrench
    config=ContactConfig(); command=np.zeros(13); command[12]=5.
    pos=np.zeros(3); ref=np.array([.01,.02,100.])
    wrench=hybrid_wrench(np,pos,np.eye(3),np.zeros(6),ref,np.eye(3),command,0.,0.,config,True)
    assert wrench[0]>0 and wrench[1]>0
    np.testing.assert_allclose(wrench[2],-5.25)  # z position error projected away
    command[12]=0.
    released=hybrid_wrench(np,pos,np.eye(3),np.zeros(6),ref,np.eye(3),command,0.,0.,config)
    assert released[2]>1000  # position control restored on release


def test_substep_wrench_governor_projects_tangent_and_releases_growth_guard():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE, GOVERNOR_BRAKE_SUBSTEPS, GOVERNOR_NOMINAL,
        GOVERNOR_RECONTACT, GOVERNOR_SUPPORT_DWELL_SUBSTEPS,
        govern_substep_wrench, spatial_guide_config,
    )

    config=replace(spatial_guide_config("xarm7"),execution_profile="substep_wrench_governor")
    desired=np.array([6.,-2.,1.,.1,.2,.3])
    measured=np.array([0.,0.,18.,0.,0.,0.]); previous_contact=np.array([0.,0.,10.,0.,0.,0.]);position=np.array([.39,-.20,.587]); twist=np.array([.008,0.,.001,0.,0.,0.])
    governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),measured,position,twist,
        False,True,GOVERNOR_NOMINAL,0,previous_contact,0,0,config)
    assert active and phase==GOVERNOR_BRAKE and age==1 and dwell==0
    assert np.linalg.norm(governed[:2])<=6.+1e-12
    assert abs(governed[2])<=3.
    np.testing.assert_array_equal(governed[3:],desired[3:])
    # Four consecutive stable-support substeps release tangent braking into
    # normal-only recontact while the route remains inside the guide prefix.
    stable_support_substeps = 0
    for _ in range(24):
        governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
            np,desired,np.array([.0075,0.]),np.array([1.,0.,2.,0.,0.,0.]),
            position,np.zeros(6),True,True,phase,age,previous,dwell,release,config)
        stable_support_substeps += 1
        if phase==GOVERNOR_RECONTACT:
            break
    assert stable_support_substeps == 4
    assert phase==GOVERNOR_RECONTACT and active and dwell >= 4
    np.testing.assert_array_equal(governed[:2],desired[:2])
    governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),np.array([1.,0.,2.,0.,0.,0.]),
        position,np.zeros(6),True,False,phase,age,previous,dwell,release,config)
    assert phase==GOVERNOR_NOMINAL and not active

    legacy=spatial_guide_config("xarm7")
    unchanged,*_=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),measured,position,twist,
        False,True,GOVERNOR_NOMINAL,0,measured,0,0,legacy)
    np.testing.assert_array_equal(unchanged,desired)

    # The already-qualified +y route retains its nominal tangent effort.
    y_desired=np.array([2.,8.,1.,.1,.2,.3])
    y_governed,y_phase,y_age,y_previous,y_dwell,y_route,active=govern_substep_wrench(
        np,y_desired,np.array([0.,.0075]),np.zeros(6),position,twist,
        False,True,GOVERNOR_NOMINAL,0,previous_contact,0,0,config)
    np.testing.assert_array_equal(y_governed[:2],y_desired[:2])
    np.testing.assert_array_equal(y_governed[3:],y_desired[3:])
    assert y_governed[2]<y_desired[2] and active and y_route==2
    supported,*_=govern_substep_wrench(
        np,y_desired,np.array([0.,.0075]),np.array([0.,0.,2.,0.,0.,0.]),
        position,twist,True,True,GOVERNOR_RECONTACT,1,previous_contact,1,0,config)
    np.testing.assert_array_equal(supported,y_desired)




def test_substep_wrench_governor_has_finite_brake_without_support():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE, GOVERNOR_BRAKE_SUBSTEPS, GOVERNOR_NOMINAL,
        GOVERNOR_RECONTACT,
        govern_substep_wrench, spatial_guide_config,
    )

    config=replace(spatial_guide_config("xarm7"),execution_profile="substep_wrench_governor")
    desired=np.array([8.,0.,-2.,0.,0.,0.]);measured=np.zeros(6)
    position=np.array([.40,-.20,.588]);twist=np.zeros(6)
    phase,age,previous,dwell,release=GOVERNOR_BRAKE,0,measured,0,0
    active_steps=0
    for _ in range(GOVERNOR_BRAKE_SUBSTEPS):
        governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
            np,desired,np.array([.0075,0.]),measured,position,twist,
            False,True,phase,age,previous,dwell,release,config)
        active_steps += int(active)
    assert active_steps == GOVERNOR_BRAKE_SUBSTEPS
    assert phase == GOVERNOR_RECONTACT and age == GOVERNOR_BRAKE_SUBSTEPS and active
    governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),measured,position,twist,
        False,False,phase,age,previous,dwell,release,config)
    assert phase == GOVERNOR_NOMINAL and age == 0 and not active


def test_substep_reference_antiwindup_is_brake_only_and_bounds_stored_energy():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE, GOVERNOR_NOMINAL, GOVERNOR_RECONTACT,
        govern_substep_reference,
        spatial_guide_config,
    )

    config=replace(spatial_guide_config("xarm7"),execution_profile="substep_wrench_governor")
    position=np.array([.40,-.20,.588]);reference=position+np.array([.012,-.002,0.])
    command=np.zeros(13);command[:2]=[.0075,0.];command[12]=2.
    twist=np.array([.001,0.,0.,0.,0.,0.]);applied=np.array([2.,0.,-3.,0.,0.,0.])
    admitted,energy,projected,allowance,effort_norm,spring_norm,admitted_tangent=govern_substep_reference(
        np,reference,position,twist,command,applied,GOVERNOR_BRAKE,config)
    stiffness=800.*np.exp(command[6:8]);spring=stiffness*(admitted[:2]-position[:2])
    assert projected
    damping=2.*np.sqrt(stiffness)*(command[:2]-twist[:2])
    assert np.linalg.norm(spring)<=.8*config.lateral_force_limit+1e-12
    assert np.linalg.norm(spring+damping)<=allowance+1e-12
    np.testing.assert_allclose(effort_norm,np.linalg.norm(spring+damping))
    np.testing.assert_allclose(admitted_tangent,spring+damping)
    np.testing.assert_allclose(spring_norm,np.linalg.norm(spring))
    np.testing.assert_allclose(energy,.5*np.sum(spring**2/stiffness))

    predictive,_,projected,allowance,effort_norm,spring_norm,admitted_tangent=govern_substep_reference(
        np,position+np.array([.02,-.002,0.]),position,twist,command,
        applied,GOVERNOR_BRAKE,config)
    predictive_spring=stiffness*(predictive[:2]-position[:2])
    assert projected and spring_norm<=.8*config.lateral_force_limit+1e-12
    assert effort_norm<=allowance+1e-12
    assert np.linalg.norm(admitted_tangent)<=allowance+1e-12

    for phase in (GOVERNOR_NOMINAL,GOVERNOR_RECONTACT):
        unchanged,*_=govern_substep_reference(
            np,reference,position,twist,command,applied,phase,config)
        np.testing.assert_array_equal(unchanged,reference)

    legacy=spatial_guide_config("xarm7")
    unchanged,legacy_energy,legacy_projected,*_=govern_substep_reference(
        np,reference,position,twist,command,applied,GOVERNOR_BRAKE,legacy)
    np.testing.assert_array_equal(unchanged,reference)
    assert legacy_energy==0. and not legacy_projected


def test_recontact_reference_candidate_projects_without_changing_frozen_profile():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_RECONTACT, govern_substep_reference, spatial_guide_config,
    )

    frozen = replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor",
    )
    candidate = replace(
        frozen,
        execution_profile="substep_wrench_governor_recontact_reference",
    )
    position = np.array([.40, -.20, .588])
    reference = position + np.array([.012, -.002, 0.])
    twist = np.array([.001, 0., 0., 0., 0., 0.])
    command = np.zeros(13)
    command[:2] = [.0075, 0.]
    command[12] = 2.
    applied = np.array([2., 0., -3., 0., 0., 0.])

    unchanged, *_ = govern_substep_reference(
        np, reference, position, twist, command, applied,
        GOVERNOR_RECONTACT, frozen,
    )
    admitted, energy, projected, allowance, effort, spring, tangent = (
        govern_substep_reference(
            np, reference, position, twist, command, applied,
            GOVERNOR_RECONTACT, candidate,
        )
    )
    np.testing.assert_array_equal(unchanged, reference)
    assert projected and energy > 0.
    assert effort <= allowance + 1e-12
    assert spring <= .8 * candidate.lateral_force_limit + 1e-12
    assert np.linalg.norm(tangent) <= allowance + 1e-12
    assert admitted[2] == reference[2]


def test_settling_reference_candidate_is_finite_after_support_loss():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_RECONTACT,
        GOVERNOR_SUPPORT_DWELL_SUBSTEPS,
        govern_substep_reference,
        spatial_guide_config,
    )

    candidate = replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor_settling_reference",
    )
    position = np.array([.40, -.20, .588])
    reference = position + np.array([.012, -.002, 0.])
    twist = np.array([.001, 0., 0., 0., 0., 0.])
    command = np.zeros(13)
    command[:2] = [.0075, 0.]
    command[12] = 2.
    applied = np.array([2., 0., -3., 0., 0., 0.])

    settling, _, projected, *_ = govern_substep_reference(
        np, reference, position, twist, command, applied,
        GOVERNOR_RECONTACT, candidate, support_dwell=0,
    )
    released, _, released_projected, *_ = govern_substep_reference(
        np, reference, position, twist, command, applied,
        GOVERNOR_RECONTACT, candidate,
        support_dwell=GOVERNOR_SUPPORT_DWELL_SUBSTEPS,
    )
    assert projected
    assert not np.array_equal(settling, reference)
    assert not released_projected
    np.testing.assert_array_equal(released, reference)
    with pytest.raises(ValueError, match="support dwell"):
        govern_substep_reference(
            np, reference, position, twist, command, applied,
            GOVERNOR_RECONTACT, candidate,
        )


def test_substep_reference_preserves_deadline_reachable_tangent_authority():
    """A recorded safe state admits the 7.5 mm/s request above the 6.13 floor."""
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE, govern_substep_reference, spatial_guide_config,
    )

    config=replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor",
    )
    position=np.array([.4025,-.20,.58835])
    reference=position+np.array([.008,0.,0.])
    twist=np.array([.0013,0.,0.,0.,0.,0.])
    command=np.zeros(13);command[:2]=[.0075,0.];command[12]=2.
    applied=np.array([2.5,0.,-3.,0.,0.,0.])
    admitted,_,projected,allowance,effort_norm,spring_norm,admitted_tangent=govern_substep_reference(
        np,reference,position,twist,command,applied,GOVERNOR_BRAKE,config)
    np.testing.assert_array_equal(admitted,reference)
    assert not projected
    assert command[0]>.00613
    assert effort_norm<=allowance+1e-12
    assert np.linalg.norm(admitted_tangent)<=allowance+1e-12
    assert spring_norm<=.8*config.lateral_force_limit+1e-12


def test_substep_wrench_governor_carries_last_substep_event_across_control_tick():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE, GOVERNOR_NOMINAL, GOVERNOR_RECONTACT,
        govern_substep_wrench, spatial_guide_config,
    )

    config=replace(spatial_guide_config("xarm7"),execution_profile="substep_wrench_governor")
    desired=np.array([12.,-1.,-4.,0.,0.,0.])
    position=np.array([.382,-.20,.5863]);twist=np.array([.005,0.,0.,0.,0.,0.])
    # A separation on the final physics substep activates normal recontact,
    # not tangent braking; the tangent channel remains available for progress.
    governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),np.zeros(6),position,twist,
        False,True,GOVERNOR_NOMINAL,0,np.array([0.,0.,5.,0.,0.,0.]),0,0,config)
    assert active and phase==GOVERNOR_RECONTACT and age==1
    assert not np.array_equal(governed[:2],desired[:2])
    assert np.linalg.norm(governed[:2])<=6.+1e-12
    assert release==1
    # Every state component crosses the control-call boundary unchanged.
    next_governed,phase,age,previous,dwell,release,active=govern_substep_wrench(
        np,desired,np.array([.0075,0.]),np.array([0.,0.,2.,0.,0.,0.]),
        position+np.array([0.,0.,.001]),twist,True,True,
        phase,age,previous,dwell,release,config)
    assert active and phase==GOVERNOR_RECONTACT and age==2 and dwell==1
    np.testing.assert_array_equal(next_governed[:2],desired[:2])
    np.testing.assert_array_equal(previous,np.array([0.,0.,2.,0.,0.,0.]))


def test_separation_edge_candidate_enters_finite_brake():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_BRAKE,
        GOVERNOR_NOMINAL,
        GOVERNOR_RECONTACT,
        govern_substep_wrench,
        spatial_guide_config,
    )

    frozen = replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor",
    )
    candidate = replace(
        frozen,
        execution_profile="substep_wrench_governor_separation_edge",
    )
    desired = np.array([12., -1., -4., 0., 0., 0.])
    issued = np.array([.0075, 0.])
    measured = np.zeros(6)
    previous = np.array([0., 0., 3.5, 0., 0., 0.])
    position = np.array([.382, -.20, .5863])
    twist = np.array([.005, 0., 0., 0., 0., 0.])
    common = (
        np, desired, issued, measured, position, twist,
        False, True, GOVERNOR_RECONTACT, 12, previous, 0, 1,
    )
    frozen_result = govern_substep_wrench(*common, frozen)
    candidate_result = govern_substep_wrench(*common, candidate)
    frozen_wrench, frozen_phase = frozen_result[:2]
    candidate_wrench, candidate_phase, candidate_age = candidate_result[:3]
    active = candidate_result[-1]
    assert frozen_phase == GOVERNOR_RECONTACT
    assert candidate_phase == GOVERNOR_BRAKE
    assert candidate_age == 1 and active
    np.testing.assert_array_equal(candidate_wrench, frozen_wrench)

    returned = np.array([1., 0., 2., 0., 0., 0.])
    frozen_next = govern_substep_wrench(
        np, desired, issued, returned, position, twist, True, True,
        frozen_result[1], frozen_result[2], frozen_result[3],
        frozen_result[4], frozen_result[5], frozen,
    )[0]
    candidate_next = govern_substep_wrench(
        np, desired, issued, returned, position, twist, True, True,
        candidate_result[1], candidate_result[2], candidate_result[3],
        candidate_result[4], candidate_result[5], candidate,
    )[0]
    np.testing.assert_array_equal(frozen_next, desired)
    assert np.linalg.norm(candidate_next[:2]) < np.linalg.norm(frozen_next[:2])

    # No prior-contact drop means the candidate is bitwise identical.
    stable = np.array([0., 0., 3.5, 0., 0., 0.])
    unchanged = govern_substep_wrench(
        np, desired, issued, stable, position, twist,
        True, True, GOVERNOR_NOMINAL, 0, stable, 4, 0, candidate,
    )[0]
    np.testing.assert_array_equal(unchanged, desired)


def test_recontact_brakes_only_outward_surface_normal_motion():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_RECONTACT, govern_substep_wrench, spatial_guide_config,
    )

    config=replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor",
    )
    desired=np.array([6.,0.,-0.5,0.,0.,0.])
    position=np.array([.411,-.20,.5893])
    common=(np,desired,np.array([.0075,0.]),np.zeros(6),position)
    outward=govern_substep_wrench(
        *common,np.array([.0075,0.,.001,0.,0.,0.]),False,True,
        GOVERNOR_RECONTACT,1,np.zeros(6),0,1,config,
    )[0]
    stationary=govern_substep_wrench(
        *common,np.array([.0075,0.,.00075,0.,0.,0.]),False,True,
        GOVERNOR_RECONTACT,1,np.zeros(6),0,1,config,
    )[0]
    inward=govern_substep_wrench(
        *common,np.array([.0075,0.,-.00025,0.,0.,0.]),False,True,
        GOVERNOR_RECONTACT,1,np.zeros(6),0,1,config,
    )[0]
    assert outward[2] < stationary[2] < inward[2]


def test_mode0_recontact_settling_has_finite_wrench_window():
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        GOVERNOR_RECONTACT,
        GOVERNOR_SUPPORT_DWELL_SUBSTEPS,
        govern_substep_wrench,
        spatial_guide_config,
    )

    frozen = replace(
        spatial_guide_config("xarm7"),
        execution_profile="substep_wrench_governor",
    )
    candidate = replace(
        frozen,
        execution_profile="substep_wrench_governor_recontact_settling",
    )
    desired = np.array([6., 1., -2., 0., 0., 0.])
    measured = np.array([1., 0., 2., 0., 0., 0.])
    position = np.array([.382, -.20, .5863])
    twist = np.zeros(6)
    common = (
        np, desired, np.array([.0075, 0.]), measured, position, twist,
        True, True, GOVERNOR_RECONTACT, 1, measured,
    )
    frozen_wrench = govern_substep_wrench(
        *common, 1, 1, frozen,
    )[0]
    settling_wrench = govern_substep_wrench(
        *common, 1, 1, candidate,
    )[0]
    released_wrench = govern_substep_wrench(
        *common, GOVERNOR_SUPPORT_DWELL_SUBSTEPS, 1, candidate,
    )[0]
    np.testing.assert_array_equal(frozen_wrench, desired)
    assert not np.array_equal(settling_wrench[:3], desired[:3])
    np.testing.assert_array_equal(released_wrench, desired)

    # The qualified y-first route keeps its existing route-memory behavior.
    y_desired = np.array([1., 6., -2., 0., 0., 0.])
    y_wrench = govern_substep_wrench(
        np, y_desired, np.array([0., .0075]), measured, position, twist,
        True, True, GOVERNOR_RECONTACT, 1, measured, 1, 2, candidate,
    )[0]
    np.testing.assert_array_equal(y_wrench, y_desired)



def test_substep_wrench_governor_numpy_jax_pure_function_parity():
    jnp=pytest.importorskip("jax.numpy")
    from dataclasses import replace
    from genedynamics.tasks.transfer.contact import (
        govern_substep_reference, govern_substep_wrench, spatial_guide_config,
    )

    config=replace(spatial_guide_config("xarm7"),execution_profile="substep_wrench_governor")
    values=(np.array([6.,2.,-1.,0.,0.,0.]),np.array([.0075,0.]),
            np.array([7.,3.,18.,0.,0.,0.]),np.array([.39,-.20,.587]),
            np.array([.004,-.002,.001,0.,0.,0.]))
    prior=np.array([2.,1.,8.,0.,0.,0.])
    reference=govern_substep_wrench(
        np,*values,True,True,0,0,prior,0,0,config)
    compiled=govern_substep_wrench(
        jnp,*(jnp.asarray(v) for v in values),True,True,0,0,
        jnp.asarray(prior),0,0,config)
    for actual,expected in zip(compiled,reference):
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=2e-6,atol=2e-6)
    ref_values=(np.array([.408,-.202,.588]),np.array([.40,-.20,.588]),
                values[4],np.r_[values[1],0.,np.zeros(9),2.],
                np.array([2.,1.,-3.,0.,0.,0.]))
    reference=govern_substep_reference(
        np,*ref_values,1,config)
    compiled=govern_substep_reference(
        jnp,*(jnp.asarray(v) for v in ref_values),1,config)
    for actual,expected in zip(compiled,reference):
        np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=2e-6,atol=2e-6)


def test_contact_physics_timestep_refinement():
    # Refine physics only, preserving the 50Hz skill/controller contract.
    rows=[]
    for dt in (.002,.001):
        runner=ContactRunner(config=ContactConfig(physics_dt=dt))
        state=runner.reset_batch(RandomRequest(9,(91,)))
        command=np.zeros((1,300,13),np.float32);command[...,12]=5.
        result=runner.collect_chunk(state,command)
        assert result.state.status[0]==0 and result.trace.allowed_contact.any()
        rows.append(result.trace)
    np.testing.assert_allclose(rows[0].tcp_position[:,-1],rows[1].tcp_position[:,-1],atol=.001)
    np.testing.assert_allclose(rows[0].normal_force[:,-25:].mean(),rows[1].normal_force[:,-25:].mean(),atol=1.)
    assert abs(rows[0].force_peak.max()-rows[1].force_peak.max())<3.
