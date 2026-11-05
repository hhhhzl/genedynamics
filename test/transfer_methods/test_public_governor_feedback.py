"""Continuation, geometric descent gating and public command lead limits."""
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("mujoco")
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.governor_feedback import PublicGovernorFeedbackPolicy,PublicGovernorFeedbackSpec


def test_public_lead_governor_unwinds_reference_and_blocks_off_axis_descent():
    task=instantiate_task("T3",robot="panda");public=task.reset((31000,),seed=31000)
    public["elapsed_seconds"][:]=.02;public["tcp_position"][:]=[.446,-.2,.606]
    public["reference_position"][:]=[.456,-.2,.606];public["tcp_rotation"][:]=np.diag([1.,-1.,-1.]);public["tcp_twist"][:]=0.
    actor=PublicGovernorFeedbackPolicy(task.spec);actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]))
    state=actor.get_state();command=actor.act(public)
    assert command[0,0]<0 and command[0,2]>0 and command[0,12]==0
    actor.restore(state);np.testing.assert_array_equal(actor.act(public),command)
    assert not any(k in vars(actor) for k in ("task","physics","mechanism"))


def test_guard_settles_on_public_ticks_and_preserves_seated_force_holding():
    task=instantiate_task("T3",robot="panda");public=task.reset((31001,),seed=31001)
    public["tcp_position"][:]=[.44,-.2,.614];public["reference_position"][:]=public["tcp_position"]
    public["tcp_rotation"][:]=public["reference_rotation"][:]=np.diag([1.,-1.,-1.]);public["tcp_twist"][:]=0.
    actor=PublicGovernorFeedbackPolicy(task.spec);actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]))
    for i in range(4):
        public["elapsed_seconds"][:]=i*.02
        command=actor.act(public)
        assert (command[0,2]>=0) if i<3 else (command[0,2]<0)
    state=actor.get_state();actor.act(public);assert actor.clear_count[0]==state["governor"]["clear_count"][0]
    actor.restore(state);public["elapsed_seconds"][:]=.08;public["tcp_position"][:]=[.4405,-.2,task.config.goal[2]]
    public["reference_position"][:]=public["tcp_position"];public["measured_wrench"][:,2]=5.
    assert abs(actor.act(public)[0,2])<1e-7


def test_governor_batch_state_reorder_is_exact():
    task=instantiate_task("T3",robot="panda");public=task.reset((31002,31003),seed=31002)
    actor=PublicGovernorFeedbackPolicy(task.spec,modes=[0,1]);actor.act(public)
    other=PublicGovernorFeedbackPolicy(task.spec,modes=[0,1]);other.restore(actor.get_state())
    actor.reorder([1,0]);other.reorder([1,0]);public={k:v[[1,0]] for k,v in task.step(np.zeros((2,13))).observation.items()}
    np.testing.assert_array_equal(actor.act(public),other.act(public))


def test_insertion_lead_is_phase_scoped_and_default_preserving():
    base=PublicGovernorFeedbackSpec()
    explicit=PublicGovernorFeedbackSpec(insertion_lead_z_m=.015)
    actor=PublicGovernorFeedbackPolicy(instantiate_task("T3",robot="panda").spec,feedback=explicit)
    np.testing.assert_array_equal(actor._translation_lead(np.array([2])),[[base.lead_xy_m,base.lead_xy_m,base.lead_z_m]])
    np.testing.assert_array_equal(actor._translation_lead(np.array([3])),[[base.lead_xy_m,base.lead_xy_m,.015]])
    default_actor=PublicGovernorFeedbackPolicy(actor.spec,feedback=base)
    np.testing.assert_array_equal(default_actor._translation_lead(np.array([2,3])),[[.004,.004,.01],[.004,.004,.01]])
    with pytest.raises(ValueError,match="outside declared public support"):
        PublicGovernorFeedbackSpec(insertion_lead_z_m=.021)


def test_public_reference_governor_enable_is_explicit_and_changes_only_lead_projection():
    task=instantiate_task("T3",robot="panda");public=task.reset((31004,),seed=31004)
    public["tcp_position"][:]=[.46,-.2,.614]
    public["reference_position"][:]=[.444,-.2,.614]
    public["tcp_rotation"][:]=public["reference_rotation"][:]=np.diag([1.,-1.,-1.])
    public["tcp_twist"][:]=0.;public["elapsed_seconds"][:]=0.
    enabled=PublicGovernorFeedbackPolicy(task.spec,feedback=PublicGovernorFeedbackSpec())
    disabled=PublicGovernorFeedbackPolicy(
        task.spec,feedback=PublicGovernorFeedbackSpec(reference_governor_enabled=False),
    )
    for actor in (enabled,disabled):
        actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]))
    governed=enabled.act(deepcopy(public));ungoverned=disabled.act(deepcopy(public))
    assert governed[0,0]>0
    assert ungoverned[0,0]<0
    np.testing.assert_array_equal(governed[:,6:],ungoverned[:,6:])
    with pytest.raises(TypeError,match="enable must be boolean"):
        PublicGovernorFeedbackSpec(reference_governor_enabled=1)
