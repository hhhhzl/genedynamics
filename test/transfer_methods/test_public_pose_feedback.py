"""Executable support and causal continuation of the separately versioned servo."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.pose_feedback import PublicPoseFeedbackPolicy,PublicPoseFeedbackSpec,pose_transport


@pytest.fixture(autouse=True)
def native_library(monkeypatch):
    monkeypatch.setenv("CROSS_EBDM_NATIVE_LIBRARY",str(Path(__file__).resolve().parents[2]/".runtime/cross_ebdm/native/libcross_ebdm.so"))


def test_pose_servo_controls_all_six_directions_from_actual_sensors():
    task=instantiate_task("T3",robot="panda");public=task.reset((28600,),seed=28600)
    public["elapsed_seconds"][:]=.02
    public["tcp_position"][:]=[.442,-.198,.614]
    public["tcp_rotation"][:]=np.diag([1.,-1.,-1.]);public["tcp_twist"][:]=0.
    actor=PublicPoseFeedbackPolicy(task.spec)
    actor.observer.memory=actor.observer.memory._replace(phase=np.array([2]))
    before=actor.get_state();expected=actor.act(public)
    for axis in range(3):
        changed=deepcopy(public);changed["tcp_position"][:,axis]+=.001
        actor.restore(before);actual=actor.act(changed)
        assert actual[0,axis]<expected[0,axis]
    from genedynamics.core.control.transfer import integrate_rotation
    for axis in range(3):
        changed=deepcopy(public);delta=np.zeros(3);delta[axis]=.002
        changed["tcp_rotation"][0]=integrate_rotation(np,public["tcp_rotation"][0],delta,1.)
        actor.restore(before);actual=actor.act(changed)
        assert abs(actual[0,axis+3]-expected[0,axis+3])>.001
    actor.restore(before);changed=deepcopy(public);changed["reference_position"]+=.02
    np.testing.assert_array_equal(actor.act(changed),expected)
    task.state.force_integral[:]=9.;actor.restore(before)
    np.testing.assert_array_equal(actor.act(public),expected)


def test_normal_residual_survives_admittance_and_is_finitely_bounded(tmp_path):
    task=instantiate_task("T3",robot="panda");public=task.reset((28601,),seed=28601)
    public["measured_wrench"][:,2]=5.
    transport=pose_transport(insertion_bias=[0.,0.,-.001,0.,0.,0.])
    transport.save(tmp_path/"transport")
    from genedynamics.solvers.single.cross_ebdm.learning.reference_transport import ReferenceTransport
    loaded=ReferenceTransport.load(tmp_path/"transport",backend="native",device="cpu")
    actor=PublicPoseFeedbackPolicy(task.spec,feedback=PublicPoseFeedbackSpec(insertion_force_scale=1.),transport=loaded)
    actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]))
    state=actor.get_state();action=actor.act(public)
    assert action[0,2]==pytest.approx(-.001) and action[0,12]==0.
    actor.restore(state);public["measured_wrench"][:,2]=-100.
    assert actor.act(public)[0,2]==pytest.approx(-actor.pose_feedback.normal_velocity_limit)


def test_phase_transport_exposes_angular_and_vertical_support():
    bias=np.array([.001,-.001,.002,.01,-.01,.02])
    transport=pose_transport(alignment_bias=bias,insertion_bias=-bias)
    out=transport.apply(np.zeros((3,1,13),np.float32),phase=np.array([1,2,3]))
    np.testing.assert_array_equal(out[0],np.zeros((1,13)))
    np.testing.assert_allclose(out[1,0,:6],bias)
    np.testing.assert_allclose(out[2,0,:6],-bias)
