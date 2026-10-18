"""P5 liveness release and frozen-v1 compatibility."""
from copy import deepcopy
import numpy as np
import pytest
import hashlib
import torch

pytest.importorskip('mujoco')
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.governor_feedback import PublicGovernorFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.liveness import SafeSettledRetreatReleaseMixin
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.solvers.single.cross_ebdm.learning.local_transport import PublicLocalResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import PublicIssuedResponseEnsemble
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.mobile_dense import PublicLivenessMobileDenseTransportActor,PublicLivenessMobileResponseMPCActor
from genedynamics.tasks.transfer.controllers.recovery_execution import PublicLivenessRecoveryExecution
from test.transfer_methods.test_public_observed_response import record


class LivenessGovernor(SafeSettledRetreatReleaseMixin,PublicGovernorFeedbackPolicy):
    pass


def public_row(task,z,tick,*,safe):
    row=task.observe()
    row['elapsed_seconds'][:]=tick*.02
    row['tcp_position'][:]=[.44 if safe else .445,-.2,z]
    row['reference_position'][:]=row['tcp_position']
    row['tcp_rotation'][:]=row['reference_rotation'][:]=np.diag([1.,-1.,-1.])
    row['tcp_twist'][:]=0.
    return row


def force_phase3(actor):
    actor.observer.memory=actor.observer.memory._replace(phase=np.array([3]))


def test_safe_settled_releases_latch_without_changing_height_release():
    task=instantiate_task('T3',robot='panda');task.reset((31100,),seed=31100);old=PublicGovernorFeedbackPolicy(task.spec);new=LivenessGovernor(task.spec)
    force_phase3(old);force_phase3(new)
    # Unsafe below the v1 height certificate latches both versions.
    unsafe=public_row(task,.605,0,safe=False)
    assert old.act(deepcopy(unsafe))[0,2]>0 and new.act(deepcopy(unsafe))[0,2]>0
    assert old.retreating[0] and new.retreating[0]
    # The same existing safe predicate and settle clock release only v2.
    for tick in range(1,5):
        safe=public_row(task,.605,tick,safe=True);old_value=old.act(deepcopy(safe));new_value=new.act(deepcopy(safe))
    assert old.retreating[0] and old_value[0,2]>0
    assert not new.retreating[0] and new_value[0,2]<0
    # The original absolute-height certificate still releases both.
    high=public_row(task,.611,5,safe=True);old.act(deepcopy(high));new.act(deepcopy(high))
    assert not old.retreating[0] and not new.retreating[0]


def test_v1_hook_refactor_preserves_frozen_command_sequence():
    task=instantiate_task('T3',robot='panda');task.reset((31101,),seed=31101);actor=PublicGovernorFeedbackPolicy(task.spec);force_phase3(actor)
    rows=[public_row(task,.605,0,safe=False)]+[public_row(task,.605,i,safe=True) for i in range(1,5)]
    values=[actor.act(deepcopy(row)) for row in rows]
    assert actor.retreating[0]
    assert all(value[0,2]>0 for value in values)
    state=actor.get_state();value=actor.act(deepcopy(rows[-1]));actor.restore(state)
    np.testing.assert_array_equal(actor.act(deepcopy(rows[-1])),value)


def test_mobile_liveness_binding_is_shared_by_dense_mpc_and_common_prefix(tmp_path):
    torch.set_num_threads(1);public,actions=record();path=tmp_path/'source.npz';np.savez_compressed(path,**{'public.'+k:v for k,v in public.items()},actions=actions)
    response_spec=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=32,members=2,hidden=(8,));bank=PublicSkillBank([dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=0)],response_spec)
    policy=PublicLocalResidualTransport(response_spec,PublicTransportSpec(hidden=(8,),residual_scale=.025));response=PublicIssuedResponseEnsemble(response_spec);spec=instantiate_task('T4',robot='go2_z1').spec;feedback=PublicBiasFeedbackSpec(guard_prediction_seconds=0.,compensation_gain=2.)
    dense=PublicLivenessMobileDenseTransportActor(spec,policy,bank,feedback);mpc=PublicLivenessMobileResponseMPCActor(spec,policy,bank,feedback,response=response);wrapped=PublicLivenessRecoveryExecution(dense,spec,feedback,mode=0,seed=0)
    assert isinstance(dense.base,PublicLivenessMatingFeedbackPolicy) and isinstance(mpc.base,PublicLivenessMatingFeedbackPolicy)
    assert isinstance(wrapped.common,PublicLivenessMatingFeedbackPolicy)
    before=wrapped.get_state();row={k:v[:1].copy() for k,v in public.items()};row['joint_position']=np.zeros((1,25));row['joint_velocity']=np.zeros((1,24));row['body_position']=np.zeros((1,3));value=wrapped.act(row);wrapped.restore(before)
    np.testing.assert_array_equal(wrapped.act(row),value)
