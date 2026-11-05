"""P5 liveness release and frozen-v1 compatibility."""
from copy import deepcopy
import numpy as np
import pytest
import hashlib
import torch
from types import SimpleNamespace

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
from genedynamics.tasks.transfer.controllers.mating_feedback import (
    PublicEarlyContactMatingFeedbackPolicy, PublicLivenessMatingFeedbackPolicy,
    PublicMatchedReferenceUnloadingMatingFeedbackPolicy,
    PublicReferenceUnloadingMatingFeedbackPolicy,
)
from genedynamics.tasks.transfer.controllers.cross_ebdm.mobile_dense import PublicLivenessMobileDenseTransportActor,PublicLivenessMobileResponseMPCActor
from genedynamics.tasks.transfer.controllers.recovery_execution import (
    PublicCompletionDominantRecoveryExecution,
    PublicLivenessRecoveryExecution,
)
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


def test_early_contact_release_uses_only_public_height_and_is_checkpointable():
    task=instantiate_task('T3',robot='xarm7');task.reset((31102,),seed=31102)
    actor=PublicEarlyContactMatingFeedbackPolicy(task.spec,modes=[0])
    low=public_row(task,.5964,0,safe=False)
    action=actor.act(deepcopy(low))
    assert actor.early_release_seen[0] and actor.early_guard_active[0]
    assert actor.observer.memory.phase[0]==1
    np.testing.assert_array_equal(action[0,:6],np.asarray([0.,0.,.008,0.,0.,0.],dtype=np.float32))
    assert action[0,12]==0.
    state=actor.get_state()
    np.testing.assert_array_equal(actor.act(deepcopy(low)),action)
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(deepcopy(low)),action)
    clear=public_row(task,.605,1,safe=False)
    actor.act(clear)
    assert not actor.early_guard_active[0]


def test_reference_unloading_is_bounded_public_and_checkpointable():
    task=instantiate_task('T3',robot='xarm7');task.reset((31103,),seed=31103)
    actor=PublicReferenceUnloadingMatingFeedbackPolicy(task.spec,modes=[0])
    low=public_row(task,.5964,0,safe=False)
    low['reference_position'][0,:2]=low['tcp_position'][0,:2]+[.003,-.001]
    action=actor.act(deepcopy(low))
    np.testing.assert_allclose(action[0,:3],[-.025,.0125,.008],rtol=0,atol=1e-7)
    state=actor.get_state()
    np.testing.assert_array_equal(actor.act(deepcopy(low)),action)
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(deepcopy(low)),action)
    clear=public_row(task,.605,1,safe=False)
    old=PublicEarlyContactMatingFeedbackPolicy(task.spec,modes=[0])
    old.act(deepcopy(low))
    np.testing.assert_array_equal(actor.act(deepcopy(clear)),old.act(deepcopy(clear)))


def test_matched_reference_unloading_respects_original_planar_bound():
    task=instantiate_task('T3',robot='xarm7');task.reset((31104,),seed=31104)
    actor=PublicMatchedReferenceUnloadingMatingFeedbackPolicy(task.spec,modes=[0])
    low=public_row(task,.5964,0,safe=False)
    low['reference_position'][0,:2]=low['tcp_position'][0,:2]+[.003,-.001]
    action=actor.act(low)
    np.testing.assert_allclose(action[0,:3],[-.02,.0125,.008],rtol=0,atol=1e-7)
    assert np.max(np.abs(action[0,:2])) <= .02000001


def test_mobile_liveness_binding_is_shared_by_dense_mpc_and_common_prefix(tmp_path):
    torch.set_num_threads(1);public,actions=record();path=tmp_path/'source.npz';np.savez_compressed(path,**{'public.'+k:v for k,v in public.items()},actions=actions)
    response_spec=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=32,members=2,hidden=(8,));bank=PublicSkillBank([dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=0)],response_spec)
    policy=PublicLocalResidualTransport(response_spec,PublicTransportSpec(hidden=(8,),residual_scale=.025));response=PublicIssuedResponseEnsemble(response_spec);spec=instantiate_task('T4',robot='go2_z1').spec;feedback=PublicBiasFeedbackSpec(guard_prediction_seconds=0.,compensation_gain=2.)
    dense=PublicLivenessMobileDenseTransportActor(spec,policy,bank,feedback);mpc=PublicLivenessMobileResponseMPCActor(spec,policy,bank,feedback,response=response);wrapped=PublicLivenessRecoveryExecution(dense,spec,feedback,mode=0,seed=0)
    assert isinstance(dense.base,PublicLivenessMatingFeedbackPolicy) and isinstance(mpc.base,PublicLivenessMatingFeedbackPolicy)
    assert isinstance(wrapped.common,PublicLivenessMatingFeedbackPolicy)
    before=wrapped.get_state();row={k:v[:1].copy() for k,v in public.items()};row['joint_position']=np.zeros((1,25));row['joint_velocity']=np.zeros((1,24));row['body_position']=np.zeros((1,3));value=wrapped.act(row);wrapped.restore(before)
    np.testing.assert_array_equal(wrapped.act(row),value)


class _FixedAction:
    def __init__(self, action):
        self.action = np.asarray(action, np.float32)[None]
        self.calls = 0

    def act(self, public):
        self.calls += 1
        return self.action.copy()

    def reset(self, **kwargs):
        self.calls = 0

    def get_state(self):
        return {"calls": self.calls, "action": self.action.copy()}

    def restore(self, state):
        self.calls = int(state["calls"])
        self.action = np.asarray(state["action"], np.float32).copy()


class _EnabledGate:
    def __init__(self):
        self.calls = 0

    def observe(self, public):
        self.calls += 1
        return True

    def reset(self):
        self.calls = 0

    def get_state(self):
        return {"calls": self.calls}

    def restore(self, state):
        self.calls = int(state["calls"])


def _completion_wrapper(common, candidate):
    value = object.__new__(PublicCompletionDominantRecoveryExecution)
    value.spec = SimpleNamespace(
        alias="T4",
        config=SimpleNamespace(
            control_dt=.02,
            position_tolerance=.002,
            orientation_tolerance=np.deg2rad(2.),
        ),
    )
    value.mode = 0
    value.seed = 0
    value.common = _FixedAction(common)
    value.candidate = _FixedAction(candidate)
    value.gate = _EnabledGate()
    value.common_controls = 0
    value.learned_controls = 0
    value._reset_completion_admission()
    return value


def _completion_row(tick, position):
    rotation = np.diag([1., -1., -1.])[None]
    return {
        "elapsed_seconds": np.asarray([[tick * .02]]),
        "tcp_position": np.asarray([position], float),
        "goal_position": np.asarray([[.44, -.20, .58]], float),
        "tcp_rotation": rotation.copy(),
        "goal_rotation": rotation.copy(),
    }


def test_completion_admission_falls_back_for_regress_and_terminal_hold():
    common = np.zeros(13, np.float32)
    common[0] = .01
    candidate = np.zeros(13, np.float32)
    candidate[0] = -.02
    actor = _completion_wrapper(common, candidate)

    row = _completion_row(0, (.40, -.20, .58))
    before = actor.get_state()
    np.testing.assert_array_equal(actor.act(row), common[None])
    assert actor.last_completion_blend == 0.
    assert actor.last_completion_code == 3
    actor.restore(before)
    np.testing.assert_array_equal(actor.act(row), common[None])

    actor.candidate.action[0, 0] = .02
    row = _completion_row(1, (.41, -.20, .58))
    np.testing.assert_array_equal(actor.act(row), actor.candidate.action)
    assert actor.last_completion_blend == 1.
    assert actor.last_completion_code == 1

    row = _completion_row(2, (.439, -.20, .58))
    np.testing.assert_array_equal(actor.act(row), common[None])
    assert actor.last_terminal_near
    assert actor.last_completion_code == 2
    assert actor.completion_terminal_controls == 1
