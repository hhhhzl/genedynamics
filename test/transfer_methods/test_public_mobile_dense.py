"""T4 target identity/source isolation and exact public execution continuation."""
import hashlib
import numpy as np
import torch
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.issued_response import PublicIssuedResponseEnsemble
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.solvers.single.cross_ebdm.learning.local_transport import PublicLocalResidualTransport
from genedynamics.tasks.transfer.controllers.mating_feedback import PublicMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.mobile_dense import PublicMobileDenseTransportActor,PublicMobileResponseMPCActor
from test.transfer_methods.test_public_observed_response import record


def test_mobile_target_chart_keeps_real_body_and_panda_query_and_restores(tmp_path):
    torch.set_num_threads(1);p,a=record();path=tmp_path/'source.npz';np.savez_compressed(path,**{'public.'+k:v for k,v in p.items()},actions=a)
    s=PublicResponseSpec(history_len=3,horizon=2,maximum_joints=32,members=2,hidden=(8,))
    bank=PublicSkillBank([dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=0)],s)
    policy=PublicLocalResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025));spec=task_spec('T4',robot='go2_z1');fb=PublicBiasFeedbackSpec(guard_prediction_seconds=0.,compensation_gain=2.)
    actor=PublicMobileDenseTransportActor(spec,policy,bank,fb);base=PublicMatingFeedbackPolicy(spec,feedback=fb)
    for i in range(8):
        row={k:v[i:i+1].copy() for k,v in p.items()};row['joint_position']=np.zeros((1,25));row['joint_position'][0,3]=1.;row['joint_velocity']=np.zeros((1,24));row['body_position']=np.array([[.01,.02,.3]])
        before=actor.get_state();batch,_=actor.prepare(row)
        np.testing.assert_array_equal(batch['target_features'][0,-11:-7],[0,0,0,1]);np.testing.assert_array_equal(batch['target_features'][0,-7:-3],[0,0,0,1])
        np.testing.assert_array_equal(batch['source_features'][0,-11:-7],[1,0,0,0]);np.testing.assert_array_equal(batch['source_features'][0,-7:-3],[0,0,1,0])
        actor.restore(before);value=actor.act(row);actor.restore(before);np.testing.assert_array_equal(actor.act(row),value)
        np.testing.assert_allclose(value,base.act(row),rtol=0,atol=2e-8)
    actor.reset();assert actor.spec.alias==actor.base.spec.alias=='T4' and actor.base.observer.spec.alias=='T3'
    response=PublicIssuedResponseEnsemble(s);frozen={k:v.clone() for k,v in response.state_dict().items()}
    mpc=PublicMobileResponseMPCActor(spec,policy,bank,fb,response=response);row={k:v[:1].copy() for k,v in p.items()}
    before=mpc.get_state();value=mpc.act(row);mpc.restore(before);np.testing.assert_array_equal(mpc.act(row),value)
    assert mpc.spec.alias=='T4' and mpc.irreversible_model_calls==12 and mpc.irreversible_action_updates==10
    assert all(torch.equal(v,frozen[k]) for k,v in response.state_dict().items())
