"""Use the real guide phase generator and source bank, not an invented label mock."""
import hashlib
import numpy as np
import torch
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import PublicCorridorGuideDenseActor,PublicModeConditionedContactEstablishedCorridorGuideDenseActor
from test.transfer_methods.test_public_observed_response import record
def test_real_guide_source_bank_categories_support_all_target_corridor_stages(tmp_path):
    entries=[];s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1)
    for mode in (0,1):
        public,actions=record(80);public['tcp_position'][:]=[.38,-.2,.596]
        if mode==0:public['tcp_position'][20:,0]=.41;public['tcp_position'][40:,:2]=[.46,-.18]
        else:public['tcp_position'][20:,1]=-.18;public['tcp_position'][40:,0]=.46
        public['reference_position']=public['tcp_position'].copy();public['goal_position'][:]=[.46,-.18,.596]
        path=tmp_path/f'source_{mode}.npz';np.savez(path,actions=actions,**{'public.'+k:v for k,v in public.items()});entries.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=mode))
    bank=PublicSkillBank(entries,s,task='T2')
    assert set(bank.groups)=={(phase,mode) for phase in (0,1,3) for mode in (0,1)}
    p,_=record();spec=task_spec('T2',robot='xarm7');policy=PublicTaskResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025));actor=PublicCorridorGuideDenseActor(spec,policy,bank,mode=0)
    for t,(stage,pos) in enumerate(zip([0,1,2,4],[[.38,-.2,.605],[.38,-.2,.596],[.398,-.2,.596],[.46,-.18,.596]])):
        actor.base.stage=stage
        row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=t*.02;row['tcp_position'][:]=pos;row['reference_position'][:]=pos;row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation'];row['measured_wrench'][:]=0
        batch,_=actor.prepare(row)
        assert int(batch['phase'][0])==[0,0,1,3][t]


def test_response_normalized_wrapper_preserves_transport_phase_interface(tmp_path):
    entries=[];s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1)
    for mode in (0,1):
        public,actions=record(80);public['tcp_position'][:]=[.38,-.2,.596]
        if mode==0:public['tcp_position'][20:,0]=.41;public['tcp_position'][40:,:2]=[.46,-.18]
        else:public['tcp_position'][20:,1]=-.18;public['tcp_position'][40:,0]=.46
        public['reference_position']=public['tcp_position'].copy();public['goal_position'][:]=[.46,-.18,.596]
        path=tmp_path/f'response_source_{mode}.npz';np.savez(path,actions=actions,**{'public.'+k:v for k,v in public.items()});entries.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=mode))
    bank=PublicSkillBank(entries,s,task='T2')
    base=task_spec('T2',robot='xarm7')
    from dataclasses import replace
    spec=replace(base,config=replace(base.config,guide_reference_admission='response_normalized_completion',log_stiffness_limit=1.5,tangential_wrench_limit=16.))
    policy=PublicTaskResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025))
    actor=PublicCorridorGuideDenseActor(spec,policy,bank,mode=1)
    p,_=record();row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=0.;row['tcp_position'][:]=[.38,-.2,.596];row['reference_position'][:]=row['tcp_position'];row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation'];row['measured_wrench'][:]=0
    batch,_=actor.prepare(row)
    assert int(batch['phase'][0]) == 0
    actor.base.switched=True;actor.base.completion.reset();actor.base.completion.act(row);actor.last_tick=-1;actor.frames=[]
    batch,_=actor.prepare(row)
    assert int(batch['phase'][0]) in {1,3}


def test_contact_establishment_admission_projects_until_public_dwell(tmp_path):
    entries=[];s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1)
    for mode in (0,1):
        public,actions=record(80);public['tcp_position'][:]=[.38,-.2,.596];public['reference_position']=public['tcp_position'].copy();public['goal_position'][:]=[.46,-.18,.596]
        path=tmp_path/f'admission_source_{mode}.npz';np.savez(path,actions=actions,**{'public.'+k:v for k,v in public.items()});entries.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=mode))
    bank=PublicSkillBank(entries,s,task='T2')
    policy=PublicTaskResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025))
    with torch.no_grad():policy.network[-1].bias.fill_(100.)
    spec=task_spec('T2',robot='xarm7');actor=PublicModeConditionedContactEstablishedCorridorGuideDenseActor(spec,policy,bank,mode=0)
    public,_=record();row={k:v[:1].copy() for k,v in public.items()};row['tcp_position'][:]=[.38,-.2,.596];row['reference_position'][:]=row['tcp_position'];row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation']
    from genedynamics.solvers.single.cross_ebdm.learning.task_transport import T2_TRANSPORT_COORDINATES
    for tick in range(10):
        row['elapsed_seconds'][:]=tick*.02;row['measured_wrench'][:]=0 if tick==0 else [0,0,5,0,0,0]
        action=actor.act(row)
        np.testing.assert_array_equal(action[:,list(T2_TRANSPORT_COORDINATES)],actor.last_common_action[:,list(T2_TRANSPORT_COORDINATES)])
        assert actor.last_selector_code==actor.PROJECTED_TO_PUBLIC_BASE
        before=actor.contact_establishment_projected_controls
        np.testing.assert_array_equal(actor.act(row),action)
        assert actor.contact_establishment_projected_controls==before
    state=actor.get_state();row['elapsed_seconds'][:]=.2;row['measured_wrench'][:]=[0,0,5,0,0,0]
    admitted=actor.act(row)
    assert actor.contact_establishment_admitted and actor.contact_establishment_tick==10
    assert actor.last_selector_code==0 and actor.contact_establishment_transport_controls==1
    assert not np.array_equal(admitted[:,list(T2_TRANSPORT_COORDINATES)],actor.last_common_action[:,list(T2_TRANSPORT_COORDINATES)])
    actor.restore(state)
    np.testing.assert_array_equal(actor.act(row),admitted)
    assert actor.contact_establishment_tick==10
    actor.reset()
    assert not actor.contact_establishment_admitted
    assert actor.contact_establishment_projected_controls==0
    mode_one=PublicModeConditionedContactEstablishedCorridorGuideDenseActor(spec,policy,bank,mode=1)
    row['elapsed_seconds'][:]=0;row['measured_wrench'][:]=0
    action=mode_one.act(row)
    assert not mode_one.contact_establishment_applicable
    assert mode_one.contact_establishment_admitted
    assert mode_one.contact_establishment_projected_controls==0
    assert mode_one.contact_establishment_transport_controls==1
    assert mode_one.last_selector_code==0
    assert not np.array_equal(action[:,list(T2_TRANSPORT_COORDINATES)],mode_one.last_common_action[:,list(T2_TRANSPORT_COORDINATES)])
