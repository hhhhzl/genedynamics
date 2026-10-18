"""Use the real guide phase generator and source bank, not an invented label mock."""
import hashlib
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.skill_bank import PublicSkillBank
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor_v2 import PublicCorridorGuideDenseActor
from genedynamics.experiments.plugins.methods.transfer.fada.guide_corridor_v2 import PublicCorridorGuideMemory
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
    p,_=record();spec=task_spec('T2',robot='xarm7');policy=PublicTaskResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025));actor=PublicCorridorGuideDenseActor(spec,policy,bank,mode=0);memory=PublicCorridorGuideMemory(spec,[0])
    for t,pos in enumerate([[.38,-.2,.605],[.38,-.2,.634],[.398,-.2,.634],[.444,-.18,.634]]):
        row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=t*.02;row['tcp_position'][:]=pos;row['reference_position'][:]=pos;row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation'];row['measured_wrench'][:]=0
        batch,_=actor.prepare(row);state=memory.get_state();context=memory.update(row);memory.restore(state);np.testing.assert_array_equal(context,memory.update(row))
        assert int(batch['phase'][0])==int(np.argmax(context[0,22:26]))==[0,1,1,3][t]
