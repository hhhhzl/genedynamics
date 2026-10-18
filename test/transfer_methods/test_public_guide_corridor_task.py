"""Source phase gaps are explicit; target memory and deployment use one clock."""
from types import SimpleNamespace
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.guide_observation import context_sequence
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.tasks.transfer.controllers.cross_ebdm.guide_corridor import PublicCorridorGuideDenseActor
from genedynamics.experiments.plugins.methods.transfer.fada.guide_corridor import PublicCorridorGuideMemory,corridor_context_sequence
from test.transfer_methods.test_public_observed_response import record
def test_phase_semantics_memory_continuation_and_frozen_source_context():
    p,_=record();s=PublicResponseSpec(history_len=4,horizon=10,hidden=(8,),members=1)
    def query(public,phase,modes):
        assert int(phase[0]) in (0,2,3)
        return dict(features=np.zeros((1,s.feature_dim),np.float32),commands=np.zeros((1,10,13),np.float32)),np.zeros(1),np.zeros(1,np.int64)
    bank=SimpleNamespace(spec=s,task='T2',query=query);policy=PublicTaskResidualTransport(s,PublicTransportSpec(hidden=(8,),residual_scale=.025));spec=task_spec('T2',robot='xarm7');actor=PublicCorridorGuideDenseActor(spec,policy,bank,mode=0);memory=PublicCorridorGuideMemory(spec,[0]);rows=[];contexts=[]
    for t,pos in enumerate([[.38,-.2,.605],[.38,-.2,.634],[.398,-.2,.634],[.444,-.18,.634]]):
        row={k:v[:1].copy() for k,v in p.items()};row['elapsed_seconds'][:]=t*.02;row['tcp_position'][:]=pos;row['reference_position'][:]=pos;row['tcp_rotation'][:]=np.diag([1.,-1.,-1.]);row['reference_rotation'][:]=row['tcp_rotation'];row['measured_wrench'][:]=0
        batch,_=actor.prepare(row);state=memory.get_state();value=memory.update(row);memory.restore(state);np.testing.assert_array_equal(value,memory.update(row));np.testing.assert_array_equal(value,memory.update(row))
        assert value.shape==(1,53) and int(np.argmax(value[0,22:26]))==int(batch['phase'][0])==[0,2,2,3][t]
        rows.append(row);contexts.append(value[0])
    stream={k:np.concatenate([r[k] for r in rows]) for k in rows[0]};np.testing.assert_array_equal(corridor_context_sequence(stream,spec,0),contexts)
    source=task_spec('T2',robot='panda');np.testing.assert_array_equal(corridor_context_sequence(p,source,0),context_sequence(p,source,0))
