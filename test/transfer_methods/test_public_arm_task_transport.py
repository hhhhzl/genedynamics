"""Native T1/T2 limiter parity and task-explicit local action support."""
import numpy as np
import torch
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.task_response import project_task_requests
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec,BOUNDS
from genedynamics.tasks.transfer.runtime import SkillSpec
from genedynamics.core.control.transfer import bounded_command
from genedynamics.tasks.transfer.contact import bounded_contact_command
from genedynamics.tasks.transfer.specs import task_spec


def inputs(task,n=2):
    spec=PublicResponseSpec(hidden=(8,),members=1);f=torch.zeros(n,spec.feature_dim);f[:,-7+(task=='T2')]=1
    return spec,f


def test_known_task_issuance_matches_both_native_limiters_in_original_units():
    rng=np.random.default_rng(95000)
    for task in ('T1','T2'):
        spec,f=inputs(task);caps=np.asarray([.12]*3+[.6]*3+[1.]*6+[0.]) if task=='T1' else BOUNDS;previous=rng.uniform(-1,1,(2,13))*caps;previous[:,12]=np.abs(previous[:,12]);commands=rng.uniform(-2,2,(2,6,13))*caps;commands[:,:,12]=np.abs(commands[:,:,12]);offset=(spec.history_len-1)*(spec.frame_dim+1)+60;f[:,offset:offset+13]=torch.tensor(previous/BOUNDS,dtype=torch.float32)
        actual=project_task_requests(f,torch.tensor(commands/BOUNDS,dtype=torch.float32),spec).numpy();expected=[]
        for i in range(2):
            value=previous[i].astype(np.float32);rows=[]
            for raw in commands[i].astype(np.float32):
                value=bounded_command(np,raw,value,SkillSpec(),.02) if task=='T1' else bounded_contact_command(np,raw,value,task_spec('T2',robot='xarm7').config);chart=value/BOUNDS;chart[12]+=(2 if task=='T2' and raw[12]>.1 else 0);rows.append(chart.copy())
            expected.append(rows)
        np.testing.assert_allclose(actual,np.asarray(expected),atol=3e-7,rtol=2e-6)


def test_task_transport_preserves_T1_box_and_force_off_and_checkpoint():
    for task in ('T1','T2'):
        spec,f=inputs(task);model=PublicTaskResidualTransport(spec,PublicTransportSpec(residual_scale=.025,hidden=(8,)));base=torch.zeros(2,10,13);base[:,:,:6]=1.49 if task=='T1' else .99
        with torch.no_grad():model.network[-1].bias.fill_(100.)
        y=model(f,f,torch.zeros_like(base),base,torch.zeros(2,dtype=torch.long),torch.zeros(2,9));bound=1.5 if task=='T1' else 1.
        assert torch.all(y[:,:,:6]<=bound) and torch.all(y[:,:,:6]>=base[:,:,:6]) and torch.all(y[:,:,12]==0)
        loaded=PublicTaskResidualTransport.from_checkpoint(model.checkpoint());torch.testing.assert_close(y,loaded(f,f,torch.zeros_like(base),base,torch.zeros(2,dtype=torch.long),torch.zeros(2,9)),rtol=0,atol=0)


def test_ordinary_adam_mpc_risk_gate_counts_and_cached_tick():
    from types import SimpleNamespace
    from genedynamics.solvers.single.response_mpc.arm import PublicArmTaskAdamMPCActor
    from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import task_proxy
    class Model(torch.nn.Module):
        def forward(self,f,commands):
            mean=commands.mean(1);zero=mean[:,0:1]*0.;outputs=torch.cat((mean[:,:1],zero.expand(-1,17),3.5+20*mean[:,:1],zero),dim=-1)
            return outputs[None]
    for task in ('T1','T2'):
        spec,f=inputs(task,1);base=torch.zeros(1,10,13);current=dict(position=torch.zeros(1,3),rotation=torch.eye(3)[None],wrench=torch.zeros(1,6));reference=torch.cat((torch.tensor([[.00025,0.,0.]]),torch.eye(3).reshape(1,9),torch.zeros(1,1)),dim=1)
        actor=PublicArmTaskAdamMPCActor.__new__(PublicArmTaskAdamMPCActor);actor.spec=SimpleNamespace(config=SimpleNamespace(control_dt=.02));actor.device=torch.device('cpu');actor.response=Model();actor.last_action_tick=-1;actor.cached_action=None;actor.fallback_controls=0;actor.irreversible_model_calls=0;actor.irreversible_action_updates=0;actor.model_acceptance_fallbacks=0;actor.maximum_query_distance=5.
        batch=dict(target_features=f.numpy(),base_proto=base.numpy(),current={k:v.numpy() for k,v in current.items()},task_reference=reference.numpy(),source_query_distance=np.array([0.]));actor.prepare=lambda _: (batch,np.zeros((1,13),np.float32))
        p=dict(elapsed_seconds=np.array([[0.]],np.float32));action=actor.act(p);assert actor.irreversible_model_calls==6 and actor.irreversible_action_updates==5;assert action[0,0]<0 and action[0,12]==0
        assert np.max(np.abs(action[0,:3]))<=.025*(.12 if task=='T1' else .08)+1e-9
        np.testing.assert_array_equal(action,actor.act(p));assert actor.irreversible_model_calls==6
        def objective(value):
            pred=actor.response(f,value);peaks=torch.expm1(pred[...,18:20]).clamp(min=0);risk=((peaks[...,0]/25-1).clamp(min=0)**2+(peaks[...,1]/10-1).clamp(min=0)**2).max();return task_proxy(pred,current,reference).mean()+risk+.01*((value-base)**2).mean()
        selected=base.clone();selected[:,0]=torch.tensor(action/BOUNDS)
        # All H slots are optimized identically in this symmetric model.
        selected[:]=selected[:,:1].clone()
        assert float(objective(selected))<=float(objective(base))
