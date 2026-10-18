"""Torch reference transport; full-batch CPU gradients and one-update conformance.

No claim of compiled training chunks, CUDA or memory-bounded microbatch training.
"""
import torch
from ...models import LearnedModel,ModelSpec,TrainState
from .models import forward


class TorchTransport:
    def __init__(self,response,spec,device):
        if response.model.backend!="torch": raise ValueError("Torch transport requires Torch response")
        self.response,self.spec=response,spec
        d=response.metadata["layout"]["carry_dim"]
        self.active_dim=response.metadata.get("active_command_dim",12)
        self.semantic_slice=slice(*response.metadata["layout"].get("semantic_slice",[14,32]))
        self.model=LearnedModel(ModelSpec(d+spec.horizon*self.active_dim*2+3,spec.horizon*self.active_dim,hidden=(64,64),learning_rate=spec.learning_rate),backend="torch",device=device)
        self.bounds=self.model._impl._tensor(response.metadata.get("command_bounds",[.12]*3+[.6]*3+[1.]*6))

    def initialize(self,seed):
        state=self.model.initialize(seed)
        return state._replace(parameters=(*state.parameters[:-1],tuple(x*.01 for x in state.parameters[-1])))

    def apply(self,parameters,batch,noise):
        b=batch.valid.shape[0]; source=batch.source_commands[...,:self.active_dim]/self.bounds
        inputs=torch.cat((batch.target_carry,source.reshape(b,-1),batch.context,noise.reshape(b,-1)),dim=-1)
        residual=torch.tanh(forward(parameters,inputs)[0]).reshape(b,self.spec.horizon,self.active_dim)
        envelope=torch.cat(((source[...,:6].abs()*5.).clamp(self.spec.motion_residual_floor,1.),torch.ones_like(source[...,6:])),dim=-1)
        value=(source+self.spec.residual_scale*residual*envelope).clamp(-1.,1.)*self.bounds
        if self.active_dim==13:
            force=torch.where(batch.source_commands[...,12]>0,value[...,12].clamp(min=0.),0.)
            return torch.cat((value[...,:12],force[...,None]),dim=-1)
        return torch.cat((value,torch.zeros((b,self.spec.horizon,1),device=value.device,dtype=value.dtype)),dim=-1)

    def _loss(self,parameters,batch,noise):
        commands=self.apply(parameters,batch,noise); b=batch.valid.shape[0]
        tensor=self.model._impl._tensor
        source_body=tensor([1.,0.]).expand(b,2) if batch.source_body is None else batch.source_body
        target_body=tensor([0.,1.]).expand(b,2) if batch.target_body is None else batch.target_body
        source_inputs=torch.cat((batch.source_carry,batch.source_commands.reshape(b,-1),source_body,torch.zeros_like(batch.context)),dim=-1)
        target_inputs=torch.cat((batch.target_carry,commands.reshape(b,-1),target_body,batch.context),dim=-1)
        source=self.response.predict(source_inputs).detach(); target=self.response.predict(target_inputs)
        section=self.semantic_slice
        error=(batch.target_carry[None,:,section]+target[:,:,section]-batch.source_carry[None,:,section]-source[:,:,section])/tensor([.01]*3+[.1]*9+[.1]*6)
        matching=error.square().mean(dim=(0,2))
        if "force_index" in self.response.metadata["layout"]:
            index=self.response.metadata["layout"]["force_index"]
            matching+=((batch.target_carry[None,:,index]+target[:,:,index]-batch.source_carry[None,:,index]-source[:,:,index])/5.).square().mean(dim=0)
        if self.spec.task_weight:
            if batch.task_reference is None: raise ValueError("task loss requires a phase task_reference")
            predicted=batch.target_carry[None,:,section]+target[:,:,section]
            task=((predicted[...,:12]-batch.task_reference[None,:,:12])/tensor([.01]*3+[.1]*9)).square().mean(dim=(0,2))
            if "force_index" in self.response.metadata["layout"]:
                task+=((batch.target_carry[None,:,index]+target[:,:,index]-batch.task_reference[None,:,12])/5.).square().mean(dim=0)
            matching+=self.spec.task_weight*task
        change=((commands[...,:self.active_dim]-batch.source_commands[...,:self.active_dim])/self.bounds).square().mean(dim=(1,2))
        if self.response.metadata.get("risk_kind")=="contact":
            risk=(target[:,:,-3]/25.-1.).clamp(min=0.)+target[:,:,-2].clamp(min=0.)+target[:,:,-1].clamp(min=0.)
        else:
            risk=((.002-target[:,:,-3])/.002).clamp(min=0.)+(target[:,:,-2]-2.5).clamp(min=0.)+(target[:,:,-1]-1.).clamp(min=0.)
        return matching+self.spec.change_weight*change,risk.T

    def train_step(self,state,batch,noise):
        parameters=tuple(tuple(x.detach().requires_grad_() for x in layer) for layer in state.parameters)
        objective,risk=self._loss(parameters,batch,noise); risk=risk.amax(dim=-1)
        valid=batch.valid.bool(); mass=(valid.sum()*self.spec.tail_fraction).clamp(min=1e-8)
        sorted_risk=torch.sort(torch.where(valid,risk,torch.full_like(risk,-torch.inf)),descending=True).values
        threshold=sorted_risk[(torch.ceil(mass).to(torch.int64)-1).clamp(0,risk.numel()-1)]
        above=valid&(risk>threshold); tied=valid&(risk==threshold)
        weight=(above.to(risk.dtype)+tied.to(risk.dtype)*(mass-above.sum()).clamp(min=0.)/tied.sum().clamp(min=1))/mass
        weight=torch.where(valid.sum()>0,weight,torch.zeros_like(weight)).detach()
        loss=(objective*batch.valid).sum()/batch.valid.sum().clamp(min=1.)+self.spec.risk_multiplier*(risk*weight).sum()
        flat=tuple(x for layer in parameters for x in layer)
        gradient=torch.autograd.grad(loss,flat)
        first=tuple(.9*m+.1*g for m,g in zip((x for l in state.first for x in l),gradient))
        second=tuple(.999*v+.001*g*g for v,g in zip((x for l in state.second for x in l),gradient))
        step=state.step+1
        updated=tuple((p-self.spec.learning_rate*(m/(1-.9**step))/(torch.sqrt(v/(1-.999**step))+1e-8)).detach() for p,m,v in zip(flat,first,second))
        paired=lambda xs:tuple((xs[i],xs[i+1]) for i in range(0,len(xs),2))
        choose=lambda new,old:tuple(tuple(torch.where(valid.any(),n,o) for n,o in zip(nl,ol)) for nl,ol in zip(new,old))
        return TrainState(choose(paired(updated),state.parameters),choose(paired(first),state.first),choose(paired(second),state.second),torch.where(valid.any(),step,state.step)),loss.detach()

    def train_chunk(self,state,batches,noises):
        losses=[]
        for i in range(noises.shape[0]):
            state,loss=self.train_step(state,type(batches)(*(None if x is None else x[i] for x in batches)),noises[i]); losses.append(loss)
        return state,torch.stack(losses)
