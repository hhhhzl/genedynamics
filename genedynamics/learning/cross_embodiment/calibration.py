"""Budget-gated hidden target interactions and frozen-model context fitting.

The gate is an experimental information boundary, not a security sandbox.
Mechanism truth never appears in its returned observation or probe records.
Each reset plus probe rollout is charged once, before either can fail.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeRecord:
    attempt: int
    purpose: str
    observation: object
    commands: object
    result: object
    initial_state: object


class HiddenTargetEnv:
    def __init__(self, runner, observer, budget):
        self.__runner,self.__observer,self.budget=runner,observer,budget
        self.__sealed=False
        self.records=[]

    def probe(self,request,commands,*,purpose="fit",initial="clear"):
        if self.__sealed:
            raise RuntimeError("calibration sealed; final evaluation cannot feed calibration")
        if request is not None and len(request.sample_ids)!=1:
            raise ValueError("one charged probe is one world/reset/rollout; batch probes need individual reservations")
        attempt=self.budget.reserve(purpose)
        try:
            state=self.__runner.reset_batch(request) if initial=="clear" else self.__runner.reset_batch(request,initial=initial)
            observation=self.__observer.observe(self.__runner,state)
            result=self.__runner.collect_chunk(state,commands)
            self.__runner.synchronize(result)
            record=ProbeRecord(attempt,purpose,observation,commands,result,state)
            self.records.append(record)
            # Physical failure still consumes a complete attempt.
            import numpy as np
            status=np.asarray(result.state.status)
            self.budget.complete(attempt,"success" if np.all(status<=1) else "failed")
            return record
        except BaseException:
            if self.budget.attempts[attempt]["outcome"]=="pending":
                self.budget.complete(attempt,"failed")
            raise

    def probe_policy(self,request,policy,*,chunks,purpose="fit",initial="clear"):
        """One charged, feedback-driven experiment including its approach.

        The policy receives public observation/state only. All executed commands,
        failure steps and approach are retained in one complete trace. A failed
        or interrupted experiment consumes its reservation exactly as probe().
        """
        if self.__sealed: raise RuntimeError("calibration sealed")
        if len(request.sample_ids)!=1 or chunks<1: raise ValueError("one world and positive chunks required")
        attempt=self.budget.reserve(purpose)
        try:
            import numpy as np
            state=self.__runner.reset_batch(request,initial=initial)
            first=state; observation=self.__observer.observe(self.__runner,state)
            traces=[]; commands=[]; body_commands=[]
            for index in range(chunks):
                command=policy(self.__observer.observe(self.__runner,state),state,index)
                result=self.__runner.collect_chunk(state,command)
                state=result.state; traces.append(result.trace)
                commands.append(command.skill if hasattr(command,"skill") else command)
                body_commands.append(command.body_twist if hasattr(command,"body_twist") else None)
                if np.all(np.asarray(state.status)!=0): break
            self.__runner.synchronize(state)
            trace=type(traces[0])(*(np.concatenate([getattr(t,n) for t in traces],axis=1) for n in traces[0]._fields))
            result=type(result)(state,trace)
            executed=np.concatenate(commands,axis=1)
            if any(b is not None for b in body_commands):
                if any(b is None for b in body_commands):raise ValueError("probe changed body command interface between chunks")
                from .contact import ContactCommands
                executed=ContactCommands(executed,np.concatenate(body_commands,axis=1))
            record=ProbeRecord(attempt,purpose,observation,executed,result,first)
            self.records.append(record)
            self.budget.complete(attempt,"success" if np.all(np.asarray(state.status)<=1) else "failed")
            return record
        except BaseException:
            if self.budget.attempts[attempt]["outcome"]=="pending": self.budget.complete(attempt,"failed")
            raise

    def seal(self):
        self.__sealed=True

    def evaluate(self,request,policy,*,chunks):
        if not self.__sealed:
            raise RuntimeError("seal calibration before final evaluation")
        state=self.__runner.reset_batch(request)
        traces=[]
        for index in range(chunks):
            observation=self.__observer.observe(self.__runner,state)
            result=self.__runner.evaluate_chunk(state,policy(observation,state,index))
            state=result.state; traces.append(result.trace)
        self.__runner.synchronize(state)
        return state,traces


def fit_context(response,base_inputs,targets,weights,*,initial=None,updates=200,learning_rate=.03,prior_weight=.001):
    """Only context is differentiated; all model weights remain frozen.

    base_inputs contains public carry/commands/body descriptor, omitting the
    last context_dim coordinates. All probe outcomes are retained; weights
    explicitly select fit rows. Validation rows are supplied with zero weight.
    """
    if response.model.backend=="torch":
        return _fit_context_torch(response,base_inputs,targets,weights,initial,updates,learning_rate,prior_weight)
    if response.model.backend!="jax": raise NotImplementedError("native context fitting is not supported")
    import jax
    import jax.numpy as jnp
    from .backends.jax.models import adam_update
    from .models import TrainState
    dim=response.metadata["layout"]["context_dim"]
    initial=jnp.zeros(dim,jnp.float32) if initial is None else jnp.asarray(initial,dtype=jnp.float32)
    if initial.shape!=(dim,): raise ValueError("context dimension mismatch")
    inputs,targets,weights=map(jnp.asarray,(base_inputs,targets,weights))
    bounds=jnp.asarray(response.metadata.get("context_bounds",[[-2.,2.]]*dim),jnp.float32)
    if bounds.shape!=(dim,2): raise ValueError("context bounds layout mismatch")
    initial=jnp.clip(initial,bounds[:,0],bounds[:,1])
    def loss(context):
        value=jnp.concatenate((inputs,jnp.broadcast_to(context,(inputs.shape[0],dim))),axis=-1)
        prediction=response.predict(value)
        error=(prediction-targets[None])/response.oscale
        data=jnp.sum(error**2*weights[None,:,None])/jnp.maximum(weights.sum()*error.shape[-1]*error.shape[0],1.)
        return data+prior_weight*jnp.mean(context**2)
    def step(state,_):
        value,gradient=jax.value_and_grad(loss)(state.parameters)
        updated=adam_update(state,gradient,learning_rate)
        # Keep fitting in the public-development context box.
        updated=updated._replace(parameters=jnp.clip(updated.parameters,bounds[:,0],bounds[:,1]))
        updated=jax.tree.map(lambda new,old:jnp.where(weights.sum()>0,new,old),updated,state)
        return updated,value
    state=TrainState(initial,jnp.zeros_like(initial),jnp.zeros_like(initial),jnp.int32(0))
    state,history=jax.jit(lambda s:jax.lax.scan(step,s,None,length=updates))(state)
    return state.parameters,history


def _fit_context_torch(response,inputs,targets,weights,initial,updates,rate,prior_weight):
    import torch
    convert=response.model._impl._tensor
    inputs,targets,weights=map(convert,(inputs,targets,weights))
    dim=response.metadata["layout"]["context_dim"]
    context=convert([0.]*dim) if initial is None else convert(initial)
    bounds=convert(response.metadata.get("context_bounds",[[-2.,2.]]*dim))
    if tuple(bounds.shape)!=(dim,2): raise ValueError("context bounds layout mismatch")
    context=torch.maximum(torch.minimum(context,bounds[:,1]),bounds[:,0])
    first=torch.zeros_like(context); second=torch.zeros_like(context); history=[]
    for step in range(1,updates+1):
        context=context.detach().requires_grad_()
        value=torch.cat((inputs,context.expand(len(inputs),dim)),dim=-1)
        error=(response.predict(value)-targets[None])/response.oscale
        data=(error.square()*weights[None,:,None]).sum()/(weights.sum()*error.shape[-1]*error.shape[0]).clamp(min=1.)
        loss=data+prior_weight*context.square().mean()
        gradient=torch.autograd.grad(loss,context)[0]
        first=.9*first+.1*gradient; second=.999*second+.001*gradient.square()
        new=context-rate*(first/(1-.9**step))/(torch.sqrt(second/(1-.999**step))+1e-8)
        new=torch.maximum(torch.minimum(new,bounds[:,1]),bounds[:,0])
        context=torch.where(weights.sum()>0,new,context).detach(); history.append(loss.detach())
    return context,torch.stack(history)
