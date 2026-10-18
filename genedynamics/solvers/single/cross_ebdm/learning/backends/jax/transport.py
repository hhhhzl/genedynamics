"""Bounded stochastic skills, frozen response gradients, global risk reduction."""
import jax
import jax.numpy as jnp
from ...models import LearnedModel,ModelSpec
from .models import forward,adam_update
from .risk import microbatch_value_and_grad


class JaxTransport:
    def __init__(self,response,spec,device):
        if response.model.backend!="jax": raise ValueError("JAX transport requires JAX response weights")
        self.response,self.spec=response,spec
        d=response.metadata["layout"]["carry_dim"]
        self.carry_dim=d
        self.semantic_slice=slice(*response.metadata["layout"].get("semantic_slice",[14,32]))
        self.active_dim=response.metadata.get("active_command_dim",12)
        self.model=LearnedModel(ModelSpec(d+spec.horizon*self.active_dim*2+3,spec.horizon*self.active_dim,
                                        hidden=(64,64),learning_rate=spec.learning_rate),device=device)
        self.bounds=jnp.array(response.metadata.get("command_bounds",[.12]*3+[.6]*3+[1.]*6),jnp.float32)
        self.apply=jax.jit(self._apply,device=self.model._impl.device)
        self.train_step=jax.jit(self._step,device=self.model._impl.device)
        self.train_chunk=jax.jit(self._chunk,device=self.model._impl.device)

    def initialize(self,seed):
        state=self.model.initialize(seed)
        # Small nonzero final weights preserve transport-noise sensitivity and
        # hidden-layer gradients while starting near the source action law.
        parameters=(*state.parameters[:-1],tuple(x*.01 for x in state.parameters[-1]))
        return state._replace(parameters=parameters)

    def _apply(self,parameters,batch,noise):
        b=batch.valid.shape[0]
        source=batch.source_commands[...,:self.active_dim]/self.bounds
        inputs=jnp.concatenate((batch.target_carry,source.reshape(b,-1),batch.context,noise.reshape(b,-1)),axis=-1)
        residual=jnp.tanh(forward(parameters,inputs)[0]).reshape(b,self.spec.horizon,self.active_dim)
        # Floor zero reproduces the legacy gate. A positive floor permits
        # target equilibrium corrections after source feed-forward stops;
        # actual-state feedback, rather than this gate, must stabilize them.
        envelope=jnp.concatenate((jnp.clip(jnp.abs(source[...,:6])*5.,self.spec.motion_residual_floor,1.),jnp.ones_like(source[...,6:])),axis=-1)
        residual=residual*envelope
        value=jnp.clip(source+self.spec.residual_scale*residual,-1.,1.)*self.bounds
        if self.active_dim==13:
            force=jnp.where(batch.source_commands[...,12]>0,jnp.maximum(value[...,12],0.),0.)
            return value.at[...,12].set(force)
        return jnp.concatenate((value,jnp.zeros((b,self.spec.horizon,1),jnp.float32)),axis=-1)

    def _loss(self,parameters,values):
        batch,noise=values
        commands=self._apply(parameters,batch,noise)
        b=batch.valid.shape[0]
        source_body=jnp.broadcast_to(jnp.array([1.,0.]),(b,2)) if batch.source_body is None else batch.source_body
        target_body=jnp.broadcast_to(jnp.array([0.,1.]),(b,2)) if batch.target_body is None else batch.target_body
        source_inputs=jnp.concatenate((batch.source_carry,batch.source_commands.reshape(b,-1),source_body,jnp.zeros((b,3))),axis=-1)
        target_inputs=jnp.concatenate((batch.target_carry,commands.reshape(b,-1),target_body,batch.context),axis=-1)
        # The exact same sampled U_s enters both source response and transport.
        source=jax.lax.stop_gradient(self.response.predict(source_inputs))
        target=self.response.predict(target_inputs)
        source_semantic=batch.source_carry[None,:,self.semantic_slice]+source[:,:,self.semantic_slice]
        target_semantic=batch.target_carry[None,:,self.semantic_slice]+target[:,:,self.semantic_slice]
        scale=jnp.array([.01]*3+[.1]*9+[.1]*6)
        matching=jnp.mean(((target_semantic-source_semantic)/scale)**2,axis=(0,2))
        if "force_index" in self.response.metadata["layout"]:
            index=self.response.metadata["layout"]["force_index"]
            error=(batch.target_carry[None,:,index]+target[:,:,index]-batch.source_carry[None,:,index]-source[:,:,index])/5.
            matching+=jnp.mean(error**2,axis=0)
        if self.spec.task_weight:
            if batch.task_reference is None: raise ValueError("task loss requires a phase task_reference")
            pose_error=(target_semantic[...,:12]-batch.task_reference[None,:,:12])/scale[:12]
            task=jnp.mean(pose_error**2,axis=(0,2))
            if "force_index" in self.response.metadata["layout"]:
                actual=batch.target_carry[None,:,index]+target[:,:,index]
                task+=jnp.mean(((actual-batch.task_reference[None,:,12])/5.)**2,axis=0)
            matching+=self.spec.task_weight*task
        change=jnp.mean(((commands[...,:self.active_dim]-batch.source_commands[...,:self.active_dim])/self.bounds)**2,axis=(1,2))
        if self.response.metadata.get("risk_kind")=="contact":
            risk=jnp.maximum(target[:,:,-3]/25.-1.,0.)+jnp.maximum(target[:,:,-2],0.)+jnp.maximum(target[:,:,-1],0.)
        else:
            risk=jnp.maximum((.002-target[:,:,-3])/.002,0.)+jnp.maximum(target[:,:,-2]-2.5,0.)+jnp.maximum(target[:,:,-1]-1.,0.)
        return matching+self.spec.change_weight*change,risk.T

    def _step(self,state,batch,noise):
        loss,gradient=microbatch_value_and_grad(self._loss,state.parameters,(batch,noise),batch.valid,
            microbatch_size=self.spec.microbatch_size,tail_fraction=self.spec.tail_fraction,multiplier=self.spec.risk_multiplier)
        updated=adam_update(state,gradient,self.spec.learning_rate)
        updated=jax.tree.map(lambda new,old:jnp.where(batch.valid.sum()>0,new,old),updated,state)
        return updated,loss

    def _chunk(self,state,batches,noises):
        return jax.lax.scan(lambda s,v:self._step(s,*v),state,(batches,noises))
