"""Device-resident learned contact policy; explicit skill/body command streams."""
import jax
import jax.numpy as jp
from ...solver import PolicyChunk
from ...contact_query import observe_contact
from ...contact_data import contact_carry,pad_contact_carry,body_descriptor
from ...contact import ContactCommands
from ...learning_data import online_prior_condition
from ...transport import TransportBatch
from .execution import _fp32_scope


class JaxContactPolicy:
    def __init__(self,runner,query,prior,transport,parameters):
        if not hasattr(runner._impl,"_chunk_one") or not hasattr(query._impl,"_one"):
            raise ValueError("fused contact policy requires JAX execution and query")
        if prior.regressor.model.backend!="jax": raise ValueError("JAX prior required")
        if (transport is None)!=(parameters is None): raise ValueError("transport and parameters must be supplied together")
        self.runner,self.query,self.prior,self.transport,self.parameters=runner,query,prior,transport,parameters
        with jax.default_device(runner._impl.device):
            self.goal=jp.asarray(runner.config.goal,jp.float32)
            self.rotation=jp.diag(jp.array([1.,-1.,-1.],jp.float32))
            self.source_body=jp.asarray(body_descriptor(query.source.config))
            self.target_body=jp.asarray(body_descriptor(runner.config))
        self._run=jax.jit(self._chunk,device=runner._impl.device)

    def initial_source_seed(self,batch):
        return jp.broadcast_to(self.query.source._impl.start,(batch,self.query.source._impl.scene.model.nq))

    def _chunk(self,state,seed,modes,prior_noise,transport_noise,context,body_commands=None):
        r=self.runner._impl; source=self.query.source._impl; b=state.sample_id.shape[0]
        def window(previous,noise):
            state,seed=previous
            semantic=observe_contact(state,self.runner,jp)
            # A legal source controller starts fresh each query; target carry
            # and contact integrators never enter this source reset.
            base=jax.vmap(source._reset_one,in_axes=(None,0,0))(source.start,state.sample_id,jp.zeros((b,source.start.size)))
            result=jax.vmap(self.query._impl._one)(semantic,base,seed)
            failed=(state.status==0)&~result.support
            state=state._replace(status=jp.where(failed,7,state.status))
            source_semantic=observe_contact(result.state,self.query.source,jp)
            condition=online_prior_condition(source_semantic,self.goal,self.rotation,state.steps*self.runner.config.control_dt,jp)
            commands=self.prior.sample(condition,modes,noise[0])
            if self.transport is not None:
                d=self.transport._impl.response.metadata["layout"]["carry_dim"]
                batch=TransportBatch(pad_contact_carry(contact_carry(result.state,self.query.source,jp),d,jp),pad_contact_carry(contact_carry(state,self.runner,jp),d,jp),commands,
                    context,result.support.astype(jp.float32),jp.broadcast_to(self.source_body,(b,3)),jp.broadcast_to(self.target_body,(b,3)))
                commands=self.transport._impl._apply(self.parameters,batch,noise[1])
            # Independent supplied body stream, or known fixture intention.
            body=jp.zeros((b,self.prior.horizon,3),jp.float32)
            if body_commands is not None: body=noise[2]
            elif self.runner.config.mobile:
                body=body.at[...,0].set(jp.where((state.steps*self.runner.config.control_dt<4.)[:,None],.005,0.))
            executed=r._collect_batched(r.models,r.gains,r.delays,state,commands,body) if r.heterogeneous else r._collect(state,commands,body)
            return (executed.state,result.state.physics.qpos),(executed.trace,result.support,failed,commands)
        (state,seed),(trace,ok,failed,commands)=jax.lax.scan(window,(state,seed),(prior_noise,transport_noise,body_commands))
        return PolicyChunk(state,seed,trace,ok,failed,commands)

    @_fp32_scope
    def run_chunk(self,state,seed,modes,prior_noise,transport_noise,context,*,body_commands=None):
        b=state.sample_id.shape[0]; h=self.prior.horizon
        if prior_noise.ndim!=4 or prior_noise.shape[1:]!=(b,h,13) or transport_noise.shape!=prior_noise.shape:
            raise ValueError("contact noise must be [K,B,H,13]")
        if context.shape!=(b,3) or modes.shape!=(b,): raise ValueError("context/mode batch mismatch")
        if body_commands is not None and body_commands.shape!=(*prior_noise.shape[:3],3): raise ValueError("body commands must be [K,B,H,3]")
        return self._run(state,seed,modes,prior_noise,transport_noise,context,body_commands)

    @staticmethod
    def synchronize(value): jax.block_until_ready(value)
