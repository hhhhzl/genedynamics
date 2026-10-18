"""Device-only source-query → prior → transport → MJX outer scan."""
import jax
import jax.numpy as jp
from mujoco.mjx._src import support
from ...cross_ebdm import PolicyChunk
from genedynamics.envs.domains.manipulation.transfer_workcell.queries.source import TaskSemantics
from genedynamics.solvers.single.cross_ebdm.learning.learning_data import online_prior_condition,online_response_carry
from genedynamics.solvers.single.cross_ebdm.learning.transport import TransportBatch
from genedynamics.envs.domains.manipulation.transfer_workcell.backends.jax.execution import _fp32_scope


class JaxPolicy:
    def __init__(self,runner,query,prior,transport,parameters):
        if runner.execution.backend!="jax" or query.execution.backend!="jax" or prior.regressor.model.backend!="jax":
            raise ValueError("fused JAX policy requires JAX execution/query/prior")
        if (transport is None)!=(parameters is None): raise ValueError("transport and parameters must be supplied together")
        self.runner,self.query,self.prior=runner._impl,query._impl,prior
        self.transport,self.parameters=transport,parameters
        self._run=jax.jit(self._chunk,device=self.runner.device)

    def initial_source_seed(self,batch):
        return jp.broadcast_to(self.query.runner.start_qpos,(batch,self.query.runner.scene.model.nq))

    def _observe_one(self,model,data):
        runner=self.runner
        p,r=support.jac(model,data,data.site_xpos[runner.sid],runner.body)
        return TaskSemantics(data.site_xpos[runner.sid],data.site_xmat[runner.sid],jp.concatenate((p.T@data.qvel,r.T@data.qvel)),jp.int32(0))

    def _chunk(self,state,seed,modes,prior_noise,transport_noise,context):
        runner=self.runner
        def window(carry,noises):
            state,seed=carry
            semantics=jax.vmap(self._observe_one,in_axes=(0 if runner.heterogeneous else None,0))(
                runner.models if runner.heterogeneous else runner.model,state.physics)
            result=jax.vmap(self.query._one,in_axes=(None,0,0,0))(self.query.runner.model,semantics,state.sample_id,seed)
            failed=(state.status==0)&~result.support
            state=state._replace(status=jp.where(failed,7,state.status))
            condition=online_prior_condition(result.semantics,runner.goal,runner.goal_rotation,
                state.steps*runner.task.control_dt,jp)
            commands=self.prior.sample(condition,modes,noises[0])
            if self.transport is not None:
                batch=TransportBatch(online_response_carry(result.virtual_state,result.semantics,jp),
                    online_response_carry(state,semantics,jp),commands,context,result.support.astype(jp.float32))
                commands=self.transport._impl._apply(self.parameters,batch,noises[1])
            executed=(runner._collect_batched(runner.models,runner.gain,runner.delay,state,commands) if runner.heterogeneous
                      else runner._collect(runner.model,runner.gain,runner.delay,state,commands))
            return (executed.state,result.virtual_state.physics.qpos),(executed.trace,result.support,failed,commands)
        (state,seed),(trace,ok,failed,commands)=jax.lax.scan(window,(state,seed),(prior_noise,transport_noise))
        return PolicyChunk(state,seed,trace,ok,failed,commands)

    @_fp32_scope
    def run_chunk(self,state,seed,modes,prior_noise,transport_noise,context):
        b=state.sample_id.shape[0]; h=self.prior.horizon
        if prior_noise.ndim!=4 or prior_noise.shape[1:]!=(b,h,12) or transport_noise.shape!=prior_noise.shape:
            raise ValueError("noise must be [K,B,H,12] for independent explicit streams")
        if context.shape!=(b,3) or modes.shape!=(b,): raise ValueError("context/mode batch mismatch")
        return self._run(state,seed,modes,prior_noise,transport_noise,context)

    @staticmethod
    def synchronize(value): jax.block_until_ready(value)
