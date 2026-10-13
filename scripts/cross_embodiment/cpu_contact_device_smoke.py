"""Fused contact query/prior/transport/MJX CPU chunk with strict transfer guard."""
import argparse
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactConfig
from genedynamics.learning.cross_embodiment.contact_query import ContactSourceQuery
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,GaussianSkillPrior
from genedynamics.learning.cross_embodiment.transport import StochasticTransport
from genedynamics.learning.cross_embodiment.solver import CrossEmbodimentSolver
from genedynamics.learning.cross_embodiment import RandomRequest
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle
from genedynamics.learning.cross_embodiment.random_streams import jax_normal,RandomStream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True); p.add_argument('--transport',type=Path)
    p.add_argument('--task',default='guided_contact_alignment'); p.add_argument('--robot',default='xarm7')
    p.add_argument('--windows',type=int,default=1); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); c=ContactConfig(environment=a.task,robot=a.robot,mobile=a.task.startswith('mobile'))
    r=ContactRunner(config=c,backend='jax'); q=ContactSourceQuery(c,backend='jax')
    prior=GaussianSkillPrior(FrozenRegressor.load(a.models/'source_prior'))
    transport=parameters=None
    if a.transport: transport,parameters=StochasticTransport.load(FrozenRegressor.load(a.models/'response'),a.transport)
    solver=CrossEmbodimentSolver(r,q,prior,transport=transport,parameters=parameters)
    state=r.reset_batch(RandomRequest(7,(71,))); seed=solver.initial_source_seed(1)
    mode=jp.zeros(1,jp.int32); context=jp.zeros((1,3),jp.float32)
    noise=lambda stream:jax.vmap(lambda k:jax_normal(8,state.sample_id,k,stream,(10,13)))(jp.arange(a.windows))
    pn,tn=noise(RandomStream.PRIOR_SKILL),noise(RandomStream.TRANSPORT)
    start=time.perf_counter(); result=solver.run_chunk(state,seed,mode,pn,tn,context); solver.synchronize(result); warm=time.perf_counter()-start
    with jax.transfer_guard('disallow'):
        start=time.perf_counter(); replay=solver.run_chunk(state,seed,mode,pn,tn,context); solver.synchronize(replay); duration=time.perf_counter()-start
    np.testing.assert_array_equal(result.state.physics.qpos,replay.state.physics.qpos)
    report={'task':a.task,'robot':a.robot,'transfer_guard':'passed','warmup_seconds':warm,'warm_chunk_seconds':duration,
            'support':np.asarray(result.support).tolist(),'status':np.asarray(result.state.status).tolist(),
            'scope':'short fused contact device chunk; physical full-task results recorded separately','gpu':'not_run'}
    save_bundle(a.output,{'commands':result.commands,'tcp':result.trace.tcp_position,'support':result.support,
        'prior_noise':pn,'transport_noise':tn,'mode':mode,'context':context,
        'initial_qpos':state.physics.qpos,'final_qpos':result.state.physics.qpos},report)
    print(report)


if __name__=='__main__':main()
