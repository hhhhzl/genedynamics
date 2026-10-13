"""Fused JAX CPU learned policy/MJX smoke with strict transfer guard after warmup."""
import argparse
from pathlib import Path
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from genedynamics.learning.cross_embodiment import CrossEmbodiment,ExecutionSpec,ObstacleTransferConfig,RandomRequest
from genedynamics.learning.cross_embodiment.source_query import SourceQueryAdapter
from genedynamics.learning.cross_embodiment.learned import FrozenRegressor,GaussianSkillPrior
from genedynamics.learning.cross_embodiment.solver import CrossEmbodimentSolver
from genedynamics.learning.cross_embodiment.core.artifacts import save_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,default=Path("tmp/cross_embodiment/learned_cpu_v2"))
    parser.add_argument("--output",type=Path,default=Path("tmp/cross_embodiment/device_pipeline"))
    parser.add_argument("--windows",type=int,default=2)
    parser.add_argument("--transport",type=Path)
    args=parser.parse_args()
    execution=ExecutionSpec(); task=ObstacleTransferConfig(robot="xarm7",timeout_seconds=30.,reset_joint_noise=0.)
    runner=CrossEmbodiment(execution=execution,task=task)
    query=SourceQueryAdapter(execution=execution)
    prior=GaussianSkillPrior(FrozenRegressor.load(args.models/"source_prior"))
    transport=parameters=None
    if args.transport:
        from genedynamics.learning.cross_embodiment.transport import StochasticTransport
        transport,parameters=StochasticTransport.load(FrozenRegressor.load(args.models/"response"),args.transport)
    solver=CrossEmbodimentSolver(runner,query,prior,transport=transport,parameters=parameters)
    state=runner.reset_batch(RandomRequest(7,(71,))); seed=solver.initial_source_seed(1)
    modes=jnp.array([0],jnp.int32); context=jnp.zeros((1,3),jnp.float32)
    pnoise=jax.random.normal(jax.random.PRNGKey(8),(args.windows,1,prior.horizon,12))
    tnoise=jax.random.normal(jax.random.PRNGKey(9),pnoise.shape)
    start=time.perf_counter(); result=solver.run_chunk(state,seed,modes,pnoise,tnoise,context); solver.synchronize(result)
    warm=time.perf_counter()-start
    with jax.transfer_guard("disallow"):
        start=time.perf_counter(); replay=solver.run_chunk(state,seed,modes,pnoise,tnoise,context); solver.synchronize(replay)
        duration=time.perf_counter()-start
    np.testing.assert_array_equal(result.state.physics.qpos,replay.state.physics.qpos)
    report={"compile_and_warmup_seconds":warm,"warm_chunk_seconds":duration,"windows":args.windows,
        "support":np.asarray(result.support).tolist(),"status":np.asarray(result.state.status).tolist(),
        "transfer_guard":"passed","scope":"short fused learned-policy/MJX chunk; GPU not_run"}
    save_bundle(args.output,{"commands":result.commands,"tcp":result.trace.tcp_position,"support":result.support,
        "prior_noise":pnoise,"transport_noise":tnoise,"mode":modes,"context":context,
        "initial_qpos":state.physics.qpos,"final_qpos":result.state.physics.qpos},report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
