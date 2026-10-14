"""GPU model search for the finite study; CPU validates every selected map.

This worker only accepts model-training roles. Actual target episodes and
source demonstrations cannot enter this entrypoint.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from scripts.cross_embodiment.run_p2 import prepare,read,gpu_report
from genedynamics.experiments.artifacts import atomic_json


def collect_gpu(jobs,root,batch_size,device):
    import numpy as np
    import jax
    from genedynamics.learning.cross_embodiment.contact import ContactConfig,ContactMechanismSpec,ContactRunner
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.finite_transport import make_mapping
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle,save_bundle
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    from genedynamics.learning.cross_embodiment.backends.jax.phase_contact import JaxPhaseContactPolicy,affine_bank,replay_noise
    from scripts.cross_embodiment.run_refinement import checksum
    root=Path(root);groups={};code=code_fingerprint();saved=0;calls=0;padding=0;timings=[]
    for job in jobs:
        if job['role'] not in ('training_calibrated','training_nominal'):
            raise ValueError('GPU finite worker only accepts declared model training roles')
        if job['code']!=code['aggregate']:raise ValueError('GPU collection code changed')
        key=json.dumps([job['config'],job['mechanism'],job['prior']],sort_keys=True)
        groups.setdefault(key,[]).append(job)
    for key,group in groups.items():
        config,mechanism,prior_path=json.loads(key);config=ContactConfig(**config)
        runner=ContactRunner(config=config,mechanism=ContactMechanismSpec(**mechanism),backend='jax',device=device)
        prior=PhaseGaussianPrior(FrozenRegressor.load(prior_path,backend='jax',device=device))
        policy=JaxPhaseContactPolicy(runner,prior);dev=runner._impl.device
        for start in range(0,len(group),batch_size):
            logical=group[start:start+batch_size];pending=[]
            for job in logical:
                path=Path(job['path']);context=dict(request_sha256=checksum(job),role=job['role'],protocol=job.get('protocol','finite_refinement_v1'))
                if (path/'metadata.json').exists():
                    _,meta=load_bundle(path)
                    if meta.get('artifact_context')!=context or meta['code_fingerprint']['aggregate']!=code['aggregate'] or meta.get('physics_provider')!='mjx-jax':
                        raise ValueError('cached GPU episode provenance mismatch')
                    continue
                if (path/'attempt.json').exists():raise RuntimeError(f'incomplete charged GPU model attempt: {path}')
                pending.append(job)
            if not pending:continue
            fixed=pending+[pending[-1]]*(batch_size-len(pending));padding+=len(fixed)-len(pending)
            paths=[]
            for job in fixed:
                if job['parameters'] is None:paths.append(None);continue
                target=root/'_gpu_maps'/checksum(job['parameters'])
                if not (target/'metadata.json').exists():make_mapping(job['parameters']).save(target)
                paths.append(target)
            weights,biases=affine_bank(paths);seeds=[j['seed'] for j in fixed];modes=[j['mode'] for j in fixed]
            state=policy.reset_episodes(seeds,modes);memory=policy.initial_memory(modes)
            noise,weights,biases=jax.device_put((replay_noise(seeds,config.timeout_steps//10),weights,biases),dev)
            for job in pending:
                atomic_json(Path(job['path'])/'attempt.json',dict(request_sha256=checksum(job),status='pending',
                    started_unix=time.time(),role=job['role'],physics_provider='mjx-jax'))
            begin=time.monotonic()
            # Compilation embeds frozen prior/model constants. Subsequent
            # batches must execute without implicit device/host transfers.
            if start==0:
                result=policy.run_chunk(state,memory,noise,weights,biases);runner.synchronize(result)
            else:
                with jax.transfer_guard('disallow'):
                    result=policy.run_chunk(state,memory,noise,weights,biases);runner.synchronize(result)
            elapsed=time.monotonic()-begin;calls+=len(fixed)
            host=jax.device_get(result)
            for index,job in enumerate(pending):
                trace={n:np.asarray(getattr(host.trace,n)[index:index+1]) for n in host.trace._fields}
                phases=np.asarray(host.phases[:,index:index+1]);context=dict(request_sha256=checksum(job),role=job['role'],protocol=job.get('protocol','finite_refinement_v1'))
                report=gpu_report(trace,phases,host.state.status[index],host.state.steps[index],job,config,elapsed/len(fixed))
                save_bundle(job['path'],{**trace,'phase':phases,'reference':np.asarray(host.references[:,index:index+1]),
                    **{'event.'+n:np.asarray(v[index:index+1]) for n,v in host.events.items()}},
                    dict(**report,config=asdict(config),diagnostic_mechanism=mechanism,code_fingerprint=code,
                         artifact_context=context,role='calibrated/public model training; never target truth'))
                _,meta=load_bundle(job['path'])
                atomic_json(Path(job['path'])/'attempt.json',dict(request_sha256=checksum(job),status='complete',
                    tensor_sha256=meta['tensor_sha256'],role=job['role'],physics_provider='mjx-jax'))
                saved+=1
            timings.append(dict(group_sha256=checksum(key),start=start,logical=len(pending),physical=len(fixed),
                compile_and_run=start==0,seconds=elapsed,memory_stats=dev.memory_stats()))
            atomic_json(root/'_gpu_collection_progress.json',dict(saved=saved,new_physical_model_calls=calls,padding_model_calls=padding,batches=timings))
            print(dict(gpu_model_saved=saved,planned=len(jobs),batch_seconds=elapsed),flush=True)
    atomic_json(root/'gpu_collection_report.json',dict(complete=True,new_saved=saved,new_physical_model_calls=calls,
        padding_model_calls=padding,batches=timings,scope='model search only; separate CPU selected-map validation and CPU target evaluation',
        new_target_episodes=0,new_probes=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='gpu:0')
    args=p.parse_args();prepare(True)
    if args.batch_size<1:raise ValueError('positive GPU batch size required')
    import jax
    with jax.experimental.enable_x64(),jax.default_matmul_precision('highest'):
        collect_gpu(read(args.manifest),args.root,args.batch_size,args.device)


if __name__=='__main__':main()
