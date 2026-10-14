"""Replay frozen single-card GPU model fixtures on another identical card."""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from scripts.cross_embodiment.run_p2 import prepare,read,digest
    prepare(True)
    import os,subprocess,hashlib
    import numpy as np
    import jax
    from genedynamics.experiments.artifacts import atomic_json
    from genedynamics.learning.cross_embodiment.contact import ContactRunner,ContactMechanismSpec,mating_contact_config
    from genedynamics.learning.cross_embodiment.contact_behavior import PhaseGaussianPrior
    from genedynamics.learning.cross_embodiment.learned import FrozenRegressor
    from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    from genedynamics.learning.cross_embodiment.backends.jax.phase_contact import JaxPhaseContactPolicy,affine_bank,replay_noise
    contract=read(args.root/'workflow_contract.json');current=code_fingerprint()['files']
    if any(current[n]!=h for n,h in contract['code']['files'].items() if n.startswith('genedynamics/')):
        raise ValueError('qualified numerical implementation changed')
    cases=[c for c in contract['cases'] if c['robot']=='xarm7' and c['gain']==1.]
    batch=contract['config']['method_params']['qualification_batch'];padded=(cases*((batch+len(cases)-1)//len(cases)))[:batch]
    with jax.experimental.enable_x64(),jax.default_matmul_precision('highest'):
        runner=ContactRunner(config=mating_contact_config('xarm7'),mechanism=ContactMechanismSpec(),backend='jax',device='gpu:0')
        prior=PhaseGaussianPrior(FrozenRegressor.load(contract['prior']['path'],backend='jax',device='gpu:0'))
        policy=JaxPhaseContactPolicy(runner,prior);seeds=[c['seed'] for c in padded];modes=[c['mode'] for c in padded]
        state=policy.reset_episodes(seeds,modes);memory=policy.initial_memory(modes)
        weights,bias=affine_bank([c['mapping'] for c in padded]);noise,weights,bias=jax.device_put((replay_noise(seeds,75),weights,bias),runner._impl.device)
        t=time.monotonic();first=policy.run_chunk(state,memory,noise,weights,bias);runner.synchronize(first);cold=time.monotonic()-t
        t=time.monotonic()
        with jax.transfer_guard('disallow'):
            second=policy.run_chunk(state,memory,noise,weights,bias);runner.synchronize(second)
        warm=time.monotonic()-t;host=jax.device_get(second);rows=[]
        for i,case in enumerate(cases):
            data,meta=load_bundle(args.root/'gpu_fixtures'/case['name'])
            errors=[float(np.max(np.abs(np.asarray(getattr(host.trace,n)[i:i+1]).astype(float)-data[n].astype(float)),initial=0.)) for n in host.trace._fields]
            errors+=[float(np.max(np.abs(np.asarray(host.events[n][i:i+1]).astype(float)-data['event.'+n].astype(float)),initial=0.)) for n in host.events]
            errors.append(float(np.max(np.abs(np.asarray(host.references[:,i:i+1])-data['reference']))))
            rows.append(dict(name=case['name'],max_error=max(errors),status=int(host.state.status[i]),
                reference_status=meta['status'],passed=max(errors)<=1e-10 and int(host.state.status[i])==meta['status']))
    result=dict(passed=all(r['passed'] for r in rows),records=rows,cold_seconds=cold,warm_seconds=warm,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],device_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv'],text=True),
        qualification_contract_sha256=digest(args.root/'workflow_contract.json'),new_model_replay_episodes=2*batch,
        new_target_episodes=0,new_probes=0,analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='identical-card model replay; independent single-card episode scheduling, no distributed learner qualification')
    atomic_json(args.output,result)
    if not result['passed']:raise RuntimeError('second physical GPU replay differs')
    print(dict(passed=result['passed'],cases=len(rows),max_error=max(r['max_error'] for r in rows)))


if __name__=='__main__':main()
