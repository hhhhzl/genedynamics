"""Serial CPU qualification with isolated tests, logs and reproducible provenance.

No GPU is selected or provisioned. Dataset generation and long training are
explicit opt-ins; existing artifact paths can be reused without target probes.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('tmp/cross_embodiment/qualification'))
    p.add_argument('--device-chunks',action='store_true',help='serial warmup and strict transfer guards')
    p.add_argument('--learned-loops',action='store_true',help='new charged probes, training and four baselines per task')
    p.add_argument('--artifacts',type=Path,default=Path('tmp/cross_embodiment'))
    p.add_argument('--skip-tests',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise SystemExit('Set JAX_PLATFORMS=cpu for CPU qualification')
    from genedynamics.learning.cross_embodiment.data import code_fingerprint
    report={'code_fingerprint':code_fingerprint(),'python':sys.version,'flags':{k:os.environ.get(k) for k in ('JAX_PLATFORMS','XLA_FLAGS','MUJOCO_GL')},
        'versions':{n:importlib.metadata.version(n) for n in ('jax','mujoco','torch','numpy','scipy','osqp')},'gpu':'not_run','stages':[]}
    def stage(name,args):
        start=time.perf_counter(); logfile=a.output/(name+'.log')
        with logfile.open('w') as stream:
            result=subprocess.run([sys.executable,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
        row={'name':name,'command':[sys.executable,*map(str,args)],'exit_code':result.returncode,'seconds':time.perf_counter()-start,'log':str(logfile)}
        report['stages'].append(row); (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{name}: {"PASS" if result.returncode==0 else "FAIL"} ({row["seconds"]:.1f}s)',flush=True)
    # A process per module releases compiled MJX executables before the next
    # topology. This also avoids interpreting concurrent-compiler OOM as a bug.
    if not a.skip_tests:
        for module in sorted(Path('test/cross_embodiment').glob('test_*.py')):
            stage(module.stem,['-m','pytest','-q',module,'--junitxml='+str(a.output/(module.stem+'.xml'))])
    if a.device_chunks:
        stage('obstacle_device',['-m','scripts.cross_embodiment.cpu_device_pipeline_smoke','--models',a.artifacts/'learned_cpu_v2',
            '--transport',a.artifacts/'learned_closed_loop_v4/transport','--output',a.output/'obstacle_device'])
        stage('contact_device',['-m','scripts.cross_embodiment.cpu_contact_device_smoke','--models',a.artifacts/'guided_models_cpu_final',
            '--transport',a.artifacts/'guided_learned_v3/transport','--output',a.output/'contact_device'])
    if a.learned_loops:
        for task,models,data in [('guided_contact_alignment','guided_models_cpu_final','guided_response_v3'),
                ('keyed_insertion_recovery','keyed_models_cpu_final','keyed_response_v3'),
                ('mobile_keyed_insertion_recovery','keyed_models_cpu_mobile','mobile_keyed_response_v4')]:
            stage(task,['-m','scripts.cross_embodiment.cpu_contact_learned_loop','--task',task,'--models',a.artifacts/models,
                '--response-data',a.artifacts/data,'--output',a.output/task])
    report['functional_pass']=all(s['exit_code']==0 for s in report['stages']) and bool(report['stages'])
    report['scope']='software CPU qualification; learned task success and failure counts remain in each task report; not research superiority'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['functional_pass'] else 1


if __name__=='__main__':sys.exit(main())
