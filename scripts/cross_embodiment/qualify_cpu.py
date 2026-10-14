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
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modules',nargs='+',help='explicit module names; default is the current complete suite')
    p.add_argument('--timeout',type=int,default=3600,help='per-module timeout in seconds')
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
        from genedynamics.experiments.process import run_stage
        from genedynamics.experiments.artifacts import atomic_json
        row=run_stage(name,[sys.executable,*map(str,args)],a.output,timeout=a.timeout)
        report['stages'].append(row)
        atomic_json(a.output/'report.json',report)
    # A process per module releases compiled MJX executables before the next
    # topology. This also avoids interpreting concurrent-compiler OOM as a bug.
    if not a.skip_tests:
        (a.output/'temporary').mkdir(parents=True,exist_ok=True)
        modules=sorted(Path('test/cross_embodiment').glob('test_*.py'))
        if a.modules:
            unknown=set(a.modules)-{m.stem for m in modules}
            if unknown:raise SystemExit(f'unknown modules: {sorted(unknown)}')
            modules=[m for m in modules if m.stem in a.modules]
        report['modules']=[m.stem for m in modules]
        for module in modules:
            stage(module.stem,['-m','pytest','-q',module,'--junitxml='+str(a.output/(module.stem+'.xml')),
                  '--basetemp='+str(a.output/'temporary'/module.stem)])
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
    import xml.etree.ElementTree as ET
    totals=dict(tests=0,failures=0,errors=0,skipped=0)
    for path in a.output.glob('test_*.xml'):
        if a.modules and path.stem not in a.modules:continue
        for suite in ET.parse(path).getroot().iter('testsuite'):
            for key in totals:totals[key]+=int(suite.get(key,0))
    report['test_counts']=totals
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['functional_pass'] else 1


if __name__=='__main__':sys.exit(main())
