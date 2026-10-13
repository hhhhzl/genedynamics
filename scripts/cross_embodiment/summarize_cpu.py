"""Aggregate latest unique CPU testcases and physical results without hiding failures."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from genedynamics.learning.cross_embodiment.data import code_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,default=Path('tmp/cross_embodiment/qualification_final'))
    p.add_argument('--artifacts',type=Path,default=Path('tmp/cross_embodiment'))
    p.add_argument('--minimum-tests',type=int,default=71);a=p.parse_args()
    cases={}
    for path in sorted(a.qualification.glob('*.xml'),key=lambda p:p.stat().st_mtime):
        for case in ET.parse(path).iter('testcase'):
            key=case.get('classname')+'::'+case.get('name')
            status='failed' if case.find('failure') is not None or case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
            cases[key]={'status':status,'junit':str(path)}
    physical={}
    for name in ('learned_closed_loop_v4','guided_learned_cpu_final','keyed_learned_cpu_final','ur5e_keyed_learned_cpu',
                 'mobile_keyed_learned_cpu_final','go2_z1_learned_cpu'):
        path=a.artifacts/name/'report.json'
        if path.exists():physical[name]=json.loads(path.read_text())['evaluation']
    guards={}
    for name in ('device_pipeline_transport_retry','contact_device_pipeline_final'):
        path=a.artifacts/name/'metadata.json'
        guards[name]=json.loads(path.read_text()) if path.exists() else {'transfer_guard':'not_run'}
    report={'code_fingerprint':code_fingerprint(),'testcases':cases,'unique_tests':len(cases),
        'counts':{status:sum(c['status']==status for c in cases.values()) for status in ('passed','failed','skipped')},
        'device_guards':guards,'physical_evaluation':physical,'gpu_qualification':'not_run',
        'scope':'CPU functional qualification; physical failures retained; no statistical method superiority claim'}
    report['functional_pass']=len(cases)>=a.minimum_tests and all(c['status']=='passed' for c in cases.values()) and all(g.get('transfer_guard')=='passed' for g in guards.values())
    (a.qualification/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print({'unique_tests':report['unique_tests'],'counts':report['counts'],'functional_pass':report['functional_pass']})
    return 0 if report['functional_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
