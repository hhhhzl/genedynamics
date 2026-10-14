"""Independent P1.6/P1.7 provenance/trace audit, tables and development figures."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from genedynamics.experiments.artifacts import atomic_json, artifact_write, POLICY_ENV
from genedynamics.learning.cross_embodiment.core.artifacts import load_bundle
from scripts.cross_embodiment.run_refinement import checksum, splits, read
from scripts.cross_embodiment.audit_validation_closure import inspect
from scripts.paper.cross_ebdm.report_exports import (
    add_report_export_argument, preflight_report_exports,
)


def audit(root):
    contract=read(root/'workflow_contract.json');code=contract['code']['aggregate']
    jobs={}
    for manifest in sorted((root/'_manifests').glob('*.json')):
        for job in read(manifest):
            if job['path'] in jobs and jobs[job['path']]!=job:raise ValueError('conflicting episode manifests')
            jobs[job['path']]=job
    errors=[];roles=Counter();event_checked=0;acceptance_checked=0
    for path,job in jobs.items():
        data,meta=load_bundle(path);path=Path(path);roles[job['role']]+=1
        expected=dict(request_sha256=checksum(job),role=job['role'],protocol=job.get('protocol','finite_refinement_v1'))
        if meta['artifact_context']!=expected or meta['code_fingerprint']['aggregate']!=code:
            errors.append(dict(path=str(path),error='collection/request provenance mismatch'))
        attempt=read(path/'attempt.json')
        if attempt.get('status')!='complete' or attempt.get('tensor_sha256')!=meta['tensor_sha256']:
            errors.append(dict(path=str(path),error='attempt ledger not completed'))
        valid=data['valid'][0].astype(bool);dt=meta['config']['control_dt']
        if int(valid.sum())!=round(meta['seconds']/dt) or int(data['status'][0,valid][-1])!=meta['status']:
            errors.append(dict(path=str(path),error='clock/terminal mismatch'))
        if 'event.loss_count' in data:
            lost=data['contact_lost'][0,valid];actual=data['event.loss_count'][0,valid];count=0
            for is_lost,value in zip(lost,actual):
                count=count+1 if is_lost else 0
                if count!=value:
                    errors.append(dict(path=str(path),error='loss counter mismatch'));break
            if meta['success'] and int(data['event.hold_count'][0,valid][-1])!=round(.3/dt):
                errors.append(dict(path=str(path),error='success hold count mismatch'))
            event_checked+=1
        if job['role']=='evaluation':
            row=inspect(path);acceptance_checked+=1
            if row['accepted']!=row['reported'] or not row['mode_classification_matches']:
                errors.append(dict(path=str(path),error='physical acceptance mismatch'))
        if job.get('mapping_path'):
            _,mapping=load_bundle(job['mapping_path'])
            if mapping['tensor_sha256']!=job['mapping_sha256']:
                errors.append(dict(path=str(path),error='frozen map changed'))
    probes=0
    for path in root.glob('level_*/seed_*/calibration.json'):
        calibration=read(path)
        if calibration.get('reused_from'):
            reference=calibration['reused_from'];ledger=read(reference['ledger_path'])
            if hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()!=reference['sha256'] or hashlib.sha256(Path(reference['ledger_path']).read_bytes()).hexdigest()!=reference['ledger_sha256']:
                errors.append(dict(path=str(path),error='reused probe provenance mismatch'))
        else:ledger=read(path.parent/'probes/ledger.json');probes+=len(ledger['attempts'])
        if calibration['budget']!=ledger or len(ledger['attempts'])!=5 or any(a['outcome']=='pending' for a in ledger['attempts']):
            errors.append(dict(path=str(path),error='probe ledger mismatch'))
    report=read(root/'scientific_report.json')
    for replicate in report['replicates']:
        folder=root/('level_'+replicate['suite'])/f'seed_{replicate["seed"]}'
        freeze=read(folder/'frozen.json')
        if freeze['code']!=code:errors.append(dict(path=str(folder),error='freeze code mismatch'))
        if 'calibration_reference' in freeze:
            reference=freeze['calibration_reference'];payload=Path(reference['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=reference['sha256']:
                errors.append(dict(path=str(folder),error='reused counted calibration changed'))
    roles_by_seed={}
    for seed in contract['config']['seeds']:
        for role,values in splits(seed,contract['config']['method_params']).items():
            for value in values:
                if value in roles_by_seed and roles_by_seed[value]!=role:errors.append(dict(seed=value,error='split leakage'))
                roles_by_seed[value]=role
    if report['actual_target_evaluation_episodes']!=roles.get('evaluation',0) or report['new_probes']!=probes:
        errors.append(dict(error='reported distinct target/probe count mismatch'))
    result=dict(passed=not errors,episodes=len(jobs),role_counts=dict(roles),event_traces_checked=event_checked,
                independently_accepted_target_traces=acceptance_checked,new_probes=probes,errors=errors,
                collection_code=code,analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='all planned traces, requests, counters, frozen map checksums and charged probes; no new simulations')
    atomic_json(root/'_reports/independent-audit.json',result)
    if errors:raise RuntimeError(f'{len(errors)} independent audit failures; inspect report')
    return result


def text_file(path,text):
    payload=text.encode()
    with artifact_write(path,len(payload)+4096):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)


def tables(root,report,paper_dir):
    filename='refinement_results.tex' if report['stage']=='1.6' else 'distribution_results.tex'
    if paper_dir is not None:
        paper_dir=preflight_report_exports(paper_dir,[Path('tables')/filename])
    lines=[f'# P{report["stage"]} development results','',
           '| Condition | Calibration | Method | Success / attempted | Executed | Mode matches | Mean task | Response |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in report['replicates']:
        for name,entry in r['results'].items():
            s=entry['summary'];response='NA' if s['mean_response_cost'] is None else f'{s["mean_response_cost"]:.6f}'
            lines.append(f'| {r["suite"]} | {r["seed"]} | {name} | {s["successes"]}/{s["attempted"]} | {s["executed"]} | {s["mode_matches"]} | {s["mean_task_cost"]:.6f} | {response} |')
    lines += ['',f'New probes: {report["new_probes"]}. Distinct target evaluations: {report["actual_target_evaluation_episodes"]}.',
              f'Transport incremental value: `{report["transport_incremental_value"]}`.',
              'Three independent calibration/optimization repeats per condition; modes are clustered within resets. Development evidence only.']
    text_file(root/'_reports/refinement-summary.md','\n'.join(lines)+'\n')
    if paper_dir is None:return
    # Explicit development-report exports are copied from audited reports.
    os.environ.pop(POLICY_ENV,None)
    names=list(report['totals']);suites=sorted({r['suite'] for r in report['replicates']})
    labels={'direct':'Direct','full':'Full','task_only':'Task-only','nominal':'Nominal',
            'reweight_only':'Reweight only','task_only_reweighted':'Task-only + reweight','full_reweighted':'Full + reweight',
            'shuffled_coupling':'Shuffled coupling','extra_noise':'Extra noise','no_source_mode':'No source mode'}
    latex=['% Generated from independently audited scientific_report.json.',
           '\\begin{center}\\small','\\begin{tabular}{l'+'r'*len(suites)+'}','\\toprule',
           'Method & '+' & '.join(s.replace('xarm7_','').capitalize() for s in suites)+'\\\\','\\midrule']
    for name in names:
        values=[]
        for suite in suites:
            entries=[r['results'][name]['summary'] for r in report['replicates'] if r['suite']==suite]
            values.append(f'{sum(e["successes"] for e in entries)}/{sum(e["attempted"] for e in entries)}')
        latex.append(labels.get(name,name.replace('_',' '))+' & '+' & '.join(values)+'\\\\')
    latex+=['\\bottomrule','\\end{tabular}','\\end{center}',
             f'本阶段新探针 {report["new_probes"]} 次，实际不同目标评估 {report["actual_target_evaluation_episodes"]} 条；别名方法与拒绝另报。']
    text_file(paper_dir/'tables'/filename,'\n'.join(latex)+'\n')


def figures(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suites=sorted({r['suite'] for r in report['replicates']})
    names=list(report['totals']);fig,axes=plt.subplots(1,len(suites),figsize=(12,4),sharey=True)
    for ax,suite in zip(np.atleast_1d(axes),suites):
        ratios=[]
        for name in names:
            rows=[r['results'][name]['summary'] for r in report['replicates'] if r['suite']==suite]
            ratios.append(sum(r['successes'] for r in rows)/sum(r['attempted'] for r in rows))
        ax.bar(np.arange(len(names)),ratios,color=['#245b78' if n=='full' else '#8eabba' for n in names])
        ax.set_xticks(np.arange(len(names)),[n.replace('_',' ') for n in names],rotation=50,ha='right')
        ax.set_title(suite);ax.set_ylim(0,1.08);ax.set_ylabel('Success / all requested episodes')
    fig.suptitle(f'P{report["stage"]}: development, 3 independent calibrations per condition')
    fig.tight_layout();target=root/'_reports/success.png'
    with artifact_write(target,4*1024**2):fig.savefig(target,dpi=180)
    plt.close(fig)
    if report['stage']!='1.6':return
    columns=[];labels=[]
    for r in report['replicates']:
        p=root/('level_'+r['suite'])/f'seed_{r["seed"]}'/'calibrated_search/search.json'
        history=read(p)['history']
        for mode in (0,1):
            columns.append([np.mean([e['success'] for e in h['episodes'] if e['mode']==mode]) for h in history])
            labels.append(f'{r["suite"].replace("xarm7_","")}\n{r["seed"]}/m{mode}')
    fig,ax=plt.subplots(figsize=(12,8))
    image=ax.imshow(np.array(columns).T,aspect='auto',vmin=0,vmax=1,cmap='Blues')
    ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_yticks(range(len(columns[0])),[f'c{i}' for i in range(len(columns[0]))])
    ax.set_title('Calibrated-model candidate feasibility: same pool for full and task-only')
    fig.colorbar(image,ax=ax,label='Training reset success fraction');fig.tight_layout()
    target=root/'_reports/candidate-feasibility.png'
    with artifact_write(target,4*1024**2):fig.savefig(target,dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    add_report_export_argument(p)
    args=p.parse_args();root=args.root.resolve()
    report=read(root/'scientific_report.json')
    if args.report_export_dir is not None:
        filename='refinement_results.tex' if report['stage']=='1.6' else 'distribution_results.tex'
        preflight_report_exports(args.report_export_dir,[Path('tables')/filename])
    os.environ[POLICY_ENV]=str(root/'artifact_policy.json')
    checked=audit(root)
    figures(root,report);tables(root,report,args.report_export_dir)
    print(json.dumps(dict(audit=checked,totals=report['totals']),indent=2))


if __name__=='__main__':main()
